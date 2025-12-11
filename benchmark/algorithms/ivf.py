"""
IVF (Inverted File Index) wrapper for quantization algorithms.

This module implements IVF+Quantization similar to FAISS's IVFPQ approach:
1. Use k-means to partition the dataset into nlist coarse clusters
2. Assign vectors to their nearest centroids
3. Train a unified IVF quantizer (BaseIVFQuantizer) on all data with assignment information
4. At search time, find nprobe nearest centroids and search using the IVF quantizer

Usage:
    This module wraps around BaseIVFQuantizer instances.
"""

import numpy as np
import time
from typing import Tuple
import faiss


class IVFWrapper:
    """
    IVF wrapper that adds inverted file index functionality using BaseIVFQuantizer.

    The workflow is:
    1. Build phase:
       - Train k-means on the dataset to get nlist centroids
       - Assign each vector to its nearest centroid
       - Call ivf_quantizer.fit(data, assignments, centroids)

    2. Search phase:
       - For each query, find nprobe nearest centroids (assignments)
       - Call ivf_quantizer.query(queries, assignments, topk, ...)
    """

    def __init__(self, ivf_quantizer, nlist: int):
        """
        Initialize IVF wrapper.

        Args:
            ivf_quantizer: BaseIVFQuantizer instance (handles quantization with IVF structure)
            nlist: Number of inverted lists (coarse clusters)
        """
        self.ivf_quantizer = ivf_quantizer
        self.nlist = nlist

        # IVF structures
        self.coarse_quantizer = None  # K-means centroids (numpy array)
        self.coarse_index = None      # FAISS index for coarse search (reused in query)

        # Dataset info
        self.ndim = None
        self.ntrain = None

    def fit(self, nd: int, data: np.ndarray) -> bool:
        """
        Build the IVF index.

        Args:
            nd: Number of vectors
            data: Training data array, shape (nd, dim)

        Returns:
            bool: True if successful
        """
        print(f"\n{'='*60}")
        print(f"IVF Build Phase")
        print(f"{'='*60}")
        print(f"Training data: {nd} vectors of dimension {data.shape[1]}")
        print(f"Number of inverted lists (nlist): {self.nlist}")

        self.ntrain = nd
        self.ndim = data.shape[1]

        # Step 1: Train coarse quantizer (k-means) to partition data
        print(f"\nStep 1: Training coarse quantizer (k-means with {self.nlist} centroids)...")
        start_time = time.time()

        # Use FAISS for efficient k-means
        kmeans = faiss.Kmeans(
            d=self.ndim,
            k=self.nlist,
            niter=25,          # Number of k-means iterations
            verbose=False,
            seed=1234
        )
        kmeans.train(data.astype(np.float32))
        self.coarse_quantizer = kmeans.centroids  # Shape: (nlist, dim)

        # Build and store coarse index for reuse in query phase
        self.coarse_index = faiss.IndexFlatL2(self.ndim)
        self.coarse_index.add(self.coarse_quantizer)

        kmeans_time = time.time() - start_time
        print(f"  K-means training time: {kmeans_time:.4f}s")
        print(f"  Coarse centroids shape: {self.coarse_quantizer.shape}")

        # Step 2: Assign each training vector to nearest centroid
        print(f"\nStep 2: Assigning vectors to inverted lists...")
        start_time = time.time()

        # Find nearest centroid for each training vector
        _, assignments = self.coarse_index.search(data.astype(np.float32), 1)  # Shape: (nd, 1)
        assignments = assignments.flatten()  # Shape: (nd,)

        assign_time = time.time() - start_time
        print(f"  Assignment time: {assign_time:.4f}s")

        # Print bucket statistics
        unique, counts = np.unique(assignments, return_counts=True)
        print(f"  Bucket size statistics:")
        print(f"    Min: {np.min(counts)}, Max: {np.max(counts)}")
        print(f"    Mean: {np.mean(counts):.1f}, Median: {np.median(counts):.1f}")
        print(f"    Empty buckets: {self.nlist - len(unique)}")

        # Step 3: Train IVF quantizer
        print(f"\nStep 3: Training IVF quantizer...")
        start_time = time.time()

        success = self.ivf_quantizer.fit(data, assignments, self.coarse_quantizer)
        quantizer_train_time = time.time() - start_time

        if not success:
            print("  ERROR: IVF quantizer training failed!")
            return False

        print(f"  IVF quantizer training time: {quantizer_train_time:.4f}s")

        total_time = kmeans_time + assign_time + quantizer_train_time
        print(f"\nTotal IVF build time: {total_time:.4f}s")
        print(f"{'='*60}\n")

        return True

    def query(
        self,
        nq: int,
        queries: np.ndarray,
        topk: int,
        nprobe: int = 1,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search the IVF index.

        Args:
            nq: Number of queries
            queries: Query vectors, shape (nq, dim)
            topk: Number of nearest neighbors to return
            nprobe: Number of inverted lists to probe
            **kwargs: Additional parameters

        Returns:
            Tuple of (I, D) where:
                I: Indices of nearest neighbors, shape (nq, topk)
                D: Distances to nearest neighbors, shape (nq, topk)
        """
        if nprobe > self.nlist:
            print(f"  Warning: nprobe ({nprobe}) > nlist ({self.nlist}), using nprobe = {self.nlist}")
            nprobe = self.nlist

        # Find nprobe nearest centroids for each query
        _, assignments = self.coarse_index.search(queries.astype(np.float32), nprobe)  # Shape: (nq, nprobe)

        # Call IVF quantizer's query method
        result = self.ivf_quantizer.query(queries, assignments, topk, **kwargs)
        return result

    def searchAndRerank(
        self,
        nq: int,
        queries: np.ndarray,
        topk: int,
        nrerank: int,
        nprobe: int = 1,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search with reranking.

        Args:
            nq: Number of queries
            queries: Query vectors
            topk: Final number of results to return
            nrerank: Number of candidates to retrieve before reranking
            nprobe: Number of inverted lists to probe
            **kwargs: Additional parameters

        Returns:
            Tuple of (I, D) with reranked results
        """
        if nprobe > self.nlist:
            print(f"  Warning: nprobe ({nprobe}) > nlist ({self.nlist}), using nprobe = {self.nlist}")
            nprobe = self.nlist

        # Find nprobe nearest centroids for each query
        _, assignments = self.coarse_index.search(queries.astype(np.float32), nprobe)  # Shape: (nq, nprobe)

        # Call IVF quantizer's searchAndRerank method
        result = self.ivf_quantizer.searchAndRerank(queries, assignments, topk, nrerank, **kwargs)
        return result

    def getMemoryUsage(self) -> int:
        """
        Get total memory usage in bytes.

        Returns:
            Total memory usage (coarse quantizer + IVF quantizer)
        """
        memory = 0

        # Coarse quantizer (centroids)
        if self.coarse_quantizer is not None:
            memory += self.coarse_quantizer.nbytes

        # IVF quantizer
        memory += self.ivf_quantizer.getMemoryUsage()

        return memory

    def getCompressionRate(self) -> float:
        """
        Get compression rate.

        Formula:
            original_size / (quantizer_compressed_size + coarse_index_size)

        Returns:
            Compression rate (original_size / compressed_size)
        """
        # Original size: ntrain * ndim * 4 bytes (float32)
        original_size = self.ntrain * self.ndim * 4

        # Get compression rate from IVF quantizer
        quantizer_compression_rate = self.ivf_quantizer.getCompressionRate()

        # Compressed size from quantizer
        quantizer_compressed_size = original_size / quantizer_compression_rate

        # Coarse index size: nlist centroids * ndim * 4 bytes
        coarse_index_size = self.nlist * self.ndim * 4

        # Total compressed size
        total_compressed_size = quantizer_compressed_size + coarse_index_size

        return original_size / total_compressed_size

    def getMSE(self) -> float:
        """
        Get mean squared error of reconstruction.

        Returns:
            MSE value from IVF quantizer
        """
        return self.ivf_quantizer.getMSE()
