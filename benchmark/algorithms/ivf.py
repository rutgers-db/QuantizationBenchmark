"""
IVF (Inverted File Index) wrapper for quantization algorithms.

This module implements IVF+Quantization similar to FAISS's IVFPQ approach:
1. Use k-means to partition the dataset into nlist coarse clusters
2. Assign vectors to buckets and compute residuals (vector - centroid)
3. Train a separate quantizer for each cluster on its residual vectors
4. At search time, scan nprobe buckets and use each bucket's quantizer for search

Usage:
    This module wraps around existing quantization algorithms to add IVF functionality.
"""

import numpy as np
import time
from typing import Tuple, Optional, Any, List
import faiss


class IVFWrapper:
    """
    IVF wrapper that adds inverted file index functionality to any quantizer.

    The workflow is:
    1. Build phase:
       - Train k-means on the dataset to get nlist centroids
       - Assign each vector to its nearest centroid
       - For each cluster, compute residuals and train a separate quantizer
       - Store one quantizer per cluster

    2. Search phase:
       - For each query, find nprobe nearest centroids
       - For each of these buckets:
           * Compute query residual (query - centroid)
           * Use that bucket's quantizer to search
       - Merge and sort results across all probed buckets
    """

    def __init__(self, quantizer_class, nlist: int, **quantizer_params):
        """
        Initialize IVF wrapper.

        Args:
            quantizer_class: The underlying quantizer class (e.g., ProductQuantization)
            nlist: Number of inverted lists (coarse clusters)
            **quantizer_params: Parameters to pass to each cluster's quantizer
        """
        self.quantizer_class = quantizer_class
        self.nlist = nlist
        self.quantizer_params = quantizer_params

        # IVF structures
        self.coarse_quantizer = None       # K-means centroids (numpy array)
        self.coarse_index = None           # FAISS index for coarse search (reused in query)
        self.inverted_lists = None         # List of numpy arrays: inverted_lists[i] contains indices in bucket i
        self.quantizers = []               # List of quantizers, one per cluster

        # Dataset info
        self.train_data = None        # Keep reference to original training data for reranking
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

        self.train_data = data.copy()  # Keep a copy for reranking
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

        # Build inverted lists
        self.inverted_lists = [[] for _ in range(self.nlist)]
        for vec_id, cluster_id in enumerate(assignments):
            self.inverted_lists[cluster_id].append(vec_id)

        # Convert to numpy arrays for efficient indexing
        self.inverted_lists = [np.array(lst, dtype=np.int32) for lst in self.inverted_lists]

        assign_time = time.time() - start_time
        print(f"  Assignment time: {assign_time:.4f}s")

        # Print bucket statistics
        bucket_sizes = [len(lst) for lst in self.inverted_lists]
        print(f"  Bucket size statistics:")
        print(f"    Min: {np.min(bucket_sizes)}, Max: {np.max(bucket_sizes)}")
        print(f"    Mean: {np.mean(bucket_sizes):.1f}, Median: {np.median(bucket_sizes):.1f}")
        print(f"    Empty buckets: {sum(1 for s in bucket_sizes if s == 0)}")

        # Step 3: For each cluster, compute residuals and train a quantizer
        print(f"\nStep 3: Training quantizer for each cluster...")
        quantizer_train_start = time.time()

        self.quantizers = []
        successful_clusters = 0
        empty_clusters = 0

        for cluster_id in range(self.nlist):
            if len(self.inverted_lists[cluster_id]) == 0:
                # Empty cluster - store None
                self.quantizers.append(None)
                empty_clusters += 1
                continue

            # Get vectors in this cluster
            vec_ids = self.inverted_lists[cluster_id]
            cluster_vectors = data[vec_ids]
            centroid = self.coarse_quantizer[cluster_id]

            # Compute residuals
            residuals = cluster_vectors - centroid

            # Train quantizer on residuals
            try:
                quantizer = self.quantizer_class(**self.quantizer_params)
                success = quantizer.fit(len(vec_ids), residuals)

                if success:
                    self.quantizers.append(quantizer)
                    successful_clusters += 1
                else:
                    print(f"  Warning: Quantizer training failed for cluster {cluster_id}")
                    self.quantizers.append(None)
            except Exception as e:
                print(f"  Warning: Exception training quantizer for cluster {cluster_id}: {e}")
                self.quantizers.append(None)

        quantizer_train_time = time.time() - quantizer_train_start
        print(f"  Quantizer training time: {quantizer_train_time:.4f}s")
        print(f"  Successfully trained: {successful_clusters}/{self.nlist} clusters")
        print(f"  Empty clusters: {empty_clusters}")

        total_time = kmeans_time + assign_time + quantizer_train_time
        print(f"\nTotal IVF build time: {total_time:.4f}s")
        print(f"{'='*60}\n")

        return successful_clusters > 0

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
            **kwargs: Additional parameters (unused, for compatibility)

        Returns:
            Tuple of (I, D) where:
                I: Indices of nearest neighbors, shape (nq, topk)
                D: Distances to nearest neighbors, shape (nq, topk)
        """
        print(f"\nIVF Query: nq={nq}, topk={topk}, nprobe={nprobe}")

        if nprobe > self.nlist:
            print(f"  Warning: nprobe ({nprobe}) > nlist ({self.nlist}), using nprobe = {self.nlist}")
            nprobe = self.nlist

        # Find nprobe nearest centroids for each query using stored coarse_index
        _, probe_ids = self.coarse_index.search(queries.astype(np.float32), nprobe)  # Shape: (nq, nprobe)

        # For each query, search within probed buckets
        all_I = np.full((nq, topk), -1, dtype=np.int32)
        all_D = np.full((nq, topk), np.inf, dtype=np.float32)

        for q_idx in range(nq):
            # Collect candidates from all probed buckets
            all_candidates_ids = []
            all_candidates_dists = []

            for probe_idx in range(nprobe):
                cluster_id = probe_ids[q_idx, probe_idx]

                # Skip empty clusters or clusters without quantizer
                if len(self.inverted_lists[cluster_id]) == 0 or self.quantizers[cluster_id] is None:
                    continue

                # Get vectors in this bucket
                vec_ids = self.inverted_lists[cluster_id]
                n_bucket = len(vec_ids)

                # Compute query residual
                query_residual = queries[q_idx] - self.coarse_quantizer[cluster_id]

                # Use this cluster's quantizer to search
                # Call quantizer.query with the residual query
                # We need to search within this bucket, so we query with k = min(topk, n_bucket)
                k_search = min(topk * nprobe, n_bucket)  # Search more candidates per bucket

                try:
                    # Query the quantizer - it will search among its trained vectors
                    I_bucket, D_bucket = self.quantizers[cluster_id].query(
                        1,  # Single query
                        query_residual.reshape(1, -1),
                        k_search
                    )

                    # Map bucket-local indices back to global indices
                    global_ids = vec_ids[I_bucket[0]]
                    distances = D_bucket[0]

                    all_candidates_ids.append(global_ids)
                    all_candidates_dists.append(distances)

                except Exception as e:
                    # If quantizer.query fails, fall back to brute force on residuals
                    print(f"  Warning: Quantizer query failed for cluster {cluster_id}, using fallback: {e}")
                    # Compute residuals for all vectors in bucket
                    bucket_vectors = self.train_data[vec_ids]
                    centroid = self.coarse_quantizer[cluster_id]
                    bucket_residuals = bucket_vectors - centroid

                    # Compute L2 distances
                    distances = np.sum((bucket_residuals - query_residual) ** 2, axis=1)

                    all_candidates_ids.append(vec_ids)
                    all_candidates_dists.append(distances)

            # Merge results from all probed buckets
            if len(all_candidates_ids) > 0:
                all_cand_ids = np.concatenate(all_candidates_ids)
                all_cand_dists = np.concatenate(all_candidates_dists)

                # Get top-k
                k = min(topk, len(all_cand_dists))
                if k > 0:
                    top_indices = np.argpartition(all_cand_dists, k-1)[:k]
                    top_indices = top_indices[np.argsort(all_cand_dists[top_indices])]

                    all_I[q_idx, :k] = all_cand_ids[top_indices]
                    all_D[q_idx, :k] = all_cand_dists[top_indices]

        return all_I, all_D

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
        Search with reranking using each quantizer's searchAndRerank method.

        Args:
            nq: Number of queries
            queries: Query vectors
            topk: Final number of results to return
            nrerank: Number of candidates to retrieve from each quantizer before reranking
            nprobe: Number of inverted lists to probe
            **kwargs: Additional parameters

        Returns:
            Tuple of (I, D) with reranked results
        """
        print(f"\nIVF SearchAndRerank: nq={nq}, topk={topk}, nrerank={nrerank}, nprobe={nprobe}")

        if nprobe > self.nlist:
            print(f"  Warning: nprobe ({nprobe}) > nlist ({self.nlist}), using nprobe = {self.nlist}")
            nprobe = self.nlist

        # Find nprobe nearest centroids for each query
        _, probe_ids = self.coarse_index.search(queries.astype(np.float32), nprobe)  # Shape: (nq, nprobe)

        # For each query, collect reranked results from all probed buckets
        all_I = np.full((nq, topk), -1, dtype=np.int32)
        all_D = np.full((nq, topk), np.inf, dtype=np.float32)

        for q_idx in range(nq):
            # Collect candidates from all probed buckets
            all_candidates_ids = []
            all_candidates_dists = []

            for probe_idx in range(nprobe):
                cluster_id = probe_ids[q_idx, probe_idx]

                # Skip empty clusters or clusters without quantizer
                if len(self.inverted_lists[cluster_id]) == 0 or self.quantizers[cluster_id] is None:
                    continue

                # Get vectors in this bucket
                vec_ids = self.inverted_lists[cluster_id]
                n_bucket = len(vec_ids)

                # Compute query residual
                query_residual = queries[q_idx] - self.coarse_quantizer[cluster_id]

                # Use this cluster's quantizer's searchAndRerank method
                k_search = min(nrerank, n_bucket)

                try:
                    # Call quantizer's searchAndRerank - it will get candidates and rerank internally
                    I_bucket, D_bucket = self.quantizers[cluster_id].searchAndRerank(
                        1,  # Single query
                        query_residual.reshape(1, -1),
                        topk=min(topk, n_bucket),  # We want topk results from this bucket
                        nrerank=k_search  # But search through nrerank candidates
                    )

                    # Map bucket-local indices back to global indices
                    global_ids = vec_ids[I_bucket[0]]

                    # Note: distances from searchAndRerank are already exact distances in residual space
                    # We need to add back the coarse quantization error, but for ranking purposes
                    # within a bucket, the residual distances are sufficient
                    distances = D_bucket[0]

                    all_candidates_ids.append(global_ids)
                    all_candidates_dists.append(distances)

                except Exception as e:
                    # If quantizer doesn't support searchAndRerank, fall back to query + rerank
                    print(f"  Warning: SearchAndRerank failed for cluster {cluster_id}, using fallback: {e}")

                    try:
                        # Try regular query first
                        I_bucket, D_bucket = self.quantizers[cluster_id].query(
                            1,
                            query_residual.reshape(1, -1),
                            k_search
                        )

                        # Manual rerank using original vectors
                        bucket_indices = vec_ids[I_bucket[0]]
                        valid_mask = (bucket_indices >= 0) & (bucket_indices < self.ntrain)
                        bucket_indices = bucket_indices[valid_mask]

                        if len(bucket_indices) > 0:
                            # Compute exact distances to original vectors
                            candidate_vectors = self.train_data[bucket_indices]
                            exact_distances = np.sum((candidate_vectors - queries[q_idx]) ** 2, axis=1)

                            # Take top results
                            k = min(topk, len(exact_distances))
                            top_indices = np.argsort(exact_distances)[:k]

                            all_candidates_ids.append(bucket_indices[top_indices])
                            all_candidates_dists.append(exact_distances[top_indices])
                    except Exception as e2:
                        print(f"  Warning: Fallback also failed for cluster {cluster_id}: {e2}")
                        continue

            # Merge results from all probed buckets
            if len(all_candidates_ids) > 0:
                all_cand_ids = np.concatenate(all_candidates_ids)
                all_cand_dists = np.concatenate(all_candidates_dists)

                # Get top-k from merged results
                k = min(topk, len(all_cand_dists))
                if k > 0:
                    top_indices = np.argpartition(all_cand_dists, k-1)[:k]
                    top_indices = top_indices[np.argsort(all_cand_dists[top_indices])]

                    all_I[q_idx, :k] = all_cand_ids[top_indices]
                    all_D[q_idx, :k] = all_cand_dists[top_indices]

        return all_I, all_D

    def getMemoryUsage(self) -> int:
        """
        Get total memory usage in bytes.

        Returns:
            Total memory usage (coarse quantizer + inverted lists + all cluster quantizers)
        """
        memory = 0

        # Coarse quantizer (centroids)
        if self.coarse_quantizer is not None:
            memory += self.coarse_quantizer.nbytes

        # Inverted lists (just the indices)
        if self.inverted_lists is not None:
            for lst in self.inverted_lists:
                memory += lst.nbytes

        # All cluster quantizers
        if self.quantizers is not None:
            for quantizer in self.quantizers:
                if quantizer is not None:
                    memory += quantizer.getMemoryUsage()

        return memory

    def getCompressionRate(self) -> float:
        """
        Get compression rate.

        Returns:
            Compression rate (original_size / compressed_size)
        """
        # Original size: ntrain * ndim * 4 bytes (float32)
        original_size = self.ntrain * self.ndim * 4

        # Compressed size: our memory usage
        compressed_size = self.getMemoryUsage()

        if compressed_size == 0:
            return 1.0

        return original_size / compressed_size

    def getMSE(self) -> float:
        """
        Get mean squared error of reconstruction.

        Returns:
            Average MSE across all clusters
        """
        if self.quantizers is None or len(self.quantizers) == 0:
            return 0.0

        # Average MSE across all non-None quantizers
        mse_sum = 0.0
        count = 0
        for quantizer in self.quantizers:
            if quantizer is not None:
                mse_sum += quantizer.getMSE()
                count += 1

        return mse_sum / count if count > 0 else 0.0
