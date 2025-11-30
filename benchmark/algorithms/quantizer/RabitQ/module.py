import numpy as np
from typing import Tuple
import psutil
import sys
import os

# Add benchmark to path for importing BaseQuantizer
sys.path.insert(0, '/benchmark')
from benchmark.base import BaseQuantizer

# Import the C++ module
try:
    import rabitq_cpp
except ImportError as e:
    print(f"Warning: Could not import rabitq_cpp: {e}")
    rabitq_cpp = None


class RabitQ(BaseQuantizer):
    """
    RaBitQ (Randomized Bit Quantization) implementation using C++ backend.
    Uses IVF with C=1 for brute-force search with optimized distance computation.
    """

    def __init__(self, ndim, data_bytes, nthread=1, space="l2"):
        """
        Initialize RaBitQ quantizer.

        Args:
            ndim: Dimensionality of vectors
            data_bytes: Size of data type in bytes (typically 4 for float32)
            nthread: Number of threads to use for parallel processing
            space: Distance metric ("l2" for Euclidean)
        """
        super().__init__()
        self.ndim = ndim
        self.data_bytes = data_bytes
        self.nthread = nthread
        self.space = space

        # Round B up to multiple of 64
        self.b_dim = ((ndim + 63) // 64) * 64

        self.data = None
        self.ndata = 0
        self.trained = False
        self.norms = None

        # RaBitQ preprocessing results
        self.projection_matrix = None
        self.binary_codes = None
        self.x0_values = None
        self.dist_to_centroid = None
        self.randomized_centroid = None

        # C++ index (required)
        self.cpp_index = None

        # Select appropriate C++ class based on dimensions
        if rabitq_cpp is None:
            raise RuntimeError(
                "C++ module 'rabitq_cpp' is not available. "
                "RabitQ requires the C++ implementation for efficient search. "
                "Make sure the C++ module was built correctly in the Docker image."
            )

        if ndim == 128 and self.b_dim == 128:
            self.cpp_index = rabitq_cpp.PyIVFRN_128_128()
        elif ndim == 960 and self.b_dim == 960:
            self.cpp_index = rabitq_cpp.PyIVFRN_960_960()
        else:
            raise RuntimeError(
                f"No C++ implementation available for dim={ndim}, b_dim={self.b_dim}. "
                f"Supported configurations: (128, 128) or (960, 960)"
            )

    def _orthogonal_matrix(self, size):
        """Generate an orthogonal matrix using QR decomposition."""
        G = np.random.randn(size, size).astype('float32')
        Q, _ = np.linalg.qr(G)
        return Q.T  # Transpose to match RaBitQ convention

    def fit(self, nd: int, data: np.ndarray) -> bool:
        """
        Train the RaBitQ quantizer on the given data.

        Args:
            nd: Number of data vectors
            data: Training data of shape (nd, d) where d is the dimensionality

        Returns:
            bool: True if training was successful, False otherwise
        """
        try:
            self.data = np.ascontiguousarray(data, dtype=np.float32)
            self._original_data = self.data  # For default search_and_rerank
            self.ndata = nd

            # Pad data to b_dim
            max_bd = max(self.ndim, self.b_dim)
            data_pad = np.pad(self.data, ((0, 0), (0, max_bd - self.ndim)), 'constant').astype('float32')

            # Set random seed for reproducibility
            np.random.seed(0)

            # Generate orthogonal projection matrix
            import time
            t0 = time.time()
            P = self._orthogonal_matrix(max_bd)
            self.projection_matrix = P

            # For C=1, we use a single centroid which is the mean of the dataset
            # This is critical: centroid MUST be the dataset mean
            self.centroid_orig = np.mean(self.data, axis=0, keepdims=True).astype('float32')

            centroid_pad = np.pad(self.centroid_orig, ((0, 0), (0, max_bd - self.ndim)), 'constant').astype('float32')

            # Project data and centroid
            t0 = time.time()
            XP = data_pad @ P  # (nd, max_bd)
            t_matmul = time.time() - t0

            CP = centroid_pad @ P  # (1, max_bd)
            self.randomized_centroid = CP

            # All data points belong to cluster 0
            cluster_id = np.zeros(nd, dtype=np.uint32)

            # Compute distance to centroid (before projection)
            dist_to_c = np.linalg.norm(self.data - self.centroid_orig, axis=1).astype('float32')
            self.dist_to_centroid = dist_to_c

            # Subtract centroid from projected data
            XP_residual = XP - CP[0]  # (nd, max_bd)

            # Generate binary codes (first b_dim dimensions)
            bin_XP = (XP_residual[:, :self.b_dim] > 0).astype(np.bool_)
            self.bin_XP = bin_XP

            # Compute x0 values
            # x0 = sum(XP * sign(bin_XP) / sqrt(B)) / ||XP||
            x0 = np.sum(
                XP_residual[:, :self.b_dim] * (2 * bin_XP - 1) / np.sqrt(self.b_dim),
                axis=1,
                keepdims=True
            ) / (np.linalg.norm(XP_residual, axis=1, keepdims=True) + 1e-10)

            # Handle ill-defined x0
            x0[~np.isfinite(x0)] = 0.8
            self.x0_values = x0.flatten().astype('float32')

            # Pack binary codes into uint64
            # Flatten and pack
            bin_XP_flat = bin_XP.flatten()
            num_uint64 = self.b_dim // 64

            # Pack bits into uint64 (little-endian bit packing)
            binary_codes = np.packbits(bin_XP_flat.reshape(-1, 8, 8)[:, ::-1]).view(np.uint64)
            binary_codes = binary_codes.reshape(nd, num_uint64)
            self.binary_codes = binary_codes

            # Build C++ index if available
            if self.cpp_index is not None:
                # Prepare data for C++ index
                # IMPORTANT: Store all arrays as class members to prevent Python GC
                # C++ index may hold pointers to these arrays

                # centroids: (1, b_dim)
                self._cpp_centroids = CP[:, :self.b_dim].astype('float32')

                # dist_to_centroid: (nd,)
                self._cpp_dist_to_c = dist_to_c.astype('float32')

                # x0: (nd,)
                self._cpp_x0 = self.x0_values.astype('float32')

                # cluster_id: (nd,) all zeros
                self._cpp_cluster_id = cluster_id.astype('uint32')

                # binary_codes: (nd, b_dim//64)
                self._cpp_binary = self.binary_codes.astype('uint64')

                # Build index - pass the stored member variables
                t0 = time.time()
                self.cpp_index.build(
                    self.data,
                    self._cpp_centroids,
                    self._cpp_dist_to_c,
                    self._cpp_x0,
                    self._cpp_cluster_id,
                    self._cpp_binary
                )
            self.trained = True
            return True

        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def query(self, nq: int, queries: np.ndarray, topk: int, **search_params) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for the top-k nearest neighbors for each query.

        Args:
            nq: Number of query vectors
            queries: Query vectors of shape (nq, d) where d is the dimensionality
            topk: Number of nearest neighbors to return
            **search_params: Optional search-time parameters (ignored for C=1)

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - I: Indices of nearest neighbors, shape (nq, topk)
                - D: Distances to nearest neighbors, shape (nq, topk)
        """
        if not self.trained:
            raise RuntimeError("Index not trained. Call fit() first.")

        queries = queries.astype(np.float32)
        # nprobe is always 1 since we only have C=1 (single centroid)
        nprobe = 1

        # C++ implementation is required
        if self.cpp_index is None:
            raise RuntimeError("C++ index is not available. This should not happen after __init__ validation.")

        # Prepare queries
        max_bd = max(self.ndim, self.b_dim)
        queries_pad = np.pad(queries, ((0, 0), (0, max_bd - self.ndim)), 'constant').astype('float32')

        # Project queries (randomized queries)
        rd_queries = queries_pad @ self.projection_matrix  # (nq, max_bd)
        rd_queries = rd_queries[:, :self.b_dim].astype('float32')

        # Call C++ search
        I, D = self.cpp_index.search(queries, rd_queries, topk, nprobe)

        return I, D

    def getMemoryUsage(self) -> float:
        """
        Get the memory usage of the quantizer in KB.

        Returns:
            float: Memory usage in KB
        """
        return psutil.Process().memory_info().rss / 1024

    def getCompressionRate(self) -> float:
        """
        Get the compression rate achieved by the quantizer.

        For RaBitQ, each D-dimensional float32 vector is compressed to B bits
        (where B is D rounded up to multiple of 64).

        Original size: D * 32 bits (float32)
        Compressed size: B bits

        Returns:
            float: Compression rate (higher is better)
        """
        original_bits = self.ndim * (self.data_bytes * 8)
        compressed_bits = self.b_dim
        return compressed_bits / original_bits

    def getMSE(self) -> float:
        """
        Get the mean squared error of the quantization.

        This measures the reconstruction error of the quantized vectors.
        For RaBitQ, we approximate reconstruction from binary codes.

        Returns:
            float: Mean squared error
        """
        if not self.trained or self.data is None:
            return float('inf')

        
        bin_XP = (2 * self.bin_XP - 1)/np.sqrt(self.ndim)
        o_bar = bin_XP @ self.projection_matrix + self.centroid_orig
        mse = np.mean(np.sum((self.data - o_bar) ** 2, axis = 1))
        return mse            
            # queries = self.data

            # # C++ implementation is required
            # if self.cpp_index is None:
            #     raise RuntimeError("C++ index is not available. This should not happen after __init__ validation.")

            # # Prepare queries
            # max_bd = max(self.ndim, self.b_dim)
            # queries_pad = np.pad(queries, ((0, 0), (0, max_bd - self.ndim)), 'constant').astype('float32')

            # # Project queries (randomized queries)
            # rd_queries = queries_pad @ self.projection_matrix  # (nq, max_bd)
            # rd_queries = rd_queries[:, :self.b_dim].astype('float32')
            # return self.cpp_index.getMSE(queries, rd_queries, 1000)

        

    def search_and_rerank(self, nq: int, queries: np.ndarray, topk: int, nrerank: int, **search_params) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for the top-k nearest neighbors with reranking.

        This implementation calls the C++ search_and_rerank method which:
        1. Uses RabitQ approximate search to get nrerank candidates
        2. Reranks candidates using exact L2 distance

        Args:
            nq: Number of query vectors
            queries: Query vectors of shape (nq, d) where d is the dimensionality
            topk: Number of nearest neighbors to return
            nrerank: Number of neighbors to rerank using exact distance
            **search_params: Optional search-time parameters (ignored for C=1)

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - I: Indices of nearest neighbors, shape (nq, topk)
                - D: Distances to nearest neighbors, shape (nq, topk)
        """
        if not self.trained:
            raise RuntimeError("Index not trained. Call fit() first.")

        queries = queries.astype(np.float32)
        # nprobe is always 1 since we only have C=1 (single centroid)
        nprobe = 1

        # C++ implementation is required
        if self.cpp_index is None:
            raise RuntimeError("C++ index is not available. This should not happen after __init__ validation.")

        # Prepare queries
        max_bd = max(self.ndim, self.b_dim)
        queries_pad = np.pad(queries, ((0, 0), (0, max_bd - self.ndim)), 'constant').astype('float32')

        # Project queries (randomized queries)
        rd_queries = queries_pad @ self.projection_matrix  # (nq, max_bd)
        rd_queries = rd_queries[:, :self.b_dim].astype('float32')

        # Call C++ search_and_rerank
        I, D = self.cpp_index.search_and_rerank(queries, rd_queries, topk, nprobe, nrerank)

        return I, D
