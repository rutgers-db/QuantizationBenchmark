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
    import vaq_cpp
except ImportError as e:
    print(f"Warning: Could not import vaq_cpp: {e}")
    vaq_cpp = None


class VAQ(BaseQuantizer):
    """
    VAQ (Vector Adaptive Quantization) implementation using C++ backend.
    """

    def __init__(self, ndim, data_bytes, nthread=1, space="l2",
                 bit_budget=256, subspace_num=32, min_bits=7, max_bits=13,
                 var_explained=1.0, search_method="SORT"):
        """
        Initialize VAQ quantizer.

        Args:
            ndim: Dimensionality of vectors
            data_bytes: Size of data type in bytes (typically 4 for float32)
            nthread: Number of threads to use for parallel processing
            space: Distance metric ("l2" for Euclidean)
            bit_budget: Total bits per encoded vector (e.g., 256 bits)
            subspace_num: Number of subvectors to split the vector into
            min_bits: Minimum bits per segment
            max_bits: Maximum bits per segment
            var_explained: Variance threshold (1.0 = no compression)
            search_method: Search algorithm ("SORT", "EA", "Heap", "TI", etc.)
        """
        super().__init__()
        self.ndim = ndim
        self.data_bytes = data_bytes
        self.nthread = nthread
        self.space = space

        # VAQ-specific parameters
        self.bit_budget = bit_budget
        self.subspace_num = subspace_num
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.var_explained = var_explained
        self.search_method = search_method

        self.data = None
        self.ndata = 0
        self.trained = False
        self.encoded = False

        # C++ index (required)
        self.cpp_index = None

        if vaq_cpp is None:
            raise RuntimeError(
                "C++ module 'vaq_cpp' is not available. "
                "VAQ requires the C++ implementation. "
                "Make sure the C++ module was built correctly in the Docker image."
            )

        # Create C++ index
        self.cpp_index = vaq_cpp.PyVAQ()

        # Build method string from parameters
        # Format: VAQ{bit_budget}m{subspace_num}min{min_bits}max{max_bits}var{var_explained},{search_method}
        method_string = f"VAQ{bit_budget}m{subspace_num}min{min_bits}max{max_bits}var{var_explained},{search_method}"
        print(f"[VAQ] Using method string: {method_string}")

        # Parse method string and configure the index
        self.cpp_index.parse_method_string(method_string)

    def fit(self, nd: int, data: np.ndarray) -> bool:
        """
        Train the VAQ quantizer on the given data.

        Args:
            nd: Number of data vectors
            data: Training data of shape (nd, d) where d is the dimensionality

        Returns:
            bool: True if training was successful, False otherwise
        """
        try:
            self.data = np.ascontiguousarray(data, dtype=np.float32)
            self.ndata = nd

            # Calculate padding if needed (VAQ requires dimensions divisible by subspace_num)
            dim_padding = 0
            if self.ndim % self.subspace_num != 0:
                subvector_len = self.ndim // self.subspace_num
                if self.ndim % self.subspace_num > 0:
                    subvector_len += 1
                dim_padding = (subvector_len * self.subspace_num) - self.ndim

            if dim_padding > 0:
                print(f"[VAQ] Padding dimensions from {self.ndim} to {self.ndim + dim_padding}")
                self.data = np.pad(self.data, ((0, 0), (0, dim_padding)), 'constant').astype('float32')

            # Train the index
            print(f"[VAQ] Training on {nd} vectors with {self.data.shape[1]} dimensions...")
            self.cpp_index.train(self.data, verbose=True)
            self.trained = True
            self.encoded = True

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
            **search_params: Optional search-time parameters

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - I: Indices of nearest neighbors, shape (nq, topk)
                - D: Distances to nearest neighbors, shape (nq, topk)
        """
        if not self.trained or not self.encoded:
            raise RuntimeError("Index not trained or encoded. Call fit() first.")

        queries = queries.astype(np.float32)

        # Pad queries if needed
        dim_padding = self.data.shape[1] - self.ndim
        if dim_padding > 0:
            queries = np.pad(queries, ((0, 0), (0, dim_padding)), 'constant').astype('float32')

        # Call C++ search
        I, D = self.cpp_index.search(queries, topk, verbose=False)

        return I, D

    def searchAndRerank(self, nq: int, queries: np.ndarray, topk: int, nrerank: int, **search_params) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for the top-k nearest neighbors with reranking.

        Args:
            nq: Number of query vectors
            queries: Query vectors of shape (nq, d) where d is the dimensionality
            topk: Number of nearest neighbors to return
            nrerank: Number of neighbors to rerank using exact distance
            **search_params: Optional search-time parameters

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - I: Indices of nearest neighbors, shape (nq, topk)
                - D: Distances to nearest neighbors, shape (nq, topk)
        """
        if not self.trained or not self.encoded:
            raise RuntimeError("Index not trained or encoded. Call fit() first.")

        queries = queries.astype(np.float32)

        # Pad queries if needed
        dim_padding = self.data.shape[1] - self.ndim
        if dim_padding > 0:
            queries = np.pad(queries, ((0, 0), (0, dim_padding)), 'constant').astype('float32')

        # Call C++ search_and_rerank
        I, D = self.cpp_index.search_and_rerank(queries, self.data, topk, nrerank, verbose=False)

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

        For VAQ, each D-dimensional float32 vector is compressed to bit_budget bits.

        Original size: D * 32 bits (float32)
        Compressed size: bit_budget bits

        Returns:
            float: Compression rate (compressed_bits / original_bits)
        """
        original_bits = self.ndim * (self.data_bytes * 8)
        compressed_bits = self.bit_budget
        return compressed_bits / original_bits

    def getMSE(self) -> float:
        """
        Get the mean squared error of the quantization.

        This would require implementing reconstruction from VAQ codes,
        which is complex.

        Returns:
            float: Mean squared error
        """
        return self.cpp_index.get_mse()
    
    def set_query(self, query: np.ndarray, thread_id: int):
        query = query.astype(np.float32).reshape(1, self.ndim)
        self.cpp_index.set_query(query)
        
    def estimate_distance(self, idx, thread_id):
        return self.cpp_index.estimate_distance(idx)
