import numpy as np
from typing import Tuple
import psutil
import sys

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
        super().__init__()
        self.ndim = ndim
        self.data_bytes = data_bytes
        self.nthread = nthread
        self.space = space

        self.bit_budget = bit_budget
        self.subspace_num = subspace_num
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.var_explained = var_explained
        self.search_method = search_method

        self.data = None
        self._original_data = None
        self.ndata = 0
        self.trained = False
        self.encoded = False

        self.cpp_index = None

        if vaq_cpp is None:
            raise RuntimeError(
                "C++ module 'vaq_cpp' is not available. "
                "VAQ requires the C++ implementation. "
                "Make sure the C++ module was built correctly in the Docker image."
            )

        self.method_string = (
            f"VAQ{bit_budget}m{subspace_num}min{min_bits}max{max_bits}"
            f"var{var_explained},{search_method}"
        )
        print(f"[VAQ] Using method string: {self.method_string}")

        self.padded_dim = self._compute_padded_dim()
        self._create_index()

    def _compute_padded_dim(self) -> int:
        if self.ndim % self.subspace_num == 0:
            return self.ndim
        subvector_len = self.ndim // self.subspace_num
        if self.ndim % self.subspace_num > 0:
            subvector_len += 1
        return subvector_len * self.subspace_num

    def _pad_vectors(self, data: np.ndarray) -> np.ndarray:
        padded = np.ascontiguousarray(data, dtype=np.float32)
        dim_padding = self.padded_dim - self.ndim
        if dim_padding > 0:
            padded = np.pad(padded, ((0, 0), (0, dim_padding)), 'constant').astype(np.float32)
        return padded

    def _create_index(self):
        self.cpp_index = vaq_cpp.PyVAQ()
        self.cpp_index.parse_method_string(self.method_string)

    def fit(self, nd: int, data: np.ndarray) -> bool:
        self._create_index()
        return self.train(nd, data) and self.add(nd, data)

    def train(self, nd: int, data: np.ndarray) -> bool:
        try:
            self.data = self._pad_vectors(data)
            self.ndata = nd
            self.trained = False
            self.encoded = False
            self._create_index()

            if self.padded_dim > self.ndim:
                print(f"[VAQ] Padding dimensions from {self.ndim} to {self.padded_dim}")

            print(f"[VAQ] Training on {nd} vectors with {self.data.shape[1]} dimensions...")
            self.cpp_index.train(self.data, verbose=True)
            self.trained = True
            return True
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def add(self, nd: int, data: np.ndarray) -> bool:
        try:
            self.data = self._pad_vectors(data)
            self._original_data = np.ascontiguousarray(data, dtype=np.float32)
            self.ndata = nd

            print(f"[VAQ] Encoding {nd} database vectors...")
            self.cpp_index.add(self.data, verbose=True)
            self.encoded = True
            return True
        except Exception as e:
            print(f"Add error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def query(self, nq: int, queries: np.ndarray, topk: int, **search_params) -> Tuple[np.ndarray, np.ndarray]:
        if not self.trained or not self.encoded:
            raise RuntimeError("Index not trained or encoded. Call fit() first.")

        queries = self._pad_vectors(queries)
        I, D = self.cpp_index.search(queries, topk, verbose=False)
        return I, D

    def getMemoryUsage(self) -> float:
        return psutil.Process().memory_info().rss / 1024

    def getCompressionRate(self) -> float:
        original_bits = self.ndim * (self.data_bytes * 8)
        compressed_bits = self.bit_budget
        return compressed_bits / original_bits

    def getMSE(self) -> float:
        return self.cpp_index.get_mse()

    def set_query(self, query: np.ndarray, thread_id: int):
        query = self._pad_vectors(np.asarray(query, dtype=np.float32).reshape(1, self.ndim))
        self.cpp_index.set_query(query)

    def estimate_distance(self, idx, thread_id):
        return self.cpp_index.estimate_distance(idx)
