import numpy as np
from typing import Tuple
import psutil
import sys

sys.path.insert(0, '/benchmark')
from benchmark.base import BaseQuantizer

try:
    import rq_cpp
except ImportError as e:
    print(f"Warning: Could not import rq_cpp: {e}")
    rq_cpp = None


class WeaviateRotationalQuantization(BaseQuantizer):
    """
    Weaviate Rotational Quantization.

    Parameters:
        ndim:      Vector dimensionality.
        bits:      1 (BRQ path) or 2 | 4 | 8 (uniform RQ path).
        metric:    "l2" | "ip".
        nthread:   OpenMP threads for add()/search().
        data_bytes:Bytes per original float element (4 for float32).
    """

    def __init__(self, ndim: int, bits: int = 8, data_bytes: int = 4,
                 nthread: int = 1, space: str = "l2", seed: int = 0x517cc1b727220a95):
        super().__init__()
        if rq_cpp is None:
            raise RuntimeError(
                "C++ module 'rq_cpp' is not available. "
                "Make sure it was built inside the Docker image."
            )
        self.ndim       = ndim
        self.bitwidth   = int(bits)
        self.data_bytes = data_bytes
        self.nthread    = nthread
        self.space      = space

        self.cpp_index = rq_cpp.PyRotationalQuantizer(
            d=ndim,
            bits=int(bits),
            metric=space,
            seed=int(seed),
            num_threads=int(nthread),
        )
        self.ndata   = 0
        self.trained = False

    def fit(self, nd: int, data: np.ndarray) -> bool:
        return self.train(nd, data) and self.add(nd, data)

    def train(self, nd: int, data: np.ndarray) -> bool:
        # RQ rotation is data-independent; train() is a no-op on the C++ side
        # but we forward the call for symmetry with other quantizers.
        data = np.ascontiguousarray(data, dtype=np.float32)
        try:
            self.cpp_index.train(data)
        except Exception as e:
            print(f"WeaviateRotationalQuantization train error: {e}")
            return False
        return True

    def add(self, nd: int, data: np.ndarray) -> bool:
        data = np.ascontiguousarray(data, dtype=np.float32)
        self._original_data = data
        self.ndata = nd
        try:
            self.cpp_index.add(data)
            self.trained = True
        except Exception as e:
            print(f"WeaviateRotationalQuantization add error: {e}")
            return False
        return True

    def query(self, nq: int, queries: np.ndarray, topk: int,
              **search_params) -> Tuple[np.ndarray, np.ndarray]:
        if not self.trained:
            raise RuntimeError("Index not trained. Call fit() or train()+add() first.")
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        I, D = self.cpp_index.search(queries, topk)
        return I, D

    def getMemoryUsage(self) -> float:
        return psutil.Process().memory_info().rss / 1024

    def getCompressionRate(self) -> float:
        return float(self.bitwidth) / float(self.data_bytes * 8)

    def getMSE(self) -> float:
        # RQ C++ backend does not expose reconstruction; report 0.
        return 0.0

    def set_query(self, query: np.ndarray, thread_id: int) -> None:
        raise NotImplementedError(
            "WeaviateRotationalQuantization does not implement per-vector "
            "set_query/estimate_distance."
        )

    def estimate_distance(self, idx: int, thread_id: int) -> float:
        raise NotImplementedError(
            "WeaviateRotationalQuantization does not implement per-vector "
            "set_query/estimate_distance."
        )
