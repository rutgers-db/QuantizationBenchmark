import numpy as np
from typing import Tuple
import psutil
import sys

sys.path.insert(0, '/benchmark')
from benchmark.base import BaseQuantizer

try:
    import bq_cpp
except ImportError as e:
    print(f"Warning: Could not import bq_cpp: {e}")
    bq_cpp = None


class QdrantBinaryQuantization(BaseQuantizer):
    """
    Qdrant-style Binary Quantization.

    Parameters:
        ndim:           Vector dimensionality.
        encoding:       1 = OneBit thermometer, 2 = TwoBits (Qdrant default).
        query_encoding: "same" | "scalar4" | "scalar8" — query representation.
        metric:         "l2" | "ip" | "hamming".
        data_bytes:     Bytes per original float element (4 for float32).
        nthread:        OpenMP threads for add()/search().
    """

    def __init__(self, ndim: int, data_bytes: int = 4,
                 nthread: int = 1, space: str = "l2",
                 encoding: int = 1,
                 query_encoding: str = "same"):
        super().__init__()
        if bq_cpp is None:
            raise RuntimeError(
                "C++ module 'bq_cpp' is not available. "
                "Make sure it was built inside the Docker image."
            )
        self.ndim           = ndim
        self.data_bytes     = data_bytes
        self.nthread        = nthread
        self.space          = space
        self.encoding       = int(encoding)
        self.query_encoding = query_encoding

        self.cpp_index = bq_cpp.PyBinaryQuantizer(
            d=ndim,
            encoding=self.encoding,
            query_encoding=self.query_encoding,
            metric=self.space,
            num_threads=int(nthread),
        )
        self.ndata   = 0
        self.trained = False

    def fit(self, nd: int, data: np.ndarray) -> bool:
        return self.train(nd, data) and self.add(nd, data)

    def train(self, nd: int, data: np.ndarray) -> bool:
        data = np.ascontiguousarray(data, dtype=np.float32)
        try:
            self.cpp_index.train(data)
        except Exception as e:
            print(f"QdrantBinaryQuantization train error: {e}")
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
            print(f"QdrantBinaryQuantization add error: {e}")
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
        # bits-per-dim / (data_bytes * 8)
        k_db = self.cpp_index.k_db()
        return float(k_db) / float(self.data_bytes * 8)

    def getMSE(self) -> float:
        # Binary quantization has no meaningful reconstruction MSE in the
        # float domain; report 0 so the pipeline keeps moving.
        return 0.0

    def set_query(self, query: np.ndarray, thread_id: int) -> None:
        raise NotImplementedError(
            "QdrantBinaryQuantization does not implement per-vector "
            "set_query/estimate_distance."
        )

    def estimate_distance(self, idx: int, thread_id: int) -> float:
        raise NotImplementedError(
            "QdrantBinaryQuantization does not implement per-vector "
            "set_query/estimate_distance."
        )
