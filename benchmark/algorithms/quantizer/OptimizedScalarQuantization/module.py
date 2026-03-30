import os
import sys
from typing import Tuple

import numpy as np
import psutil

sys.path.insert(0, '/benchmark')
sys.path.insert(0, os.path.dirname(__file__))
from benchmark.base import BaseQuantizer

try:
    import osq_cpp
except ImportError as exc:
    print(f"Warning: Could not import osq_cpp: {exc}")
    osq_cpp = None


class OptimizedScalarQuantization(BaseQuantizer):
    def __init__(self, ndim, nbit, data_bytes, nthread=1, space="l2", query_nbit=None):
        super().__init__()
        if osq_cpp is None:
            raise RuntimeError("C++ module 'osq_cpp' is not available. Make sure the binding was built in the image.")

        self.ndim = int(ndim)
        self.nbit = int(nbit)
        self.query_nbit = -1 if query_nbit is None else int(query_nbit)
        self.data_bytes = int(data_bytes)
        self.nthread = int(nthread)
        self.space = str(space).lower()

        self.data = None
        self._original_data = None
        self.ndata = 0
        self.index = osq_cpp.PyOSQIndex(self.ndim, self.space, self.nbit, self.query_nbit)
        self.index.set_num_threads(self.nthread)

    def fit(self, nd: int, data: np.ndarray) -> bool:
        self.data = np.ascontiguousarray(data.astype(np.float32, copy=False))
        self._original_data = self.data
        self.ndata = int(nd)
        try:
            self.index.build(self.data)
        except Exception as exc:
            print(f"Training error: {exc}")
            return False
        return True

    def query(self, nq: int, query: np.ndarray, topk: int, **search_params) -> Tuple[np.ndarray, np.ndarray]:
        queries = np.ascontiguousarray(query.astype(np.float32, copy=False))
        labels, scores = self.index.search(queries, topk)
        labels = np.asarray(labels, dtype=np.int64)
        scores = np.asarray(scores, dtype=np.float32)
        return labels, self._scores_to_distances(scores)

    def getMemoryUsage(self) -> float:
        return psutil.Process().memory_info().rss / 1024

    def getCompressionRate(self) -> float:
        return self.nbit / (self.data_bytes * 8)

    def getCompressionMemory(self) -> float:
        resolved_query_nbit = self.nbit if self.query_nbit < 0 else self.query_nbit
        effective_nbit = min(self.nbit, resolved_query_nbit)
        code_bits = self.ndata * self.ndim * effective_nbit
        correction_bits = self.ndata * (3 * 32 + 32)
        centroid_bits = self.ndim * 32
        return code_bits + correction_bits + centroid_bits

    def getMSE(self) -> float:
        if self.data is None or self.ndata <= 0:
            return 0.0
        reconstructed = np.asarray(self.index.reconstruct_all(), dtype=np.float32)
        se_per_row = np.sum((reconstructed - self.data) ** 2, axis=1)
        return float(np.mean(se_per_row))

    def set_query(self, query, thread_id):
        query = np.ascontiguousarray(np.asarray(query, dtype=np.float32))
        self.index.set_query(query)

    def estimate_distance(self, idx, thread_id):
        score = float(self.index.score_id(int(idx)))
        return float(self._scores_to_distances(np.asarray([score], dtype=np.float32))[0])

    def _scores_to_distances(self, scores: np.ndarray) -> np.ndarray:
        valid = np.isfinite(scores) & (scores > 0)
        if self.space == "l2":
            distances = np.full_like(scores, np.inf, dtype=np.float32)
            distances[valid] = (1.0 / scores[valid]) - 1.0
            return distances
        distances = np.full_like(scores, np.inf, dtype=np.float32)
        distances[np.isfinite(scores)] = -scores[np.isfinite(scores)]
        return distances
