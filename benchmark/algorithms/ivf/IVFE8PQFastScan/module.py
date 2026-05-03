import numpy as np
from typing import Tuple
import psutil
import sys

sys.path.insert(0, '/benchmark')
from benchmark.base import BaseQuantizer

try:
    import e8pq_cpp
except ImportError as e:
    print(f"Warning: Could not import e8pq_cpp: {e}")
    e8pq_cpp = None


class IVFE8PQFastScan(BaseQuantizer):
    """
    IVFE8PQ with PQ-fastscan-style register-resident LUT.

    Same training pipeline as IVFE8PQ (FHT-Kac rotation, IVF coarse quantizer,
    learned PQ codebook on the normalized residual `o = (x_r - c) / ||x_r - c||`,
    RaBitQ scoring factors). Search differs: codes are stored in 32-wide
    block-major tiles and the per-query LUT is quantized to uint8 for an
    in-register lookup. Two backends, dispatched by ``nbit``:
      * nbit=8 → e8pq_cpp.IVFE8PQFastScan: 256-entry LUT, AVX512VBMI
        ``_mm512_permutex2var_epi8`` (zero L1 traffic), or a 32-wide gather
        fallback on non-VBMI machines.
      * nbit=4 → e8pq_cpp.IVFE8PQFastScan4: 16-entry LUT, classic PQ4 fastscan
        kernel using ``_mm256_shuffle_epi8`` on broadcast 16-byte LUTs, with
        nibble-paired packed codes (M/2 bytes per vector).
    """

    def __init__(self, ndim, nlist, nsubvec, nbit=8, data_bytes=4,
                 nthread=1, space="l2"):
        super().__init__()
        self.ndim = ndim
        self.nlist = nlist
        self.nsubvec = nsubvec
        self.nbit = int(nbit)
        self.data_bytes = data_bytes
        self.nthread = nthread
        self.space = space

        self.data = None
        self.ndata = 0
        self.trained = False
        self.index = None

        if e8pq_cpp is None:
            raise RuntimeError(
                "C++ module 'e8pq_cpp' is not available. "
                "Make sure it was built in the Docker image."
            )
        if self.nbit == 8:
            self._cpp_cls_name = 'IVFE8PQFastScan'
        elif self.nbit == 4:
            self._cpp_cls_name = 'IVFE8PQFastScan4'
        else:
            raise ValueError(
                f"IVFE8PQFastScan supports nbit ∈ {{4, 8}}, got nbit={self.nbit}"
            )
        if not hasattr(e8pq_cpp, self._cpp_cls_name):
            raise RuntimeError(
                f"e8pq_cpp.{self._cpp_cls_name} not found — rebuild with the "
                "updated e8pq_binding.cpp."
            )

    def fit(self, nd: int, data: np.ndarray) -> bool:
        return self.train(nd, data) and self.add(nd, data)

    def train(self, nd: int, data: np.ndarray) -> bool:
        return True

    def add(self, nd: int, data: np.ndarray) -> bool:
        try:
            self.data = np.ascontiguousarray(data, dtype=np.float32)
            self._original_data = self.data
            self.ndata = nd

            metric_str = "ip" if self.space == "ip" else "l2"
            cpp_cls = getattr(e8pq_cpp, self._cpp_cls_name)
            self.index = cpp_cls(
                nd,
                self.ndim,
                int(self.nlist),
                int(self.nsubvec),
                int(self.nbit),
                self.nthread,
                metric_str,
            )
            self.index.fit(self.data)
            self.trained = True
            return True
        except Exception as e:
            print(f"Add error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def query(self, nq: int, queries: np.ndarray, topk: int,
              **search_params) -> Tuple[np.ndarray, np.ndarray]:
        if not self.trained:
            raise RuntimeError("Index not trained. Call fit() first.")

        queries = np.ascontiguousarray(queries, dtype=np.float32)
        nprobe = search_params.get("nprobe", 1)
        nprobe = max(1, min(int(nprobe), int(self.nlist)))

        I, D = self.index.search_batch(queries, topk, nprobe)
        return I, D

    def getMemoryUsage(self) -> float:
        return psutil.Process().memory_info().rss / 1024

    def getCompressionRate(self) -> float:
        return (self.nbit * self.nsubvec) / (self.ndim * self.data_bytes * 8.0)

    def getMSE(self) -> float:
        return 0.0

    def set_query(self, query, thread_id):
        pass

    def estimate_distance(self, idx: int, thread_id: int) -> float:
        return 0.0
