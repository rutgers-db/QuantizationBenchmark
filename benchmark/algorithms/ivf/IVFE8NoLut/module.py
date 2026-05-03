import numpy as np
from typing import Tuple
import psutil
import sys
import faiss

sys.path.insert(0, '/benchmark')
from benchmark.base import BaseQuantizer

try:
    import e8nolut_cpp
except ImportError as e:
    print(f"Warning: Could not import e8nolut_cpp: {e}")
    e8nolut_cpp = None


class IVFE8NoLut(BaseQuantizer):
    """
    IVF + E8-lattice 1-bit quantization, LUT-free search path.

    Same quantizer as IVFE8 (240 min-norm E8 shell vectors, FHT-Kac rotation,
    RaBitQ-style factors, 1 byte per 8-dim block). The code byte is repacked as
    [flag:1 | payload:7]: flag=0 Type A (5-bit pair idx + 2 sign bits), flag=1
    Type B (7 sign bits, 8th sign recovered via even parity). Search computes
    <q_b, codeword> directly from the packed code via permute + masked sub,
    avoiding the 256-entry per-query LUT and its gather.
    """

    def __init__(self, ndim, nlist, data_bytes=4, nthread=1, space="l2", bits=1):
        super().__init__()
        self.ndim = ndim
        self.nlist = nlist
        self.data_bytes = data_bytes
        if bits not in (1, 1.5):
            raise ValueError(f"bits must be 1 or 1.5, got {bits}")
        self.bits = bits
        self.nthread = nthread
        self.space = space
        faiss.omp_set_num_threads(self.nthread)
        self.data = None
        self.ndata = 0
        self.trained = False

        self.coarse_index = None
        self._trained_centroids = None
        self.index = None

        if e8nolut_cpp is None:
            raise RuntimeError(
                "C++ module 'e8nolut_cpp' is not available. "
                "Make sure the C++ module was built correctly in the Docker image."
            )

        if self.bits == 1:
            self._cpp_class = e8nolut_cpp.IVFE8NoLut
        else:
            self._cpp_class = e8nolut_cpp.IVFE8NoLut15

    def fit(self, nd: int, data: np.ndarray) -> bool:
        return self.train(nd, data) and self.add(nd, data)

    def train(self, nd: int, data: np.ndarray) -> bool:
        try:
            train_data = np.ascontiguousarray(data, dtype=np.float32)

            if self.nlist == 1:
                centroids = np.ascontiguousarray(
                    train_data.mean(axis=0, keepdims=True).astype(np.float32)
                )
            else:
                faiss.omp_set_num_threads(self.nthread)
                kmeans = faiss.Kmeans(
                    d=self.ndim,
                    k=self.nlist,
                    niter=25,
                    verbose=False,
                    seed=1234,
                )
                kmeans.train(train_data)
                centroids = np.ascontiguousarray(kmeans.centroids.astype(np.float32))

            self._trained_centroids = centroids
            self.coarse_index = faiss.IndexFlatL2(self.ndim)
            self.coarse_index.add(centroids)

            self.trained = False
            return True
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def add(self, nd: int, data: np.ndarray) -> bool:
        if self._trained_centroids is None or self.coarse_index is None:
            raise RuntimeError("Index not trained. Call train() first.")
        try:
            self.data = np.ascontiguousarray(data, dtype=np.float32)
            self._original_data = self.data
            self.ndata = nd

            _, assignments = self.coarse_index.search(self.data, 1)
            cluster_ids = np.ascontiguousarray(assignments.flatten().astype(np.uint32))

            metric_str = "ip" if self.space == "ip" else "l2"

            self.index = self._cpp_class(
                nd,
                self.ndim,
                int(self._trained_centroids.shape[0]),
                self.nthread,
                metric_str,
            )
            self.index.construct(self.data, self._trained_centroids, cluster_ids)

            self.trained = True
            return True
        except Exception as e:
            print(f"Add error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def query(self, nq: int, queries: np.ndarray, topk: int, **search_params) -> Tuple[np.ndarray, np.ndarray]:
        if not self.trained:
            raise RuntimeError("Index not trained. Call fit() first.")

        queries = np.ascontiguousarray(queries, dtype=np.float32)
        nprobe = search_params.get("nprobe", 1)
        nprobe = max(1, min(int(nprobe), int(self._trained_centroids.shape[0])))

        I, D = self.index.search_batch(queries, topk, nprobe)
        return I, D

    def getMemoryUsage(self) -> float:
        return psutil.Process().memory_info().rss / 1024

    def getCompressionRate(self) -> float:
        return self.bits / (self.data_bytes * 8)

    def getMSE(self) -> float:
        return 0.0

    def set_query(self, query, thread_id):
        pass

    def estimate_distance(self, idx: int, thread_id: int) -> float:
        return 0.0
