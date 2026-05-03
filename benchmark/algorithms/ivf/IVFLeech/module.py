import numpy as np
from typing import Tuple
import psutil
import sys
import faiss

sys.path.insert(0, '/benchmark')
from benchmark.base import BaseQuantizer

try:
    import leech_cpp
except ImportError as e:
    print(f"Warning: Could not import leech_cpp: {e}")
    leech_cpp = None


class IVFLeech(BaseQuantizer):
    """
    IVF + Leech-lattice (Λ24) quantization.

    Mirrors the IVFE8NoLut pipeline (FHT-Kac rotation, residual from cluster
    centroid, RaBitQ-style f_add / f_rescale factors) but with a 24-dim block
    quantized to one of the 196,560 minimum-norm Leech vectors. Codes are
    packed as 18 bits in 3 bytes per block — 1 bit per dimension, matching
    the IVFE8 1-bit budget.

    Build encodes via brute-force AVX-512 argmax over all codewords; search
    decodes 16 packed indices in parallel and computes 16 inner products via
    24 broadcast-FMAs against the rotated query block.
    """

    def __init__(self, ndim, nlist, data_bytes=4, nthread=1, space="l2"):
        super().__init__()
        self.ndim = ndim
        self.nlist = nlist
        self.data_bytes = data_bytes
        # 18-bit code per 24-dim block -> 3 bytes per block -> 1 bit/dim,
        # matching the bit budget exposed by IVFE8/IVFE8NoLut. The padding to
        # 24 bits is for byte alignment only.
        self.bits = 1
        self.nthread = nthread
        self.space = space
        faiss.omp_set_num_threads(self.nthread)
        self.data = None
        self.ndata = 0
        self.trained = False

        self.coarse_index = None
        self._trained_centroids = None
        self.index = None

        if leech_cpp is None:
            raise RuntimeError(
                "C++ module 'leech_cpp' is not available. "
                "Make sure the C++ module was built correctly in the Docker image."
            )

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

            self.index = leech_cpp.IVFLeech(
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
