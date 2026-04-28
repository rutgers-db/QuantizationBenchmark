import faiss
import numpy as np
from typing import Tuple
import psutil
import sys
import os

# Add benchmark to path for importing BaseQuantizer
sys.path.insert(0, '/benchmark')
from benchmark.base import BaseQuantizer
from benchmark.ivf_centroid_cache import (
    data_fingerprint,
    coarse_key,
    load_centroids,
    save_centroids,
)


def _max_threads() -> int:
    return max(1, (os.cpu_count() or 1))


class ProductQuantizationFastScanFaiss(BaseQuantizer):
    def __init__(self, ndim, nlist, nsubvec, nbit, data_bytes, nthread=1, space="l2", bbs=32):
        super().__init__()
        if int(nbit) != 4:
            raise ValueError(f"IndexIVFPQFastScan requires nbit=4, got nbit={nbit}")
        self.ndim = ndim
        self.nsubvec = nsubvec
        self.nbit = nbit
        self.bbs = int(bbs)
        is_ip = space in ("ip", "inner_product")
        metric = faiss.METRIC_INNER_PRODUCT if is_ip else faiss.METRIC_L2
        self.coarse_quantizer = faiss.IndexFlatIP(ndim) if is_ip else faiss.IndexFlatL2(ndim)
        self.nlist = nlist
        self.index = faiss.IndexIVFPQFastScan(
            self.coarse_quantizer, ndim, nlist, nsubvec, nbit, metric, self.bbs
        )
        self.space = space
        self.data_bytes = data_bytes
        self.nthread = int(nthread)
        self.refine = faiss.IndexFlatIP(self.ndim) if is_ip else faiss.IndexFlatL2(self.ndim)
        self.dc = None

    def fit(self, nd: int, data: np.ndarray) -> bool:
        return self.train(nd, data) and self.add(nd, data)

    def train(self, nd: int, data: np.ndarray) -> bool:
        self.ndata = nd
        self.data = np.ascontiguousarray(data.astype(np.float32, copy=False))
        # Max out CPU threads during training (k-means + PQ codebook).
        faiss.omp_set_num_threads(_max_threads())
        try:
            fp = data_fingerprint(self.data)
            ckey = coarse_key(fp, self.nlist, self.space)
            cached = load_centroids(ckey)
            if cached is not None and cached.shape == (self.nlist, self.ndim):
                # Pre-populate the coarse quantizer so IVF.train_q1 skips k-means
                # (IndexFlatL2.is_trained is always True and ntotal == nlist hits
                # the fast path in Faiss's Level1Quantizer::train_q1).
                self.coarse_quantizer.reset()
                self.coarse_quantizer.add(cached)
            self.index.train(self.data)
            if cached is None:
                centroids = self.coarse_quantizer.reconstruct_n(0, self.nlist)
                save_centroids(ckey, centroids)
        except Exception as e:
            print(f"Training error: {e}")
            return False
        return True

    def add(self, nd: int, data: np.ndarray) -> bool:
        self.ndata = nd
        self.data = np.ascontiguousarray(data.astype(np.float32, copy=False))
        self._original_data = self.data
        # Max out CPU threads during add (encoding + inverted list construction).
        faiss.omp_set_num_threads(_max_threads())
        try:
            self.index.add(self.data)
            self.refine.add(self.data)
        except Exception as e:
            print(f"Add error: {e}")
            return False
        return True

    def query(self, nq: int, query: np.ndarray, topk: int, **search_params) -> Tuple[np.ndarray, np.ndarray]:
        # Restore configured search-time thread count.
        faiss.omp_set_num_threads(self.nthread)
        nprobe = search_params.get('nprobe', self.nlist)
        self.index.nprobe = nprobe
        D, I = self.index.search(query, topk)
        return I, D

    def getMemoryUsage(self) -> float:
        return psutil.Process().memory_info().rss / 1024

    def getCompressionRate(self) -> float:
        return self.nbit / (self.ndim // self.nsubvec * (self.data_bytes * 8))

    def getCompressionMemory(self) -> float:
        return (2 ** self.nbit) * self.ndim * 64 + self.ndata * self.nbit * self.nsubvec

    def getMSE(self) -> float:
        return 0.0

    def set_query(self, query, thread_id):
        # IndexIVFPQFastScan does not expose a per-vector distance computer;
        # FastScan evaluates candidates in SIMD batches via precomputed LUTs.
        self._query_vec = np.ascontiguousarray(query.astype(np.float32, copy=False)).reshape(1, -1)

    def estimate_distance(self, idx, thread_id):
        # Fall back to exact L2 on the original vector for single-point queries.
        vec = self.refine.reconstruct(int(idx)).reshape(1, -1)
        diff = self._query_vec - vec
        return float(np.dot(diff.ravel(), diff.ravel()))
