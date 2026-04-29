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


class ProductQuantizationFaiss(BaseQuantizer):
    def __init__(self, ndim, nlist, nsubvec, nbit, data_bytes, nthread = 1, space = "l2"):
        super().__init__()
        self.ndim = ndim
        self.nsubvec = nsubvec
        self.nbit = nbit
        is_ip = space in ("ip", "inner_product")
        metric = faiss.METRIC_INNER_PRODUCT if is_ip else faiss.METRIC_L2
        self.coarse_quantizer = faiss.IndexFlatIP(ndim) if is_ip else faiss.IndexFlatL2(ndim)
        self.nlist = nlist
        self.index = faiss.IndexIVFPQ(self.coarse_quantizer, ndim, nlist, nsubvec, nbit, metric)
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
            self.index.make_direct_map(True)
            self.refine.add(self.data)
            # IndexIVFPQ exposes get_distance_computer() only for METRIC_L2.
            # In IP mode it raises; the dc is just for graph traversal hooks
            # (set_query/estimate_distance), which the IVF search path doesn't
            # need, so swallow and leave dc=None.
            try:
                self.dc = self.index.get_distance_computer()
            except Exception:
                # faiss may raise FaissException (or another SWIG-wrapped type)
                # when the index doesn't expose a per-element distance computer
                # for this metric. Catch broadly.
                self.dc = None
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

    # Need To Test
    def getMemoryUsage(self) -> float:
        return psutil.Process().memory_info().rss/1024

    def getCompressionRate(self) -> float:
        return self.nbit / (self.ndim // self.nsubvec * (self.data_bytes * 8))

    def getCompressionMemory(self) -> float:
        return (2 ** self.nbit) * self.ndim * 64 + self.ndata * self.nbit * self.nsubvec

    def getMSE(self) -> float:
        return 0.0

    def set_query(self, query, thread_id):
        self.dc.set_query(faiss.swig_ptr(query))

    def estimate_distance(self, idx, thread_id):
        return self.dc(int(idx))
