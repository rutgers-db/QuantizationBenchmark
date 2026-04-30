import faiss
import numpy as np
from typing import Tuple
import psutil
import sys

sys.path.insert(0, '/benchmark')
from benchmark.base import BaseQuantizer


class PLSQFaiss(BaseQuantizer):
    def __init__(self, ndim, nsplits, nsubvec, nbit, data_bytes, nthread=1, space="l2"):
        super().__init__()
        self.ndim = ndim
        self.nsplits = nsplits
        self.nsubvec = nsubvec
        self.nbit = nbit
        self.total_subquantizers = nsplits * nsubvec
        self.space = space
        self.data_bytes = data_bytes
        self.data = None
        self.ndata = 0
        self.nthread = nthread
        faiss.omp_set_num_threads(nthread)
        self.dc = None

        metric = faiss.METRIC_L2 if str(space).lower() == "l2" else faiss.METRIC_INNER_PRODUCT
        try:
            self.index = faiss.IndexProductLocalSearchQuantizer(ndim, nsplits, nsubvec, nbit, metric)
        except TypeError:
            self.index = faiss.IndexProductLocalSearchQuantizer(ndim, nsplits, nsubvec, nbit)

    def fit(self, nd: int, data: np.ndarray) -> bool:
        return self.train(nd, data) and self.add(nd, data)

    def train(self, nd: int, data: np.ndarray) -> bool:
        self.data = np.ascontiguousarray(data.astype(np.float32, copy=False))
        self.ndata = nd
        try:
            self.index.train(self.data)
        except Exception as e:
            print(f"Training error: {e}")
            return False
        return True

    def add(self, nd: int, data: np.ndarray) -> bool:
        self.data = np.ascontiguousarray(data.astype(np.float32, copy=False))
        self._original_data = self.data
        self.ndata = nd
        try:
            self.index.add(self.data)
            # Some metrics don't expose get_distance_computer(); the dc is only
            # used by graph-traversal hooks, so tolerate failure.
            try:
                self.dc = self.index.get_distance_computer()
            except Exception:
                self.dc = None
        except Exception as e:
            print(f"Add error: {e}")
            return False
        return True

    def query(self, nq: int, query: np.ndarray, topk: int, **search_params) -> Tuple[np.ndarray, np.ndarray]:
        q = query.astype(np.float32, copy=False)
        D, I = self.index.search(q, topk)
        return I, D

    def getMemoryUsage(self) -> float:
        return psutil.Process().memory_info().rss / 1024

    def getCompressionRate(self) -> float:
        original_bits = self.ndim * self.data_bytes * 8
        compressed_bits = self.total_subquantizers * self.nbit
        return compressed_bits / original_bits

    def getCompressionMemory(self) -> float:
        codebook_size = (2 ** self.nbit) * self.ndim * 64
        code_size = self.ndata * self.total_subquantizers * self.nbit
        return codebook_size + code_size

    def getMSE(self) -> float:
        if self.data is None or self.ndata <= 0:
            return 0.0
        recons = np.zeros((self.ndata, self.ndim), dtype=np.float32)
        self.index.reconstruct_n(0, self.ndata, recons)
        se_per_row = np.sum((recons - self.data) ** 2, axis=1)
        return float(np.mean(se_per_row))

    def set_query(self, query, thread_id):
        query = np.ascontiguousarray(query, dtype=np.float32)
        self.dc.set_query(faiss.swig_ptr(query))

    def estimate_distance(self, idx, thread_id):
        return self.dc(int(idx))
