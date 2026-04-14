import faiss
import numpy as np
from typing import Tuple
import psutil
import sys
import os

# Add benchmark to path for importing BaseQuantizer
sys.path.insert(0, '/benchmark')
from benchmark.base import BaseQuantizer


class ProductQuantizationFaiss(BaseQuantizer):
    def __init__(self, ndim, nlist, nsubvec, nbit, data_bytes, nthread = 1, space = "l2"):
        super().__init__()
        self.ndim = ndim
        self.nsubvec = nsubvec
        self.nbit = nbit
        self.coarse_quantizer = faiss.IndexFlatL2(ndim)
        self.nlist = nlist
        self.index = faiss.IndexIVFPQ(self.coarse_quantizer, ndim, nlist, nsubvec, nbit)
        self.space = space
        self.data_bytes = data_bytes
        self.nthread = nthread
        faiss.omp_set_num_threads(nthread)
        self.refine = faiss.IndexFlatL2(self.ndim)
        self.dc = None




    def fit(self, nd: int, data: np.ndarray) -> bool:
        return self.train(nd, data) and self.add(nd, data)

    def train(self, nd: int, data: np.ndarray) -> bool:
        self.ndata = nd
        self.data = np.ascontiguousarray(data.astype(np.float32, copy=False))
        try:
            self.index.train(self.data)
        except Exception as e:
            print(f"Training error: {e}")
            return False
        return True

    def add(self, nd: int, data: np.ndarray) -> bool:
        self.ndata = nd
        self.data = np.ascontiguousarray(data.astype(np.float32, copy=False))
        self._original_data = self.data
        try:
            self.index.add(self.data)
            self.index.make_direct_map(True)
            self.refine.add(self.data)
            self.dc = self.index.get_distance_computer()
        except Exception as e:
            print(f"Add error: {e}")
            return False
        return True


    def query(self, nq: int, query: np.ndarray, topk: int, **search_params) -> Tuple[np.ndarray, np.ndarray]:
        # Faiss search expects (queries, k), not (nq, queries, k)
        # search_params are ignored for PQ (no search-time parameters)
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
