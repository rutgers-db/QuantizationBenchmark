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
    def __init__(self, ndim, nsubvec, nbit, data_bytes, nthread = 1, space = "l2"):
        super().__init__()
        self.ndim = ndim
        self.nsubvec = nsubvec
        self.nbit = nbit
        self.index = faiss.IndexPQ(ndim, nsubvec, nbit)
        self.space = space
        self.data_bytes = data_bytes
        self.data = None
        self.ndata = 0
        self.nthread = nthread
        faiss.omp_set_num_threads(nthread)
        self.dc = None
        pass




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
            self.dc = self.index.get_distance_computer()
        except Exception as e:
            print(f"Add error: {e}")
            return False
        return True


    def query(self, nq: int, query: np.ndarray, topk: int, **search_params) -> Tuple[np.ndarray, np.ndarray]:
        # Faiss search expects (queries, k), not (nq, queries, k)
        # search_params are ignored for PQ (no search-time parameters)
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
        recons = np.zeros_like(self.data)
        self.index.reconstruct_n(0,self.ndata,recons)
        se_per_row = np.sum((recons - self.data)**2, axis=1)

        # Calculate norms of original vectors
        norms = np.linalg.norm(self.data, axis=1)

        # Estimate inner product: (se_per_row - 2 * norm) / (-2)
        estimated_ip = (se_per_row - 2 * norms) / (-2)

        # Calculate difference between estimated IP and norms
        ip_norm_diff = estimated_ip - norms
        
        abs_ip_diff = np.abs(ip_norm_diff)
        
        ip_diff = np.mean(abs_ip_diff)
        print(ip_diff)

        mse = np.mean(se_per_row)
        return mse
    
    def set_query(self, query, thread_id):
        # Ensure query is a contiguous float32 array for Faiss SWIG interface
        self.dc.set_query(faiss.swig_ptr(query))
        
    def estimate_distance(self, idx, thread_id):
        return self.dc(int(idx))
