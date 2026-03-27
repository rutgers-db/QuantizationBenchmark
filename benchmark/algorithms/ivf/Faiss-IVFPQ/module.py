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
        self.ndata = nd
        self.data = data
        try:
            # Faiss train expects just the data, not the count
            self.index.train(data)
            # Add vectors to index for querying
            self.index.add(data)
            
            self.index.make_direct_map(True)
            
            self.refine.add(data)
            self.dc = self.index.get_distance_computer()
        except Exception as e:
            print(f"Training error: {e}")
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
        recons = np.zeros_like(self.data)
        self.index.reconstruct_n(0,self.ndata,recons)
        se_per_row = np.sum((recons - self.data)**2, axis=1)
        mse = np.mean(se_per_row)
        return mse
    
    def set_query(self, query, thread_id):
        self.dc.set_query(faiss.swig_ptr(query))
        
    def estimate_distance(self, idx, thread_id):
        return self.dc(int(idx))
