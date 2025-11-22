import faiss
import numpy as np
from typing import Tuple
import psutil
import sys

# Add benchmark to path for importing BaseQuantizer
sys.path.insert(0, '/benchmark')
from benchmark.base import BaseQuantizer


class ProductQuantizationFaiss(BaseQuantizer):
    def __init__(self, ndim, nsubvec, nbit, data_bytes, nthread = 1, space = "l2"):
        self.ndim = ndim
        self.nsubvec = nsubvec
        self.nbit = nbit
        self.index = faiss.IndexPQ(ndim, nsubvec, nbit)
        self.space = space
        self.nthread = nthread
        self.data_bytes = data_bytes
        self.data = None
        self.ndata = 0
        pass

    def fit(self, nd: int, data: np.ndarray) -> bool:
        self.data = data
        self.ndata = nd
        try:
            # Faiss train expects just the data, not the count
            self.index.train(data)
            # Add vectors to index for querying
            self.index.add(data)
        except Exception as e:
            print(f"Training error: {e}")
            return False
        return True


    def query(self, nq: int, query: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
        # Faiss search expects (queries, k), not (nq, queries, k)
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