import faiss
import numpy as np
from typing import Tuple
import psutil
class ProductQuantizationFaiss:
    def __init__(self, ndim, nsubvec, nbit, data_bit, nthread = 1, space = "l2"):
        self.ndim = ndim
        self.nsubvec = nsubvec
        self.nbit = nbit
        self.index = faiss.IndexPQ(ndim, nsubvec, nbit)
        self.space = space
        self.nthread = nthread
        self.data_bit = data_bit
        self.data = None
        self.ndata = 0
        pass

    def fit(self, nd: int, data: np.ndarray) -> bool:
        self.data = data
        self.ndata = nd
        try:
            self.index.train(nd,data)
        except:
            return False
        return True
             

    def query(self, nq: int, query: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
        D, I = self.index.search(nq,query, topk)
        return I, D

    def getMemoryUsage(self) -> float:
        return psutil.Process().memory_info().rss/1024

    def getCompressionRate(self) -> float:
        return self.nbit / (self.ndim // self.nsubvec * self.data_bit)  
    
    def getCompressionMemory(self) -> float:
        return (2 ** self.nbit) * self.ndim * self.data_bit + self.ndata * self.nbit * self.nsubvec
    def getMSE(self) -> float:
        recons = np.zeros_like(self.data)
        self.index.reconstruct_n(0,self.ndata,recons)
        se_per_row = np.sum((recons - self.data)**2, axis=1)
        mse = np.mean(se_per_row)
        return mse