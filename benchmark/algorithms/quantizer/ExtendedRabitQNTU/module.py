import sys
import numpy as np
from typing import Tuple
import psutil
sys.path.insert(0, '/benchmark')
from benchmark.base import BaseQuantizer
import ExtendedRabitQ
import faiss
class ExtendedRabitQNTU(BaseQuantizer):
    def __init__(self, ndim, nbit, data_bytes, nthread = 1):
        self.Index = ExtendedRabitQ.Index(ndim,nbit)
        self.ndim = ndim
        self.nbit = nbit
        self.nthread = nthread
        faiss.omp_set_num_threads(nthread)

        self.data_bytes = data_bytes 

    def fit(self, nd:int, data: np.ndarray):
        try:
            centroids = np.mean(data, axis = 1)
            cids = np.zeros(nd)
            self.Index.train(data,centroids,cids,nd, self.nthread)
        except Exception as e:
            print(f"Training error: {e}")
            return False
        return True


    def query(self, nq: int, queries: np.ndarray, topk: int, **search_params) -> Tuple[np.ndarray, np.ndarray]:
        I,D = self.Index.search(queries, nq, topk)
        D = np.abs(D)
        return I ,D


    def getMemoryUsage(self) -> float:
        return psutil.Process().memory_info().rss/1024

    def getCompressionRate(self) -> float:
        return self.nbit / (self.ndim * (self.data_bytes * 8))  


    def getMSE(self) -> float:
        return 0



