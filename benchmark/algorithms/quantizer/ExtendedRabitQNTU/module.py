import sys
import numpy as np
from typing import Tuple
import psutil
sys.path.insert(0, '/benchmark')
from benchmark.base import BaseQuantizer
import ExtendedRabitQ
import ExtendedRabitQ_HighAcc
import faiss
class ExtendedRabitQNTU(BaseQuantizer):
    def __init__(self, ndim, nbit, data_bytes, high_acc_flag , nthread = 1):
        self.high_acc_flag = high_acc_flag
        if high_acc_flag:
            self.Index = ExtendedRabitQ_HighAcc.Index(ndim,nbit)
        else:
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
            self.Index.train(data,centroids,nd, self.nthread)
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
        return self.nbit / (self.data_bytes * 8)  


    def getMSE(self) -> float:
        return self.Index.getMSE()

    def set_query(self, query, thread_id) :
        pass

    def estimate_distance(self, idx, thread_id):
        pass




