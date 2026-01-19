import faiss
import sys
import numpy as np
from typing import Tuple
import psutil
sys.path.insert(0, '/benchmark')
from benchmark.base import BaseQuantizer

class OptimizedProductQuantizationFaiss(BaseQuantizer):
    def __init__(self, ndim, nsubvec, nbit, data_bytes, niter, nthread = 1, space = "l2"):
        super().__init__()
        self.ndim = ndim
        self.nsubvec = nsubvec
        self.nbit = nbit
        self.PQIndex = faiss.IndexPQ(ndim, nsubvec, nbit)
        self.index = None
        self.space = space
        self.niter = niter
        self.nthread = nthread
        faiss.omp_set_num_threads(nthread)
        self.data_bytes = data_bytes
        self.data = None
        self.ndata = 0
        self.opq = None
        self.dc = None
        self.transformed_query = None
        pass




    def fit(self, nd: int, data: np.ndarray) -> bool:
        self.data = data
        self.ndata = nd
        try:
            # Faiss train expects just the data, not the count

            self.opq = faiss.OPQMatrix(self.ndim, self.nsubvec)
            self.opq.niter = self.niter
            self.opq.train(data)

            self.index = faiss.IndexPreTransform(self.opq,  self.PQIndex)
            self.index.train(data)
            self.index.add(data)
            # Add vectors to index for querying

            self.dc = self.PQIndex.get_distance_computer()
        except Exception as e:
            print(f"Training error: {e}")
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
    
    # The bit of all memory after compression 
    def getCompressionMemory(self) -> float:

        return (2 ** self.nbit) * self.ndim * 64 + self.ndata * self.nbit * self.nsubvec + self.ndim * self.ndim *  64
    
    def getMSE(self) -> float:
        recons = np.zeros_like(self.data)
        self.index.reconstruct_n(0,self.ndata,recons)
        se_per_row = np.sum((recons - self.data)**2, axis=1)
        mse = np.mean(se_per_row)
        return mse
    
    def searchAndRerank(self, nq, query, topk, nrerank):
        refine = faiss.IndexFlatL2(self.ndim)
        refine.add(self.data)

        refiner = faiss.IndexRefine(self.index, refine)
        refiner.k_factor = nrerank / topk
        D, I = refiner.search(query, topk)
        return I, D
    
    def set_query(self, query: np.ndarray, thread_id: int):
        # OPQ needs to first apply the rotation matrix to the query
        # then use the PQ distance computer
        query_2d = query.reshape(1, -1)
        self.transformed_query = self.opq.apply(query_2d).flatten().astype(np.float32)
        self.transformed_query = np.ascontiguousarray(self.transformed_query)
        self.dc.set_query(faiss.swig_ptr(self.transformed_query))

    def estimate_distance(self, idx: int, thread_id: int):
        return self.dc(int(idx))