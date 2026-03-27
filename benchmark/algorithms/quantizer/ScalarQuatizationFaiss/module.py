import faiss
import numpy as np
from typing import Tuple
import psutil
import sys
import os

# Add benchmark to path for importing BaseQuantizer
sys.path.insert(0, '/benchmark')
from benchmark.base import BaseQuantizer


class ScalarQuantizationFaiss(BaseQuantizer):
    def __init__(self, ndim, nbit, data_bytes, nthread = 1, space = "l2"):
        super().__init__()
        self.ndim = ndim
        self.nbit = nbit
        # Faiss Scalar Quantizer types:
        # QT_8bit: 8 bits per component
        # QT_4bit: 4 bits per component
        # QT_6bit: 6 bits per component
        # QT_fp16: 16 bits float per component
        if nbit == 8:
            qtype = faiss.ScalarQuantizer.QT_8bit
        elif nbit == 4:
            qtype = faiss.ScalarQuantizer.QT_4bit
        elif nbit == 6:
            qtype = faiss.ScalarQuantizer.QT_6bit
        elif nbit == 16:
            qtype = faiss.ScalarQuantizer.QT_fp16
        else:
            raise ValueError(f"Unsupported nbit value: {nbit}. Supported values are 4, 6, 8, 16")

        self.index = faiss.IndexScalarQuantizer(ndim, qtype, faiss.METRIC_L2 if space == "l2" else faiss.METRIC_INNER_PRODUCT)
        self.space = space
        self.data_bytes = data_bytes
        self.data = None
        self.ndata = 0
        self.nthread = nthread
        faiss.omp_set_num_threads(nthread)
        self.dc = None




    def fit(self, nd: int, data: np.ndarray) -> bool:
        self.data = data
        self.ndata = nd
        try:
            # Faiss train expects just the data, not the count
            self.index.train(data)
            # Add vectors to index for querying
            self.index.add(data)

            self.dc = self.index.get_distance_computer()
        except Exception as e:
            print(f"Training error: {e}")
            return False
        return True


    def query(self, nq: int, query: np.ndarray, topk: int, **search_params) -> Tuple[np.ndarray, np.ndarray]:
        # Faiss search expects (queries, k), not (nq, queries, k)
        # search_params are ignored for Scalar Quantization (no search-time parameters)
        D, I = self.index.search(query, topk)
        return I, D

    def getMemoryUsage(self) -> float:
        return psutil.Process().memory_info().rss/1024

    def getCompressionRate(self) -> float:
        # Compression rate = bits per component / original bits per component
        return self.nbit / (self.data_bytes * 8)

    def getCompressionMemory(self) -> float:
        # Scalar quantization memory: number of vectors * dimension * bits per component
        # Plus codebook overhead (min/max values per dimension)
        return self.ndata * self.ndim * self.nbit + self.ndim * 2 * 32

    def getMSE(self) -> float:
        recons = np.zeros_like(self.data)
        self.index.reconstruct_n(0, self.ndata, recons)
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
