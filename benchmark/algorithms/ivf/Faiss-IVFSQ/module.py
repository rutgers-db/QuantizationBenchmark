import faiss
import numpy as np
from typing import Tuple
import psutil
import sys
import os

# Add benchmark to path for importing BaseQuantizer
sys.path.insert(0, '/benchmark')
from benchmark.base import BaseQuantizer


class ScalarQuantizationIVFFaiss(BaseQuantizer):
    def __init__(self, ndim, nlist, nbit, data_bytes, nthread = 1, space = "l2"):
        super().__init__()
        self.ndim = ndim
        self.nbit = nbit
        self.nlist = nlist

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

        # Create coarse quantizer (for IVF clustering)
        self.coarse_quantizer = faiss.IndexFlatL2(ndim)

        # Create IVF + Scalar Quantization index
        metric = faiss.METRIC_L2 if space == "l2" else faiss.METRIC_INNER_PRODUCT
        self.index = faiss.IndexIVFScalarQuantizer(self.coarse_quantizer, ndim, nlist, qtype, metric)

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
            self.index.make_direct_map()
            self.refine.add(self.data)
            self.dc = self.index.get_distance_computer()
        except Exception as e:
            print(f"Add error: {e}")
            return False
        return True


    def query(self, nq: int, query: np.ndarray, topk: int, **search_params) -> Tuple[np.ndarray, np.ndarray]:
        # Faiss search expects (queries, k), not (nq, queries, k)
        # nprobe: number of clusters to visit during search
        nprobe = search_params.get('nprobe', self.nlist)
        self.index.nprobe = nprobe
        D, I = self.index.search(query, topk)
        return I, D

    def getMemoryUsage(self) -> float:
        return psutil.Process().memory_info().rss/1024

    def getCompressionRate(self) -> float:
        # Compression rate = bits per component / original bits per component
        return self.nbit / (self.data_bytes * 8)

    def getCompressionMemory(self) -> float:
        # IVF-SQ memory:
        # - Centroids: nlist * ndim * 4 bytes (float32)
        # - Quantized vectors: ndata * ndim * nbit bits
        # - Per-dimension min/max values for each cluster: nlist * ndim * 2 * 4 bytes
        centroid_memory = self.nlist * self.ndim * 32  # in bits
        quantized_memory = self.ndata * self.ndim * self.nbit  # in bits
        minmax_memory = self.nlist * self.ndim * 2 * 32  # in bits
        return centroid_memory + quantized_memory + minmax_memory

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
        self.dc.set_query(faiss.swig_ptr(query))

    def estimate_distance(self, idx, thread_id):
        return self.dc(int(idx))
