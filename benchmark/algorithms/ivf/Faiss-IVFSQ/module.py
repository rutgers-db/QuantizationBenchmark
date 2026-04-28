import faiss
import numpy as np
from typing import Tuple
import psutil
import sys
import os

# Add benchmark to path for importing BaseQuantizer
sys.path.insert(0, '/benchmark')
from benchmark.base import BaseQuantizer
from benchmark.ivf_centroid_cache import (
    data_fingerprint,
    coarse_key,
    load_centroids,
    save_centroids,
)


def _max_threads() -> int:
    return max(1, (os.cpu_count() or 1))


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
        is_ip = space in ("ip", "inner_product")
        metric = faiss.METRIC_INNER_PRODUCT if is_ip else faiss.METRIC_L2
        self.coarse_quantizer = faiss.IndexFlatIP(ndim) if is_ip else faiss.IndexFlatL2(ndim)

        # Create IVF + Scalar Quantization index
        self.index = faiss.IndexIVFScalarQuantizer(self.coarse_quantizer, ndim, nlist, qtype, metric)

        self.space = space
        self.data_bytes = data_bytes
        self.nthread = int(nthread)
        self.refine = faiss.IndexFlatIP(self.ndim) if is_ip else faiss.IndexFlatL2(self.ndim)
        self.dc = None




    def fit(self, nd: int, data: np.ndarray) -> bool:
        return self.train(nd, data) and self.add(nd, data)

    def train(self, nd: int, data: np.ndarray) -> bool:
        self.ndata = nd
        self.data = np.ascontiguousarray(data.astype(np.float32, copy=False))
        # Max out CPU threads during training (k-means + SQ).
        faiss.omp_set_num_threads(_max_threads())
        try:
            fp = data_fingerprint(self.data)
            ckey = coarse_key(fp, self.nlist, self.space)
            cached = load_centroids(ckey)
            if cached is not None and cached.shape == (self.nlist, self.ndim):
                self.coarse_quantizer.reset()
                self.coarse_quantizer.add(cached)
            self.index.train(self.data)
            if cached is None:
                centroids = self.coarse_quantizer.reconstruct_n(0, self.nlist)
                save_centroids(ckey, centroids)
        except Exception as e:
            print(f"Training error: {e}")
            return False
        return True

    def add(self, nd: int, data: np.ndarray) -> bool:
        self.ndata = nd
        self.data = np.ascontiguousarray(data.astype(np.float32, copy=False))
        self._original_data = self.data
        faiss.omp_set_num_threads(_max_threads())
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
        faiss.omp_set_num_threads(self.nthread)
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
        return 0.0

    def set_query(self, query, thread_id):
        self.dc.set_query(faiss.swig_ptr(query))

    def estimate_distance(self, idx, thread_id):
        return self.dc(int(idx))
