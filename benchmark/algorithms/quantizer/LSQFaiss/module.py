import faiss
import numpy as np
from typing import Tuple
import psutil
import sys

# 让 benchmark 可被导入
sys.path.insert(0, '/benchmark')
from benchmark.base import BaseQuantizer


class LSQFaiss(BaseQuantizer):
    def __init__(self, ndim, nsubvec, nbit, data_bytes, nthread=1, space="l2"):
        super().__init__()
        self.ndim = ndim
        self.nsubvec = nsubvec
        self.nbit = nbit
        # NOTE:
        # - 在 PyPI 的 faiss-cpu 里通常没有 IndexLSQ 这个符号（会触发 AttributeError）
        # - 但 faiss 提供了 LSQ（Local Search Quantizer）对应的 IndexLocalSearchQuantizer
        metric = faiss.METRIC_L2 if str(space).lower() == "l2" else faiss.METRIC_INNER_PRODUCT
        try:
            # 新版/完整绑定通常支持传 metric
            self.index = faiss.IndexLocalSearchQuantizer(ndim, nsubvec, nbit, metric)
        except TypeError:
            # 兼容部分绑定签名：不接受 metric 参数
            self.index = faiss.IndexLocalSearchQuantizer(ndim, nsubvec, nbit)
        self.space = space
        self.data_bytes = data_bytes
        self.data = None
        self.ndata = 0
        self.nthread = nthread
        faiss.omp_set_num_threads(nthread)

    def fit(self, nd: int, data: np.ndarray) -> bool:
        # Faiss 通常要求 float32
        self.data = data.astype(np.float32, copy=False)
        self._original_data = self.data  # 默认 rerank 用
        self.ndata = nd
        try:
            self.index.train(self.data)
            self.index.add(self.data)
        except Exception as e:
            print(f"Training error: {e}")
            return False
        return True

    def query(self, nq: int, query: np.ndarray, topk: int, **search_params) -> Tuple[np.ndarray, np.ndarray]:
        q = query.astype(np.float32, copy=False)
        D, I = self.index.search(q, topk)
        return I, D

    def getMemoryUsage(self) -> float:
        return psutil.Process().memory_info().rss / 1024

    def getCompressionRate(self) -> float:
        return self.nbit / (self.ndim // self.nsubvec * (self.data_bytes * 8))

    def getCompressionMemory(self) -> float:
        return (2 ** self.nbit) * self.ndim * 64 + self.ndata * self.nbit * self.nsubvec

    def getMSE(self) -> float:
        if self.data is None or self.ndata <= 0:
            return 0.0
        recons = np.zeros((self.ndata, self.ndim), dtype=np.float32)
        self.index.reconstruct_n(0, self.ndata, recons)
        se_per_row = np.sum((recons - self.data) ** 2, axis=1)
        return float(np.mean(se_per_row))