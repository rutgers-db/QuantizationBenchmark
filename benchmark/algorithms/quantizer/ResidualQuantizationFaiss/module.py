import faiss
import numpy as np
from typing import Tuple
import psutil
import sys

sys.path.insert(0, '/benchmark')
from benchmark.base import BaseQuantizer


class ResidualQuantizationFaiss(BaseQuantizer):
    def __init__(self, ndim, nsubvec, nbit, data_bytes, nthread=1, space="l2", max_beam_size=5):
        """
        Initialize Residual Quantizer.

        Args:
            ndim: Dimensionality of the vectors
            nsubvec: Number of subquantizers (M in RQ notation)
            nbit: Number of bits per subquantizer
            data_bytes: Number of bytes per element in original data
            nthread: Number of threads for parallel processing
            space: Distance metric ("l2" or "ip")
            max_beam_size: Beam size for encoding (higher = better quality, slower)
        """
        super().__init__()
        self.ndim = ndim
        self.nsubvec = nsubvec
        self.nbit = nbit
        self.max_beam_size = max_beam_size
        metric = faiss.METRIC_L2 if str(space).lower() == "l2" else faiss.METRIC_INNER_PRODUCT

        # Create IndexResidualQuantizer
        self.index = faiss.IndexResidualQuantizer(ndim, nsubvec, nbit, metric)
        # Configure the residual quantizer
        self.index.rq.train_type = faiss.ResidualQuantizer.Train_default
        self.index.rq.max_beam_size = max_beam_size

        self.space = space
        self.data_bytes = data_bytes
        self.data = None
        self.ndata = 0
        self.nthread = nthread
        faiss.omp_set_num_threads(nthread)
        self.dc = None

    def fit(self, nd: int, data: np.ndarray) -> bool:
        return self.train(nd, data) and self.add(nd, data)

    def train(self, nd: int, data: np.ndarray) -> bool:
        self.data = np.ascontiguousarray(data.astype(np.float32, copy=False))
        self.ndata = nd
        try:
            self.index.train(self.data)
        except Exception as e:
            print(f"Training error: {e}")
            return False
        return True

    def add(self, nd: int, data: np.ndarray) -> bool:
        self.data = np.ascontiguousarray(data.astype(np.float32, copy=False))
        self._original_data = self.data
        self.ndata = nd
        try:
            self.index.add(self.data)
            # Some metrics don't expose get_distance_computer(); the dc is only
            # used by graph-traversal hooks, so tolerate failure.
            try:
                self.dc = self.index.get_distance_computer()
            except Exception:
                self.dc = None
        except Exception as e:
            print(f"Add error: {e}")
            return False
        return True

    def query(self, nq: int, query: np.ndarray, topk: int, **search_params) -> Tuple[np.ndarray, np.ndarray]:
        q = query.astype(np.float32, copy=False)
        D, I = self.index.search(q, topk)
        return I, D

    def getMemoryUsage(self) -> float:
        return psutil.Process().memory_info().rss / 1024

    def getCompressionRate(self) -> float:
        # Original bits per vector: ndim * data_bytes * 8
        # Compressed bits per vector: nsubvec * nbit
        original_bits = self.ndim * self.data_bytes * 8
        compressed_bits = self.nsubvec * self.nbit
        return compressed_bits / original_bits

    def getCompressionMemory(self) -> float:
        # Codebook memory: nsubvec codebooks, each with 2^nbit entries of ndim floats
        # Code memory: ndata vectors, each with nsubvec * nbit bits
        codebook_size = self.nsubvec * (2 ** self.nbit) * self.ndim * 32  # bits
        code_size = self.ndata * self.nsubvec * self.nbit  # bits
        return codebook_size + code_size

    def getMSE(self) -> float:
        if self.data is None or self.ndata <= 0:
            return 0.0
        recons = np.zeros((self.ndata, self.ndim), dtype=np.float32)
        self.index.reconstruct_n(0, self.ndata, recons)
        se_per_row = np.sum((recons - self.data) ** 2, axis=1)
        return float(np.mean(se_per_row))

    def set_query(self, query, thread_id):
        # Ensure query is a contiguous float32 array for Faiss SWIG interface
        query = np.ascontiguousarray(query, dtype=np.float32)
        self.dc.set_query(faiss.swig_ptr(query))

    def estimate_distance(self, idx, thread_id):
        return self.dc(int(idx))
