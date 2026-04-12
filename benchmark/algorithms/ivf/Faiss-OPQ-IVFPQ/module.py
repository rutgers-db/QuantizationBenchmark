import faiss
import numpy as np
from typing import Tuple
import psutil
import sys

sys.path.insert(0, '/benchmark')
from benchmark.base import BaseQuantizer


class FaissOPQIVFPQ(BaseQuantizer):
    def __init__(
        self,
        ndim,
        nlist,
        nsubvec,
        nbit,
        data_bytes,
        nthread=1,
        space="l2",
        opq_out_dim=None,
        opq_niter=50,
    ):
        super().__init__()
        self.ndim = int(ndim)
        self.nlist = int(nlist)
        self.nsubvec = int(nsubvec)
        self.nbit = int(nbit)
        self.data_bytes = int(data_bytes)
        self.nthread = int(nthread)
        self.space = str(space).lower()
        self.opq_out_dim = None if opq_out_dim is None else int(opq_out_dim)
        self.opq_niter = int(opq_niter)

        if self.space not in {"l2", "cosine"}:
            raise ValueError(
                f"Unsupported space '{space}'. FaissOPQIVFPQ currently supports 'l2' and 'cosine'."
            )

        faiss.omp_set_num_threads(self.nthread)

        opq_prefix = f"OPQ{self.nsubvec}"
        if self.opq_out_dim is not None:
            opq_prefix = f"{opq_prefix}_{self.opq_out_dim}"
        self.factory_string = f"{opq_prefix},IVF{self.nlist},PQ{self.nsubvec}x{self.nbit}"

        self.index = faiss.index_factory(self.ndim, self.factory_string, faiss.METRIC_L2)
        self.index_ivf = faiss.extract_index_ivf(self.index)
        self.opq = faiss.downcast_VectorTransform(self.index.chain.at(0))
        self.opq.niter = self.opq_niter

        self.data = None
        self.ndata = 0
        self.dc = None
        self._current_query = None

    def fit(self, nd: int, data: np.ndarray) -> bool:
        return self.train(nd, data) and self.add(nd, data)

    def train(self, nd: int, data: np.ndarray) -> bool:
        self.ndata = int(nd)
        self.data = self._prepare_vectors(data)
        try:
            self.index.train(self.data)
        except Exception as exc:
            print(f"Training error: {exc}")
            return False
        return True

    def add(self, nd: int, data: np.ndarray) -> bool:
        self.ndata = int(nd)
        self.data = self._prepare_vectors(data)
        self._original_data = self.data
        try:
            self.index.reset()
            self.index.add(self.data)
            self.index_ivf.make_direct_map()
            self.dc = self.index.get_distance_computer()
        except Exception as exc:
            print(f"Add error: {exc}")
            return False
        return True

    def query(
        self,
        nq: int,
        query: np.ndarray,
        topk: int,
        **search_params,
    ) -> Tuple[np.ndarray, np.ndarray]:
        queries = self._prepare_vectors(query)
        nprobe = min(int(search_params.get("nprobe", self.nlist)), self.nlist)
        self.index_ivf.nprobe = nprobe
        distances, labels = self.index.search(queries, topk)
        return labels, distances

    def getMemoryUsage(self) -> float:
        return psutil.Process().memory_info().rss / 1024

    def getCompressionRate(self) -> float:
        compressed_bytes = (self.nsubvec * self.nbit) / 8.0
        original_bytes = self.ndim * self.data_bytes
        return original_bytes / compressed_bytes

    def getCompressionMemory(self) -> float:
        encoded_dim = self.opq.d_out
        codebook_bits = (2 ** self.nbit) * encoded_dim * 32
        centroid_bits = self.nlist * encoded_dim * 32
        code_bits = self.ndata * self.nsubvec * self.nbit
        rotation_bits = self.ndim * encoded_dim * 32
        return codebook_bits + centroid_bits + code_bits + rotation_bits

    def getMSE(self) -> float:
        if self.data is None or self.ndata <= 0:
            return 0.0
        recons = np.zeros_like(self.data)
        self.index.reconstruct_n(0, self.ndata, recons)
        se_per_row = np.sum((recons - self.data) ** 2, axis=1)
        return float(np.mean(se_per_row))

    def set_query(self, query: np.ndarray, thread_id: int):
        if self.dc is None:
            raise RuntimeError("Index has not been added yet.")
        prepared_query = self._prepare_vectors(
            np.asarray(query, dtype=np.float32).reshape(1, -1)
        )
        self._current_query = np.ascontiguousarray(prepared_query[0])
        self.dc.set_query(faiss.swig_ptr(self._current_query))

    def estimate_distance(self, idx: int, thread_id: int):
        if self.dc is None:
            raise RuntimeError("Index has not been added yet.")
        return self.dc(int(idx))

    def _prepare_vectors(self, vectors: np.ndarray) -> np.ndarray:
        prepared = np.ascontiguousarray(vectors.astype(np.float32, copy=False))
        if self.space == "cosine":
            prepared = prepared.copy()
            faiss.normalize_L2(prepared)
        return prepared
