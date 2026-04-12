import os
import sys
from typing import Tuple

import faiss
import numpy as np
import psutil

sys.path.insert(0, '/benchmark')
sys.path.insert(0, os.path.dirname(__file__))
from benchmark.base import BaseQuantizer

try:
    import residual_opq_ivfpq_cpp
except ImportError as exc:
    print(f"Warning: Could not import residual_opq_ivfpq_cpp: {exc}")
    residual_opq_ivfpq_cpp = None


class ResidualOPQIVFPQ(BaseQuantizer):
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
        if residual_opq_ivfpq_cpp is None:
            raise RuntimeError(
                "C++ module 'residual_opq_ivfpq_cpp' is not available. "
                "Make sure the binding was built in the image."
            )

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
                f"Unsupported space '{space}'. ResidualOPQIVFPQ currently supports 'l2' and 'cosine'."
            )
        if self.nbit not in {4, 8}:
            raise ValueError(
                f"Unsupported nbit value: {nbit}. ResidualOPQIVFPQ currently supports 4 and 8 bits."
            )

        faiss.omp_set_num_threads(self.nthread)

        self.coarse_index = None
        self.centroids = None
        self.opq = None
        self.pq = None
        self.index = None

        self.data = None
        self.ndata = 0
        self.assignments = None
        self.list_offsets = None
        self.list_ids = None

    def fit(self, nd: int, data: np.ndarray) -> bool:
        return self.train(nd, data) and self.add(nd, data)

    def train(self, nd: int, data: np.ndarray) -> bool:
        training_data = self._prepare_vectors(data)
        try:
            kmeans = faiss.Kmeans(
                d=self.ndim,
                k=self.nlist,
                niter=25,
                verbose=False,
                seed=1234,
            )
            kmeans.train(training_data)
            self.centroids = np.ascontiguousarray(
                np.asarray(kmeans.centroids, dtype=np.float32).reshape(self.nlist, self.ndim)
            )

            self.coarse_index = faiss.IndexFlatL2(self.ndim)
            self.coarse_index.add(self.centroids)

            _, assignments = self.coarse_index.search(training_data, 1)
            assignments = assignments[:, 0].astype(np.int64, copy=False)
            residuals = np.ascontiguousarray(training_data - self.centroids[assignments])

            if self.opq_out_dim is None:
                self.opq = faiss.OPQMatrix(self.ndim, self.nsubvec)
            else:
                self.opq = faiss.OPQMatrix(self.ndim, self.nsubvec, self.opq_out_dim)
            self.opq.niter = self.opq_niter
            self.opq.train(residuals)

            transformed_residuals = np.ascontiguousarray(
                self.opq.apply_py(residuals).astype(np.float32, copy=False)
            )

            self.pq = faiss.ProductQuantizer(self.opq.d_out, self.nsubvec, self.nbit)
            self.pq.train(transformed_residuals)

            self.index = residual_opq_ivfpq_cpp.ResidualOpqIvfpqIndex(
                self.ndim,
                self.opq.d_out,
                self.nsubvec,
                self.nbit,
            )
            self.index.set_num_threads(self.nthread)
        except Exception as exc:
            print(f"Training error: {exc}")
            return False
        return True

    def add(self, nd: int, data: np.ndarray) -> bool:
        if self.coarse_index is None or self.opq is None or self.pq is None or self.index is None:
            raise RuntimeError("train() must be called before add().")

        self.ndata = int(nd)
        self.data = self._prepare_vectors(data)
        self._original_data = self.data

        try:
            _, assignments = self.coarse_index.search(self.data, 1)
            self.assignments = assignments[:, 0].astype(np.int64, copy=False)
            counts = np.bincount(self.assignments, minlength=self.nlist).astype(np.int64, copy=False)
            self.list_offsets = np.empty(self.nlist + 1, dtype=np.int64)
            self.list_offsets[0] = 0
            np.cumsum(counts, out=self.list_offsets[1:])
            self.list_ids = np.argsort(self.assignments, kind="stable").astype(np.int64, copy=False)

            residuals = np.ascontiguousarray(self.data - self.centroids[self.assignments])
            transformed_residuals = np.ascontiguousarray(
                self.opq.apply_py(residuals).astype(np.float32, copy=False)
            )
            ordered_transformed = np.ascontiguousarray(transformed_residuals[self.list_ids])

            opq_matrix = np.ascontiguousarray(
                faiss.vector_to_array(self.opq.A).reshape(self.opq.d_out, self.opq.d_in).astype(np.float32)
            )
            pq_centroids = np.ascontiguousarray(
                faiss.vector_to_array(self.pq.centroids).reshape(
                    self.nsubvec,
                    1 << self.nbit,
                    self.opq.d_out // self.nsubvec,
                ).astype(np.float32)
            )

            self.index.build(
                np.ascontiguousarray(self.centroids, dtype=np.float32),
                opq_matrix,
                np.ascontiguousarray(pq_centroids.reshape(-1)),
                ordered_transformed,
                self.list_offsets,
                self.list_ids,
            )
            self.index.set_num_threads(self.nthread)
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
        _, probe_lists = self.coarse_index.search(queries, nprobe)
        labels, distances = self.index.search_preassigned(queries, probe_lists, topk)
        return np.asarray(labels, dtype=np.int64), np.asarray(distances, dtype=np.float32)

    def getMemoryUsage(self) -> float:
        return psutil.Process().memory_info().rss / 1024

    def getCompressionRate(self) -> float:
        compressed_bytes = (self.nsubvec * self.nbit) / 8.0
        original_bytes = self.ndim * self.data_bytes
        return original_bytes / compressed_bytes

    def getCompressionMemory(self) -> float:
        encoded_dim = self.opq.d_out
        codebook_bits = (2 ** self.nbit) * encoded_dim * 32
        centroid_bits = self.nlist * self.ndim * 32
        code_bits = self.ndata * self.nsubvec * self.nbit
        rotation_bits = self.ndim * encoded_dim * 32
        return codebook_bits + centroid_bits + code_bits + rotation_bits

    def getMSE(self) -> float:
        if self.data is None or self.ndata <= 0:
            return 0.0
        reconstructed = np.asarray(self.index.reconstruct_all(), dtype=np.float32)
        se_per_row = np.sum((reconstructed - self.data) ** 2, axis=1)
        return float(np.mean(se_per_row))

    def set_query(self, query: np.ndarray, thread_id: int):
        prepared_query = self._prepare_vectors(
            np.asarray(query, dtype=np.float32).reshape(1, -1)
        )
        self.index.set_query(np.ascontiguousarray(prepared_query[0], dtype=np.float32))

    def estimate_distance(self, idx: int, thread_id: int):
        return float(self.index.score_id(int(idx)))

    def _prepare_vectors(self, vectors: np.ndarray) -> np.ndarray:
        prepared = np.ascontiguousarray(vectors.astype(np.float32, copy=False))
        if self.space == "cosine":
            prepared = prepared.copy()
            faiss.normalize_L2(prepared)
        return prepared
