import os
import sys
from typing import List, Tuple

import faiss
import numpy as np
import psutil

sys.path.insert(0, '/benchmark')
sys.path.insert(0, os.path.dirname(__file__))
from benchmark.base import BaseQuantizer
from benchmark.ivf_centroid_cache import (
    data_fingerprint,
    coarse_key,
    load_centroids,
    save_centroids,
)

try:
    import ivf_rq_cpp
except ImportError as exc:
    print(f"Warning: Could not import ivf_rq_cpp: {exc}")
    ivf_rq_cpp = None


def _max_threads() -> int:
    return max(1, (os.cpu_count() or 1))


class IVFWeaviateRSQ(BaseQuantizer):
    """IVF + Weaviate Rotational Quantization on residuals.

    Build pipeline mirrors IVFOSQ / IVFQdrantBQ:
      train()  learns coarse centroids (cached across IVF methods).
      add()    assigns each db vector, builds invlists, then trains the
               RQ on residuals concatenated in invlist order.
      query()  asks the C++ side to encode each probed list's residual
               query separately and scan only that list's ids.
    """

    def __init__(self, ndim, nlist, bits=8, data_bytes=4, nthread=1,
                 space="l2", seed=0x517cc1b727220a95):
        super().__init__()
        if ivf_rq_cpp is None:
            raise RuntimeError(
                "C++ module 'ivf_rq_cpp' is not available. "
                "Make sure it was built inside the Docker image."
            )

        self.ndim       = int(ndim)
        self.nlist      = int(nlist)
        self.bitwidth   = int(bits)
        self.data_bytes = int(data_bytes)
        self.nthread    = int(nthread)
        self.space      = str(space).lower()
        self.seed       = int(seed)
        self.metric     = (
            faiss.METRIC_L2 if self.space == "l2" else faiss.METRIC_INNER_PRODUCT
        )

        self.coarse_index = None
        self.centroids    = None
        self.assignments  = None
        self.invlists: List[np.ndarray] = []
        self.list_offsets = None
        self.list_ids     = None
        self.ivf_index    = None
        self.ndata        = 0

    def fit(self, nd: int, data: np.ndarray) -> bool:
        return self.train(nd, data) and self.add(nd, data)

    def train(self, nd: int, data: np.ndarray) -> bool:
        try:
            faiss.omp_set_num_threads(_max_threads())
            training_data = np.ascontiguousarray(data.astype(np.float32, copy=False))
            if self.space == "cosine":
                training_data = training_data.copy()
                faiss.normalize_L2(training_data)

            fp = data_fingerprint(training_data)
            ckey = coarse_key(fp, self.nlist, self.space)
            cached = load_centroids(ckey)
            if cached is not None and cached.shape == (self.nlist, self.ndim):
                self.centroids = np.ascontiguousarray(cached.astype(np.float32))
            else:
                kmeans = faiss.Kmeans(d=self.ndim, k=self.nlist,
                                      niter=25, verbose=False, seed=1234)
                kmeans.train(training_data)
                self.centroids = kmeans.centroids.astype(np.float32)
                save_centroids(ckey, self.centroids)

            self.coarse_index = (
                faiss.IndexFlatL2(self.ndim)
                if self.metric == faiss.METRIC_L2
                else faiss.IndexFlatIP(self.ndim)
            )
            self.coarse_index.add(self.centroids)
            return True
        except Exception as exc:
            print(f"IVFWeaviateRSQ training error: {exc}")
            return False

    def add(self, nd: int, data: np.ndarray) -> bool:
        if self.coarse_index is None or self.centroids is None:
            raise RuntimeError("Index not trained. Call train() first.")
        try:
            faiss.omp_set_num_threads(_max_threads())

            data = np.ascontiguousarray(data.astype(np.float32, copy=False))
            self._original_data = data
            self.ndata = int(nd)

            indexed = data.copy()
            if self.space == "cosine":
                faiss.normalize_L2(indexed)

            _, assignments = self.coarse_index.search(indexed, 1)
            self.assignments = assignments[:, 0].astype(np.int64)

            buckets = [[] for _ in range(self.nlist)]
            for i, lid in enumerate(self.assignments.tolist()):
                buckets[int(lid)].append(i)
            self.invlists = [np.asarray(b, dtype=np.int64) for b in buckets]

            offsets = [0]
            flat_ids = []
            residual_chunks: List[np.ndarray] = []
            for lid, ids in enumerate(self.invlists):
                if ids.size:
                    chunk = indexed[ids] - self.centroids[lid]
                    residual_chunks.append(chunk.astype(np.float32, copy=False))
                    flat_ids.extend(ids.tolist())
                offsets.append(len(flat_ids))

            if residual_chunks:
                residuals = np.concatenate(residual_chunks, axis=0)
            else:
                residuals = np.zeros((0, self.ndim), dtype=np.float32)

            self.list_offsets = np.asarray(offsets, dtype=np.int64)
            self.list_ids     = np.asarray(flat_ids, dtype=np.int64)

            self.ivf_index = ivf_rq_cpp.PyIVFRQIndex(
                d=self.ndim,
                bits=self.bitwidth,
                metric=self.space if self.space != "cosine" else "ip",
                seed=self.seed,
            )
            self.ivf_index.set_num_threads(_max_threads())
            self.ivf_index.build(
                np.ascontiguousarray(residuals, dtype=np.float32),
                self.list_offsets,
                self.list_ids,
            )
            return True
        except Exception as exc:
            print(f"IVFWeaviateRSQ add error: {exc}")
            return False

    def query(self, nq: int, queries: np.ndarray, topk: int,
              **search_params) -> Tuple[np.ndarray, np.ndarray]:
        faiss.omp_set_num_threads(self.nthread)
        if self.ivf_index is not None:
            self.ivf_index.set_num_threads(self.nthread)
        queries = np.ascontiguousarray(queries.astype(np.float32, copy=False))
        coarse_q = queries.copy()
        if self.space == "cosine":
            faiss.normalize_L2(coarse_q)

        nprobe = min(int(search_params.get("nprobe", self.nlist)), self.nlist)
        _, probe_lists = self.coarse_index.search(coarse_q, nprobe)
        probe_lists = np.ascontiguousarray(probe_lists.astype(np.int64, copy=False))

        labels, dists = self.ivf_index.search_preassigned(
            coarse_q, self.centroids, probe_lists, topk
        )
        return np.asarray(labels, dtype=np.int64), np.asarray(dists, dtype=np.float32)

    def getMemoryUsage(self) -> float:
        return psutil.Process().memory_info().rss / 1024

    def getCompressionRate(self) -> float:
        return float(self.bitwidth) / float(self.data_bytes * 8)

    def getCompressionMemory(self) -> float:
        # RQ uses 1 byte per dim for the uniform path (bits in 2/4/8) and
        # 1 bit per dim for BRQ. Actual code_size includes padding and
        # per-vector meta floats; this is a tight lower bound.
        per_vec_bits = (
            self.ndim * 1 if self.bitwidth == 1 else self.ndim * 8
        )
        meta_bits     = self.ndata * 4 * 32
        centroid_bits = self.nlist * self.ndim * 32
        return self.ndata * per_vec_bits + meta_bits + centroid_bits

    def getMSE(self) -> float:
        return 0.0

    def set_query(self, query, thread_id):
        raise NotImplementedError(
            "IVFWeaviateRSQ does not implement per-vector set_query/estimate_distance."
        )

    def estimate_distance(self, idx, thread_id):
        raise NotImplementedError(
            "IVFWeaviateRSQ does not implement per-vector set_query/estimate_distance."
        )
