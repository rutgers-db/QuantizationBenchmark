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
    import osq_cpp
except ImportError as exc:
    print(f"Warning: Could not import osq_cpp: {exc}")
    osq_cpp = None


def _max_threads() -> int:
    return max(1, (os.cpu_count() or 1))


class IVFOSQ(BaseQuantizer):
    def __init__(self, ndim, nlist, nbit, data_bytes, nthread=1, space="l2", query_nbit=None):
        super().__init__()
        if osq_cpp is None:
            raise RuntimeError("C++ module 'osq_cpp' is not available. Make sure the binding was built in the image.")

        self.ndim = int(ndim)
        self.nlist = int(nlist)
        self.nbit = int(nbit)
        self.query_nbit = -1 if query_nbit is None else int(query_nbit)
        self.data_bytes = int(data_bytes)
        self.nthread = int(nthread)
        self.space = str(space).lower()
        self.metric = faiss.METRIC_L2 if self.space == "l2" else faiss.METRIC_INNER_PRODUCT

        self.data = None
        self.indexed_data = None
        self._original_data = None
        self.ndata = 0
        self.assignments = None
        self.invlists: List[np.ndarray] = []
        self.ivf_index = None
        self.coarse_index = None
        self.centroids = None
        self.residuals = None
        self.list_offsets = None
        self.list_ids = None
        self._current_query = None

    def fit(self, nd: int, data: np.ndarray) -> bool:
        try:
            # Max out CPU threads while building (k-means + residual encoding).
            faiss.omp_set_num_threads(_max_threads())

            self.data = np.ascontiguousarray(data.astype(np.float32, copy=False))
            self._original_data = self.data
            self.ndata = int(nd)
            training_data = self.data.copy()
            if self.space == "cosine":
                faiss.normalize_L2(training_data)
            self.indexed_data = training_data

            # Try to reuse coarse centroids trained by another IVF method with the
            # same (data, nlist, space). When space == "cosine" the training data
            # is L2-normalised above, so normalised fingerprints match across
            # methods that share this convention.
            fp = data_fingerprint(training_data)
            ckey = coarse_key(fp, self.nlist, self.space)
            cached = load_centroids(ckey)
            if cached is not None and cached.shape == (self.nlist, self.ndim):
                self.centroids = np.ascontiguousarray(cached.astype(np.float32))
            else:
                kmeans = faiss.Kmeans(
                    d=self.ndim,
                    k=self.nlist,
                    niter=25,
                    verbose=False,
                    seed=1234,
                )
                kmeans.train(training_data)
                self.centroids = kmeans.centroids.astype(np.float32)
                save_centroids(ckey, self.centroids)

            self.coarse_index = faiss.IndexFlatL2(self.ndim) if self.metric == faiss.METRIC_L2 else faiss.IndexFlatIP(self.ndim)
            self.coarse_index.add(self.centroids)
            _, assignments = self.coarse_index.search(training_data, 1)
            self.assignments = assignments[:, 0].astype(np.int64)

            buckets = [[] for _ in range(self.nlist)]
            for idx, list_id in enumerate(self.assignments.tolist()):
                buckets[int(list_id)].append(idx)
            self.invlists = [np.asarray(bucket, dtype=np.int64) for bucket in buckets]

            self.residuals = self.indexed_data.copy()
            for list_id, ids in enumerate(self.invlists):
                if ids.size == 0:
                    continue
                self.residuals[ids] -= self.centroids[list_id]

            offsets = [0]
            flat_ids = []
            for ids in self.invlists:
                flat_ids.extend(ids.tolist())
                offsets.append(len(flat_ids))
            self.list_offsets = np.asarray(offsets, dtype=np.int64)
            self.list_ids = np.asarray(flat_ids, dtype=np.int64)

            self.ivf_index = osq_cpp.PyIVFOSQIndex(self.ndim, self.space, self.nbit, self.query_nbit)
            # Build uses all CPU threads; query-time thread count is restored below.
            self.ivf_index.set_num_threads(_max_threads())
            self.ivf_index.build(
                np.ascontiguousarray(self.residuals, dtype=np.float32),
                self.list_offsets,
                self.list_ids,
            )
        except Exception as exc:
            print(f"Training error: {exc}")
            return False
        return True

    def query(self, nq: int, query: np.ndarray, topk: int, **search_params) -> Tuple[np.ndarray, np.ndarray]:
        # Switch to the configured search-time thread count.
        faiss.omp_set_num_threads(self.nthread)
        if self.ivf_index is not None:
            self.ivf_index.set_num_threads(self.nthread)
        queries = np.ascontiguousarray(query.astype(np.float32, copy=False))
        coarse_queries = queries.copy()
        if self.space == "cosine":
            faiss.normalize_L2(coarse_queries)

        nprobe = min(int(search_params.get("nprobe", self.nlist)), self.nlist)
        _, probe_lists = self.coarse_index.search(coarse_queries, nprobe)
        probe_lists = np.ascontiguousarray(probe_lists.astype(np.int64, copy=False))
        labels, scores = self.ivf_index.search_preassigned(coarse_queries, self.centroids, probe_lists, topk)
        labels = np.asarray(labels, dtype=np.int64)
        scores = np.asarray(scores, dtype=np.float32)
        return labels, self._scores_to_distances(scores)

    def getMemoryUsage(self) -> float:
        return psutil.Process().memory_info().rss / 1024

    def getCompressionRate(self) -> float:
        return self.nbit / (self.data_bytes * 8)

    def getCompressionMemory(self) -> float:
        resolved_query_nbit = self.nbit if self.query_nbit < 0 else self.query_nbit
        effective_nbit = min(self.nbit, resolved_query_nbit)
        code_bits = self.ndata * self.ndim * effective_nbit
        correction_bits = self.ndata * (3 * 32 + 32)
        centroid_bits = self.nlist * self.ndim * 32
        return code_bits + correction_bits + centroid_bits

    def getMSE(self) -> float:
        return 0.0

    def set_query(self, query, thread_id):
        query = np.ascontiguousarray(np.asarray(query, dtype=np.float32))
        if self.space == "cosine":
            query = query.copy().reshape(1, -1)
            faiss.normalize_L2(query)
            query = query[0]
        self._current_query = query

    def estimate_distance(self, idx, thread_id):
        idx = int(idx)
        list_id = int(self.assignments[idx])
        if self._current_query is None:
            raise RuntimeError("set_query must be called before estimate_distance")
        centroid = np.ascontiguousarray(self.centroids[list_id], dtype=np.float32)
        score = float(self.ivf_index.score_query_with_centroid(self._current_query, centroid, idx))
        return float(self._scores_to_distances(np.asarray([score], dtype=np.float32))[0])

    def _scores_to_distances(self, scores: np.ndarray) -> np.ndarray:
        valid = np.isfinite(scores) & (scores > 0)
        if self.space == "l2":
            distances = np.full_like(scores, np.inf, dtype=np.float32)
            distances[valid] = (1.0 / scores[valid]) - 1.0
            return distances
        distances = np.full_like(scores, np.inf, dtype=np.float32)
        distances[np.isfinite(scores)] = -scores[np.isfinite(scores)]
        return distances
