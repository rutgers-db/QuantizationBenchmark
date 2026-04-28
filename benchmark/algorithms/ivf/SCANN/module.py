import os
import numpy as np
from typing import Tuple
import psutil
import sys

sys.path.insert(0, '/benchmark')
from benchmark.base import BaseQuantizer

import scann
from threadpoolctl import threadpool_limits


def _max_threads() -> int:
    return max(1, (os.cpu_count() or 1))


class SCANN(BaseQuantizer):
    """
    ScaNN (Scalable Nearest Neighbors) IVF-style index.

    Uses tree partitioning + asymmetric hashing (AH) for approximate search.
    Maps `space` to ScaNN's native distance measures:
      - "l2"     -> "squared_l2", raw vectors, no AVQ
      - "ip"     -> "dot_product", raw vectors, anisotropic_quantization=0.2
      - "cosine" -> "dot_product", L2-normalised vectors, anisotropic_quantization=0.2

    The searcher is built WITHOUT .reorder() so query() returns pure AH-scored
    results (no hidden exact-rescoring overhead in the no-rerank baseline).
    Reranking is handled by BaseQuantizer.searchAndRerank, which calls query()
    with nrerank candidates then does exact distance in Python — this matches
    what other IVF methods do and keeps the comparison fair.

    Build params: num_leaves, dims_per_block, hash_type, data_bytes, nthread, space
    Search params: num_leaves_to_search (passed via **search_params)
    """

    def __init__(
        self,
        ndim,
        num_leaves,
        dims_per_block=2,
        hash_type="lut16",
        data_bytes=4,
        nthread=1,
        space="l2",
    ):
        super().__init__()
        self.ndim = int(ndim)
        self.num_leaves = int(num_leaves)
        self.dims_per_block = int(dims_per_block)
        self.hash_type = str(hash_type).lower()
        self.data_bytes = int(data_bytes)
        self.nthread = int(nthread)
        self.space = str(space).lower()

        if self.hash_type not in ("lut16", "lut256"):
            raise ValueError(
                f"hash_type must be 'lut16' or 'lut256', got {self.hash_type!r}"
            )
        if self.ndim % self.dims_per_block != 0:
            raise ValueError(
                f"ndim ({self.ndim}) must be divisible by dims_per_block ({self.dims_per_block})"
            )

        self.searcher = None
        self.data = None
        self._original_data = None
        self.ndata = 0
        self._current_query = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def fit(self, nd: int, data: np.ndarray) -> bool:
        self.ndata = nd
        self.data = np.ascontiguousarray(data.astype(np.float32, copy=False))
        self._original_data = self.data

        # Use at most 250 000 samples for k-means tree training (ScaNN default).
        training_sample_size = min(nd, 250000)

        # Max out threads during build (k-means + AH codebook training).
        build_threads = _max_threads()

        try:
            if self.space == "l2":
                # ScaNN supports squared_l2 natively – no transformation needed.
                builder = (
                    scann.scann_ops_pybind.builder(
                        self.data, 10, "squared_l2"
                    )
                    .tree(
                        num_leaves=self.num_leaves,
                        num_leaves_to_search=min(100, self.num_leaves),
                        training_sample_size=training_sample_size,
                    )
                    .score_ah(self.dims_per_block, hash_type=self.hash_type)
                )
            elif self.space in ("ip", "inner_product"):
                # Pure MIPS — raw vectors, ScaNN's dot_product, AVQ enabled.
                builder = (
                    scann.scann_ops_pybind.builder(
                        self.data, 10, "dot_product"
                    )
                    .tree(
                        num_leaves=self.num_leaves,
                        num_leaves_to_search=min(100, self.num_leaves),
                        training_sample_size=training_sample_size,
                    )
                    .score_ah(
                        self.dims_per_block,
                        anisotropic_quantization_threshold=0.2,
                        hash_type=self.hash_type,
                    )
                )
            else:
                # Cosine: normalise so dot_product == cosine similarity.
                normalized = self.data / np.linalg.norm(
                    self.data, axis=1, keepdims=True
                )
                builder = (
                    scann.scann_ops_pybind.builder(
                        normalized, 10, "dot_product"
                    )
                    .tree(
                        num_leaves=self.num_leaves,
                        num_leaves_to_search=min(100, self.num_leaves),
                        training_sample_size=training_sample_size,
                    )
                    .score_ah(
                        self.dims_per_block,
                        anisotropic_quantization_threshold=0.2,
                        hash_type=self.hash_type,
                    )
                )

            # threadpool_limits caps OpenMP/BLAS threads during build; ScaNN's
            # own k-means training threads are set via set_n_training_threads.
            with threadpool_limits(limits=build_threads):
                self.searcher = (
                    builder.set_n_training_threads(build_threads).build()
                )
        except Exception as e:
            print(f"ScaNN build error: {e}")
            return False
        return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def query(
        self, nq: int, query: np.ndarray, topk: int, **search_params
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pure AH-scored search – no exact reranking.
        BaseQuantizer.searchAndRerank will handle any reranking in Python so
        the ScaNN baseline is a fair apples-to-apples vs other IVF methods.
        """
        queries = np.ascontiguousarray(query.astype(np.float32, copy=False))

        # Cosine path normalises both DB and query so dot_product == cosine sim.
        # IP / MIPS uses raw vectors on both sides.
        if self.space == "cosine":
            queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

        num_leaves_to_search = min(
            int(search_params.get("num_leaves_to_search", min(100, self.num_leaves))),
            self.num_leaves,
        )
        effective_topk = min(topk, self.ndata)

        # Parallelise across queries using nthread; threadpool_limits caps
        # OpenMP/BLAS to the same count so we don't oversubscribe cores.
        with threadpool_limits(limits=self.nthread):
            I, D = self.searcher.search_batched_parallel(
                queries,
                leaves_to_search=num_leaves_to_search,
                final_num_neighbors=effective_topk,
            )

        I = self._pad_results(np.asarray(I, dtype=np.int64), nq, effective_topk)
        D = self._pad_results(np.asarray(D, dtype=np.float32), nq, effective_topk)
        return I, D

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def getMemoryUsage(self) -> float:
        return psutil.Process().memory_info().rss / 1024

    def getCompressionRate(self) -> float:
        """
        AH compression ratio depends on hash_type:
          lut16  → 4 bits per block (16 centroids)
          lut256 → 8 bits per block (256 centroids)
        Rate = (ndim * data_bytes * 8) / (ndim / dims_per_block * bits_per_block)
        """
        bits_per_block = 4 if self.hash_type == "lut16" else 8
        original_bits = self.ndim * self.data_bytes * 8
        compressed_bits = (self.ndim // self.dims_per_block) * bits_per_block
        return original_bits / compressed_bits

    def getMSE(self) -> float:
        # ScaNN does not expose reconstruction vectors in its Python API.
        return 0.0

    # ------------------------------------------------------------------
    # Graph-index interface (not used for standalone IVF search)
    # ------------------------------------------------------------------

    def set_query(self, query, thread_id):
        self._current_query = np.ascontiguousarray(
            np.asarray(query, dtype=np.float32)
        )

    def estimate_distance(self, idx, thread_id):
        """Exact squared-L2 fallback (ScaNN does not expose AH distance tables)."""
        if self._current_query is None:
            raise RuntimeError("set_query must be called before estimate_distance")
        diff = self._original_data[int(idx)] - self._current_query
        return float(np.dot(diff, diff))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pad_results(arr: np.ndarray, nq: int, k: int) -> np.ndarray:
        """Pad result array to shape (nq, k) if ScaNN returned fewer than k neighbours."""
        if arr.shape == (nq, k):
            return arr
        fill = -1 if np.issubdtype(arr.dtype, np.integer) else np.inf
        out = np.full((nq, k), fill, dtype=arr.dtype)
        rows, cols = arr.shape
        out[:rows, :cols] = arr
        return out
