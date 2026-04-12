import numpy as np
from typing import Tuple
import psutil
import sys

sys.path.insert(0, '/benchmark')
from benchmark.base import BaseQuantizer

import scann


class SCANN(BaseQuantizer):
    """
    ScaNN (Scalable Nearest Neighbors) IVF-style index.

    Uses tree partitioning + asymmetric hashing (AH) for approximate search.
    All datasets are L2; ScaNN's "squared_l2" distance is used directly.
    Reordering (exact rescoring) is built-in via ScaNN's .reorder() step and
    controlled at search time via pre_reorder_num_neighbors.

    Build params: num_leaves, dims_per_block, data_bytes, nthread, space
    Search params: num_leaves_to_search (passed via **search_params)
    Rerank: nrerank (passed to searchAndRerank; uses ScaNN's exact reorder pass)
    """

    # Maximum reorder candidates – built into the searcher at construction time.
    _MAX_REORDER = 2500

    def __init__(
        self,
        ndim,
        num_leaves,
        dims_per_block=2,
        data_bytes=4,
        nthread=1,
        space="l2",
    ):
        super().__init__()
        self.ndim = int(ndim)
        self.num_leaves = int(num_leaves)
        self.dims_per_block = int(dims_per_block)
        self.data_bytes = int(data_bytes)
        self.nthread = int(nthread)
        self.space = str(space).lower()

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
                    .score_ah(self.dims_per_block)
                    .reorder(self._MAX_REORDER)
                )
            else:
                # Fallback for non-L2 spaces: normalise and use dot_product.
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
                    )
                    .reorder(self._MAX_REORDER)
                )

            self.searcher = builder.set_n_training_threads(self.nthread).build()
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
        Approximate AH search followed by a *small* exact reorder pass
        (pre_reorder_num_neighbors = topk to minimise overhead).
        """
        queries = np.ascontiguousarray(query.astype(np.float32, copy=False))

        if self.space != "l2":
            queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

        num_leaves_to_search = min(
            int(search_params.get("num_leaves_to_search", min(100, self.num_leaves))),
            self.num_leaves,
        )
        # Cap topk at dataset size.
        effective_topk = min(topk, self.ndata)
        # Rerank only topk candidates exactly (keeps the "no-rerank" baseline fast).
        pre_reorder = min(effective_topk, self._MAX_REORDER)

        I, D = self.searcher.search_batched(
            queries,
            leaves_to_search=num_leaves_to_search,
            pre_reorder_num_neighbors=pre_reorder,
            final_num_neighbors=effective_topk,
        )

        I = self._pad_results(np.asarray(I, dtype=np.int64), nq, effective_topk)
        D = self._pad_results(np.asarray(D, dtype=np.float32), nq, effective_topk)
        return I, D

    def searchAndRerank(
        self,
        nq: int,
        queries: np.ndarray,
        topk: int,
        nrerank: int,
        **search_params,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        ScaNN-native reranking: AH coarse search → exact reorder of nrerank candidates.
        """
        queries = np.ascontiguousarray(queries.astype(np.float32, copy=False))

        if self.space != "l2":
            queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

        num_leaves_to_search = min(
            int(search_params.get("num_leaves_to_search", min(100, self.num_leaves))),
            self.num_leaves,
        )
        effective_topk = min(topk, self.ndata)
        pre_reorder = min(int(nrerank), self._MAX_REORDER)

        I, D = self.searcher.search_batched(
            queries,
            leaves_to_search=num_leaves_to_search,
            pre_reorder_num_neighbors=pre_reorder,
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
        AH with lut16 (default): 4 bits per block.
        Rate = (ndim * 32 bits) / (ndim / dims_per_block * 4 bits)
             = 8 * dims_per_block
        For dims_per_block=2 → 16x compression.
        """
        original_bits = self.ndim * self.data_bytes * 8
        compressed_bits = (self.ndim // self.dims_per_block) * 4  # lut16 = 4 bits
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
