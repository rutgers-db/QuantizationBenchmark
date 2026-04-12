import numpy as np
from typing import Tuple
import psutil
import sys
import os
import faiss

sys.path.insert(0, '/benchmark')
from benchmark.base import BaseQuantizer

# Import the C++ module
try:
    import rabitqlib_cpp
except ImportError as e:
    print(f"Warning: Could not import rabitqlib_cpp: {e}")
    rabitqlib_cpp = None


class RabitQLibrary(BaseQuantizer):
    """
    RaBitQ implementation using the rabitqlib C++ library backend.

    Uses rabitqlib::ivf::IVF which handles rotation (FHT-KAC) and quantization
    internally, unlike the manual Python preprocessing in the RabitQ variant.
    Supports arbitrary dimensions and configurable bit-widths.
    """

    def __init__(self, ndim, nlist, data_bytes, bits=4, nthread=1, space="l2"):
        """
        Initialize RaBitQLibrary quantizer.

        Args:
            ndim: Dimensionality of vectors
            nlist: Number of IVF clusters (coarse quantizer centroids)
            data_bytes: Size of data type in bytes (typically 4 for float32)
            bits: Total bits per dimension for quantization (default 4)
            nthread: Number of threads for parallel processing
            space: Distance metric ("l2" for Euclidean, "ip" for inner product)
        """
        super().__init__()
        self.ndim = ndim
        self.nlist = nlist
        self.data_bytes = data_bytes
        self.bits = bits
        self.nthread = nthread
        self.space = space

        self.data = None
        self.ndata = 0
        self.trained = False

        self.coarse_index = None
        self._trained_centroids = None

        self.index = None

        if rabitqlib_cpp is None:
            raise RuntimeError(
                "C++ module 'rabitqlib_cpp' is not available. "
                "RabitQLibrary requires the C++ implementation for efficient search. "
                "Make sure the C++ module was built correctly in the Docker image."
            )

    # ------------------------------------------------------------------
    # Training: learn IVF centroids via K-means
    # ------------------------------------------------------------------

    def fit(self, nd: int, data: np.ndarray) -> bool:
        return self.train(nd, data) and self.add(nd, data)

    def train(self, nd: int, data: np.ndarray) -> bool:
        """Learn IVF centroids on the given training sample."""
        try:
            train_data = np.ascontiguousarray(data, dtype=np.float32)

            if self.nlist == 1:
                centroids = np.ascontiguousarray(
                    train_data.mean(axis=0, keepdims=True).astype(np.float32)
                )
            else:
                faiss.omp_set_num_threads(self.nthread)
                kmeans = faiss.Kmeans(
                    d=self.ndim,
                    k=self.nlist,
                    niter=25,
                    verbose=False,
                    seed=1234,
                )
                kmeans.train(train_data)
                centroids = np.ascontiguousarray(kmeans.centroids.astype(np.float32))

            self._trained_centroids = centroids
            self.coarse_index = faiss.IndexFlatL2(self.ndim)
            self.coarse_index.add(centroids)

            self.trained = False
            return True
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            return False

    # ------------------------------------------------------------------
    # Add: encode database and build C++ IVF index
    # ------------------------------------------------------------------

    def add(self, nd: int, data: np.ndarray) -> bool:
        """
        Assign database vectors to clusters and build the C++ IVF index.
        """
        if self._trained_centroids is None or self.coarse_index is None:
            raise RuntimeError("Index not trained. Call train() first.")

        try:
            self.data = np.ascontiguousarray(data, dtype=np.float32)
            self._original_data = self.data
            self.ndata = nd

            # Assign each vector to its nearest centroid
            _, assignments = self.coarse_index.search(self.data, 1)
            cluster_ids = np.ascontiguousarray(assignments.flatten().astype(np.uint32))

            metric_str = "ip" if self.space == "ip" else "l2"

            self.index = rabitqlib_cpp.IVF(
                nd,
                self.ndim,
                int(self._trained_centroids.shape[0]),
                self.bits,
                metric_str,
            )
            self.index.construct(
                self.data,
                self._trained_centroids,
                cluster_ids,
                False,
            )

            self.trained = True
            return True
        except Exception as e:
            print(f"Add error: {e}")
            import traceback
            traceback.print_exc()
            return False

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, nq: int, queries: np.ndarray, topk: int, **search_params) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for the top-k nearest neighbors for each query.

        Args:
            nq: Number of query vectors
            queries: Query vectors of shape (nq, ndim)
            topk: Number of nearest neighbors to return
            **search_params:
                nprobe (int): Number of IVF clusters to probe (default 1)
                use_hacc (bool): Use high-accuracy fastscan (default True)

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - I: Indices of nearest neighbors, shape (nq, topk)
                - D: Distances to nearest neighbors, shape (nq, topk)
        """
        if not self.trained:
            raise RuntimeError("Index not trained. Call fit() first.")

        queries = np.ascontiguousarray(queries, dtype=np.float32)
        nprobe = search_params.get("nprobe", 1)
        nprobe = max(1, min(int(nprobe), int(self._trained_centroids.shape[0])))
        use_hacc = bool(search_params.get("use_hacc", True))

        I, D = self.index.search_batch(queries, self.data, topk, nprobe, use_hacc)
        return I, D

    # ------------------------------------------------------------------
    # Search + rerank
    # ------------------------------------------------------------------

    def searchAndRerank(
        self,
        nq: int,
        queries: np.ndarray,
        topk: int,
        nrerank: int,
        **search_params,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nrerank candidates with RaBitQ, then rerank using exact L2.

        Args:
            nq: Number of query vectors
            queries: Query vectors of shape (nq, ndim)
            topk: Number of nearest neighbors to return
            nrerank: Number of candidates to retrieve before reranking
            **search_params: Passed through to query()

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - I: Reranked indices, shape (nq, topk)
                - D: Exact L2 distances, shape (nq, topk)
        """
        if not self.trained:
            raise RuntimeError("Index not trained. Call fit() first.")

        queries = np.ascontiguousarray(queries, dtype=np.float32)
        nprobe = search_params.get("nprobe", 1)
        nprobe = max(1, min(int(nprobe), int(self._trained_centroids.shape[0])))
        use_hacc = bool(search_params.get("use_hacc", True))

        # Get nrerank approximate candidates
        I_approx, _ = self.index.search_batch(queries, self.data, nrerank, nprobe, use_hacc)

        # Rerank with exact L2 distances
        selected = self.data[I_approx]           # (nq, nrerank, ndim)
        diff = selected - queries[:, None, :]    # (nq, nrerank, ndim)
        D_exact = np.linalg.norm(diff, axis=2)   # (nq, nrerank)

        topk_idx = np.argsort(D_exact, axis=1)[:, :topk]
        I = np.take_along_axis(I_approx, topk_idx, axis=1)
        D = np.take_along_axis(D_exact, topk_idx, axis=1)
        return I, D

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def getMemoryUsage(self) -> float:
        """Return memory usage of this process in KB."""
        return psutil.Process().memory_info().rss / 1024

    def getCompressionRate(self) -> float:
        """
        Compression rate: bits per dimension / original bits per dimension.

        Original: data_bytes * 8 bits per dimension (e.g. 32 for float32).
        Compressed: self.bits bits per dimension.
        """
        return self.bits / (self.data_bytes * 8)

    def getMSE(self) -> float:
        """
        Mean squared error of the quantized representation.

        Computed as the average squared L2 distance between each database
        vector and its assigned centroid (coarse quantization error only,
        since exact reconstruction from binary codes is not exposed by the
        library's Python interface).
        """
        if not self.trained or self.data is None or self._trained_centroids is None:
            return float('inf')

        _, assignments = self.coarse_index.search(self.data, 1)
        assigned_centroids = self._trained_centroids[assignments.flatten()]
        residuals = self.data - assigned_centroids
        mse = float(np.mean(np.sum(residuals ** 2, axis=1)))
        return mse

    # ------------------------------------------------------------------
    # Graph-index hooks (not used for IVF-based search)
    # ------------------------------------------------------------------

    def set_query(self, query, thread_id):
        pass

    def estimate_distance(self, idx: int, thread_id: int) -> float:
        return 0.0
