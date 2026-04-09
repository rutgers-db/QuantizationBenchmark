import numpy as np
from typing import Tuple
import psutil
import sys
import os
import faiss

sys.path.insert(0, '/benchmark')
from benchmark.base import BaseQuantizer

try:
    import saq_cpp
except ImportError as e:
    print(f"Warning: Could not import saq_cpp: {e}")
    saq_cpp = None


class CAQ(BaseQuantizer):
    """
    CAQ (Code Adjustment Quantization) implementation using C++ backend.
    Uses PCA rotation + code adjustment without dimension segmentation.
    This is the non-segmented variant of SAQ.
    """

    def __init__(self, ndim, nbit, data_bytes, nlist=1, nthread=1, space="l2",
                 caq_adj_rd_lmt=6, vars_bound_m=4.0):
        super().__init__()
        self.ndim = ndim
        self.nbit = nbit
        self.data_bytes = data_bytes
        self.nlist = max(1, int(nlist))
        self.nthread = nthread
        self.space = space
        self.caq_adj_rd_lmt = caq_adj_rd_lmt
        self.vars_bound_m = vars_bound_m

        # Disable segmentation for CAQ
        self.enable_segmentation = False

        self.data = None
        self.ndata = 0
        self.trained = False

        # PCA components
        self.pca_matrix = None
        self.eigenvalues = None
        self.data_mean = None

        # Coarse quantizer
        self.coarse_quantizer = None
        self.coarse_index = None

        faiss.omp_set_num_threads(nthread)

        if saq_cpp is None:
            raise RuntimeError(
                "C++ module 'saq_cpp' is not available. "
                "CAQ requires the C++ implementation. "
                "Make sure the C++ module was built correctly in the Docker image."
            )

        self.cpp_index = saq_cpp.PySAQ(
            avg_bits=float(nbit),
            enable_segmentation=self.enable_segmentation,
            caq_adj_rd_lmt=caq_adj_rd_lmt,
            random_rotation=True,
            use_fastscan=True,
            caq_adj_eps=1e-8,
            vars_bound_m=vars_bound_m,
        )

    def _compute_pca(self, data):
        """Compute PCA rotation matrix and eigenvalues from data."""
        self.data_mean = data.mean(axis=0).astype(np.float32)
        centered = data - self.data_mean

        cov = np.dot(centered.T, centered) / (data.shape[0] - 1)
        cov = cov.astype(np.float64)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        self.pca_matrix = eigenvectors.T.astype(np.float32)
        self.eigenvalues = np.maximum(eigenvalues, 0).astype(np.float32)

    def _apply_pca(self, data):
        """Apply PCA rotation to data."""
        centered = data - self.data_mean
        return np.ascontiguousarray((centered @ self.pca_matrix.T).astype(np.float32))

    def fit(self, nd: int, data: np.ndarray) -> bool:
        return self.train(nd, data) and self.add(nd, data)

    def train(self, nd: int, data: np.ndarray) -> bool:
        """Learn PCA rotation and coarse centroids from training data."""
        try:
            train_data = np.ascontiguousarray(data.astype(np.float32))

            self._compute_pca(train_data)

            data_pca = self._apply_pca(train_data)

            if self.nlist == 1:
                self.coarse_quantizer = np.ascontiguousarray(
                    data_pca.mean(axis=0, keepdims=True).astype(np.float32)
                )
            else:
                kmeans = faiss.Kmeans(
                    d=self.ndim,
                    k=self.nlist,
                    niter=25,
                    verbose=False,
                    seed=1234,
                )
                kmeans.train(data_pca)
                self.coarse_quantizer = np.ascontiguousarray(
                    kmeans.centroids.astype(np.float32)
                )

            self.coarse_index = faiss.IndexFlatL2(self.ndim)
            self.coarse_index.add(self.coarse_quantizer)

            self.trained = False
            return True
        except Exception as e:
            print(f"Training error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def add(self, nd: int, data: np.ndarray) -> bool:
        """Encode database vectors with the learned PCA/centroids and build C++ index."""
        if self.coarse_index is None or self.pca_matrix is None:
            raise RuntimeError("Index not trained. Call train() first.")

        try:
            self.data = np.ascontiguousarray(data.astype(np.float32))
            self._original_data = self.data
            self.ndata = nd

            data_pca = self._apply_pca(self.data)

            _, assignments = self.coarse_index.search(data_pca, 1)
            cluster_ids = np.ascontiguousarray(assignments.reshape(-1).astype(np.uint32))

            self.cpp_index.build(
                data_pca,
                self.coarse_quantizer,
                cluster_ids,
                self.eigenvalues,
                self.nthread,
            )

            self.trained = True
            return True
        except Exception as e:
            print(f"Add error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def query(self, nq: int, queries: np.ndarray, topk: int, **search_params) -> Tuple[np.ndarray, np.ndarray]:
        if not self.trained:
            raise RuntimeError("Index not trained. Call fit() first.")

        queries = np.ascontiguousarray(queries.astype(np.float32))

        queries_pca = self._apply_pca(queries)

        nprobe = int(search_params.get("nprobe", 1))
        nprobe = max(1, min(nprobe, self.coarse_quantizer.shape[0]))

        vars_bound_m = float(search_params.get("vars_bound_m", self.vars_bound_m))

        I, D = self.cpp_index.search(queries_pca, topk, nprobe, vars_bound_m, self.nthread)

        return I, D

    def searchAndRerank(self, nq: int, queries: np.ndarray, topk: int, nrerank: int, **search_params) -> Tuple[np.ndarray, np.ndarray]:
        if not self.trained:
            raise RuntimeError("Index not trained. Call fit() first.")

        queries = np.ascontiguousarray(queries.astype(np.float32))

        I_candidates, _ = self.query(nq, queries, nrerank, **search_params)

        prepared_candidates = self.prepareRerankCandidates(queries, I_candidates)
        return self.rerankPreparedCandidates(queries, prepared_candidates, nrerank, topk)

    def getMemoryUsage(self) -> float:
        return psutil.Process().memory_info().rss / 1024

    def getCompressionRate(self) -> float:
        return self.nbit / (self.data_bytes * 8)

    def getMSE(self) -> float:
        if not self.trained or self.data is None:
            return float('inf')

        return self.cpp_index.getMSE(self.nthread)

    def set_query(self, query, thread_id):
        query = np.ascontiguousarray(query.astype(np.float32).reshape(1, -1))
        query_pca = self._apply_pca(query).flatten()
        self.cpp_index.set_query(query_pca, thread_id)

    def estimate_distance(self, idx: int, thread_id: int) -> float:
        return self.cpp_index.estimate_distance(idx, thread_id)
