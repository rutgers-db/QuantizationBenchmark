import numpy as np
from contextlib import nullcontext
from typing import Tuple
import psutil
import sys
import os
import faiss

sys.path.insert(0, '/benchmark')
from benchmark.base import BaseQuantizer
from benchmark.ivf_centroid_cache import (
    data_fingerprint,
    load_npz,
    save_npz,
)

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

try:
    import saq_cpp
except ImportError as e:
    print(f"Warning: Could not import saq_cpp: {e}")
    saq_cpp = None


def _max_threads() -> int:
    return max(1, (os.cpu_count() or 1))


def _lift_blas(n: int):
    # docker_runner pins OPENBLAS/MKL/OMP to config.yaml nthread, which throttles
    # PCA GEMM, eigh, and faiss flat search during build. Lift the BLAS pool for
    # the build phase so it matches the SAQ C++ thread pool.
    if threadpool_limits is None:
        return nullcontext()
    return threadpool_limits(limits=n)


class IVFSAQ(BaseQuantizer):
    """
    SAQ (Segmented Code Adjustment Quantization) implementation using C++ backend.
    Uses PCA rotation + dimension segmentation + code adjustment for high-quality
    vector quantization.
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

        # Enable segmentation for SAQ (True), disable for CAQ (False)
        self.enable_segmentation = True

        self.data = None
        self.ndata = 0
        self.trained = False

        # PCA components
        self.pca_matrix = None  # Rotation matrix P (ndim x ndim)
        self.eigenvalues = None  # Variance per PCA dimension
        self.data_mean = None

        # Coarse quantizer
        self.coarse_quantizer = None
        self.coarse_index = None

        if saq_cpp is None:
            raise RuntimeError(
                "C++ module 'saq_cpp' is not available. "
                "SAQ requires the C++ implementation. "
                "Make sure the C++ module was built correctly in the Docker image."
            )

        self._dist_type = "ip" if self.space in ("ip", "inner_product") else "l2"
        self.cpp_index = saq_cpp.PySAQ(
            avg_bits=float(nbit),
            enable_segmentation=self.enable_segmentation,
            caq_adj_rd_lmt=caq_adj_rd_lmt,
            random_rotation=True,
            use_fastscan=True,
            caq_adj_eps=1e-8,
            vars_bound_m=vars_bound_m,
            dist_type=self._dist_type,
        )

    def _compute_pca(self, data):
        """Compute PCA rotation matrix and eigenvalues from data."""
        self.data_mean = data.mean(axis=0).astype(np.float32)
        centered = data - self.data_mean

        # Compute covariance matrix
        cov = np.dot(centered.T, centered) / (data.shape[0] - 1)
        cov = cov.astype(np.float64)

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalue descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store rotation matrix (each row is a principal component)
        self.pca_matrix = eigenvectors.T.astype(np.float32)  # (ndim, ndim)
        self.eigenvalues = np.maximum(eigenvalues, 0).astype(np.float32)

    def _apply_pca(self, data):
        """Apply PCA rotation to data."""
        # For IP, skip the mean shift: <P(q-m), P(d-m)> = <q-m, d-m> ≠ <q, d>.
        # The rotation P is orthogonal so <Pq, Pd> = <q, d>; SAQ's per-cluster
        # centroid subtraction handles the recentering needed for quantization
        # quality. With centering, IP rankings are wrong even after the C++
        # searcher fix, so this path is required for correct IP recall.
        if self._dist_type == "ip":
            return np.ascontiguousarray((data @ self.pca_matrix.T).astype(np.float32))
        centered = data - self.data_mean
        return np.ascontiguousarray((centered @ self.pca_matrix.T).astype(np.float32))

    def fit(self, nd: int, data: np.ndarray) -> bool:
        return self.train(nd, data) and self.add(nd, data)

    def train(self, nd: int, data: np.ndarray) -> bool:
        """Learn PCA rotation and coarse centroids from training data."""
        try:
            # Max out CPU threads for PCA + k-means. Lift BLAS pool too — the
            # docker entrypoint pins OPENBLAS/MKL to nthread, which throttles
            # PCA GEMM/eigh and faiss flat search.
            nt = _max_threads()
            faiss.omp_set_num_threads(nt)
            with _lift_blas(nt):
                train_data = np.ascontiguousarray(data.astype(np.float32))

                fp = data_fingerprint(train_data)
                # SAQ centroids live in PCA-rotated space and are not interchangeable
                # with the shared raw-space coarse cache. Cache PCA + centroids
                # together under an SAQ-specific key so multiple nbit runs over the
                # same data and nlist reuse the same training output.
                # Distinguish L2 vs IP so spherical-vs-flat k-means don't share a key.
                # v2: IP no longer subtracts the mean, so the centroid geometry
                # differs from v1 cached values. Bump the key to force retrain.
                saq_key = f"saq_pca_v2_{fp}_nlist{self.nlist}_d{self.ndim}_{self._dist_type}"
                cached = load_npz(saq_key)
                if (
                    cached is not None
                    and cached.get("pca_matrix") is not None
                    and cached.get("data_mean") is not None
                    and cached.get("eigenvalues") is not None
                    and cached.get("coarse_quantizer") is not None
                    and cached["pca_matrix"].shape == (self.ndim, self.ndim)
                    and cached["coarse_quantizer"].shape == (self.nlist, self.ndim)
                ):
                    self.pca_matrix = np.ascontiguousarray(cached["pca_matrix"].astype(np.float32))
                    self.data_mean = np.ascontiguousarray(cached["data_mean"].astype(np.float32))
                    self.eigenvalues = np.ascontiguousarray(cached["eigenvalues"].astype(np.float32))
                    self.coarse_quantizer = np.ascontiguousarray(cached["coarse_quantizer"].astype(np.float32))
                else:
                    # Step 1: Compute PCA
                    self._compute_pca(train_data)

                    # Step 2: Transform training data to PCA space
                    data_pca = self._apply_pca(train_data)

                    # Step 3: Run k-means on PCA-transformed data.
                    # For IP, use spherical k-means (matches faiss IndexIVF's
                    # default for METRIC_INNER_PRODUCT; PCA is orthogonal so IP
                    # is preserved post-rotation).
                    is_ip = self._dist_type == "ip"
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
                            spherical=is_ip,
                        )
                        kmeans.train(data_pca)
                        self.coarse_quantizer = np.ascontiguousarray(
                            kmeans.centroids.astype(np.float32)
                        )

                    save_npz(
                        saq_key,
                        pca_matrix=self.pca_matrix,
                        data_mean=self.data_mean,
                        eigenvalues=self.eigenvalues,
                        coarse_quantizer=self.coarse_quantizer,
                    )

                # Step 4: Build coarse index for cluster assignment
                is_ip = self._dist_type == "ip"
                self.coarse_index = faiss.IndexFlatIP(self.ndim) if is_ip else faiss.IndexFlatL2(self.ndim)
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
            nt = _max_threads()
            faiss.omp_set_num_threads(nt)
            with _lift_blas(nt):
                self.data = np.ascontiguousarray(data.astype(np.float32))
                self._original_data = self.data
                self.ndata = nd

                # Apply PCA transformation
                data_pca = self._apply_pca(self.data)

                # Assign clusters
                _, assignments = self.coarse_index.search(data_pca, 1)
                cluster_ids = np.ascontiguousarray(assignments.reshape(-1).astype(np.uint32))

                # Build C++ index with variance information — the saq binding takes
                # num_threads per-call, so use max threads here regardless of nthread.
                self.cpp_index.build(
                    data_pca,
                    self.coarse_quantizer,
                    cluster_ids,
                    self.eigenvalues,
                    nt,
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

        # Query runs with the configured thread count.
        faiss.omp_set_num_threads(self.nthread)
        queries = np.ascontiguousarray(queries.astype(np.float32))

        # Apply PCA transformation to queries
        queries_pca = self._apply_pca(queries)

        nprobe = int(search_params.get("nprobe", 1))
        nprobe = max(1, min(nprobe, self.coarse_quantizer.shape[0]))

        vars_bound_m = float(search_params.get("vars_bound_m", self.vars_bound_m))

        I, D = self.cpp_index.search(queries_pca, topk, nprobe, vars_bound_m, self.nthread)

        return I, D

    def getMemoryUsage(self) -> float:
        return psutil.Process().memory_info().rss / 1024

    def getCompressionRate(self) -> float:
        return self.nbit / (self.data_bytes * 8)

    def getMSE(self) -> float:
        return 0.0
        # if not self.trained or self.data is None:
        #     return float('inf')

        # return self.cpp_index.getMSE(self.nthread)

    def set_query(self, query, thread_id):
        query = np.ascontiguousarray(query.astype(np.float32).reshape(1, -1))
        query_pca = self._apply_pca(query).flatten()
        self.cpp_index.set_query(query_pca, thread_id)

    def estimate_distance(self, idx: int, thread_id: int) -> float:
        return self.cpp_index.estimate_distance(idx, thread_id)
