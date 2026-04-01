import sys
import numpy as np
from typing import Tuple
import psutil

sys.path.insert(0, '/benchmark')
from benchmark.base import BaseQuantizer
import ExtendedRabitQ
import ExtendedRabitQ_HighAcc
import faiss


class ExtendedRabitQNTU(BaseQuantizer):
    def __init__(self, ndim, nbit, data_bytes, high_acc_flag, nthread=1, nlist=1):
        super().__init__()
        self.high_acc_flag = high_acc_flag
        self.ndim = ndim
        self.nbit = nbit
        self.nthread = nthread
        self.nlist = max(1, int(nlist))
        faiss.omp_set_num_threads(nthread)
        self.data_bytes = data_bytes
        self.data = None
        self.centroids = None
        self.IDs = None
        self.nd = None
        self._trained_centroids = None
        self.coarse_index = None
        self.Index = self._create_index()

    def _create_index(self):
        if self.high_acc_flag:
            return ExtendedRabitQ_HighAcc.Index(self.ndim, self.nbit, self.nlist)
        return ExtendedRabitQ.Index(self.ndim, self.nbit, self.nlist)

    def _single_cluster_centroid(self, data: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(data.mean(axis=0, keepdims=True).astype(np.float32, copy=False))

    def _build_coarse_quantizer(self, data: np.ndarray) -> np.ndarray:
        if self.nlist == 1:
            centroids = self._single_cluster_centroid(data)
        else:
            kmeans = faiss.Kmeans(
                d=self.ndim,
                k=self.nlist,
                niter=25,
                verbose=False,
                seed=1234,
            )
            kmeans.train(data)
            centroids = np.ascontiguousarray(kmeans.centroids.astype(np.float32, copy=False))

        self.coarse_index = faiss.IndexFlatL2(self.ndim)
        self.coarse_index.add(centroids)
        return centroids

    def _assign_clusters(self, data: np.ndarray) -> np.ndarray:
        if self.coarse_index is None:
            raise RuntimeError("Coarse quantizer not available. Call train() first.")
        _, assignments = self.coarse_index.search(data, 1)
        return np.ascontiguousarray(assignments.reshape(-1).astype(np.uint32, copy=False))

    def fit(self, nd: int, data: np.ndarray):
        return self.train(nd, data) and self.add(nd, data)

    def train(self, nd: int, data: np.ndarray):
        try:
            train_data = np.ascontiguousarray(data.astype(np.float32, copy=False))
            self._trained_centroids = self._build_coarse_quantizer(train_data)
            self.trained = False
        except Exception as e:
            print(f"Training error: {e}")
            return False
        return True

    def add(self, nd: int, data: np.ndarray):
        try:
            if self._trained_centroids is None:
                raise RuntimeError("Index not trained. Call train() first.")

            self.data = np.ascontiguousarray(data.astype(np.float32, copy=False))
            self._original_data = self.data
            self.centroids = np.ascontiguousarray(self._trained_centroids.copy())
            self.IDs = self._assign_clusters(self.data)
            self.nd = nd

            self.Index = self._create_index()
            self.Index.train(self.data, self.centroids, self.IDs, nd, self.nthread)
            self.trained = True
        except Exception as e:
            print(f"Add error: {e}")
            return False
        return True

    def query(self, nq: int, queries: np.ndarray, topk: int, **search_params) -> Tuple[np.ndarray, np.ndarray]:
        queries = np.ascontiguousarray(queries.astype(np.float32, copy=False))
        nprobe = int(search_params.get('nprobe', 1))
        nprobe = max(1, min(nprobe, self.centroids.shape[0]))
        I, D = self.Index.search(queries, nq, topk, nprobe, self.nthread)
        D = np.abs(D)
        return I, D

    def searchAndRerank(self, nq, query, topk, nrerank, **search_params):
        query = np.ascontiguousarray(query.astype(np.float32, copy=False))
        nprobe = int(search_params.get('nprobe', 1))
        nprobe = max(1, min(nprobe, self.centroids.shape[0]))
        I, D = self.Index.search(query, nq, nrerank, nprobe, self.nthread)
        selected = self.data[I]
        diff = selected - query[:, None, :]
        D = np.linalg.norm(diff, axis=2)
        topk_idx = np.argsort(D, axis=1)[:, :topk]
        I = np.take_along_axis(I, topk_idx, axis=1)
        D = np.take_along_axis(D, topk_idx, axis=1)
        return I, D

    def getMemoryUsage(self) -> float:
        return psutil.Process().memory_info().rss / 1024

    def getCompressionRate(self) -> float:
        return self.nbit / (self.data_bytes * 8)

    def getMSE(self) -> float:
        return self.Index.getMSE(self.data, self.centroids, self.IDs, self.nd, self.nthread)

    def set_query(self, query, thread_id):
        pass

    def estimate_distance(self, idx, thread_id):
        pass
