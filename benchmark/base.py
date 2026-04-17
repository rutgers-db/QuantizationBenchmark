from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import numpy as np
from typing import Tuple
import os

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None


def _rerank_nthread(quantizer) -> int:
    n = getattr(quantizer, "nthread", None)
    if isinstance(n, (int, float)) and int(n) > 0:
        return int(n)
    return int(os.environ.get("OMP_NUM_THREADS", 0)) or (os.cpu_count() or 1)


def _pin_blas(n: int):
    return threadpool_limits(limits=n) if threadpool_limits is not None else nullcontext()


def _parallel_l2_topk(
    I: np.ndarray,
    selected: np.ndarray,
    queries: np.ndarray,
    topk: int,
    nthread: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parallel exact-L2 rerank of per-query candidate sets.

    `selected` has shape (nq, nrerank, d); `I` has shape (nq, nrerank) with
    `-1` marking invalid candidates. Returns the top-k ids and distances.

    Split the query axis into `nthread` chunks, process each chunk in a worker
    thread, and (if threadpoolctl is available) pin the BLAS pool to 1 thread
    inside the with-block so the outer pool and BLAS don't oversubscribe.
    The q·c term inside the L2 identity is a batched BLAS matmul, which is
    the only computation worth parallelizing at this scale.
    """
    nq, nrerank = I.shape
    k = min(int(topk), int(nrerank))
    out_I = np.full((nq, topk), -1, dtype=np.int64)
    out_D = np.full((nq, topk), np.inf, dtype=np.float32)
    if nq == 0 or k <= 0:
        return out_I, out_D

    Q = queries if queries.dtype == np.float32 else queries.astype(np.float32, copy=False)

    def rerank_chunk(start: int, end: int) -> None:
        C = selected[start:end]
        Qc = Q[start:end]
        Ic = I[start:end]
        c2 = np.einsum("ijk,ijk->ij", C, C)
        q2 = np.einsum("ij,ij->i", Qc, Qc)[:, None]
        qc = np.matmul(C, Qc[:, :, None])[:, :, 0]
        D = np.sqrt(np.maximum(c2 + q2 - 2.0 * qc, 0.0)).astype(np.float32, copy=False)
        D = np.where(Ic >= 0, D, np.inf)
        order = np.argsort(D, axis=1)[:, :k]
        out_I[start:end, :k] = np.take_along_axis(Ic, order, axis=1)
        out_D[start:end, :k] = np.take_along_axis(D, order, axis=1)

    workers = max(1, min(int(nthread), nq))
    step = max(1, (nq + workers - 1) // workers)
    with _pin_blas(1):
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(lambda s: rerank_chunk(s, min(s + step, nq)), range(0, nq, step)))
    return out_I, out_D


class BaseQuantizer(ABC):
    """
    Base class for vector quantization algorithms.

    All quantizer implementations should inherit from this class and implement
    the required methods.
    """

    def __init__(self, **kwargs):
        """
        Initialize the quantizer with optional parameters.

        Args:
            **kwargs: Algorithm-specific parameters
        """
        # Store original data for default search_and_rerank implementation
        # Subclasses should set this in fit() if they want to use the default implementation
        self._original_data = None

    @abstractmethod
    def fit(self, nd: int, data: np.ndarray) -> bool:
        """
        Train the quantizer on the given data.

        Args:
            nd: Number of data vectors
            data: Training data of shape (nd, d) where d is the dimensionality

        Returns:
            bool: True if training was successful, False otherwise
        """
        pass

    def train(self, nd: int, data: np.ndarray) -> bool:
        """
        Optional train-only hook used by distribution-shift experiments.

        Quantizers that support training codebooks on one sample and adding a
        different database later should override this together with add().
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support separate train/add."
        )

    def add(self, nd: int, data: np.ndarray) -> bool:
        """
        Optional add-only hook used by distribution-shift experiments.

        Quantizers that override train() should also override add().
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support separate train/add."
        )

    @abstractmethod
    def query(self, nq: int, queries: np.ndarray, topk: int, **search_params) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for the top-k nearest neighbors for each query.

        Args:
            nq: Number of query vectors
            queries: Query vectors of shape (nq, d) where d is the dimensionality
            topk: Number of nearest neighbors to return
            **search_params: Optional search-time parameters (e.g., nprobe for IVF)

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - I: Indices of nearest neighbors, shape (nq, topk)
                - D: Distances to nearest neighbors, shape (nq, topk)
        """
        pass

    @abstractmethod
    def getMemoryUsage(self) -> float:
        """
        Get the memory usage of the quantizer in KB.

        Returns:
            float: Memory usage in KB
        """
        pass

    @abstractmethod
    def getCompressionRate(self) -> float:
        """
        Get the compression rate achieved by the quantizer.

        The compression rate is defined as:
        original_size / compressed_size

        Returns:
            float: Compression rate (higher is better)
        """
        pass

    @abstractmethod
    def getMSE(self) -> float:
        """
        Get the mean squared error of the quantization.

        This measures the reconstruction error of the quantized vectors.

        Returns:
            float: Mean squared error
        """
        pass
    
    @abstractmethod
    def set_query(self, query: np.ndarray, thread_id: int) -> None:
        """
        This should be called before estimate_distance so that the distance table of the query can be pre-computed.

        This method is used by graph algorithms to prepare for distance computations.

        Args:
            query: numpy float32 array of query vector

        Returns:
            None
        """
        pass

    @abstractmethod
    def estimate_distance(self, idx: int, thread_id: int) -> float:
        """
        Given an idx, estimate the distance between the previously set query and the idx.

        This method is used by graph algorithms for approximate distance computation during search.
        set_query() must be called before using this method.

        Args:
            idx: int - index of the data point

        Returns:
            float: estimated distance
        """
        pass


    def searchAndRerank(self, nq: int, queries: np.ndarray, topk: int, nrerank: int, **search_params) -> Tuple[np.ndarray, np.ndarray]:
        """
        Approximate-search + exact-L2 rerank, parallelized across `nthread`
        workers via `_parallel_l2_topk`.
        """
        data = self._original_data if getattr(self, "_original_data", None) is not None else getattr(self, "data", None)
        if data is None:
            raise RuntimeError(
                "Original data not available for reranking. "
                "Subclass must set self._original_data in fit() or override searchAndRerank()."
            )
        I, _ = self.query(nq, queries, nrerank, **search_params)
        safe_I = np.where(I >= 0, I, 0)
        selected = data[safe_I].astype(np.float32, copy=False)
        return _parallel_l2_topk(I, selected, queries, topk, _rerank_nthread(self))

    def prepareRerankCandidates(
        self,
        queries: np.ndarray,
        candidate_indices: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gather candidate IDs and their original vectors as uniform (nq, nrerank)
        and (nq, nrerank, d) arrays.
        """
        data = self._original_data if getattr(self, "_original_data", None) is not None else getattr(self, "data", None)
        if data is None:
            raise RuntimeError(
                "Original data not available for reranking. "
                "Quantizer must expose self._original_data or self.data."
            )
        safe_I = np.where(candidate_indices >= 0, candidate_indices, 0)
        selected = data[safe_I].astype(np.float32, copy=False)
        return candidate_indices, selected

    def rerankPreparedCandidates(
        self,
        queries: np.ndarray,
        prepared_candidates: Tuple[np.ndarray, np.ndarray],
        nrerank: int,
        topk: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exact-L2 rerank on prepared (ids, vectors) arrays of shape
        (nq, nrerank) and (nq, nrerank, d), parallelized across `nthread`.
        """
        I, selected = prepared_candidates
        return _parallel_l2_topk(I, selected, queries, topk, _rerank_nthread(self))


class BaseDimReduction(ABC):
    """
    Base class for dimensionality reduction algorithms.

    All dimensionality reduction implementations should inherit from this class
    and implement the required methods.
    """

    def __init__(self, **kwargs):
        """
        Initialize the dimensionality reduction algorithm with optional parameters.

        Args:
            **kwargs: Algorithm-specific parameters
        """
        pass

    @abstractmethod
    def fit_transform(self, n: int, data: np.ndarray) -> np.ndarray:
        """
        Fit the dimensionality reduction model and transform the data.

        Args:
            n: Number of data vectors
            data: Input data of shape (n, d) where d is the original dimensionality

        Returns:
            np.ndarray: Transformed data of shape (n, d_reduced) where d_reduced
                       is the reduced dimensionality
        """
        pass

    @abstractmethod
    def transform(self, n: int, data: np.ndarray) -> np.ndarray:
        """
        Transform new data using the fitted model.

        Args:
            n: Number of data vectors
            data: Input data of shape (n, d) where d is the original dimensionality

        Returns:
            np.ndarray: Transformed data of shape (n, d_reduced)
        """
        pass

    @abstractmethod
    def getMemoryUsage(self) -> float:
        """
        Get the memory usage of the dimensionality reduction model in KB.

        Returns:
            float: Memory usage in KB
        """
        pass

    @abstractmethod
    def getCompressionRate(self) -> float:
        """
        Get the compression rate achieved by dimensionality reduction.

        The compression rate is defined as:
        original_dimension / reduced_dimension

        Returns:
            float: Compression rate (higher is better)
        """
        pass


class BaseGraphIndex(ABC):
    """
    Base class for graph-based index algorithms (e.g., DiskANN, HNSW, NSG).

    Graph algorithms can use quantizers for approximate distance computation
    during search by calling the quantizer's set_query() and estimate_distance() methods.
    """

    def __init__(self, quantizer: BaseQuantizer, **kwargs):
        """
        Initialize the graph index with a quantizer.

        Args:
            quantizer: A quantizer instance that implements set_query() and estimate_distance()
            **kwargs: Algorithm-specific parameters
        """
        self.quantizer = quantizer
        self._original_data = None

    @abstractmethod
    def build(self, nd: int, data: np.ndarray) -> bool:
        """
        Build the graph index on the given data.

        Args:
            nd: Number of data vectors
            data: Training data of shape (nd, d) where d is the dimensionality

        Returns:
            bool: True if building was successful, False otherwise
        """
        pass

    @abstractmethod
    def search(self, nq: int, queries: np.ndarray, topk: int, **search_params) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Search for the top-k nearest neighbors for each query.

        Args:
            nq: Number of query vectors
            queries: Query vectors of shape (nq, d) where d is the dimensionality
            topk: Number of nearest neighbors to return
            **search_params: Optional search-time parameters (e.g., search_list_size, beam_width)

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - I: Indices of nearest neighbors, shape (nq, topk)
                - D: Distances to nearest neighbors, shape (nq, topk)
                - hops: Number of graph hops per query, shape (nq,), or None if not supported
                - comps: Number of distance computations per query, shape (nq,), or None if not supported
                - nrerank: Number of rerank operations per query, shape (nq,), or None if not supported
        """
        pass

    @abstractmethod
    def getMemoryUsage(self) -> float:
        """
        Get the memory usage of the graph index in KB.

        Returns:
            float: Memory usage in KB
        """
        pass
