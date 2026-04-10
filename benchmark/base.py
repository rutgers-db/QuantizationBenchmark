from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple
import os



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
        Search for the top-k nearest neighbors for each query with reranking.

        This default implementation:
        1. Calls query() with nrerank as topk to get candidate neighbors
        2. Reranks candidates using exact L2 distance
        3. Returns top-k results after reranking

        Subclasses can override this for custom reranking strategies.

        Args:
            nq: Number of query vectors
            queries: Query vectors of shape (nq, d) where d is the dimensionality
            topk: Number of nearest neighbors to return
            nrerank: Number of neighbors to rerank using exact distance
            **search_params: Optional search-time parameters (e.g., nprobe for IVF)

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - I: Indices of nearest neighbors, shape (nq, topk)
                - D: Distances to nearest neighbors, shape (nq, topk)
        """
        if self._original_data is None:
            raise RuntimeError(
                "Original data not available for reranking. "
                "Subclass must either: (1) set self._original_data in fit(), or "
                "(2) override search_and_rerank() method."
            )

        # Step 1: Get nrerank candidates using approximate search
        I_candidates, _ = self.query(nq, queries, nrerank, **search_params)

        # Step 2: Rerank using exact L2 distance
        prepared_candidates = self.prepareRerankCandidates(queries, I_candidates)
        return self.rerankPreparedCandidates(queries, prepared_candidates, nrerank, topk)

    def prepareRerankCandidates(
        self,
        queries: np.ndarray,
        candidate_indices: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Materialize candidate IDs and vectors once so multiple rerank passes can
        reuse them without repeating setup work.
        """
        original_data = getattr(self, "_original_data", None)
        if original_data is None:
            original_data = getattr(self, "data", None)

        if original_data is None:
            raise RuntimeError(
                "Original data not available for reranking. "
                "Quantizer must expose self._original_data or self.data."
            )

        prepared_candidates = []

        for i in range(candidate_indices.shape[0]):
            valid_mask = candidate_indices[i] >= 0
            valid_indices = candidate_indices[i][valid_mask]
            candidate_vectors = np.ascontiguousarray(original_data[valid_indices].astype(np.float32))
            prepared_candidates.append((valid_indices, candidate_vectors))

        return prepared_candidates

    def rerankPreparedCandidates(
        self,
        queries: np.ndarray,
        prepared_candidates: List[Tuple[np.ndarray, np.ndarray]],
        nrerank: int,
        topk: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rerank from previously prepared candidates using Faiss exact KNN.
        """
        import faiss

        nq = queries.shape[0]
        I_reranked = np.full((nq, topk), -1, dtype=np.int64)
        D_reranked = np.full((nq, topk), np.inf, dtype=np.float32)

        for i in range(nq):
            valid_indices, candidate_vectors = prepared_candidates[i]
            if valid_indices.size == 0:
                continue

            query_vector = np.ascontiguousarray(queries[i:i + 1].astype(np.float32))
            active_count = min(nrerank, valid_indices.size)
            active_indices = valid_indices[:active_count]
            active_vectors = candidate_vectors[:active_count]
            distances_sq, local_indices = faiss.knn(query_vector, active_vectors, min(topk, active_count))

            local_indices = local_indices[0]
            valid_local_mask = local_indices >= 0
            selected_indices = active_indices[local_indices[valid_local_mask]]
            selected_distances = np.sqrt(np.maximum(distances_sq[0][valid_local_mask], 0.0)).astype(np.float32)
            result_count = selected_indices.shape[0]

            I_reranked[i, :result_count] = selected_indices
            D_reranked[i, :result_count] = selected_distances

        return I_reranked, D_reranked


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
