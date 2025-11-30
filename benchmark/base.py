from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple
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


    def search_and_rerank(self, nq: int, queries: np.ndarray, topk: int, nrerank: int, **search_params) -> Tuple[np.ndarray, np.ndarray]:
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
        I_reranked = np.zeros((nq, topk), dtype=np.int64)
        D_reranked = np.zeros((nq, topk), dtype=np.float32)

        for i in range(nq):
            # Get candidate vectors
            candidate_indices = I_candidates[i]
            candidate_vectors = self._original_data[candidate_indices]

            # Compute exact L2 distances
            query_vector = queries[i]
            distances = np.linalg.norm(candidate_vectors - query_vector, axis=1)

            # Sort by distance and take top-k
            sorted_idx = np.argsort(distances)[:topk]
            I_reranked[i] = candidate_indices[sorted_idx]
            D_reranked[i] = distances[sorted_idx]

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
