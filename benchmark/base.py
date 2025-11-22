from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


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
        pass

    @abstractmethod
    def fit(self, data: np.ndarray) -> bool:
        """
        Train the quantizer on the given data.

        Args:
            data: Training data of shape (n, d) where n is the number of vectors
                  and d is the dimensionality

        Returns:
            bool: True if training was successful, False otherwise
        """
        pass

    @abstractmethod
    def query(self, queries: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for the top-k nearest neighbors for each query.

        Args:
            queries: Query vectors of shape (nq, d) where nq is the number of queries
                    and d is the dimensionality
            topk: Number of nearest neighbors to return

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - I: Indices of nearest neighbors, shape (nq, topk)
                - D: Distances to nearest neighbors, shape (nq, topk)
        """
        pass

    @abstractmethod
    def getMemoryUsage(self) -> float:
        """
        Get the memory usage of the quantizer in bytes.

        Returns:
            float: Memory usage in bytes
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
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit the dimensionality reduction model and transform the data.

        Args:
            data: Input data of shape (n, d) where n is the number of vectors
                  and d is the original dimensionality

        Returns:
            np.ndarray: Transformed data of shape (n, d_reduced) where d_reduced
                       is the reduced dimensionality
        """
        pass

    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform new data using the fitted model.

        Args:
            data: Input data of shape (n, d) where n is the number of vectors
                  and d is the original dimensionality

        Returns:
            np.ndarray: Transformed data of shape (n, d_reduced)
        """
        pass

    @abstractmethod
    def getMemoryUsage(self) -> float:
        """
        Get the memory usage of the dimensionality reduction model in bytes.

        Returns:
            float: Memory usage in bytes
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
