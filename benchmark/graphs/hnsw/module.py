import numpy as np
from typing import Tuple
import sys

# Add benchmark to path for importing BaseGraphIndex
sys.path.insert(0, '/benchmark')
from benchmark.base import BaseGraphIndex, BaseQuantizer

# Import the C++ module (will be built in Docker)
try:
    import hnswlib
except ImportError as e:
    print(f"Warning: Could not import hnswlib: {e}")
    hnswlib = None


class HNSW(BaseGraphIndex):
    """
    HNSW graph index with pluggable quantizer.

    This implementation uses hnswlib's HNSW graph construction
    and search, but replaces the built-in distance computation with
    any quantizer that implements set_query() and estimate_distance().
    """

    def __init__(self, quantizer: BaseQuantizer, M: int = 16, ef_construction: int = 200,
                 num_threads: int = 16, metric: str = "l2", **kwargs):
        """
        Initialize HNSW with a quantizer.

        Args:
            quantizer: Quantizer instance implementing set_query() and estimate_distance()
            M: Max connections per node (default: 16)
            ef_construction: Construction time search width (default: 200)
            num_threads: Number of threads for parallel operations
            metric: Distance metric ("l2" only for now)
            **kwargs: Additional parameters
        """
        super().__init__(quantizer, **kwargs)

        if hnswlib is None:
            raise RuntimeError(
                "C++ module 'hnswlib' is not available. "
                "HNSW requires the C++ implementation. "
                "Make sure the C++ module was built correctly in the Docker image."
            )

        self.M = M
        self.ef_construction = ef_construction
        self.num_threads = num_threads
        self.metric = metric
        self.cpp_index = None
        self.num_points = 0
        self.dimension = 0
        self.trained = False

    def set_quantizer(self, quantizer) -> None:
        """
        Set a new quantizer without rebuilding the graph.

        This allows testing multiple quantization parameters on the same graph structure.
        The quantizer must be already trained before calling this method.

        Args:
            quantizer: A trained quantizer instance with set_query() and estimate_distance() methods
        """
        if not self.trained:
            raise RuntimeError("Index not built yet. Call build() first before changing quantizer.")

        self.quantizer = quantizer
        # Re-initialize index with new quantizer
        old_index = self.cpp_index
        self.cpp_index = hnswlib.QuantizationIndex(self.metric, self.dimension)
        self.cpp_index.init_index(
            max_elements=self.num_points,
            M=self.M,
            ef_construction=self.ef_construction,
            random_seed=42,
            quantizer=quantizer,
            allow_replace_deleted=False
        )
        self.cpp_index.add_items(self._original_data, num_threads=self.num_threads)
        print("Quantizer updated successfully (graph rebuilt)")

    def build(self, nd: int, data: np.ndarray, train_quantizer: bool = True) -> bool:
        """
        Build the HNSW index.

        This performs the following steps:
        1. Train the quantizer on the data (optional)
        2. Build HNSW graph
        3. Add all data points to the graph

        Args:
            nd: Number of data vectors
            data: Training data of shape (nd, d)
            train_quantizer: If True, train the quantizer before building (default: True)

        Returns:
            bool: True if building was successful
        """
        try:
            self.num_points = nd
            self.dimension = data.shape[1]
            self._original_data = np.ascontiguousarray(data, dtype=np.float32)

            print(f"Building HNSW index: M={self.M}, ef_construction={self.ef_construction}, threads={self.num_threads}")

            # Train quantizer if requested
            if train_quantizer and self.quantizer:
                print("Training quantizer...")
                self.quantizer.fit(nd, self._original_data)

            # Create C++ index
            self.cpp_index = hnswlib.QuantizationIndex(self.metric, self.dimension)
            self.cpp_index.init_index(
                max_elements=nd,
                M=self.M,
                ef_construction=self.ef_construction,
                random_seed=42,
                quantizer=self.quantizer,
                allow_replace_deleted=False
            )
            self.cpp_index.set_num_threads(self.num_threads)

            # Add items to index
            print("Adding items to HNSW index...")
            self.cpp_index.add_items(self._original_data, num_threads=self.num_threads)

            self.trained = True
            print("HNSW index built successfully!")
            return True

        except Exception as e:
            print(f"Error building HNSW index: {e}")
            import traceback
            traceback.print_exc()
            return False

    def search(self, nq: int, queries: np.ndarray, topk: int, **search_params) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors using the quantizer.

        During search, the quantizer's estimate_distance() is used to
        compute approximate distances for candidate filtering.

        Args:
            nq: Number of query vectors
            queries: Query vectors of shape (nq, d)
            topk: Number of nearest neighbors to return
            **search_params: Search parameters
                - ef: Search time search width (default: max(ef_construction, 2*topk))

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - I: Indices of nearest neighbors, shape (nq, topk)
                - D: Distances to nearest neighbors, shape (nq, topk)
                - hops: Number of graph hops per query, shape (nq,)
                - comps: Number of distance computations per query, shape (nq,)
                - nrerank: Number of rerank operations per query, shape (nq,) (equals ef for each query)
        """
        if not self.trained:
            raise RuntimeError("Index not trained. Call build() first.")

        queries = np.ascontiguousarray(queries, dtype=np.float32)

        # Get search parameters
        ef = search_params.get("ef", max(self.ef_construction, 2 * topk))

        # Use C++ search with quantizer callbacks
        # Returns (indices, distances, hops, computations)
        I, D, hops, comps = self.cpp_index.knn_query(queries, topk, ef)

        # nrerank is ef for each query (number of candidates reranked with exact distances)
        nrerank = np.full(nq, ef, dtype=np.int64)

        return I.astype(np.int64), D, hops, comps, nrerank

    def getMemoryUsage(self) -> float:
        """
        Get memory usage of the index in KB.

        This includes:
        - Graph structure
        - Quantizer memory

        Returns:
            float: Memory usage in KB
        """
        if not self.trained:
            return 0.0

        # Get quantizer memory
        quantizer_mem = self.quantizer.getMemoryUsage()

        # Estimate graph memory: each node has M neighbors at level 0 and M/2 at higher levels
        # Approximate: M * 2 * num_points * 4 bytes
        graph_mem = (self.num_points * self.M * 2 * 4) / 1024  # KB

        return quantizer_mem + graph_mem

    def get_index_stats(self) -> dict:
        """Get detailed index statistics."""
        return {
            "num_points": self.num_points,
            "dimension": self.dimension,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "quantizer": self.quantizer.__class__.__name__ if self.quantizer else None
        }

    def __repr__(self):
        return (f"HNSW(M={self.M}, ef_construction={self.ef_construction}, "
                f"points={self.num_points}, dim={self.dimension}, "
                f"quantizer={self.quantizer.__class__.__name__})")
