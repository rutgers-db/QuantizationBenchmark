import numpy as np
from typing import Tuple
import sys
import os

# Add benchmark to path for importing BaseGraphIndex
sys.path.insert(0, '/benchmark')
from benchmark.base import BaseGraphIndex

# Import the SymphonyQG module (will be built in Docker)
try:
    import symphonyqg
except ImportError as e:
    print(f"Warning: Could not import symphonyqg: {e}")
    symphonyqg = None


class SymphonyQG(BaseGraphIndex):
    """
    SymphonyQG graph index with integrated quantization.

    SymphonyQG integrates quantization directly into the graph structure,
    so it does not require an external quantizer. The quantizer parameter
    is set to None.

    Reference: SIGMOD 2025 - SymphonyQG: Towards Symphonious Integration of
    Quantization and Graph for Approximate Nearest Neighbor Search
    """

    def __init__(self, quantizer=None, degree_bound: int = 32,
                 build_ef: int = 200, num_iter: int = 3,
                 num_threads: int = 16, metric: str = "L2", **kwargs):
        """
        Initialize SymphonyQG.

        Args:
            quantizer: Not used (must be None). Kept for API compatibility.
            degree_bound: Maximum out-degree of graph, must be a multiple of 32 (default: 32)
            build_ef: EF parameter for graph construction (default: 200)
            num_iter: Number of iterations for indexing (default: 3)
            num_threads: Number of threads for indexing (default: 16)
            metric: Distance metric, currently only "L2" supported (default: "L2")
            **kwargs: Additional parameters
        """
        # SymphonyQG has integrated quantization, so quantizer is always None
        super().__init__(quantizer=None, **kwargs)

        if symphonyqg is None:
            raise RuntimeError(
                "Python module 'symphonyqg' is not available. "
                "SymphonyQG requires the symphonyqg library. "
                "Make sure it was built correctly in the Docker image."
            )

        self.degree_bound = degree_bound
        self.build_ef = build_ef
        self.num_iter = num_iter
        self.num_threads = num_threads
        self.metric = metric
        self.index = None
        self.num_points = 0
        self.dimension = 0
        self.trained = False

    def build(self, nd: int, data: np.ndarray, **kwargs) -> bool:
        """
        Build the SymphonyQG index.

        Args:
            nd: Number of data vectors
            data: Training data of shape (nd, d)
            **kwargs: Additional parameters

        Returns:
            bool: True if building was successful
        """
        try:
            self.num_points = nd
            self.dimension = data.shape[1]
            self._original_data = np.ascontiguousarray(data, dtype=np.float32)

            print(f"Building SymphonyQG index...")
            print(f"  num_elements: {self.num_points}")
            print(f"  dimension: {self.dimension}")
            print(f"  degree_bound: {self.degree_bound}")
            print(f"  build_ef: {self.build_ef}")
            print(f"  num_iter: {self.num_iter}")
            print(f"  num_threads: {self.num_threads}")

            # Initialize index
            self.index = symphonyqg.Index(
                "QG",
                self.metric,
                num_elements=self.num_points,
                dimension=self.dimension,
                degree_bound=self.degree_bound
            )

            # Build the index
            # Note: SymphonyQG API uses 'num_thread' (singular)
            self.index.build_index(
                self._original_data,
                self.build_ef,
                num_iter=self.num_iter,
                num_thread=self.num_threads
            )

            self.trained = True
            print("SymphonyQG index built successfully!")
            return True

        except Exception as e:
            print(f"Error building SymphonyQG index: {e}")
            import traceback
            traceback.print_exc()
            return False

    def search(self, nq: int, queries: np.ndarray, topk: int, **search_params) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors.

        Args:
            nq: Number of query vectors
            queries: Query vectors of shape (nq, d)
            topk: Number of nearest neighbors to return
            **search_params: Search parameters
                - search_ef: Beam size for search (default: 100)

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - I: Indices of nearest neighbors, shape (nq, topk)
                - D: Distances to nearest neighbors, shape (nq, topk)
                - hops:
                - comps:
                - nrerank:
        """
        if not self.trained:
            raise RuntimeError("Index not trained. Call build() first.")

        queries = np.ascontiguousarray(queries, dtype=np.float32)

        # Get search parameters
        search_ef = search_params.get("search_ef", 100)

        # Set beam size for search
        self.index.set_ef(search_ef)

        # Allocate result arrays
        I = np.zeros((nq, topk), dtype=np.int64)
        # D = np.zeros((nq, topk), dtype=np.float32)
        hops = np.zeros(nq, dtype=np.int32)
        approx_comps = np.zeros(nq, dtype = np.int32)
        exact_comps = np.zeros(nq, dtype=np.int32)

        # Search each query
        # SymphonyQG's search returns only indices, we compute distances separately
        for i in range(nq):
            query = queries[i]
            result, hop, approx_comp, exact_comp = self.index.search(query, topk)
            I[i] = result[:topk]
            # Compute L2 squared distances since SymphonyQG only returns indices
            # D[i] = np.sum((self._original_data[I[i]] - query) ** 2, axis=1)
            hops[i] = hop
            approx_comps[i] = approx_comp
            exact_comps[i] = exact_comp

        # Return: I, D (None), hops, comps (approx_comps), nrerank (exact_comps)
        return I, None, hops, approx_comps, exact_comps

    def getMemoryUsage(self) -> float:
        """
        Get memory usage of the index in KB.

        Returns:
            float: Memory usage in KB
        """
        if not self.trained:
            return 0.0

        # Estimate memory usage based on index structure
        # Graph edges: num_points * degree_bound * 4 bytes (int32)
        # Quantized data: depends on internal compression
        # Rough estimate: graph + compressed vectors

        graph_mem = (self.num_points * self.degree_bound * 4) / 1024  # KB

        # SymphonyQG uses 4-bit quantization with FastScan
        # Estimate: dimension / 2 bytes per vector (4-bit per dimension)
        quantized_data_mem = (self.num_points * self.dimension / 2) / 1024  # KB

        return graph_mem + quantized_data_mem

    def __repr__(self):
        return (f"SymphonyQG(degree_bound={self.degree_bound}, "
                f"build_ef={self.build_ef}, num_iter={self.num_iter}, "
                f"points={self.num_points}, dim={self.dimension})")
