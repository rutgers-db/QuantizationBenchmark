import numpy as np
from typing import Tuple
import sys
import os
import tempfile

# Add benchmark to path for importing BaseGraphIndex
sys.path.insert(0, '/benchmark')
from benchmark.base import BaseGraphIndex, BaseQuantizer

# Import the C++ module (will be built in Docker)
try:
    import diskann_cpp
except ImportError as e:
    print(f"Warning: Could not import diskann_cpp: {e}")
    diskann_cpp = None


class DiskANN(BaseGraphIndex):
    """
    DiskANN graph index with pluggable quantizer.

    This implementation uses DiskANN's disk-based index construction
    and search, but replaces the built-in Product Quantization with
    any quantizer that implements set_query() and estimate_distance().
    """

    def __init__(self, quantizer: BaseQuantizer, R: int = 64, L: int = 100,
                 num_threads: int = 16, metric: str = "l2", **kwargs):
        """
        Initialize DiskANN with a quantizer.

        Args:
            quantizer: Quantizer instance implementing set_query() and estimate_distance()
            R: Max out-degree for graph (default: 64)
            L: Build complexity/search list size (default: 100)
            num_threads: Number of threads for parallel operations
            metric: Distance metric ("l2" or "mips")
            **kwargs: Additional parameters
        """
        super().__init__(quantizer, **kwargs)

        if diskann_cpp is None:
            raise RuntimeError(
                "C++ module 'diskann_cpp' is not available. "
                "DiskANN requires the C++ implementation. "
                "Make sure the C++ module was built correctly in the Docker image."
            )

        self.R = R
        self.L = L
        self.num_threads = num_threads
        self.metric = metric
        self.index_prefix = None
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
        self.cpp_index.set_quantizer(quantizer)
        print("Quantizer updated successfully (graph structure unchanged)")

    def build(self, nd: int, data: np.ndarray, train_quantizer: bool = True) -> bool:
        """
        Build the DiskANN index.

        This performs the following steps:
        1. Train the quantizer on the data (optional)
        2. Save data to disk
        3. Build Vamana graph using DiskANN's build process
        4. Save quantizer state

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

            # Create temporary directory for index files (only if not already created)
            if not hasattr(self, 'index_prefix') or self.index_prefix is None:
                temp_dir = tempfile.mkdtemp(prefix="diskann_")
                self.index_prefix = os.path.join(temp_dir, "index")
                print(f"Building DiskANN index in {temp_dir}")
            else:
                print(f"Reusing existing index directory at {self.index_prefix}")

            print(f"Parameters: R={self.R}, L={self.L}, threads={self.num_threads}")

            # Create C++ index wrapper (only if not already created)
            if not hasattr(self, 'cpp_index') or self.cpp_index is None:
                self.cpp_index = diskann_cpp.DiskANNIndex(
                    quantizer=self.quantizer,
                    index_prefix=self.index_prefix,
                    R=self.R,
                    L=self.L,
                    num_threads=self.num_threads,
                    metric=self.metric
                )

            # Build the index (optionally train quantizer, then build graph)
            build_memory_gb = 8.0  # Memory budget for building
            self.cpp_index.build(self._original_data, build_memory_gb, train_quantizer)

            self.trained = True
            print("DiskANN index built successfully!")
            return True

        except Exception as e:
            print(f"Error building DiskANN index: {e}")
            import traceback
            traceback.print_exc()
            return False

    def search(self, nq: int, queries: np.ndarray, topk: int, **search_params) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors using the quantizer.

        During search, the quantizer's estimate_distance() is used to
        compute approximate distances for candidate filtering, then
        exact distances are computed for reranking.

        Args:
            nq: Number of query vectors
            queries: Query vectors of shape (nq, d)
            topk: Number of nearest neighbors to return
            **search_params: Search parameters
                - search_L: Search list size (default: max(L, 2*topk))
                - beam_width: Beam width (default: 4)

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - I: Indices of nearest neighbors, shape (nq, topk)
                - D: Distances to nearest neighbors, shape (nq, topk)
        """
        if not self.trained:
            raise RuntimeError("Index not trained. Call build() first.")

        queries = np.ascontiguousarray(queries, dtype=np.float32)

        # Get search parameters
        search_L = search_params.get("search_L", max(self.L, 2 * topk))
        beam_width = search_params.get("beam_width", 4)

        # Use C++ search with quantizer callbacks
        I, D = self.cpp_index.search(queries, topk, search_L, beam_width)

        return I, D

    def getMemoryUsage(self) -> float:
        """
        Get memory usage of the index in KB.

        This includes:
        - Graph structure
        - Quantizer memory
        - Cached data

        Returns:
            float: Memory usage in KB
        """
        if not self.trained:
            return 0.0

        # Get quantizer memory
        quantizer_mem = self.quantizer.getMemoryUsage()

        # Get index memory (graph + cached nodes)
        # For now, estimate based on graph structure
        # Each node has at most R neighbors (4 bytes each)
        graph_mem = (self.num_points * self.R * 4) / 1024  # KB

        return quantizer_mem + graph_mem

    def get_index_stats(self) -> dict:
        """Get detailed index statistics."""
        if self.cpp_index is None:
            return {}

        return self.cpp_index.get_stats()

    def __repr__(self):
        return (f"DiskANN(R={self.R}, L={self.L}, "
                f"points={self.num_points}, dim={self.dimension}, "
                f"quantizer={self.quantizer.__class__.__name__})")
