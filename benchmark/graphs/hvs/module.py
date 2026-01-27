import numpy as np
from typing import Tuple
import sys
import os
import tempfile

# Add benchmark to path for importing BaseGraphIndex
sys.path.insert(0, '/benchmark')
from benchmark.base import BaseGraphIndex

# Import the HVS module (will be built in Docker)
try:
    import hvs
except ImportError as e:
    print(f"Warning: Could not import hvs: {e}")
    hvs = None


class HVS(BaseGraphIndex):
    """
    HVS (Hierarchical Vector Search) graph index with integrated quantization.

    HVS integrates hierarchical quantization directly into the graph structure,
    so it does not require an external quantizer.
    """

    def __init__(self, quantizer=None, max_level: int = 1, delta: float = 0.5,
                 efConstruction: int = 500, M: int = 16,
                 num_threads: int = 16, **kwargs):
        super().__init__(quantizer=None, **kwargs)

        if hvs is None:
            raise RuntimeError(
                "Python module 'hvs' is not available. "
                "Make sure it was built correctly in the Docker image."
            )

        self.max_level = max_level
        self.delta = delta
        self.efConstruction = efConstruction
        self.M = M
        self.num_threads = num_threads
        self.index = None
        self.num_points = 0
        self.dimension = 0
        self.trained = False
        self._tmpdir = None

    def build(self, nd: int, data: np.ndarray, **kwargs) -> bool:
        try:
            self.num_points = nd
            self.dimension = data.shape[1]
            self._original_data = np.ascontiguousarray(data, dtype=np.float32)

            # Create temp directory for index files
            self._tmpdir = tempfile.mkdtemp(prefix="hvs_")
            index_path = os.path.join(self._tmpdir, "index.bin")
            index2_path = os.path.join(self._tmpdir, "index2.bin")
            quantizer_path = os.path.join(self._tmpdir, "quantizer.gt")
            searching_path = os.path.join(self._tmpdir, "searching.gt")

            print(f"Building HVS index...")
            print(f"  num_elements: {self.num_points}")
            print(f"  dimension: {self.dimension}")
            print(f"  max_level: {self.max_level}")
            print(f"  delta: {self.delta}")
            print(f"  efConstruction: {self.efConstruction}")
            print(f"  M: {self.M}")
            print(f"  index dir: {self._tmpdir}")

            # Build index (module-level function, writes files to disk)
            hvs.build(
                self._original_data,
                max_level=self.max_level,
                delta=self.delta,
                index_path=index_path,
                index2_path=index2_path,
                quantizer_path=quantizer_path,
                searching_path=searching_path,
                efConstruction=self.efConstruction,
                M=self.M
            )

            # Load the built index for searching
            self.index = hvs.HVSIndex()
            self.index.load(
                index_path=index_path,
                index2_path=index2_path,
                quantizer_path=quantizer_path,
                searching_path=searching_path,
                max_level=self.max_level,
                vecdim=self.dimension
            )

            self.trained = True
            print("HVS index built successfully!")
            return True

        except Exception as e:
            print(f"Error building HVS index: {e}")
            import traceback
            traceback.print_exc()
            return False

    def search(self, nq: int, queries: np.ndarray, topk: int, **search_params) -> Tuple[np.ndarray, np.ndarray]:
        if not self.trained:
            raise RuntimeError("Index not trained. Call build() first.")

        queries = np.ascontiguousarray(queries, dtype=np.float32)

        efsearch = search_params.get("efsearch", 1000)

        # HVS search returns (result_ids as uint32, time_per_query_us as float)
        result_ids, _ = self.index.search(queries, topk=topk, efsearch=efsearch)

        # Convert uint32 -> int64 for framework compatibility
        I = result_ids.astype(np.int64)

        # Compute L2 squared distances (HVS doesn't return distances)
        D = np.zeros((nq, topk), dtype=np.float32)
        for i in range(nq):
            D[i] = np.sum((self._original_data[I[i]] - queries[i]) ** 2, axis=1)

        return I, D

    def getMemoryUsage(self) -> float:
        if not self.trained or self._tmpdir is None:
            return 0.0

        # Sum up index file sizes
        total_bytes = 0
        for fname in ["index.bin", "index2.bin", "quantizer.gt", "searching.gt"]:
            fpath = os.path.join(self._tmpdir, fname)
            if os.path.exists(fpath):
                total_bytes += os.path.getsize(fpath)

        return total_bytes / 1024.0  # KB

    def __repr__(self):
        return (f"HVS(max_level={self.max_level}, delta={self.delta}, "
                f"efConstruction={self.efConstruction}, M={self.M}, "
                f"points={self.num_points}, dim={self.dimension})")
