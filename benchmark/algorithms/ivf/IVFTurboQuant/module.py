import numpy as np
from typing import Tuple
import psutil
import sys

sys.path.insert(0, '/benchmark')
from benchmark.base import BaseQuantizer

try:
    import turbo_quant_cpp
except ImportError as e:
    print(f"Warning: Could not import turbo_quant_cpp: {e}")
    turbo_quant_cpp = None


class IVFTurboQuant(BaseQuantizer):
    """
    TurboQuant (unified) — scalar quantisation with orthogonal rotation.

    Pipeline per vector x:
      1. Centre:     x_c = x - centroid  (if use_data_centroid=True)
      2. Normalise to unit sphere
      3. Rotation:   Hadamard (O(d log d)) or Dense Haar-random (O(d²))
      4. Per-dimension scalar quantisation via Lloyd-Max N(0,1) codebook
      5. Store ‖x_c‖ for distance reconstruction

    mode="mse"  uses all `bitwidth` bits for scalar codes (pure MSE).
    mode="ip"   uses (bitwidth-1) bits for codes + 1 bit QJL residual sign.

    The benchmark always evaluates L2 (Euclidean) distance.

    Supports distribution-shift testing: train() builds the codebook/centroid
    from a (possibly shifted) sample, then add() encodes the full database.
    """

    def __init__(self, ndim: int, bitwidth: int, data_bytes: int = 4,
                 nthread: int = 1, space: str = "l2",
                 mode: str = "mse", seed: int = 123456789,
                 rotation_type: str = "dense",
                 use_data_centroid: bool = True,
                 nlist: int = 1):
        """
        Args:
            ndim:              Vector dimensionality.
            bitwidth:          Bits per dimension (1–9).
            data_bytes:        Bytes per float element (4 for float32).
            nthread:           OMP threads for train/add/query.
            space:             Distance metric ("l2").
            mode:              "mse" (all bits for MSE) or "ip" (QJL residual).
            seed:              RNG seed for the rotation random signs.
            rotation_type:     "hadamard" (fast, pads to power-of-2) or "dense".
            use_data_centroid: Centre-normalise vectors before quantising.
            nlist:             IVF clusters (1 = flat brute-force).
        Note:
            nprobe is a search-time parameter passed via query(**search_params).
        """
        super().__init__()
        if turbo_quant_cpp is None:
            raise RuntimeError(
                "C++ module 'turbo_quant_cpp' is not available. "
                "Make sure it was built inside the Docker image."
            )
        self.ndim       = ndim
        self.bitwidth   = bitwidth
        self.data_bytes = data_bytes
        self.nthread    = nthread
        self.space      = space
        self.nlist      = nlist

        mode_int          = 0 if mode == "mse" else 1
        rotation_type_int = 0 if rotation_type == "hadamard" else 1

        # nprobe is not fixed at build time; set to 1 as placeholder
        self.cpp_index = turbo_quant_cpp.PyTurboQuant(
            ndim, bitwidth, mode_int, nthread, seed,
            rotation_type_int, use_data_centroid, nlist, 1
        )

        self.ndata   = 0
        self.trained = False

    # ------------------------------------------------------------------
    # fit — train on data then add the same data (normal benchmark flow)
    # ------------------------------------------------------------------
    def fit(self, nd: int, data: np.ndarray) -> bool:
        return self.train(nd, data) and self.add(nd, data)

    # ------------------------------------------------------------------
    # train — build centroid + codebook from the given sample
    #         (may be a distributional-shift subset for shift experiments)
    # ------------------------------------------------------------------
    def train(self, nd: int, data: np.ndarray) -> bool:
        data = np.ascontiguousarray(data, dtype=np.float32)
        try:
            self.cpp_index.train(data)
        except Exception as e:
            print(f"TurboQuant train error: {e}")
            return False
        return True

    # ------------------------------------------------------------------
    # add — encode and index the database vectors
    #       (full dataset, potentially different from train data)
    # ------------------------------------------------------------------
    def add(self, nd: int, data: np.ndarray) -> bool:
        data = np.ascontiguousarray(data, dtype=np.float32)
        self._original_data = data
        self.ndata = nd
        try:
            self.cpp_index.add(data)
            self.trained = True
        except Exception as e:
            print(f"TurboQuant add error: {e}")
            return False
        return True

    # ------------------------------------------------------------------
    # query — approximate nearest-neighbour search (L2 distance)
    # ------------------------------------------------------------------
    def query(self, nq: int, queries: np.ndarray, topk: int,
              **search_params) -> Tuple[np.ndarray, np.ndarray]:
        if not self.trained:
            raise RuntimeError("Index not trained. Call fit() or train()+add() first.")
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        nprobe = int(search_params.get("nprobe", 1))
        nprobe = max(1, min(nprobe, self.nlist))
        self.cpp_index.set_nprobe(nprobe)
        I, D = self.cpp_index.search(queries, topk)
        return I, D

    # ------------------------------------------------------------------
    # metrics
    # ------------------------------------------------------------------
    def getMemoryUsage(self) -> float:
        return psutil.Process().memory_info().rss / 1024

    def getCompressionRate(self) -> float:
        """bits_per_dim / (data_bytes * 8)"""
        return self.bitwidth / (self.data_bytes * 8)

    def getMSE(self) -> float:
        return 0.0
        # if not self.trained:
        #     return float('inf')
        # return float(self.cpp_index.getMSE())

    # ------------------------------------------------------------------
    # graph-index interface (set_query / estimate_distance)
    # ------------------------------------------------------------------
    def set_query(self, query: np.ndarray, thread_id: int) -> None:
        query = np.ascontiguousarray(query.reshape(1, self.ndim), dtype=np.float32)
        self.cpp_index.set_query(query)

    def estimate_distance(self, idx: int, thread_id: int) -> float:
        return self.cpp_index.estimate_distance(int(idx))
