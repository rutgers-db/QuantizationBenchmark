"""
Shared IVF coarse-centroid cache used across IVF-based quantizers.

The cache lives on disk under ``IVF_CACHE_DIR`` (default ``/tmp/quantbench_ivf_cache``)
and persists across Docker container runs because ``/tmp`` is bind-mounted from the
host.  Different IVF methods that train centroids from the same data and the same
(nlist, space) combination can reuse the same k-means centroids, which guarantees
both fairness across methods and a large reduction in training time when many
parameter combinations are benchmarked.

Typical use in an IVF quantizer's ``train`` method::

    from benchmark.ivf_centroid_cache import (
        data_fingerprint, load_centroids, save_centroids,
    )

    fp = data_fingerprint(train_data)
    key = f"coarse_l2_{fp}_nlist{nlist}"
    centroids = load_centroids(key)
    if centroids is None:
        centroids = run_kmeans(train_data, nlist)
        save_centroids(key, centroids)
"""

import hashlib
import os
import numpy as np


_DEFAULT_CACHE_DIR = "/tmp/quantbench_ivf_cache"


def get_cache_dir() -> str:
    return os.environ.get("IVF_CACHE_DIR", _DEFAULT_CACHE_DIR)


def data_fingerprint(data: np.ndarray) -> str:
    """Return a lightweight, deterministic fingerprint of a training dataset.

    Uses shape, dtype, and an MD5 over a head/tail sample so the function stays
    cheap even on very large datasets.  The fingerprint is stable for identical
    arrays regardless of memory layout.
    """
    data = np.ascontiguousarray(data)
    n = int(data.shape[0])
    d = int(data.shape[1]) if data.ndim > 1 else 1
    dtype = str(data.dtype)

    sample_n = min(4096, n)
    if n <= sample_n * 2:
        sample = data
    else:
        sample = np.concatenate([data[:sample_n], data[-sample_n:]], axis=0)

    h = hashlib.md5(np.ascontiguousarray(sample).tobytes()).hexdigest()[:16]
    return f"n{n}_d{d}_{dtype}_{h}"


def _cache_path(key: str) -> str:
    cache_dir = get_cache_dir()
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError:
        pass
    return os.path.join(cache_dir, f"{key}.npz")


def load_centroids(key: str):
    """Return cached centroids for ``key`` or ``None`` if not present/corrupt."""
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    try:
        with np.load(path) as f:
            centroids = f["centroids"]
        print(f"[ivf-cache] HIT  {key} -> {path} (shape={centroids.shape})")
        return np.ascontiguousarray(centroids.astype(np.float32, copy=False))
    except Exception as exc:
        print(f"[ivf-cache] Failed to load {path}: {exc}")
        return None


def save_centroids(key: str, centroids: np.ndarray) -> None:
    """Atomically persist ``centroids`` under ``key`` in the shared cache."""
    path = _cache_path(key)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        # Use a file object so np.savez does NOT auto-append ``.npz`` to tmp
        # (which would break the os.replace below).
        with open(tmp, "wb") as f:
            np.savez(f, centroids=np.ascontiguousarray(centroids.astype(np.float32, copy=False)))
        os.replace(tmp, path)
        print(f"[ivf-cache] SAVE {key} -> {path} (shape={centroids.shape})")
    except Exception as exc:
        print(f"[ivf-cache] Failed to save {path}: {exc}")
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def load_npz(key: str):
    """Return a dict of arrays for a generic ``.npz`` cache entry, or None."""
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    try:
        with np.load(path) as f:
            return {name: f[name] for name in f.files}
    except Exception as exc:
        print(f"[ivf-cache] Failed to load {path}: {exc}")
        return None


def save_npz(key: str, **arrays: np.ndarray) -> None:
    """Save a dict of named numpy arrays atomically under ``key``."""
    path = _cache_path(key)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "wb") as f:
            np.savez(f, **{k: np.ascontiguousarray(v) for k, v in arrays.items()})
        os.replace(tmp, path)
        shapes = {k: v.shape for k, v in arrays.items()}
        print(f"[ivf-cache] SAVE {key} -> {path} ({shapes})")
    except Exception as exc:
        print(f"[ivf-cache] Failed to save {path}: {exc}")
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def coarse_key(data_fp: str, nlist: int, space: str = "l2",
               niter: int = 25, seed: int = 1234) -> str:
    """Key for shared raw-space IVF coarse centroids.

    Any method that trains IVF centroids by running faiss k-means on the raw
    training data with ``(nlist, space, niter, seed)`` can read/write using this
    key, letting different quantizers (Faiss-IVFPQ, Faiss-IVFSQ, IVFOSQ,
    IVFRabitQLibrary, ...) share the same centroids.
    """
    return f"coarse_{space}_{data_fp}_nlist{nlist}_niter{niter}_seed{seed}"
