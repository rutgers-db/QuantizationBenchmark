"""Benchmark FAISS binary quantization on fvecs/ivecs datasets.

Two modes:

  * Symmetric (default): both db and query are binarized (sign around
    per-dim mean, same as BinaryQuantizer::train in main.cpp). Search is
    brute-force Hamming via faiss.IndexBinaryFlat.

  * Asymmetric (--asymmetric): only the db is binarized; the query stays
    float32. Each db vector's bits are interpreted as +/-1 and the score
    is <q, signs>, computed as a single BLAS matmul <q, bits> (the
    -sum(q) / d constants are query-only and don't change the ranking).
    This trades speed for better recall — same coarse code, but the query
    keeps its full precision.

Usage:
    python faiss_bq_benchmark.py <base.fvecs> <query.fvecs> <gt.ivecs>
                                 [--k 10] [--asymmetric]
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

try:
    import faiss
except ImportError:
    sys.stderr.write("faiss not installed. Try: pip install faiss-cpu\n")
    sys.exit(1)


def read_xvecs(path: str, dtype: np.dtype) -> np.ndarray:
    """Read fvecs / ivecs. Each record is [int32 dim][dim * dtype]."""
    assert np.dtype(dtype).itemsize == 4, "fvecs/ivecs elements are 4 bytes"
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise RuntimeError(f"empty file: {path}")
    dim = int(raw[0])
    stride = dim + 1
    if raw.size % stride != 0:
        raise RuntimeError(f"{path}: size not divisible by record stride")
    n = raw.size // stride
    data = raw.reshape(n, stride)[:, 1:].copy()
    if dtype != np.int32:
        data = data.view(dtype)
    return data.reshape(n, dim)


def bits_sign(x: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Per-dim sign binarization, returns a (n, d) uint8 matrix of 0/1 bits."""
    return (x > thresholds).astype(np.uint8)


def pack_bits(bits: np.ndarray) -> np.ndarray:
    """Pack (n, d) {0,1} -> (n, ceil(d/8)) uint8 for IndexBinaryFlat.

    FAISS stores dim j in bit (j % 8) of byte (j // 8), i.e. little-endian
    within a byte. np.packbits defaults to MSB-first, so pass bitorder='little'.
    """
    d = bits.shape[1]
    pad = (-d) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros((bits.shape[0], pad), dtype=np.uint8)], axis=1)
    return np.packbits(bits, axis=1, bitorder="little")


def search_symmetric(base_bits: np.ndarray, query_bits: np.ndarray, k: int):
    """Brute-force Hamming via IndexBinaryFlat. Returns (timings_dict, D, I)."""
    timings = {}

    t0 = time.perf_counter()
    base_codes = pack_bits(base_bits)
    d_bits = base_codes.shape[1] * 8
    index = faiss.IndexBinaryFlat(d_bits)
    index.add(base_codes)
    timings["encode+add"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    query_codes = pack_bits(query_bits)
    timings["encode query"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    D, I = index.search(query_codes, k)
    timings["search"] = time.perf_counter() - t0

    timings["code_bytes"] = int(base_codes.nbytes)
    timings["bytes_per_code"] = int(base_codes.shape[1])
    timings["d_bits"] = d_bits
    return timings, D, I


def search_asymmetric(base_bits: np.ndarray, query: np.ndarray, k: int):
    """Float query vs. binarized db.

    For db bits b in {0,1} interpreted as signs s = 2b - 1:
        IP(q, s)    = 2 <q, b> - sum(q)
        ||q - s||^2 = ||q||^2 - 2 IP(q, s) + d
    Both are rank-equivalent to <q, b> (descending), so we just compute
    that one n_query x n_base matmul and take top-k.

    Since we skipped the expensive popcount kernel we also skip the
    compression benefit at search time: db is materialized as float32 for
    BLAS. We do keep the *trained thresholds* as the binarization step,
    so recall isolates the "keep query in float" change vs. pure symmetric.
    """
    timings = {}

    t0 = time.perf_counter()
    # n x d float matrix of 0/1 (what asymmetric scoring actually multiplies).
    db = base_bits.astype(np.float32)
    timings["prep db"] = time.perf_counter() - t0
    timings["code_bytes"] = int(db.nbytes)           # float32, no bit packing
    timings["bytes_per_code"] = int(db.shape[1] * 4)
    timings["d_bits"] = int(db.shape[1])

    t0 = time.perf_counter()
    # scores[i, j] = <query[i], db[j]> ; higher = closer under both IP and L2.
    scores = query @ db.T                            # (nq, n) float32
    # argpartition for top-k largest, then sort within the picked slice.
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    row_idx = np.arange(scores.shape[0])[:, None]
    top_scores = scores[row_idx, part]
    order = np.argsort(-top_scores, axis=1)
    I = part[row_idx, order]
    D = top_scores[row_idx, order]
    timings["search"] = time.perf_counter() - t0

    return timings, D, I


def main() -> int:
    ap = argparse.ArgumentParser(description="FAISS binary quantization benchmark")
    ap.add_argument("base", help="base .fvecs")
    ap.add_argument("query", help="query .fvecs")
    ap.add_argument("gt", help="groundtruth .ivecs")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument(
        "--asymmetric",
        action="store_true",
        help="keep query as float32 (no query-side binarization); "
             "scores via <q, db_bits> matmul",
    )
    args = ap.parse_args()

    k = args.k
    mode = "asymmetric (float query)" if args.asymmetric else "symmetric (binary query)"
    print(f"FAISS binary quantization benchmark — {mode}  (k={k})")
    print(f"  base:  {args.base}")
    print(f"  query: {args.query}")
    print(f"  gt:    {args.gt}")

    base = read_xvecs(args.base, np.float32)
    query = read_xvecs(args.query, np.float32)
    gt = read_xvecs(args.gt, np.int32)
    n_base, d = base.shape
    n_query = query.shape[0]
    assert query.shape[1] == d, f"dim mismatch: base d={d}, query d={query.shape[1]}"
    assert gt.shape[0] == n_query, "gt rows != query rows"
    assert gt.shape[1] >= k, f"gt has {gt.shape[1]} cols but k={k}"
    print(f"  n_base={n_base}  n_query={n_query}  d={d}")
    print(f"  faiss {faiss.__version__}  threads={faiss.omp_get_max_threads()}")

    # Shared step: learn thresholds and binarize the db.
    t0 = time.perf_counter()
    thresholds = base.mean(axis=0, dtype=np.float32)
    base_bits = bits_sign(base, thresholds)       # (n, d) uint8 0/1
    train_ms = (time.perf_counter() - t0) * 1000

    if args.asymmetric:
        timings, D, I = search_asymmetric(base_bits, query, k)
    else:
        query_bits = bits_sign(query, thresholds)
        timings, D, I = search_symmetric(base_bits, query_bits, k)

    mb = timings["code_bytes"] / (1024 * 1024)
    search_ms = timings["search"] * 1000
    print(f"train+binarize db: {train_ms:8.2f} ms")
    for key in ("encode+add", "prep db", "encode query"):
        if key in timings:
            print(f"{key:<18} {timings[key] * 1000:8.2f} ms")
    print(
        f"codes: {mb:.2f} MB, {timings['bytes_per_code']} B/code, d_bits={timings['d_bits']}"
    )
    print(
        f"search:            {search_ms:8.2f} ms total, "
        f"{search_ms * 1000 / n_query:.2f} us/query, "
        f"{n_query * 1000 / search_ms:.0f} QPS"
    )

    gt_k = gt[:, :k]
    top1_hits = int(np.sum(I[:, 0] == gt_k[:, 0]))
    topk_hits = int(np.sum(np.any(I[:, :, None] == gt_k[:, None, :], axis=2)))
    print(f"recall@1:          {top1_hits / n_query:.4f}")
    print(f"recall@{k}:         {topk_hits / (n_query * k):.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
