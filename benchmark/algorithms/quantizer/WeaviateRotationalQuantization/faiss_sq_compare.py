#!/usr/bin/env python3
"""
Baseline: faiss IndexScalarQuantizer (QT_8bit_uniform) — no rotation, per-dim
uniform 8-bit scalar quantization.

Usage:
    python faiss_sq_compare.py <base.fvecs> <query.fvecs> <groundtruth.ivecs> [k=10] [threads=0]

`threads=0` keeps the faiss default (OpenMP auto). Pass an explicit number to
make the comparison apples-to-apples with the C++ binary run at OMP_NUM_THREADS=N.
"""

import os
import sys
import time
import numpy as np


def read_fvecs(path: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    d = int(raw[0])
    row = d + 1
    assert raw.size % row == 0, f"corrupt fvecs: {path}"
    n = raw.size // row
    return raw.reshape(n, row)[:, 1:].copy().view(np.float32)


def read_ivecs(path: str) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32)
    w = int(raw[0])
    row = w + 1
    assert raw.size % row == 0, f"corrupt ivecs: {path}"
    n = raw.size // row
    return raw.reshape(n, row)[:, 1:].copy()


def recall_at_k(I: np.ndarray, gt: np.ndarray, k: int) -> float:
    use = min(k, gt.shape[1])
    hits = 0
    for q in range(I.shape[0]):
        truth = set(gt[q, :use].tolist())
        hits += sum(1 for x in I[q, :k] if int(x) in truth)
    return hits / (I.shape[0] * use)


def main():
    if len(sys.argv) < 4:
        print("usage: faiss_sq_compare.py <base.fvecs> <query.fvecs> "
              "<groundtruth.ivecs> [k=10] [threads=0]", file=sys.stderr)
        sys.exit(1)

    base_path = sys.argv[1]
    query_path = sys.argv[2]
    gt_path = sys.argv[3]
    k = int(sys.argv[4]) if len(sys.argv) >= 5 else 10
    threads = int(sys.argv[5]) if len(sys.argv) >= 6 else 0

    import faiss  # import after argparsing for faster failure on bad args

    if threads > 0:
        faiss.omp_set_num_threads(threads)
    eff_threads = faiss.omp_get_max_threads()

    base = np.ascontiguousarray(read_fvecs(base_path), dtype=np.float32)
    query = np.ascontiguousarray(read_fvecs(query_path), dtype=np.float32)
    gt = read_ivecs(gt_path)

    nb, d = base.shape
    nq = query.shape[0]
    print(f"d={d} nb={nb} nq={nq} k={k} threads={eff_threads}")

    # --- faiss SQ8 uniform -----------------------------------------------
    index = faiss.IndexScalarQuantizer(
        d, faiss.ScalarQuantizer.QT_8bit_uniform, faiss.METRIC_L2)

    t0 = time.perf_counter()
    index.train(base)
    t1 = time.perf_counter()
    index.add(base)
    t2 = time.perf_counter()
    # A single search call inside faiss is already parallelized over queries.
    D, I = index.search(query, k)
    t3 = time.perf_counter()

    train_ms = (t1 - t0) * 1000.0
    add_ms = (t2 - t1) * 1000.0
    search_ms = (t3 - t2) * 1000.0
    qps = nq / (t3 - t2)
    rec = recall_at_k(I, gt, k)

    # code size: SQ8 = 1 byte per dim (no metadata; scale stored once).
    print(f"faiss SQ8_uniform  code={d:4d}B  train={train_ms:.1f}ms  "
          f"add={add_ms:6.1f}ms ({add_ms*1000.0/nb:.2f} us/vec)  "
          f"search={search_ms:6.1f}ms  {qps:7.1f} QPS  recall@{k}={rec:.4f}")


if __name__ == "__main__":
    main()
