"""Generate a uniform [-1, 1] random fvecs/ivecs dataset and compute exact
L2 ground truth, so both the C++ and the qdrant BQ benchmarks can be run
on data that matches qdrant's "expected input" range.

Writes three files to --out-dir:
    uniform_base.fvecs      (n_base   x d, float32 in [-1, 1])
    uniform_query.fvecs     (n_query  x d, float32 in [-1, 1])
    uniform_gt.ivecs        (n_query  x k, int32, exact L2 top-k ids)

Usage:
    pip install faiss-cpu numpy
    python gen_uniform_dataset.py --n=1000000 --nq=10000 --d=128 --k=100 \\
                                  --out-dir=/tmp
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

try:
    import faiss
except ImportError:
    sys.stderr.write("need faiss (pip install faiss-cpu)\n")
    sys.exit(1)


def write_fvecs(path: str, data: np.ndarray) -> None:
    assert data.dtype == np.float32 and data.ndim == 2
    n, d = data.shape
    out = np.empty((n, d + 1), dtype=np.int32)
    out[:, 0] = d
    # Reinterpret the payload columns as int32 for a single-shot tofile.
    out[:, 1:] = data.view(np.int32)
    out.tofile(path)


def write_ivecs(path: str, data: np.ndarray) -> None:
    assert data.dtype == np.int32 and data.ndim == 2
    n, d = data.shape
    out = np.empty((n, d + 1), dtype=np.int32)
    out[:, 0] = d
    out[:, 1:] = data
    out.tofile(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",  type=int, default=1_000_000, help="#base vectors")
    ap.add_argument("--nq", type=int, default=10_000,    help="#query vectors")
    ap.add_argument("--d",  type=int, default=128)
    ap.add_argument("--k",  type=int, default=100,
                    help="top-k saved in ground truth")
    ap.add_argument("--out-dir", default="/tmp")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"generating n={args.n} nq={args.nq} d={args.d} uniform[-1,1]")
    t0 = time.perf_counter()
    base  = rng.uniform(-1.0, 1.0, size=(args.n,  args.d)).astype(np.float32)
    query = rng.uniform(-1.0, 1.0, size=(args.nq, args.d)).astype(np.float32)
    print(f"  base:  {base.nbytes / 1e6:.1f} MB")
    print(f"  query: {query.nbytes / 1e6:.1f} MB")

    base_path  = os.path.join(args.out_dir, "uniform_base.fvecs")
    query_path = os.path.join(args.out_dir, "uniform_query.fvecs")
    gt_path    = os.path.join(args.out_dir, "uniform_gt.ivecs")

    write_fvecs(base_path,  base)
    write_fvecs(query_path, query)
    print(f"wrote fvecs in {time.perf_counter() - t0:.1f} s")

    # Exact L2 top-k via faiss.IndexFlatL2 (BLAS-backed).
    t0 = time.perf_counter()
    index = faiss.IndexFlatL2(args.d)
    index.add(base)
    _, I = index.search(query, args.k)
    gt = I.astype(np.int32)
    print(f"exact L2 top-{args.k} in {time.perf_counter() - t0:.1f} s")

    write_ivecs(gt_path, gt)
    print("wrote:")
    print(f"  {base_path}")
    print(f"  {query_path}")
    print(f"  {gt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
