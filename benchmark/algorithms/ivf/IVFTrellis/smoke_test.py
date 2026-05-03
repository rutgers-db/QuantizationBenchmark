"""Smoke test that calls trellis_cpp directly with hand-rolled centroids
(skips faiss / module.py to keep this dev-host runnable)."""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "build"))
import trellis_cpp

np.random.seed(0)
DIM = 128
N = 20_000
NQ = 200
K = 10

base = np.random.randn(N, DIM).astype(np.float32)
queries = np.random.randn(NQ, DIM).astype(np.float32)

# Brute-force GT (L2)
gt_dist = ((queries[:, None, :] - base[None, :, :]) ** 2).sum(-1)
gt = np.argsort(gt_dist, axis=1)[:, :K]

# Hand-rolled mini k-means: random init + 5 iters
NLIST = 64
rng = np.random.default_rng(7)
centroids = base[rng.choice(N, NLIST, replace=False)].copy()
for _ in range(5):
    d = ((base[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
    a = d.argmin(axis=1)
    new = np.zeros_like(centroids)
    cnt = np.bincount(a, minlength=NLIST)
    np.add.at(new, a, base)
    nz = cnt > 0
    new[nz] /= cnt[nz, None]
    new[~nz] = centroids[~nz]
    centroids = new.astype(np.float32)
d = ((base[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
cluster_ids = d.argmin(axis=1).astype(np.uint32)

idx = trellis_cpp.IVFTrellis(N, DIM, NLIST, 8, "l2", "fht")
t0 = time.time()
idx.construct(base, centroids, cluster_ids)
print(f"[build]  {time.time() - t0:.2f}s   N={N}, dim={DIM}, nlist={NLIST}")

print("\nraw search (no rerank):")
for nprobe in [4, 8, 16, 32]:
    t0 = time.time()
    I, D = idx.search_batch(queries, K, nprobe)
    t_q = time.time() - t0
    rec = (I[:, :, None] == gt[:, None, :]).any(-1).sum() / (NQ * K)
    print(f"  nprobe={nprobe:3d}  recall@{K}={rec:.3f}  t={t_q*1000:.1f}ms")

print("\nwith rerank from original vectors:")
for nprobe, nrerank in [(8, 100), (16, 200), (32, 500), (32, 1000)]:
    t0 = time.time()
    I, D = idx.search_batch(queries, nrerank, nprobe)
    safe = np.where(I >= 0, I, 0)
    cand = base[safe]
    diff = queries[:, None, :] - cand
    d_exact = (diff * diff).sum(-1)
    order = np.argsort(d_exact, axis=1)[:, :K]
    I_rr = np.take_along_axis(I, order, axis=1)
    t_q = time.time() - t0
    rec = (I_rr[:, :, None] == gt[:, None, :]).any(-1).sum() / (NQ * K)
    print(f"  nprobe={nprobe:3d} nrerank={nrerank:4d}  recall@{K}={rec:.3f}  t={t_q*1000:.1f}ms")
