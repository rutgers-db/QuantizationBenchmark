"""
Head-to-head IVFE8 vs IVFRabitQLibrary on SIFT1M (IVF-only, no rerank).

Both use the same coarse KMeans centroids; the only thing that differs is
the residual quantizer (E8 lattice vs sign / RaBitQ multi-bit).
"""

import sys
import os
import time

import numpy as np
import faiss

sys.path.insert(
    0,
    "/common/home/yz1391/QuantizationBenchmark/benchmark/algorithms/ivf/IVFE8/build",
)
sys.path.insert(
    0,
    "/common/home/yz1391/QuantizationBenchmark/benchmark/algorithms/ivf/IVFRabitQLibrary/build",
)

import e8_cpp
import rabitqlib_cpp


def fvecs_read(path):
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    a = a.reshape(-1, d + 1)[:, 1:].copy()
    return a.view(np.float32)


def ivecs_read(path):
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def recall_at_k(I, gt, k=10):
    nq = I.shape[0]
    hits = 0
    gt_top = gt[:, :k]
    for i in range(nq):
        hits += len(set(I[i, :k].tolist()) & set(gt_top[i].tolist()))
    return hits / (nq * k)


def bench(idx, queries, gt, topk, nprobes, label, nthread=8):
    print(f"\n--- {label} ---")
    print(f"{'nprobe':>8}{'recall@10':>12}{'QPS':>10}{'t/q(ms)':>10}")
    for np_ in nprobes:
        t0 = time.perf_counter()
        if hasattr(idx, "search_batch") and "use_hacc" in idx.search_batch.__doc__ if idx.search_batch.__doc__ else False:
            I, _ = idx.search_batch(queries, topk, np_, True)
        else:
            # IVFE8 has search_batch(queries, k, nprobe); IVFRabitQLibrary adds use_hacc.
            try:
                I, _ = idx.search_batch(queries, topk, np_)
            except TypeError:
                I, _ = idx.search_batch(queries, topk, np_, True)
        dt = time.perf_counter() - t0
        r = recall_at_k(I, gt, topk)
        qps = queries.shape[0] / dt
        print(f"{np_:>8}{r:>12.4f}{qps:>10.0f}{dt*1000/queries.shape[0]:>10.3f}")


def main():
    base = "/data/local/embedding_dataset/sift1M"
    xb = fvecs_read(f"{base}/sift_base.fvecs")
    xq = fvecs_read(f"{base}/sift_query.fvecs")
    gt = ivecs_read(f"{base}/sift_groundtruth.ivecs")
    print(f"xb {xb.shape}, xq {xq.shape}, gt {gt.shape}")

    nthread = 8
    faiss.omp_set_num_threads(nthread)

    d = xb.shape[1]
    nb = xb.shape[0]
    nq = xq.shape[0]
    topk = 10

    for nlist in [1024, 4096]:
        print(f"\n========== nlist={nlist} ==========")

        t0 = time.perf_counter()
        km = faiss.Kmeans(d=d, k=nlist, niter=25, seed=1234, verbose=False)
        km.train(xb)
        centroids = np.ascontiguousarray(km.centroids.astype(np.float32))
        quant = faiss.IndexFlatL2(d)
        quant.add(centroids)
        _, assign = quant.search(xb, 1)
        cluster_ids = np.ascontiguousarray(assign.flatten().astype(np.uint32))
        print(f"KMeans + assign: {time.perf_counter()-t0:.1f}s")

        # Build E8
        t0 = time.perf_counter()
        e8 = e8_cpp.IVFE8(nb, d, nlist, nthread, "l2", "fht")
        e8.construct(xb, centroids, cluster_ids)
        print(f"E8 build: {time.perf_counter()-t0:.1f}s")

        # Build RaBitQ 1-bit
        t0 = time.perf_counter()
        rb1 = rabitqlib_cpp.IVF(nb, d, nlist, 1, nthread, "l2")
        rb1.construct(xb, centroids, cluster_ids, False)
        print(f"RaBitQ 1-bit build: {time.perf_counter()-t0:.1f}s")

        # Build RaBitQ 4-bit (sanity baseline)
        t0 = time.perf_counter()
        rb4 = rabitqlib_cpp.IVF(nb, d, nlist, 4, nthread, "l2")
        rb4.construct(xb, centroids, cluster_ids, False)
        print(f"RaBitQ 4-bit build: {time.perf_counter()-t0:.1f}s")

        nprobes = [10, 30, 100] if nlist >= 4096 else [10, 30, 100, 300]

        bench(e8, xq, gt, topk, nprobes, f"IVFE8  (nlist={nlist})", nthread)
        bench(rb1, xq, gt, topk, nprobes, f"IVFRaBitQ 1-bit (nlist={nlist})", nthread)
        bench(rb4, xq, gt, topk, nprobes, f"IVFRaBitQ 4-bit (nlist={nlist})", nthread)

        # ---------- FAISS IVFPQ ----------
        # PQ16x8 = 16 bytes/vec = 1 bit/dim (same budget as E8-1bit & RaBitQ-1bit)
        # PQ64x8 = 64 bytes/vec = 4 bits/dim (same budget as RaBitQ-4bit)
        for (nsub, label) in [(16, "PQ16x8 (1 bit/dim)"), (64, "PQ64x8 (4 bit/dim)")]:
            t0 = time.perf_counter()
            coarse = faiss.IndexFlatL2(d)
            idx_pq = faiss.IndexIVFPQ(coarse, d, nlist, nsub, 8)
            faiss.omp_set_num_threads(nthread)
            idx_pq.train(xb)
            idx_pq.add(xb)
            print(f"{label} build: {time.perf_counter()-t0:.1f}s")

            print(f"\n--- IVF{label} (nlist={nlist}) ---")
            print(f"{'nprobe':>8}{'recall@10':>12}{'QPS':>10}{'t/q(ms)':>10}")
            for np_ in nprobes:
                idx_pq.nprobe = np_
                t0 = time.perf_counter()
                D_pq, I_pq = idx_pq.search(xq, topk)
                dt = time.perf_counter() - t0
                r = recall_at_k(I_pq, gt, topk)
                qps = nq / dt
                print(f"{np_:>8}{r:>12.4f}{qps:>10.0f}{dt*1000/nq:>10.3f}")


if __name__ == "__main__":
    main()
