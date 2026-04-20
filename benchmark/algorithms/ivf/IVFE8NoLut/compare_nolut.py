"""
Head-to-head IVFE8 (LUT + gather) vs IVFE8NoLut (direct-compute, no LUT) on
SIFT1M. Same centroids, same assignments -> any recall delta is FP noise.
"""

import sys
import time

import numpy as np
import faiss

sys.path.insert(
    0,
    "/common/home/yz1391/QuantizationBenchmark/benchmark/algorithms/ivf/IVFE8/build",
)
sys.path.insert(
    0,
    "/common/home/yz1391/QuantizationBenchmark/benchmark/algorithms/ivf/IVFE8NoLut/build",
)

import e8_cpp
import e8nolut_cpp


def fvecs_read(path):
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    a = a.reshape(-1, d + 1)[:, 1:].copy()
    return a.view(np.float32)


def ivecs_read(path):
    a = np.fromfile(path, dtype=np.int32)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def recall_at_k(I, gt, k):
    nq = I.shape[0]
    hits = 0
    gt_top = gt[:, :k]
    for i in range(nq):
        hits += len(set(I[i, :k].tolist()) & set(gt_top[i].tolist()))
    return hits / (nq * k)


def bench(idx, queries, gt, topk, nprobes, label):
    print(f"\n--- {label} ---")
    print(f"{'nprobe':>8}{'recall@10':>12}{'QPS':>10}{'t/q(ms)':>10}")
    results = {}
    for np_ in nprobes:
        # warm-up
        idx.search_batch(queries[:100], topk, np_)
        t0 = time.perf_counter()
        I, _ = idx.search_batch(queries, topk, np_)
        dt = time.perf_counter() - t0
        r = recall_at_k(I, gt, topk)
        qps = queries.shape[0] / dt
        print(f"{np_:>8}{r:>12.4f}{qps:>10.0f}{dt*1000/queries.shape[0]:>10.3f}")
        results[np_] = (r, qps, I)
    return results


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

        t0 = time.perf_counter()
        lut = e8_cpp.IVFE8(nb, d, nlist, nthread, "l2", "fht")
        lut.construct(xb, centroids, cluster_ids)
        print(f"IVFE8    (LUT) build:   {time.perf_counter()-t0:.1f}s")

        t0 = time.perf_counter()
        nlu = e8nolut_cpp.IVFE8NoLut(nb, d, nlist, nthread, "l2", "fht")
        nlu.construct(xb, centroids, cluster_ids)
        print(f"IVFE8NoLut     build:   {time.perf_counter()-t0:.1f}s")

        nprobes = [10, 30, 100] if nlist >= 4096 else [10, 30, 100, 300]

        r_lut = bench(lut, xq, gt, topk, nprobes, f"IVFE8 LUT (nlist={nlist})")
        r_nlu = bench(nlu, xq, gt, topk, nprobes, f"IVFE8NoLut (nlist={nlist})")

        print(f"\n--- delta (NoLut vs LUT) nlist={nlist} ---")
        print(f"{'nprobe':>8}{'dRecall':>10}{'QPS ratio':>12}{'id overlap':>12}")
        for np_ in nprobes:
            r0, q0, I0 = r_lut[np_]
            r1, q1, I1 = r_nlu[np_]
            overlap = 0
            for i in range(I0.shape[0]):
                overlap += len(set(I0[i].tolist()) & set(I1[i].tolist()))
            overlap /= (I0.shape[0] * I0.shape[1])
            print(f"{np_:>8}{r1-r0:>10.4f}{q1/q0:>12.3f}{overlap:>12.4f}")


if __name__ == "__main__":
    main()
