"""IVFE8 (LUT) vs IVFE8NoLut on high-dim HDF5 datasets.

LUT size per query = n_blocks * 256 * 4B = (padded_dim/8) * 1KB. L1 is ~32KB.
SIFT (d=128): 8KB -> fits L1, gather wins.
text2image/paper (d=200, padded=256): 32KB -> right at L1 edge.
gist (d=960, padded=960): 120KB -> spills to L2.
video (d=1024): 128KB -> spills to L2.
"""
import sys, time
import h5py
import numpy as np
import faiss

sys.path.insert(0, "/common/home/yz1391/QuantizationBenchmark/benchmark/algorithms/ivf/IVFE8/build")
sys.path.insert(0, "/common/home/yz1391/QuantizationBenchmark/benchmark/algorithms/ivf/IVFE8NoLut/build")

import e8_cpp
import e8nolut_cpp


def load(path):
    with h5py.File(path, "r") as f:
        xb = np.ascontiguousarray(f["train"][:], dtype=np.float32)
        xq = np.ascontiguousarray(f["test"][:], dtype=np.float32)
        gt = np.ascontiguousarray(f["neighbors"][:], dtype=np.int32)
    return xb, xq, gt


def recall_at_k(I, gt, k):
    nq = I.shape[0]
    hits = 0
    gt_top = gt[:, :k]
    for i in range(nq):
        hits += len(set(I[i, :k].tolist()) & set(gt_top[i].tolist()))
    return hits / (nq * k)


def bench(idx, xq, gt, topk, nprobes, label):
    print(f"\n--- {label} ---")
    print(f"{'nprobe':>8}{'recall@10':>12}{'QPS':>10}{'t/q(ms)':>10}")
    out = {}
    for np_ in nprobes:
        idx.search_batch(xq[:100], topk, np_)          # warm
        t0 = time.perf_counter()
        I, _ = idx.search_batch(xq, topk, np_)
        dt = time.perf_counter() - t0
        r = recall_at_k(I, gt, topk)
        qps = xq.shape[0] / dt
        print(f"{np_:>8}{r:>12.4f}{qps:>10.0f}{dt*1000/xq.shape[0]:>10.3f}")
        out[np_] = (r, qps)
    return out


def run(path, nlists, nprobes, nthread=8):
    print(f"\n================ {path} ================")
    xb, xq, gt = load(path)
    print(f"xb {xb.shape}, xq {xq.shape}, gt {gt.shape}")
    d = xb.shape[1]
    padded = ((d + 63) // 64) * 64
    n_blocks = padded // 8
    lut_kb = n_blocks * 256 * 4 / 1024
    print(f"d={d} padded_dim={padded} n_blocks={n_blocks} LUT={lut_kb:.0f}KB per query")

    topk = 10
    faiss.omp_set_num_threads(nthread)

    for nlist in nlists:
        print(f"\n---- nlist={nlist} ----")
        t0 = time.perf_counter()
        km = faiss.Kmeans(d=d, k=nlist, niter=25, seed=1234, verbose=False)
        km.train(xb)
        centroids = np.ascontiguousarray(km.centroids.astype(np.float32))
        quant = faiss.IndexFlatL2(d); quant.add(centroids)
        _, assign = quant.search(xb, 1)
        cluster_ids = np.ascontiguousarray(assign.flatten().astype(np.uint32))
        print(f"KMeans+assign: {time.perf_counter()-t0:.1f}s")

        t0 = time.perf_counter()
        lut = e8_cpp.IVFE8(xb.shape[0], d, nlist, nthread, "l2", "fht")
        lut.construct(xb, centroids, cluster_ids)
        print(f"IVFE8     build: {time.perf_counter()-t0:.1f}s")

        t0 = time.perf_counter()
        nlu = e8nolut_cpp.IVFE8NoLut(xb.shape[0], d, nlist, nthread, "l2", "fht")
        nlu.construct(xb, centroids, cluster_ids)
        print(f"IVFE8NoLut build: {time.perf_counter()-t0:.1f}s")

        r_lut = bench(lut, xq, gt, topk, nprobes, f"IVFE8 LUT (nlist={nlist})")
        r_nlu = bench(nlu, xq, gt, topk, nprobes, f"IVFE8NoLut (nlist={nlist})")

        print(f"\n-- QPS ratio (NoLut / LUT) --")
        print(f"{'nprobe':>8}{'ratio':>10}{'dRecall':>10}")
        for np_ in nprobes:
            r0, q0 = r_lut[np_]; r1, q1 = r_nlu[np_]
            print(f"{np_:>8}{q1/q0:>10.3f}{r1-r0:>+10.4f}")


if __name__ == "__main__":
    base = "/data/local/embedding_dataset/hdf5"
    run(f"{base}/paper-200-euclidean.hdf5",      [1024, 4096], [10, 30, 100])
    run(f"{base}/gist-960-euclidean.hdf5",       [1024, 4096], [10, 30, 100])
    run(f"{base}/video-1024-euclidean.hdf5",     [1024, 4096], [10, 30, 100])
