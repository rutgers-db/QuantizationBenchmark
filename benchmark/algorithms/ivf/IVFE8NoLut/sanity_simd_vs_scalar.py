"""Same NoLut index, two searches: SIMD path vs scalar tail path. Same
rotator, same codes -> any delta must come from the SIMD kernel."""
import sys, numpy as np, faiss
sys.path.insert(0, "/common/home/yz1391/QuantizationBenchmark/benchmark/algorithms/ivf/IVFE8NoLut/build")
import e8nolut_cpp

def fvecs_read(path):
    a = np.fromfile(path, dtype=np.int32); d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy().view(np.float32)

xb = fvecs_read("/data/local/embedding_dataset/sift1M/sift_base.fvecs")
xq = fvecs_read("/data/local/embedding_dataset/sift1M/sift_query.fvecs")[:1000]

nlist = 256
d = xb.shape[1]
faiss.omp_set_num_threads(8)
km = faiss.Kmeans(d=d, k=nlist, niter=10, seed=1234, verbose=False)
km.train(xb)
centroids = np.ascontiguousarray(km.centroids.astype(np.float32))
quant = faiss.IndexFlatL2(d); quant.add(centroids)
_, assign = quant.search(xb, 1)
cluster_ids = np.ascontiguousarray(assign.flatten().astype(np.uint32))

idx = e8nolut_cpp.IVFE8NoLut(xb.shape[0], d, nlist, 8, "l2", "fht")
idx.construct(xb, centroids, cluster_ids)

np_ = 32
idx.force_scalar = False
I_simd, D_simd = idx.search_batch(xq, 100, np_)
idx.force_scalar = True
I_scal, D_scal = idx.search_batch(xq, 100, np_)

# Per-query: look up each query's returned ids in scalar map, compare distances
diffs = []
id_overlap = 0
for q in range(len(xq)):
    m_simd = dict(zip(I_simd[q].tolist(), D_simd[q].tolist()))
    m_scal = dict(zip(I_scal[q].tolist(), D_scal[q].tolist()))
    common = set(m_simd) & set(m_scal)
    id_overlap += len(common)
    for cid in common:
        if m_scal[cid] != 0 and not np.isinf(m_scal[cid]):
            diffs.append(abs(m_simd[cid] - m_scal[cid]) / max(abs(m_scal[cid]), 1e-9))
print(f"common_ids_per_q = {id_overlap/len(xq):.2f}/100")
print(f"rel_dist_diff: median={np.median(diffs):.2e} p99={np.percentile(diffs,99):.2e} max={np.max(diffs):.2e}")
