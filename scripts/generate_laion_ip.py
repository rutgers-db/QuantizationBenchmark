import os
import time
import h5py
import numpy as np
import faiss

BASE = "/data/local/embedding_dataset/laion-5b/sample/laion-en-2M.f32vecs"
QUERY = "/data/local/embedding_dataset/laion-5b/sample/laion-en-query-100k.f32vecs"
OUT = "/data/local/embedding_dataset/hdf5/laion-768-ip.hdf5"
DIM = 768
N_TEST = 1000
TOPK = 100

def load_f32vecs(path: str, dim: int) -> np.ndarray:
    size = os.path.getsize(path)
    assert size % (dim * 4) == 0, f"{path}: size {size} not divisible by dim*4"
    n = size // (dim * 4)
    return np.memmap(path, dtype=np.float32, mode="r", shape=(n, dim))

print(f"Loading base from {BASE}")
base_mm = load_f32vecs(BASE, DIM)
print(f"  base: {base_mm.shape}")

print(f"Loading queries from {QUERY}")
query_mm = load_f32vecs(QUERY, DIM)
print(f"  query: {query_mm.shape}")

train = np.ascontiguousarray(base_mm[:], dtype=np.float32)
test = np.ascontiguousarray(query_mm[:N_TEST], dtype=np.float32)
print(f"train shape: {train.shape}, test shape: {test.shape}")

n_gpu = faiss.get_num_gpus()
assert n_gpu > 0, "no GPU available; run this on a GPU machine"
print(f"Using {n_gpu} GPU(s) for brute-force IP search")

cpu_index = faiss.IndexFlatIP(DIM)
co = faiss.GpuMultipleClonerOptions()
co.shard = True  # shard base across GPUs if multiple
co.useFloat16 = False
index = faiss.index_cpu_to_all_gpus(cpu_index, co=co) if n_gpu > 1 else \
        faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_index)

t0 = time.time()
index.add(train)
print(f"  add done in {time.time()-t0:.1f}s, ntotal={index.ntotal}")

print(f"Searching top-{TOPK} for {test.shape[0]} queries (inner product)")
t0 = time.time()
D, I = index.search(test, TOPK)
print(f"  search done in {time.time()-t0:.1f}s")
print(f"  D[0,:5] = {D[0,:5]}")
print(f"  I[0,:5] = {I[0,:5]}")

print(f"Writing {OUT}")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with h5py.File(OUT, "w") as f:
    f.attrs["type"] = "dense"
    f.attrs["distance"] = "inner_product"
    f.attrs["dimension"] = DIM
    f.attrs["point_type"] = "float"
    f.create_dataset("train", data=train, dtype=np.float32)
    f.create_dataset("test", data=test, dtype=np.float32)
    f.create_dataset("neighbors", data=I.astype(np.int32), dtype=np.int32)
    f.create_dataset("distances", data=D.astype(np.float32), dtype=np.float32)
print("done")
