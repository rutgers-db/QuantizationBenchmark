# Rotational Quantization (RQ) — C++

A compact, SIMD-optimized C++ implementation of Rotational Quantization with a
faiss-like interface (`train` / `add` / `search`).

Two modes:

| bits | Scheme                          | Storage per vector     | Inner loop              |
|------|---------------------------------|------------------------|-------------------------|
| 8    | Classic RQ (uniform uint8)      | `out_dim` B + 16 B meta | AVX2 uint8 dot product |
| 1    | Binary RQ (sign + scale)        | `out_dim/8` B + 8 B meta| XOR + popcnt           |

`out_dim` is `d` padded up to a multiple of 64.

## Design

1. **Rotation**: 3 rounds of `{random swaps → random ±1 sign flips → block-wise
   Fast Walsh–Hadamard Transform (block=64)}`. Orthonormal → preserves L2
   norm and inner product. Structured so computation is `O(d log 64)` per
   rotate, no dense matrix stored.
2. **Encoding** (bits=8): scan rotated vector for `[min, max]`, pick
   `step = (max-min)/255`, quantize each dim to `uint8` via rounding.
   Metadata stored: `{lower, step, step·Σc, ‖x‖²}`.
3. **Encoding** (bits=1): store `sign(rx[i])` as one bit; scale =
   `mean(|rx[i]|)`. Metadata: `{scale, ‖x‖²}`.
4. **Distance** (bits=8): the inner product estimate decomposes into four
   terms; the only per-dim term is an 8-bit × 8-bit dot, computed with AVX2
   `vpunpck` + `vpmaddwd` (two accumulators to hide latency).
5. **Distance** (bits=1): hamming distance via 64-bit XOR + `popcntq`, then
   `⟨x,y⟩ ≈ s_x s_y (D − 2h)`.
6. **Storage**: SoA — codes in one contiguous buffer, metadata in another
   (`4 floats` per vector). Memory-bandwidth-friendly for the scan phase.

## Build

```bash
cd cpp_impl
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Requires a CPU with AVX2 + FMA + POPCNT (any x86-64 chip from ~2013 onward).
A scalar fallback exists if `__AVX2__` is not defined.

## Run on SIFT1M (fvecs/ivecs)

```bash
./build/rq_example sift_base.fvecs sift_query.fvecs sift_groundtruth.ivecs 10
```

Expected output fields:
```
bits=8  code=128B  add=...  search=... us/q  recall@10=...
bits=1  code=16B   add=...  search=... us/q  recall@10=...
```

## Library usage

```cpp
#include "rq.h"

rq::RotationalQuantizer index(/*d=*/768, /*bits=*/8, rq::Metric::L2);
index.train(0, nullptr);          // no-op; RQ is data-independent
index.add(nb, base_vectors);      // [nb, d] float32, row-major

std::vector<float>   D(nq * k);
std::vector<int64_t> I(nq * k);
index.search(nq, queries, k, D.data(), I.data());
```

For inner-product / cosine (cosine = IP on L2-normalized vectors), pass
`rq::Metric::IP`. The returned `D` is ordered ascending for L2 and
descending for IP.
