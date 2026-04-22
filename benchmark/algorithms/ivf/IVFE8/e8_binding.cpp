// pybind11 binding for a self-contained IVF + E8-lattice 1-bit index.
//
// Quantization pipeline mirrors RaBitQ 1-bit: FHT-Kac rotation, residual from
// cluster centroid, RaBitQ factors (f_add, f_rescale, f_error). The residual
// is quantized per-8-dim-block to one of the 240 E8 root vectors, packed as
// 1 byte per block (d/8 bytes per vector — same budget as RaBitQ 1-bit).
//
// IVFE8 (gather / "LUT" variant):
//   Per-query 256-entry float32 LUT per block. Codes stored in block-major
//   tiles of 16 vectors; one _mm512_i32gather_ps loads 16 float LUT entries.
//
// IVFE8FastScan (PQ-fastscan-style, register-resident uint8 LUT):
//   Inspired by Faiss PQFastScan.  The float LUT is globally quantised to
//   uint8 (256 bytes/block).  At query time the 256-byte per-block LUT is
//   loaded into four ZMM registers (zmm_lut0-3), making it fully
//   register-resident — zero L1 traffic during the scan inner loop.  Lookup
//   uses _mm512_permutex2var_epi8 (AVX512VBMI).  Results are accumulated as
//   uint16 per tile of 32 vectors and converted to float once at tile end.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <omp.h>

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"

#include "e8_codebook.h"

namespace py = pybind11;

using PID = rabitqlib::PID;
using e8lib::kBlockDim;
using e8lib::kCodebookSize;
using e8lib::get_e8_codebook;

namespace {

constexpr float kConstEpsilon = 1.9f;
constexpr size_t kTile   = 16;  // IVFE8 tile width (gather variant)
constexpr size_t kTileFS = 32;  // IVFE8FastScan tile width

inline float dot8(const float* a, const float* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3] +
           a[4] * b[4] + a[5] * b[5] + a[6] * b[6] + a[7] * b[7];
}

/* Per-list storage for IVFE8 (16-wide tiles). */
struct ListStorage {
    std::vector<PID> ids;
    std::vector<uint8_t> codes_bm;       // block-major, 16-wide tiles
    std::vector<float> f_add;
    std::vector<float> f_rescale;
    size_t size() const { return ids.size(); }
};

/* Per-list storage for IVFE8FastScan (32-wide tiles). */
struct ListStorageFS {
    std::vector<PID> ids;
    std::vector<uint8_t> codes_bm;       // block-major, 32-wide tiles
    std::vector<float> f_add;
    std::vector<float> f_rescale;
    size_t size() const { return ids.size(); }
};

}  // anonymous

class IVFE8 {
public:
    IVFE8(size_t n, size_t dim, size_t nlist, int nthread,
          const std::string& metric, const std::string& rotator)
        : n_(n), dim_(dim), nlist_(nlist), nthread_(nthread) {

        (void)metric;

        padded_dim_ = rabitqlib::round_up_to_multiple(dim_, 64);
        if (padded_dim_ % kBlockDim != 0) {
            throw std::runtime_error("padded_dim must be multiple of 8");
        }
        n_blocks_ = padded_dim_ / kBlockDim;

        rabitqlib::RotatorType rtype =
            (rotator == "matrix") ? rabitqlib::RotatorType::MatrixRotator
                                  : rabitqlib::RotatorType::FhtKacRotator;
        std::srand(dim_ + padded_dim_);
        rotator_.reset(rabitqlib::choose_rotator<float>(dim_, rtype, padded_dim_));

        rotated_centroids_.assign(nlist_ * padded_dim_, 0.0f);
        centroids_.assign(nlist_ * dim_, 0.0f);
        lists_.resize(nlist_);
    }

    void construct(py::array_t<float, py::array::c_style | py::array::forcecast> data,
                   py::array_t<float, py::array::c_style | py::array::forcecast> centroids,
                   py::array_t<uint32_t, py::array::c_style | py::array::forcecast> cluster_ids) {
        auto d = data.request();
        auto c = centroids.request();
        auto ids = cluster_ids.request();
        if (d.ndim != 2) throw std::runtime_error("data must be 2-D");
        if (c.ndim != 2) throw std::runtime_error("centroids must be 2-D");
        if (ids.ndim != 1) throw std::runtime_error("cluster_ids must be 1-D");

        size_t nb = static_cast<size_t>(d.shape[0]);
        const float* dptr = static_cast<const float*>(d.ptr);
        const float* cptr = static_cast<const float*>(c.ptr);
        const uint32_t* idptr = static_cast<const uint32_t*>(ids.ptr);

        std::memcpy(centroids_.data(), cptr, nlist_ * dim_ * sizeof(float));
        for (size_t i = 0; i < nlist_; ++i) {
            rotator_->rotate(centroids_.data() + i * dim_,
                             rotated_centroids_.data() + i * padded_dim_);
        }

        std::vector<size_t> counts(nlist_, 0);
        for (size_t i = 0; i < nb; ++i) {
            if (idptr[i] >= nlist_) throw std::runtime_error("cluster id out of range");
            ++counts[idptr[i]];
        }
        for (size_t l = 0; l < nlist_; ++l) {
            size_t sz = counts[l];
            size_t padded_sz = ((sz + kTile - 1) / kTile) * kTile;
            lists_[l].ids.resize(sz);
            lists_[l].codes_bm.assign(padded_sz * n_blocks_, 0);
            lists_[l].f_add.assign(padded_sz, 0.0f);
            lists_[l].f_rescale.assign(padded_sz, 0.0f);
        }

        std::vector<size_t> cursor(nlist_, 0);
        std::vector<size_t> pos(nb);
        for (size_t i = 0; i < nb; ++i) {
            uint32_t cid = idptr[i];
            pos[i] = cursor[cid]++;
            lists_[cid].ids[pos[i]] = static_cast<PID>(i);
        }

        omp_set_num_threads(nthread_);
        const float* cb = get_e8_codebook().data();

        #pragma omp parallel
        {
            std::vector<float> rotated(padded_dim_);
            std::vector<float> residual(padded_dim_);
            std::vector<float> reconstructed(padded_dim_);

            #pragma omp for schedule(static)
            for (int64_t i = 0; i < (int64_t)nb; ++i) {
                uint32_t cid = idptr[i];
                size_t p = pos[i];

                rotator_->rotate(dptr + i * dim_, rotated.data());
                const float* cent_rot = rotated_centroids_.data() + cid * padded_dim_;
                for (size_t k = 0; k < padded_dim_; ++k) {
                    residual[k] = rotated[k] - cent_rot[k];
                }

                // Quantize + build reconstruction for factor computation.
                uint8_t code_scratch[4096 / kBlockDim];  // >= n_blocks_ for padded_dim<=4096
                for (size_t b = 0; b < n_blocks_; ++b) {
                    uint8_t k = get_e8_codebook().encode_block(residual.data() + b * kBlockDim);
                    code_scratch[b] = k;
                    const float* cw = cb + k * kBlockDim;
                    float* rec = reconstructed.data() + b * kBlockDim;
                    for (size_t dd = 0; dd < kBlockDim; ++dd) rec[dd] = cw[dd];
                }

                // Scatter codes into block-major tile layout.
                size_t tile = p / kTile;
                size_t lane = p % kTile;
                uint8_t* tile_base = lists_[cid].codes_bm.data() + tile * kTile * n_blocks_;
                for (size_t b = 0; b < n_blocks_; ++b) {
                    tile_base[b * kTile + lane] = code_scratch[b];
                }

                float l2_sqr = rabitqlib::l2norm_sqr<float>(residual.data(), padded_dim_);
                float ip_resi_qr = rabitqlib::dot_product<float>(residual.data(),
                                                                 reconstructed.data(),
                                                                 padded_dim_);
                float ip_cent_qr = rabitqlib::dot_product<float>(cent_rot,
                                                                 reconstructed.data(),
                                                                 padded_dim_);
                float norm_qr_sq = rabitqlib::l2norm_sqr<float>(reconstructed.data(),
                                                                padded_dim_);

                if (ip_resi_qr == 0.0f) {
                    ip_resi_qr = std::numeric_limits<float>::infinity();
                }

                float f_add     = l2_sqr + (2.0f * l2_sqr * ip_cent_qr / ip_resi_qr);
                float f_rescale = -2.0f * l2_sqr / ip_resi_qr;
                lists_[cid].f_add[p]     = f_add;
                lists_[cid].f_rescale[p] = f_rescale;
            }
        }
    }

    std::pair<py::array_t<int64_t>, py::array_t<float>>
    search_batch(py::array_t<float, py::array::c_style | py::array::forcecast> queries,
                 size_t k, size_t nprobe) {
        auto q = queries.request();
        if (q.ndim != 2) throw std::runtime_error("queries must be 2-D");
        size_t nq = static_cast<size_t>(q.shape[0]);
        const float* qptr = static_cast<const float*>(q.ptr);

        py::array_t<int64_t> I({static_cast<py::ssize_t>(nq),
                                static_cast<py::ssize_t>(k)});
        py::array_t<float> D({static_cast<py::ssize_t>(nq),
                              static_cast<py::ssize_t>(k)});
        int64_t* Iptr = static_cast<int64_t*>(I.request().ptr);
        float* Dptr = static_cast<float*>(D.request().ptr);

        omp_set_num_threads(nthread_);

        const size_t padded_dim = padded_dim_;
        const size_t n_blocks = n_blocks_;

        #pragma omp parallel
        {
            std::vector<float> rotated(padded_dim);
            std::vector<float> lut(n_blocks * kCodebookSize);
            std::vector<std::pair<float, size_t>> probe_list(nlist_);
            std::vector<float> ip_tile(kTile);

            #pragma omp for schedule(dynamic, 8)
            for (int64_t qi = 0; qi < (int64_t)nq; ++qi) {
                const float* query = qptr + qi * dim_;
                rotator_->rotate(query, rotated.data());

                for (size_t l = 0; l < nlist_; ++l) {
                    float dsq = rabitqlib::euclidean_sqr<float>(
                        rotated.data(),
                        rotated_centroids_.data() + l * padded_dim,
                        padded_dim);
                    probe_list[l] = {dsq, l};
                }
                size_t np = std::min(nprobe, nlist_);
                std::partial_sort(probe_list.begin(),
                                  probe_list.begin() + np,
                                  probe_list.end());

                // Build LUT: for each block, compute 256 inner products
                // <q_b, cb[k]> using the transposed codebook. Each iteration
                // produces 16 LUT entries via 8 broadcast-FMAs.
                const float* cb_T = get_e8_codebook().data_transposed();
                for (size_t b = 0; b < n_blocks; ++b) {
                    const float* qb = rotated.data() + b * kBlockDim;
                    float* row = lut.data() + b * kCodebookSize;
                    // Hoist broadcasts outside the slab loop.
                    __m512 q0 = _mm512_set1_ps(qb[0]);
                    __m512 q1 = _mm512_set1_ps(qb[1]);
                    __m512 q2 = _mm512_set1_ps(qb[2]);
                    __m512 q3 = _mm512_set1_ps(qb[3]);
                    __m512 q4 = _mm512_set1_ps(qb[4]);
                    __m512 q5 = _mm512_set1_ps(qb[5]);
                    __m512 q6 = _mm512_set1_ps(qb[6]);
                    __m512 q7 = _mm512_set1_ps(qb[7]);
                    for (size_t ck = 0; ck < kCodebookSize; ck += 16) {
                        __m512 acc = _mm512_mul_ps(
                            _mm512_load_ps(cb_T + 0 * kCodebookSize + ck), q0);
                        acc = _mm512_fmadd_ps(
                            _mm512_load_ps(cb_T + 1 * kCodebookSize + ck), q1, acc);
                        acc = _mm512_fmadd_ps(
                            _mm512_load_ps(cb_T + 2 * kCodebookSize + ck), q2, acc);
                        acc = _mm512_fmadd_ps(
                            _mm512_load_ps(cb_T + 3 * kCodebookSize + ck), q3, acc);
                        acc = _mm512_fmadd_ps(
                            _mm512_load_ps(cb_T + 4 * kCodebookSize + ck), q4, acc);
                        acc = _mm512_fmadd_ps(
                            _mm512_load_ps(cb_T + 5 * kCodebookSize + ck), q5, acc);
                        acc = _mm512_fmadd_ps(
                            _mm512_load_ps(cb_T + 6 * kCodebookSize + ck), q6, acc);
                        acc = _mm512_fmadd_ps(
                            _mm512_load_ps(cb_T + 7 * kCodebookSize + ck), q7, acc);
                        _mm512_storeu_ps(row + ck, acc);
                    }
                }

                std::priority_queue<std::pair<float, PID>> topk;

                for (size_t pp = 0; pp < np; ++pp) {
                    float g_add = probe_list[pp].first;
                    size_t l = probe_list[pp].second;
                    const auto& L = lists_[l];
                    size_t sz = L.size();
                    if (sz == 0) continue;

                    size_t n_full_tiles = sz / kTile;
                    size_t tail = sz - n_full_tiles * kTile;

                    // ---------------- SIMD full-tile path ----------------
                    const uint8_t* L_codes = L.codes_bm.data();
                    const float* L_fadd = L.f_add.data();
                    const float* L_fres = L.f_rescale.data();
                    const PID* L_ids = L.ids.data();
                    const size_t tile_stride = kTile * n_blocks;
                    // Prefetch first tile before entering the loop.
                    if (n_full_tiles > 0) {
                        for (size_t b = 0; b < n_blocks; b += 4) {
                            _mm_prefetch(reinterpret_cast<const char*>(L_codes + b * kTile),
                                         _MM_HINT_T0);
                        }
                    }
                    for (size_t t = 0; t < n_full_tiles; ++t) {
                        const uint8_t* tile_base = L_codes + t * tile_stride;
                        // Prefetch next tile's codes.
                        if (t + 1 < n_full_tiles) {
                            const uint8_t* next_tile = L_codes + (t + 1) * tile_stride;
                            for (size_t b = 0; b < n_blocks; b += 4) {
                                _mm_prefetch(reinterpret_cast<const char*>(next_tile + b * kTile),
                                             _MM_HINT_T0);
                            }
                        }

                        // Dual accumulators let the OOO core issue two
                        // gathers in flight (gather throughput is sub-1 per
                        // cycle on most Intel cores).
                        __m512 acc0 = _mm512_setzero_ps();
                        __m512 acc1 = _mm512_setzero_ps();
                        size_t b = 0;
                        size_t b_pair_end = (n_blocks / 2) * 2;
                        for (; b < b_pair_end; b += 2) {
                            __m128i c0 = _mm_loadu_si128(
                                reinterpret_cast<const __m128i*>(tile_base + (b + 0) * kTile));
                            __m128i c1 = _mm_loadu_si128(
                                reinterpret_cast<const __m128i*>(tile_base + (b + 1) * kTile));
                            __m512i i0 = _mm512_cvtepu8_epi32(c0);
                            __m512i i1 = _mm512_cvtepu8_epi32(c1);
                            const float* lut0 = lut.data() + (b + 0) * kCodebookSize;
                            const float* lut1 = lut.data() + (b + 1) * kCodebookSize;
                            __m512 v0 = _mm512_i32gather_ps(i0, lut0, 4);
                            __m512 v1 = _mm512_i32gather_ps(i1, lut1, 4);
                            acc0 = _mm512_add_ps(acc0, v0);
                            acc1 = _mm512_add_ps(acc1, v1);
                        }
                        for (; b < n_blocks; ++b) {
                            __m128i codes16 = _mm_loadu_si128(
                                reinterpret_cast<const __m128i*>(tile_base + b * kTile));
                            __m512i idx = _mm512_cvtepu8_epi32(codes16);
                            const float* lut_b = lut.data() + b * kCodebookSize;
                            __m512 vals = _mm512_i32gather_ps(idx, lut_b, 4);
                            acc0 = _mm512_add_ps(acc0, vals);
                        }
                        __m512 acc = _mm512_add_ps(acc0, acc1);
                        __m512 fadd = _mm512_loadu_ps(L_fadd + t * kTile);
                        __m512 fres = _mm512_loadu_ps(L_fres + t * kTile);
                        __m512 est  = _mm512_fmadd_ps(fres, acc,
                                        _mm512_add_ps(_mm512_set1_ps(g_add), fadd));

                        // Bound filter: if heap is full, only lanes below
                        // current top-k bound may evict the max. Skip the
                        // entire tile when no lane qualifies.
                        if (topk.size() >= k) {
                            __m512 bound = _mm512_set1_ps(topk.top().first);
                            __mmask16 win_mask =
                                _mm512_cmp_ps_mask(est, bound, _CMP_LT_OQ);
                            if (win_mask == 0) continue;

                            _mm512_storeu_ps(ip_tile.data(), est);
                            const PID* ids16 = L_ids + t * kTile;
                            while (win_mask) {
                                unsigned v = __builtin_ctz(win_mask);
                                float e = ip_tile[v];
                                if (e < topk.top().first) {
                                    topk.pop();
                                    topk.emplace(e, ids16[v]);
                                }
                                win_mask &= (win_mask - 1);
                            }
                        } else {
                            _mm512_storeu_ps(ip_tile.data(), est);
                            const PID* ids16 = L_ids + t * kTile;
                            for (size_t v = 0; v < kTile; ++v) {
                                float e = ip_tile[v];
                                if (topk.size() < k) {
                                    topk.emplace(e, ids16[v]);
                                } else if (e < topk.top().first) {
                                    topk.pop();
                                    topk.emplace(e, ids16[v]);
                                }
                            }
                        }
                    }

                    // ---------------- scalar tail path ----------------
                    if (tail > 0) {
                        const uint8_t* tile_base =
                            L.codes_bm.data() + n_full_tiles * kTile * n_blocks;
                        for (size_t v = 0; v < tail; ++v) {
                            float ip = 0.0f;
                            for (size_t b = 0; b < n_blocks; ++b) {
                                uint8_t c = tile_base[b * kTile + v];
                                ip += lut[b * kCodebookSize + c];
                            }
                            size_t p = n_full_tiles * kTile + v;
                            float est = g_add + L.f_add[p] + L.f_rescale[p] * ip;
                            PID id = L.ids[p];
                            if (topk.size() < k) {
                                topk.emplace(est, id);
                            } else if (est < topk.top().first) {
                                topk.pop();
                                topk.emplace(est, id);
                            }
                        }
                    }
                }

                // Write top-k results.
                for (size_t j = 0; j < k; ++j) {
                    Iptr[qi * k + j] = -1;
                    Dptr[qi * k + j] = std::numeric_limits<float>::infinity();
                }
                std::vector<std::pair<float, PID>> ordered;
                ordered.reserve(topk.size());
                while (!topk.empty()) {
                    ordered.push_back(topk.top());
                    topk.pop();
                }
                std::reverse(ordered.begin(), ordered.end());
                for (size_t j = 0; j < ordered.size(); ++j) {
                    Iptr[qi * k + j] = static_cast<int64_t>(ordered[j].second);
                    Dptr[qi * k + j] = ordered[j].first;
                }
            }
        }
        return {I, D};
    }

private:
    size_t n_;
    size_t dim_;
    size_t padded_dim_;
    size_t nlist_;
    size_t n_blocks_;
    int nthread_;

    std::unique_ptr<rabitqlib::Rotator<float>> rotator_;
    std::vector<float> centroids_;
    std::vector<float> rotated_centroids_;

    std::vector<ListStorage> lists_;
};

// ============================================================
// IVFE8FastScan — PQ-fastscan-style register-resident LUT
// ============================================================
//
// The 256-byte per-block LUT is quantised to uint8 and loaded into four ZMM
// registers (zmm_lut0..3), making it fully register-resident during the scan.
// Lookup uses _mm512_permutex2var_epi8 (requires AVX512VBMI).
// Tile width is 32; accumulation uses uint16 to avoid overflow.
//
// Layout of the in-register LUT for one block:
//   zmm_lut0 : uint8 distances for LUT[  0.. 63]
//   zmm_lut1 : uint8 distances for LUT[ 64..127]
//   zmm_lut2 : uint8 distances for LUT[128..191]
//   zmm_lut3 : uint8 distances for LUT[192..255]  (255 is a padding entry)
//
// Lookup pipeline for 32 codes (zero-extended to 64-byte ZMM):
//   1. zmm_dist_low  = permutex2var(lut0, idx,         lut1)  // IDs 0..127
//   2. zmm_idx_hi    = idx & 0x7F                              // strip bit-7
//   3. zmm_dist_high = permutex2var(lut2, zmm_idx_hi,  lut3)  // IDs 128..255
//   4. mask          = movepi8_mask(idx)                       // bit-7 → mask
//   5. zmm_final     = mask_blend(mask, dist_low, dist_high)   // select half
//   6. lower 32 bytes of zmm_final → cvtepu8_epi16 → acc_u16  // accumulate

class IVFE8FastScan {
public:
    IVFE8FastScan(size_t n, size_t dim, size_t nlist, int nthread,
                  const std::string& metric, const std::string& rotator)
        : n_(n), dim_(dim), nlist_(nlist), nthread_(nthread) {

        (void)metric;

        padded_dim_ = rabitqlib::round_up_to_multiple(dim_, 64);
        if (padded_dim_ % kBlockDim != 0) {
            throw std::runtime_error("padded_dim must be multiple of 8");
        }
        n_blocks_ = padded_dim_ / kBlockDim;

        rabitqlib::RotatorType rtype =
            (rotator == "matrix") ? rabitqlib::RotatorType::MatrixRotator
                                  : rabitqlib::RotatorType::FhtKacRotator;
        std::srand(dim_ + padded_dim_);
        rotator_.reset(rabitqlib::choose_rotator<float>(dim_, rtype, padded_dim_));

        rotated_centroids_.assign(nlist_ * padded_dim_, 0.0f);
        centroids_.assign(nlist_ * dim_, 0.0f);
        lists_.resize(nlist_);
    }

    void construct(py::array_t<float, py::array::c_style | py::array::forcecast> data,
                   py::array_t<float, py::array::c_style | py::array::forcecast> centroids,
                   py::array_t<uint32_t, py::array::c_style | py::array::forcecast> cluster_ids) {
        auto d = data.request();
        auto c = centroids.request();
        auto ids = cluster_ids.request();
        if (d.ndim != 2) throw std::runtime_error("data must be 2-D");
        if (c.ndim != 2) throw std::runtime_error("centroids must be 2-D");
        if (ids.ndim != 1) throw std::runtime_error("cluster_ids must be 1-D");

        size_t nb = static_cast<size_t>(d.shape[0]);
        const float* dptr = static_cast<const float*>(d.ptr);
        const float* cptr = static_cast<const float*>(c.ptr);
        const uint32_t* idptr = static_cast<const uint32_t*>(ids.ptr);

        std::memcpy(centroids_.data(), cptr, nlist_ * dim_ * sizeof(float));
        for (size_t i = 0; i < nlist_; ++i) {
            rotator_->rotate(centroids_.data() + i * dim_,
                             rotated_centroids_.data() + i * padded_dim_);
        }

        std::vector<size_t> counts(nlist_, 0);
        for (size_t i = 0; i < nb; ++i) {
            if (idptr[i] >= nlist_) throw std::runtime_error("cluster id out of range");
            ++counts[idptr[i]];
        }
        for (size_t l = 0; l < nlist_; ++l) {
            size_t sz = counts[l];
            size_t padded_sz = ((sz + kTileFS - 1) / kTileFS) * kTileFS;
            lists_[l].ids.resize(sz);
            lists_[l].codes_bm.assign(padded_sz * n_blocks_, 0);
            lists_[l].f_add.assign(padded_sz, 0.0f);
            lists_[l].f_rescale.assign(padded_sz, 0.0f);
        }

        std::vector<size_t> cursor(nlist_, 0);
        std::vector<size_t> pos(nb);
        for (size_t i = 0; i < nb; ++i) {
            uint32_t cid = idptr[i];
            pos[i] = cursor[cid]++;
            lists_[cid].ids[pos[i]] = static_cast<PID>(i);
        }

        omp_set_num_threads(nthread_);
        const float* cb = get_e8_codebook().data();

        #pragma omp parallel
        {
            std::vector<float> rotated(padded_dim_);
            std::vector<float> residual(padded_dim_);
            std::vector<float> reconstructed(padded_dim_);

            #pragma omp for schedule(static)
            for (int64_t i = 0; i < (int64_t)nb; ++i) {
                uint32_t cid = idptr[i];
                size_t p = pos[i];

                rotator_->rotate(dptr + i * dim_, rotated.data());
                const float* cent_rot = rotated_centroids_.data() + cid * padded_dim_;
                for (size_t k = 0; k < padded_dim_; ++k)
                    residual[k] = rotated[k] - cent_rot[k];

                uint8_t code_scratch[4096 / kBlockDim];
                for (size_t b = 0; b < n_blocks_; ++b) {
                    uint8_t k = get_e8_codebook().encode_block(residual.data() + b * kBlockDim);
                    code_scratch[b] = k;
                    const float* cw = cb + k * kBlockDim;
                    float* rec = reconstructed.data() + b * kBlockDim;
                    for (size_t dd = 0; dd < kBlockDim; ++dd) rec[dd] = cw[dd];
                }

                // Scatter codes into 32-wide block-major tile layout.
                size_t tile = p / kTileFS;
                size_t lane = p % kTileFS;
                uint8_t* tile_base = lists_[cid].codes_bm.data() + tile * kTileFS * n_blocks_;
                for (size_t b = 0; b < n_blocks_; ++b)
                    tile_base[b * kTileFS + lane] = code_scratch[b];

                float l2_sqr    = rabitqlib::l2norm_sqr<float>(residual.data(), padded_dim_);
                float ip_resi_qr = rabitqlib::dot_product<float>(residual.data(),
                                                                  reconstructed.data(), padded_dim_);
                float ip_cent_qr = rabitqlib::dot_product<float>(cent_rot,
                                                                  reconstructed.data(), padded_dim_);

                if (ip_resi_qr == 0.0f)
                    ip_resi_qr = std::numeric_limits<float>::infinity();

                lists_[cid].f_add[p]     = l2_sqr + 2.0f * l2_sqr * ip_cent_qr / ip_resi_qr;
                lists_[cid].f_rescale[p] = -2.0f * l2_sqr / ip_resi_qr;
            }
        }
    }

    std::pair<py::array_t<int64_t>, py::array_t<float>>
    search_batch(py::array_t<float, py::array::c_style | py::array::forcecast> queries,
                 size_t k, size_t nprobe) {
        auto q = queries.request();
        if (q.ndim != 2) throw std::runtime_error("queries must be 2-D");
        size_t nq = static_cast<size_t>(q.shape[0]);
        const float* qptr = static_cast<const float*>(q.ptr);

        py::array_t<int64_t> I({static_cast<py::ssize_t>(nq), static_cast<py::ssize_t>(k)});
        py::array_t<float>   D({static_cast<py::ssize_t>(nq), static_cast<py::ssize_t>(k)});
        int64_t* Iptr = static_cast<int64_t*>(I.request().ptr);
        float*   Dptr = static_cast<float*>(D.request().ptr);

        omp_set_num_threads(nthread_);

        const size_t padded_dim = padded_dim_;
        const size_t n_blocks   = n_blocks_;

        #pragma omp parallel
        {
            std::vector<float>   rotated(padded_dim);
            // Float LUT: [n_blocks][kCodebookSize]
            std::vector<float>   lut_f(n_blocks * kCodebookSize);
            // Uint8 LUT for in-register lookup: [n_blocks][kCodebookSize]
            // Each block's 256 bytes are stored contiguously and aligned to 64
            // so they can be loaded directly into four ZMM registers.
            alignas(64) std::vector<uint8_t> lut_u8(n_blocks * kCodebookSize);

            std::vector<std::pair<float, size_t>> probe_list(nlist_);
            std::vector<float> ip_tile(kTileFS);  // temp store for 32 estimates

            #pragma omp for schedule(dynamic, 8)
            for (int64_t qi = 0; qi < (int64_t)nq; ++qi) {
                const float* query = qptr + qi * dim_;
                rotator_->rotate(query, rotated.data());

                // -------- Coarse quantiser: find nearest nprobe centroids --------
                for (size_t l = 0; l < nlist_; ++l) {
                    float dsq = rabitqlib::euclidean_sqr<float>(
                        rotated.data(),
                        rotated_centroids_.data() + l * padded_dim,
                        padded_dim);
                    probe_list[l] = {dsq, l};
                }
                size_t np = std::min(nprobe, nlist_);
                std::partial_sort(probe_list.begin(),
                                  probe_list.begin() + np,
                                  probe_list.end());

                // -------- Build float LUT (same as IVFE8) --------
                const float* cb_T = get_e8_codebook().data_transposed();
                for (size_t b = 0; b < n_blocks; ++b) {
                    const float* qb  = rotated.data() + b * kBlockDim;
                    float*       row = lut_f.data() + b * kCodebookSize;
                    __m512 q0 = _mm512_set1_ps(qb[0]), q1 = _mm512_set1_ps(qb[1]);
                    __m512 q2 = _mm512_set1_ps(qb[2]), q3 = _mm512_set1_ps(qb[3]);
                    __m512 q4 = _mm512_set1_ps(qb[4]), q5 = _mm512_set1_ps(qb[5]);
                    __m512 q6 = _mm512_set1_ps(qb[6]), q7 = _mm512_set1_ps(qb[7]);
                    for (size_t ck = 0; ck < kCodebookSize; ck += 16) {
                        __m512 acc = _mm512_mul_ps(
                            _mm512_load_ps(cb_T + 0 * kCodebookSize + ck), q0);
                        acc = _mm512_fmadd_ps(
                            _mm512_load_ps(cb_T + 1 * kCodebookSize + ck), q1, acc);
                        acc = _mm512_fmadd_ps(
                            _mm512_load_ps(cb_T + 2 * kCodebookSize + ck), q2, acc);
                        acc = _mm512_fmadd_ps(
                            _mm512_load_ps(cb_T + 3 * kCodebookSize + ck), q3, acc);
                        acc = _mm512_fmadd_ps(
                            _mm512_load_ps(cb_T + 4 * kCodebookSize + ck), q4, acc);
                        acc = _mm512_fmadd_ps(
                            _mm512_load_ps(cb_T + 5 * kCodebookSize + ck), q5, acc);
                        acc = _mm512_fmadd_ps(
                            _mm512_load_ps(cb_T + 6 * kCodebookSize + ck), q6, acc);
                        acc = _mm512_fmadd_ps(
                            _mm512_load_ps(cb_T + 7 * kCodebookSize + ck), q7, acc);
                        _mm512_storeu_ps(row + ck, acc);
                    }
                }

#ifdef __AVX512VBMI__
                // ---- Quantise float LUT → uint8 (VBMI in-register path only) ----
                float global_min = lut_f[0], global_max = lut_f[0];
                for (float v : lut_f) {
                    if (v < global_min) global_min = v;
                    if (v > global_max) global_max = v;
                }
                float q_scale  = (global_max > global_min)
                                 ? (global_max - global_min) / 255.0f : 1.0f;
                float q_offset = (float)n_blocks * global_min;
                {
                    float q_inv = 1.0f / q_scale;
                    for (size_t i = 0; i < n_blocks * kCodebookSize; ++i) {
                        float v = (lut_f[i] - global_min) * q_inv;
                        lut_u8[i] = (uint8_t)(v < 0.0f ? 0 : v > 255.0f ? 255
                                                              : (uint8_t)(v + 0.5f));
                    }
                }
#endif  // __AVX512VBMI__

                // -------- Per-probe scan --------
                std::priority_queue<std::pair<float, PID>> topk;

                for (size_t pp = 0; pp < np; ++pp) {
                    float  g_add = probe_list[pp].first;
                    size_t l     = probe_list[pp].second;
                    const auto& L = lists_[l];
                    size_t sz = L.size();
                    if (sz == 0) continue;

                    size_t n_full_tiles = sz / kTileFS;
                    size_t tail         = sz - n_full_tiles * kTileFS;

                    const uint8_t* L_codes = L.codes_bm.data();
                    const float*   L_fadd  = L.f_add.data();
                    const float*   L_fres  = L.f_rescale.data();
                    const PID*     L_ids   = L.ids.data();
                    const size_t   tile_stride = kTileFS * n_blocks;

#ifdef __AVX512VBMI__
                    // ---- Register-resident LUT path (AVX512VBMI) ----
                    //
                    // Each block's 256-byte uint8 LUT is pre-loaded into 4 ZMM registers.
                    // _mm512_permutex2var_epi8 performs the lookup with zero L1 traffic.
                    {
                    const __m512i zmm_7f   = _mm512_set1_epi8(0x7f);
                    const __m512  zmm_qs   = _mm512_set1_ps(q_scale);
                    const __m512  zmm_qof  = _mm512_set1_ps(q_offset);
                    const __m512  zmm_gadd = _mm512_set1_ps(g_add);

                    for (size_t t = 0; t < n_full_tiles; ++t) {
                        const uint8_t* tile_base = L_codes + t * tile_stride;
                        if (t + 1 < n_full_tiles) {
                            const uint8_t* next = L_codes + (t + 1) * tile_stride;
                            for (size_t b = 0; b < n_blocks; b += 4)
                                _mm_prefetch(reinterpret_cast<const char*>(next + b * kTileFS),
                                             _MM_HINT_T0);
                        }

                        __m512i acc = _mm512_setzero_si512();
                        for (size_t b = 0; b < n_blocks; ++b) {
                            const uint8_t* lut_b = lut_u8.data() + b * kCodebookSize;
                            __m512i zmm_lut0 = _mm512_loadu_si512(lut_b +   0);
                            __m512i zmm_lut1 = _mm512_loadu_si512(lut_b +  64);
                            __m512i zmm_lut2 = _mm512_loadu_si512(lut_b + 128);
                            __m512i zmm_lut3 = _mm512_loadu_si512(lut_b + 192);
                            __m256i codes256 = _mm256_loadu_si256(
                                reinterpret_cast<const __m256i*>(tile_base + b * kTileFS));
                            __m512i zmm_idx    = _mm512_zextsi256_si512(codes256);
                            __m512i d_lo       = _mm512_permutex2var_epi8(zmm_lut0, zmm_idx, zmm_lut1);
                            __m512i zmm_idx_hi = _mm512_and_si512(zmm_idx, zmm_7f);
                            __m512i d_hi       = _mm512_permutex2var_epi8(zmm_lut2, zmm_idx_hi, zmm_lut3);
                            __mmask64 hi_mask  = _mm512_movepi8_mask(zmm_idx);
                            __m512i zmm_dist   = _mm512_mask_blend_epi8(hi_mask, d_lo, d_hi);
                            acc = _mm512_add_epi16(acc, _mm512_cvtepu8_epi16(
                                _mm512_castsi512_si256(zmm_dist)));
                        }

                        // uint16 → float, reverse quantisation: ip_float = ip_u8*q_scale + q_offset
                        __m512 f_lo = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(_mm512_castsi512_si256(acc)));
                        __m512 f_hi = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(
                            _mm512_extracti64x4_epi64(acc, 1)));
                        f_lo = _mm512_fmadd_ps(f_lo, zmm_qs, zmm_qof);
                        f_hi = _mm512_fmadd_ps(f_hi, zmm_qs, zmm_qof);

                        // RaBitQ scoring: est = g_add + f_add + f_rescale * ip
                        __m512 est_lo = _mm512_fmadd_ps(_mm512_loadu_ps(L_fres + t * kTileFS), f_lo,
                                            _mm512_add_ps(zmm_gadd, _mm512_loadu_ps(L_fadd + t * kTileFS)));
                        __m512 est_hi = _mm512_fmadd_ps(_mm512_loadu_ps(L_fres + t * kTileFS + 16), f_hi,
                                            _mm512_add_ps(zmm_gadd, _mm512_loadu_ps(L_fadd + t * kTileFS + 16)));

                        // Top-k update
                        const PID* ids32 = L_ids + t * kTileFS;
                        if (topk.size() >= k) {
                            __m512 bound   = _mm512_set1_ps(topk.top().first);
                            __mmask16 wl   = _mm512_cmp_ps_mask(est_lo, bound, _CMP_LT_OQ);
                            __mmask16 wh   = _mm512_cmp_ps_mask(est_hi, bound, _CMP_LT_OQ);
                            if ((wl | wh) == 0) continue;
                            _mm512_storeu_ps(ip_tile.data(),      est_lo);
                            _mm512_storeu_ps(ip_tile.data() + 16, est_hi);
                            while (wl) {
                                unsigned v = __builtin_ctz(wl);
                                float e = ip_tile[v];
                                if (e < topk.top().first) { topk.pop(); topk.emplace(e, ids32[v]); }
                                wl &= wl - 1;
                            }
                            while (wh) {
                                unsigned v = __builtin_ctz(wh);
                                float e = ip_tile[v + 16];
                                if (e < topk.top().first) { topk.pop(); topk.emplace(e, ids32[v + 16]); }
                                wh &= wh - 1;
                            }
                        } else {
                            _mm512_storeu_ps(ip_tile.data(),      est_lo);
                            _mm512_storeu_ps(ip_tile.data() + 16, est_hi);
                            for (size_t v = 0; v < kTileFS; ++v) {
                                float e = ip_tile[v];
                                if (topk.size() < k) topk.emplace(e, ids32[v]);
                                else if (e < topk.top().first) { topk.pop(); topk.emplace(e, ids32[v]); }
                            }
                        }
                    }  // for t (VBMI)
                    }

#else  // !__AVX512VBMI__ — Cascade Lake: float gather from lut_f, 32-wide tiles
                    //
                    // Uses _mm512_i32gather_ps from the float LUT (same as IVFE8) but
                    // processes 32 vectors per tile.  Adjacent blocks are unrolled in
                    // pairs so that 4 independent gathers are in flight per iteration,
                    // hiding gather latency on CPUs without AVX512VBMI.
                    {
                    const __m512 zmm_gadd = _mm512_set1_ps(g_add);

                    for (size_t t = 0; t < n_full_tiles; ++t) {
                        const uint8_t* tile_base = L_codes + t * tile_stride;
                        if (t + 1 < n_full_tiles) {
                            const uint8_t* next = L_codes + (t + 1) * tile_stride;
                            for (size_t b = 0; b < n_blocks; b += 4)
                                _mm_prefetch(reinterpret_cast<const char*>(next + b * kTileFS),
                                             _MM_HINT_T0);
                        }

                        // acc0 = ip sum for lower 16 vectors; acc1 = upper 16 vectors.
                        __m512 acc0 = _mm512_setzero_ps();
                        __m512 acc1 = _mm512_setzero_ps();

                        // Dual-block unroll: issue 4 independent gathers per iteration.
                        size_t b = 0;
                        const size_t b_pair_end = (n_blocks / 2) * 2;
                        for (; b < b_pair_end; b += 2) {
                            const uint8_t* col0  = tile_base + (b + 0) * kTileFS;
                            const uint8_t* col1  = tile_base + (b + 1) * kTileFS;
                            const float*   lut0f = lut_f.data() + (b + 0) * kCodebookSize;
                            const float*   lut1f = lut_f.data() + (b + 1) * kCodebookSize;
                            __m512i i0_lo = _mm512_cvtepu8_epi32(
                                _mm_loadu_si128(reinterpret_cast<const __m128i*>(col0)));
                            __m512i i0_hi = _mm512_cvtepu8_epi32(
                                _mm_loadu_si128(reinterpret_cast<const __m128i*>(col0 + 16)));
                            __m512i i1_lo = _mm512_cvtepu8_epi32(
                                _mm_loadu_si128(reinterpret_cast<const __m128i*>(col1)));
                            __m512i i1_hi = _mm512_cvtepu8_epi32(
                                _mm_loadu_si128(reinterpret_cast<const __m128i*>(col1 + 16)));
                            acc0 = _mm512_add_ps(acc0,
                                _mm512_add_ps(_mm512_i32gather_ps(i0_lo, lut0f, 4),
                                              _mm512_i32gather_ps(i1_lo, lut1f, 4)));
                            acc1 = _mm512_add_ps(acc1,
                                _mm512_add_ps(_mm512_i32gather_ps(i0_hi, lut0f, 4),
                                              _mm512_i32gather_ps(i1_hi, lut1f, 4)));
                        }
                        for (; b < n_blocks; ++b) {
                            const uint8_t* col = tile_base + b * kTileFS;
                            const float*   lbf = lut_f.data() + b * kCodebookSize;
                            acc0 = _mm512_add_ps(acc0, _mm512_i32gather_ps(
                                _mm512_cvtepu8_epi32(_mm_loadu_si128(
                                    reinterpret_cast<const __m128i*>(col))), lbf, 4));
                            acc1 = _mm512_add_ps(acc1, _mm512_i32gather_ps(
                                _mm512_cvtepu8_epi32(_mm_loadu_si128(
                                    reinterpret_cast<const __m128i*>(col + 16))), lbf, 4));
                        }

                        // RaBitQ scoring (acc0/acc1 are already float ip sums)
                        __m512 est_lo = _mm512_fmadd_ps(_mm512_loadu_ps(L_fres + t * kTileFS), acc0,
                                            _mm512_add_ps(zmm_gadd, _mm512_loadu_ps(L_fadd + t * kTileFS)));
                        __m512 est_hi = _mm512_fmadd_ps(_mm512_loadu_ps(L_fres + t * kTileFS + 16), acc1,
                                            _mm512_add_ps(zmm_gadd, _mm512_loadu_ps(L_fadd + t * kTileFS + 16)));

                        // Top-k update
                        const PID* ids32 = L_ids + t * kTileFS;
                        if (topk.size() >= k) {
                            __m512 bound   = _mm512_set1_ps(topk.top().first);
                            __mmask16 wl   = _mm512_cmp_ps_mask(est_lo, bound, _CMP_LT_OQ);
                            __mmask16 wh   = _mm512_cmp_ps_mask(est_hi, bound, _CMP_LT_OQ);
                            if ((wl | wh) == 0) continue;
                            _mm512_storeu_ps(ip_tile.data(),      est_lo);
                            _mm512_storeu_ps(ip_tile.data() + 16, est_hi);
                            while (wl) {
                                unsigned v = __builtin_ctz(wl);
                                float e = ip_tile[v];
                                if (e < topk.top().first) { topk.pop(); topk.emplace(e, ids32[v]); }
                                wl &= wl - 1;
                            }
                            while (wh) {
                                unsigned v = __builtin_ctz(wh);
                                float e = ip_tile[v + 16];
                                if (e < topk.top().first) { topk.pop(); topk.emplace(e, ids32[v + 16]); }
                                wh &= wh - 1;
                            }
                        } else {
                            _mm512_storeu_ps(ip_tile.data(),      est_lo);
                            _mm512_storeu_ps(ip_tile.data() + 16, est_hi);
                            for (size_t v = 0; v < kTileFS; ++v) {
                                float e = ip_tile[v];
                                if (topk.size() < k) topk.emplace(e, ids32[v]);
                                else if (e < topk.top().first) { topk.pop(); topk.emplace(e, ids32[v]); }
                            }
                        }
                    }  // for t (Cascade Lake)
                    }

#endif  // __AVX512VBMI__

                    // ---- Scalar tail (< kTileFS vectors) ----
                    if (tail > 0) {
                        const uint8_t* tail_base =
                            L_codes + n_full_tiles * tile_stride;
                        for (size_t v = 0; v < tail; ++v) {
                            float ip = 0.0f;
                            for (size_t b = 0; b < n_blocks; ++b) {
                                uint8_t c = tail_base[b * kTileFS + v];
                                ip += lut_f[b * kCodebookSize + c];
                            }
                            size_t p = n_full_tiles * kTileFS + v;
                            float est = g_add + L.f_add[p] + L.f_rescale[p] * ip;
                            PID id = L.ids[p];
                            if (topk.size() < k) {
                                topk.emplace(est, id);
                            } else if (est < topk.top().first) {
                                topk.pop();
                                topk.emplace(est, id);
                            }
                        }
                    }
                }  // for each probe

                // Write results
                for (size_t j = 0; j < k; ++j) {
                    Iptr[qi * k + j] = -1;
                    Dptr[qi * k + j] = std::numeric_limits<float>::infinity();
                }
                std::vector<std::pair<float, PID>> ordered;
                ordered.reserve(topk.size());
                while (!topk.empty()) {
                    ordered.push_back(topk.top());
                    topk.pop();
                }
                std::reverse(ordered.begin(), ordered.end());
                for (size_t j = 0; j < ordered.size(); ++j) {
                    Iptr[qi * k + j] = static_cast<int64_t>(ordered[j].second);
                    Dptr[qi * k + j] = ordered[j].first;
                }
            }
        }
        return {I, D};
    }

private:
    size_t n_, dim_, padded_dim_, nlist_, n_blocks_;
    int nthread_;
    std::unique_ptr<rabitqlib::Rotator<float>> rotator_;
    std::vector<float> centroids_;
    std::vector<float> rotated_centroids_;
    std::vector<ListStorageFS> lists_;
};

PYBIND11_MODULE(e8_cpp, m) {
    m.doc() = "IVF + E8-lattice 1-bit quantization (AVX-512 gather)";
    py::class_<IVFE8>(m, "IVFE8")
        .def(py::init<size_t, size_t, size_t, int,
                      const std::string&, const std::string&>(),
             py::arg("n"),
             py::arg("dim"),
             py::arg("nlist"),
             py::arg("nthread") = 1,
             py::arg("metric") = "l2",
             py::arg("rotator") = "fht")
        .def("construct", &IVFE8::construct,
             py::arg("data"), py::arg("centroids"), py::arg("cluster_ids"))
        .def("search_batch", &IVFE8::search_batch,
             py::arg("queries"), py::arg("k"), py::arg("nprobe"));
    py::class_<IVFE8FastScan>(m, "IVFE8FastScan")
        .def(py::init<size_t, size_t, size_t, int,
                      const std::string&, const std::string&>(),
             py::arg("n"),
             py::arg("dim"),
             py::arg("nlist"),
             py::arg("nthread") = 1,
             py::arg("metric") = "l2",
             py::arg("rotator") = "fht")
        .def("construct", &IVFE8FastScan::construct,
             py::arg("data"), py::arg("centroids"), py::arg("cluster_ids"))
        .def("search_batch", &IVFE8FastScan::search_batch,
             py::arg("queries"), py::arg("k"), py::arg("nprobe"));
}
