// pybind11 binding for IVF + E8-lattice 1-bit with a LUT-free search path.
//
// Quantizer: same 240 min-norm E8 vectors as IVFE8 (FHT-Kac rotation,
// per-block 1-byte code, RaBitQ-style unbiased factors). Each code byte is
// stored as [flag:1 | payload:7]: flag 0 = Type A (5-bit pair idx + 2 signs),
// flag 1 = Type B (7 signs; 8th sign recovered by even parity).
//
// Search kernel: per block the kernel computes both paths for all 16 lanes
// of a tile and blends by the flag bit.
//   A path: vpermi2d on (i,j) tables -> vpermps on broadcast-q -> mask_sub
//           for signs -> add.
//   B path: 7 mask_sub accumulations from explicit bits + one from the
//           XOR-folded parity bit -> multiply by 0.5.
// No per-query LUT is built, no gather is issued.

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
#include "e8_codebook12.h"
#include "e8_codebook13.h"
#include "e8_codebook15.h"

namespace py = pybind11;

using PID = rabitqlib::PID;
using e8lib::kBlockDim;
using e8lib::kCodebookSize;
using e8lib::get_e8_codebook;
using e8lib::kFlagMask;
using e8lib::kPayloadMask;
using e8lib::kPairShift;
using e8lib::kPairMask;
using e8lib::kSignIBit;
using e8lib::kSignJBit;

namespace {

constexpr size_t kTile = 16;  // AVX-512 tile width

inline float scalar_ip_packed(uint8_t packed, const float* q_b) {
    float cw[8];
    get_e8_codebook().decode_packed(packed, cw);
    float s = 0.0f;
    for (int d = 0; d < 8; ++d) s += cw[d] * q_b[d];
    return s;
}

/* Per-list storage. codes are laid out in block-major tiles: for tiles of 16
 * vectors, block b of tile t occupies 16 consecutive bytes at
 *    codes_bm[t * n_blocks * 16 + b * 16 + v].
 * Tail (< 16 vectors) uses the same layout; scalar tail path reads
 * `codes_bm[n_full_tiles * kTile * n_blocks + b * kTile + v]`. */
struct ListStorage {
    std::vector<PID> ids;
    std::vector<uint8_t> codes_bm;
    std::vector<float> f_add;
    std::vector<float> f_rescale;
    size_t size() const { return ids.size(); }
};

}  // anonymous

class IVFE8NoLut {
public:
    IVFE8NoLut(size_t n, size_t dim, size_t nlist, int nthread,
               const std::string& metric, const std::string& rotator = "fht")
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
        const auto& cb = get_e8_codebook();

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

                uint8_t code_scratch[4096 / kBlockDim];
                for (size_t b = 0; b < n_blocks_; ++b) {
                    uint8_t raw_k = cb.encode_block_raw(residual.data() + b * kBlockDim);
                    uint8_t packed = cb.raw_to_packed(raw_k);
                    code_scratch[b] = packed;
                    const float* cw = cb.data() + raw_k * kBlockDim;
                    float* rec = reconstructed.data() + b * kBlockDim;
                    for (size_t dd = 0; dd < kBlockDim; ++dd) rec[dd] = cw[dd];
                }

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

        const auto& cb = get_e8_codebook();
        const int32_t* pair_i_ptr = cb.pair_to_i_arr();   // 32 int32 entries
        const int32_t* pair_j_ptr = cb.pair_to_j_arr();

        // Split pair tables into low/high halves for vpermi2d.
        const __m512i pair_i_lo = _mm512_load_si512(
            reinterpret_cast<const __m512i*>(pair_i_ptr));
        const __m512i pair_i_hi = _mm512_load_si512(
            reinterpret_cast<const __m512i*>(pair_i_ptr + 16));
        const __m512i pair_j_lo = _mm512_load_si512(
            reinterpret_cast<const __m512i*>(pair_j_ptr));
        const __m512i pair_j_hi = _mm512_load_si512(
            reinterpret_cast<const __m512i*>(pair_j_ptr + 16));

        const __m512i k_pair_mask_v  = _mm512_set1_epi32(kPairMask);
        const __m512i k_sign_i_v     = _mm512_set1_epi32(kSignIBit);
        const __m512i k_sign_j_v     = _mm512_set1_epi32(kSignJBit);
        const __m512i k_flag_v       = _mm512_set1_epi32(kFlagMask);
        const __m512i k_one_v        = _mm512_set1_epi32(1);
        const __m512  k_half_v       = _mm512_set1_ps(0.5f);
        const __m512  k_zero_v       = _mm512_setzero_ps();

        #pragma omp parallel
        {
            std::vector<float> rotated(padded_dim);
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

                std::priority_queue<std::pair<float, PID>> topk;

                for (size_t pp = 0; pp < np; ++pp) {
                    float g_add = probe_list[pp].first;
                    size_t l = probe_list[pp].second;
                    const auto& L = lists_[l];
                    size_t sz = L.size();
                    if (sz == 0) continue;

                    size_t n_full_tiles = sz / kTile;
                    size_t tail = sz - n_full_tiles * kTile;

                    if (force_scalar_) {
                        // Per-position scalar path used as correctness baseline
                        // for the SIMD kernel. Walks each position, translates
                        // to (tile, lane), reads code via the canonical layout.
                        for (size_t p = 0; p < sz; ++p) {
                            size_t tile = p / kTile;
                            size_t lane = p % kTile;
                            const uint8_t* tb =
                                L.codes_bm.data() + tile * kTile * n_blocks;
                            float ip = 0.0f;
                            for (size_t b = 0; b < n_blocks; ++b) {
                                uint8_t packed = tb[b * kTile + lane];
                                ip += scalar_ip_packed(packed,
                                                       rotated.data() + b * kBlockDim);
                            }
                            float est = g_add + L.f_add[p] + L.f_rescale[p] * ip;
                            PID id = L.ids[p];
                            if (topk.size() < k) {
                                topk.emplace(est, id);
                            } else if (est < topk.top().first) {
                                topk.pop();
                                topk.emplace(est, id);
                            }
                        }
                        continue;
                    }

                    const uint8_t* L_codes = L.codes_bm.data();
                    const float* L_fadd = L.f_add.data();
                    const float* L_fres = L.f_rescale.data();
                    const PID* L_ids = L.ids.data();
                    const size_t tile_stride = kTile * n_blocks;

                    if (n_full_tiles > 0) {
                        for (size_t b = 0; b < n_blocks; b += 4) {
                            _mm_prefetch(reinterpret_cast<const char*>(L_codes + b * kTile),
                                         _MM_HINT_T0);
                        }
                    }

                    for (size_t t = 0; t < n_full_tiles; ++t) {
                        const uint8_t* tile_base = L_codes + t * tile_stride;
                        if (t + 1 < n_full_tiles) {
                            const uint8_t* next_tile = L_codes + (t + 1) * tile_stride;
                            for (size_t b = 0; b < n_blocks; b += 4) {
                                _mm_prefetch(reinterpret_cast<const char*>(next_tile + b * kTile),
                                             _MM_HINT_T0);
                            }
                        }

                        __m512 acc = _mm512_setzero_ps();
                        for (size_t b = 0; b < n_blocks; ++b) {
                            const float* q_b = rotated.data() + b * kBlockDim;

                            // q_bcast = [q_b[0..7] | q_b[0..7]] across 16 lanes
                            __m256 q_b_256 = _mm256_loadu_ps(q_b);
                            __m512 q_bcast = _mm512_broadcast_f32x8(q_b_256);

                            // Load 16 code bytes -> 16 int32
                            __m128i c16 = _mm_loadu_si128(
                                reinterpret_cast<const __m128i*>(tile_base + b * kTile));
                            __m512i c32 = _mm512_cvtepu8_epi32(c16);

                            // ----- A path -----
                            __m512i pair_id = _mm512_and_epi32(
                                _mm512_srli_epi32(c32, kPairShift), k_pair_mask_v);
                            __m512i i_lane = _mm512_permutex2var_epi32(
                                pair_i_lo, pair_id, pair_i_hi);
                            __m512i j_lane = _mm512_permutex2var_epi32(
                                pair_j_lo, pair_id, pair_j_hi);
                            __m512 q_i = _mm512_permutexvar_ps(i_lane, q_bcast);
                            __m512 q_j = _mm512_permutexvar_ps(j_lane, q_bcast);
                            __mmask16 mi = _mm512_test_epi32_mask(c32, k_sign_i_v);
                            __mmask16 mj = _mm512_test_epi32_mask(c32, k_sign_j_v);
                            q_i = _mm512_mask_sub_ps(q_i, mi, k_zero_v, q_i);
                            q_j = _mm512_mask_sub_ps(q_j, mj, k_zero_v, q_j);
                            __m512 ip_A = _mm512_add_ps(q_i, q_j);

                            // ----- B path -----
                            __m512 acc_B = _mm512_setzero_ps();
                            // d = 0..6: bit from code byte
                            #pragma GCC unroll 7
                            for (int d = 0; d < 7; ++d) {
                                __mmask16 mb = _mm512_test_epi32_mask(
                                    c32, _mm512_set1_epi32(1 << d));
                                __m512 qd = _mm512_set1_ps(q_b[d]);
                                __m512 sel = _mm512_mask_sub_ps(qd, mb, k_zero_v, qd);
                                acc_B = _mm512_add_ps(acc_B, sel);
                            }
                            // d = 7: parity of low 7 bits (XOR fold)
                            __m512i low7 = _mm512_and_epi32(
                                c32, _mm512_set1_epi32(kPayloadMask));
                            __m512i x = _mm512_xor_epi32(
                                low7, _mm512_srli_epi32(low7, 4));
                            x = _mm512_xor_epi32(x, _mm512_srli_epi32(x, 2));
                            x = _mm512_xor_epi32(x, _mm512_srli_epi32(x, 1));
                            __mmask16 mb7 = _mm512_test_epi32_mask(x, k_one_v);
                            __m512 qd7 = _mm512_set1_ps(q_b[7]);
                            __m512 sel7 = _mm512_mask_sub_ps(qd7, mb7, k_zero_v, qd7);
                            acc_B = _mm512_add_ps(acc_B, sel7);
                            __m512 ip_B = _mm512_mul_ps(acc_B, k_half_v);

                            // Blend: flag bit set -> B, clear -> A
                            __mmask16 flag_B = _mm512_test_epi32_mask(c32, k_flag_v);
                            __m512 ip = _mm512_mask_mov_ps(ip_A, flag_B, ip_B);
                            acc = _mm512_add_ps(acc, ip);
                        }
                        __m512 fadd = _mm512_loadu_ps(L_fadd + t * kTile);
                        __m512 fres = _mm512_loadu_ps(L_fres + t * kTile);
                        __m512 est  = _mm512_fmadd_ps(fres, acc,
                                        _mm512_add_ps(_mm512_set1_ps(g_add), fadd));

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

                    // scalar tail
                    if (tail > 0) {
                        const uint8_t* tile_base =
                            L.codes_bm.data() + n_full_tiles * kTile * n_blocks;
                        for (size_t v = 0; v < tail; ++v) {
                            float ip = 0.0f;
                            for (size_t b = 0; b < n_blocks; ++b) {
                                uint8_t packed = tile_base[b * kTile + v];
                                ip += scalar_ip_packed(packed,
                                                       rotated.data() + b * kBlockDim);
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
public:
    bool force_scalar_ = false;
};

/* ===========================================================================
 * 1.5-bit (12-bit/block) variant.
 *
 * Same outer scaffolding (IVF + FHT-Kac rotation + RaBitQ-style factors) as
 * IVFE8NoLut, but the per-block codebook is the 2160-vector second shell of E8
 * (norm^2 = 4) and codes are packed at 12 bits per block. The search kernel
 * computes <q_b, codeword> directly from the bits along three paths (D / C / E)
 * and blends per lane based on flag bits; no per-query LUT, no gather.
 *
 * Storage per (tile, block) is 24 bytes:
 *   bytes [0..15]  : low 8 bits of each lane's code (lane v at offset v)
 *   bytes [16..23] : high 4 bits, two lanes per byte
 *                    (lane 2k -> low nibble, lane 2k+1 -> high nibble)
 * ===========================================================================
 */

namespace {

using e8lib12::kCodeBitsPerBlock;
using e8lib12::kPackedBytesPerTileBlock;

constexpr size_t kBlockBytes15 = kPackedBytesPerTileBlock;  // 24

inline uint16_t read_packed_code15(const uint8_t* tile_base, size_t b, size_t lane) {
    const uint8_t* blk = tile_base + b * kBlockBytes15;
    uint16_t lo  = blk[lane];
    uint8_t  hib = blk[16 + (lane >> 1)];
    uint8_t  nib = (lane & 1) ? static_cast<uint8_t>(hib >> 4) : static_cast<uint8_t>(hib & 0x0F);
    return static_cast<uint16_t>(lo | (static_cast<uint16_t>(nib) << 8));
}

inline void write_packed_code15(uint8_t* tile_base, size_t b, size_t lane, uint16_t code) {
    uint8_t* blk = tile_base + b * kBlockBytes15;
    blk[lane] = static_cast<uint8_t>(code & 0xFF);
    uint8_t nib = static_cast<uint8_t>((code >> 8) & 0x0F);
    size_t hi_off = 16 + (lane >> 1);
    if (lane & 1) {
        blk[hi_off] = static_cast<uint8_t>((blk[hi_off] & 0x0F) | (nib << 4));
    } else {
        blk[hi_off] = static_cast<uint8_t>((blk[hi_off] & 0xF0) | nib);
    }
}

inline float scalar_ip_packed15(uint16_t packed, const float* q_b) {
    float cw[8];
    e8lib12::get_e8_shell2_codebook().decode_packed(packed, cw);
    float s = 0.0f;
    for (int d = 0; d < 8; ++d) s += cw[d] * q_b[d];
    return s;
}

}  // anonymous

class IVFE8NoLut12 {
public:
    IVFE8NoLut12(size_t n, size_t dim, size_t nlist, int nthread,
                 const std::string& metric, const std::string& rotator = "fht")
        : n_(n), dim_(dim), nlist_(nlist), nthread_(nthread) {

        (void)metric;

        padded_dim_ = rabitqlib::round_up_to_multiple(dim_, 64);
        if (padded_dim_ % e8lib12::kBlockDim != 0) {
            throw std::runtime_error("padded_dim must be multiple of 8");
        }
        n_blocks_ = padded_dim_ / e8lib12::kBlockDim;

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
        // Per-tile storage allocation (kBlockBytes15 = 24 bytes per block per tile).
        for (size_t l = 0; l < nlist_; ++l) {
            size_t sz = counts[l];
            size_t n_tiles = (sz + kTile - 1) / kTile;
            size_t padded_sz = n_tiles * kTile;
            lists_[l].ids.resize(sz);
            lists_[l].codes_bm.assign(n_tiles * n_blocks_ * kBlockBytes15, 0);
            lists_[l].f_add.assign(padded_sz, 0.0f);
            lists_[l].f_rescale.assign(padded_sz, 0.0f);
        }

        std::vector<size_t> cursor(nlist_, 0);
        for (size_t i = 0; i < nb; ++i) {
            uint32_t cid = idptr[i];
            size_t p = cursor[cid]++;
            lists_[cid].ids[p] = static_cast<PID>(i);
        }

        // Tile-major work list, so each tile has a single owning thread (the
        // 12-bit packed layout shares high-nibble bytes between two lanes).
        struct TileWork { uint32_t cid; uint32_t tile; uint32_t n_lanes; };
        std::vector<TileWork> tile_works;
        tile_works.reserve(nb / kTile + nlist_);
        for (size_t l = 0; l < nlist_; ++l) {
            size_t sz = lists_[l].ids.size();
            size_t n_full = sz / kTile;
            for (size_t t = 0; t < n_full; ++t) {
                tile_works.push_back({static_cast<uint32_t>(l),
                                      static_cast<uint32_t>(t),
                                      static_cast<uint32_t>(kTile)});
            }
            size_t tail = sz - n_full * kTile;
            if (tail > 0) {
                tile_works.push_back({static_cast<uint32_t>(l),
                                      static_cast<uint32_t>(n_full),
                                      static_cast<uint32_t>(tail)});
            }
        }

        omp_set_num_threads(nthread_);
        const auto& cb = e8lib12::get_e8_shell2_codebook();

        #pragma omp parallel
        {
            std::vector<float> rotated(padded_dim_);
            std::vector<float> residual(padded_dim_);
            std::vector<float> reconstructed(padded_dim_);

            #pragma omp for schedule(dynamic, 1)
            for (int64_t ti = 0; ti < (int64_t)tile_works.size(); ++ti) {
                TileWork tw = tile_works[ti];
                uint8_t* tile_base = lists_[tw.cid].codes_bm.data()
                                   + tw.tile * (n_blocks_ * kBlockBytes15);
                // High-nibble bytes are written via OR; clear the whole tile so
                // unused tail lanes encode the zero-vector codeword (0x000 -> D
                // subset 0 with all-positive signs, harmless for tail since the
                // tail is read by the scalar path).
                std::memset(tile_base, 0, n_blocks_ * kBlockBytes15);

                for (uint32_t v = 0; v < tw.n_lanes; ++v) {
                    size_t p = static_cast<size_t>(tw.tile) * kTile + v;
                    PID id = lists_[tw.cid].ids[p];

                    rotator_->rotate(dptr + static_cast<size_t>(id) * dim_, rotated.data());
                    const float* cent_rot =
                        rotated_centroids_.data() + static_cast<size_t>(tw.cid) * padded_dim_;
                    for (size_t kk = 0; kk < padded_dim_; ++kk) {
                        residual[kk] = rotated[kk] - cent_rot[kk];
                    }

                    for (size_t b = 0; b < n_blocks_; ++b) {
                        uint16_t raw = cb.encode_block_raw(
                            residual.data() + b * e8lib12::kBlockDim);
                        uint16_t packed = cb.raw_to_packed(raw);
                        cb.decode_packed(packed,
                                         reconstructed.data() + b * e8lib12::kBlockDim);
                        write_packed_code15(tile_base, b, v, packed);
                    }

                    float l2_sqr = rabitqlib::l2norm_sqr<float>(residual.data(), padded_dim_);
                    float ip_resi_qr = rabitqlib::dot_product<float>(
                        residual.data(), reconstructed.data(), padded_dim_);
                    float ip_cent_qr = rabitqlib::dot_product<float>(
                        cent_rot, reconstructed.data(), padded_dim_);

                    if (ip_resi_qr == 0.0f) {
                        ip_resi_qr = std::numeric_limits<float>::infinity();
                    }
                    float f_add     = l2_sqr + (2.0f * l2_sqr * ip_cent_qr / ip_resi_qr);
                    float f_rescale = -2.0f * l2_sqr / ip_resi_qr;
                    lists_[tw.cid].f_add[p]     = f_add;
                    lists_[tw.cid].f_rescale[p] = f_rescale;
                }
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

        const auto& cb = e8lib12::get_e8_shell2_codebook();

        // Quad-position table for D-path (128 int32, padded). Each entry packs
        // (d0 | d1<<8 | d2<<16 | d3<<24).
        const int32_t* qpt = cb.d_quad_pos_arr();
        const __m512i qpt_a0 = _mm512_load_si512(reinterpret_cast<const __m512i*>(qpt +   0));
        const __m512i qpt_a1 = _mm512_load_si512(reinterpret_cast<const __m512i*>(qpt +  16));
        const __m512i qpt_a2 = _mm512_load_si512(reinterpret_cast<const __m512i*>(qpt +  32));
        const __m512i qpt_a3 = _mm512_load_si512(reinterpret_cast<const __m512i*>(qpt +  48));
        const __m512i qpt_a4 = _mm512_load_si512(reinterpret_cast<const __m512i*>(qpt +  64));
        const __m512i qpt_a5 = _mm512_load_si512(reinterpret_cast<const __m512i*>(qpt +  80));
        const __m512i qpt_a6 = _mm512_load_si512(reinterpret_cast<const __m512i*>(qpt +  96));
        const __m512i qpt_a7 = _mm512_load_si512(reinterpret_cast<const __m512i*>(qpt + 112));

        const __m512i k_subset_mask_v = _mm512_set1_epi32(e8lib12::kSubsetMask);
        const __m512i k_pos_mask_v    = _mm512_set1_epi32(e8lib12::kPosMask);
        const __m512i k_e_payload_v   = _mm512_set1_epi32(e8lib12::kEPayloadMask);
        const __m512i k_dsigns_v      = _mm512_set1_epi32(e8lib12::kDSignsMask);
        const __m512i k_ce_flag_v     = _mm512_set1_epi32(e8lib12::kFlagCEMask);
        const __m512i k_e_flag_v      = _mm512_set1_epi32(e8lib12::kFlagETypeBit);
        const __m512i k_csign_v       = _mm512_set1_epi32(e8lib12::kCSignBit);
        const __m512i k_one_v         = _mm512_set1_epi32(1);
        const __m512i k_byte_mask_v   = _mm512_set1_epi32(0xFF);
        const __m512i k_nib_mask_v    = _mm512_set1_epi32(0x0F);
        const __m512i k_bit5_v        = _mm512_set1_epi32(0x20);
        const __m512i k_bit6_v        = _mm512_set1_epi32(0x40);
        const __m512  k_half_v        = _mm512_set1_ps(0.5f);
        const __m512  k_zero_v        = _mm512_setzero_ps();
        const __m512i hi_nib_shifts   = _mm512_setr_epi32(
            0,4,0,4, 0,4,0,4, 0,4,0,4, 0,4,0,4);

        #pragma omp parallel
        {
            std::vector<float> rotated(padded_dim);
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

                std::priority_queue<std::pair<float, PID>> topk;

                for (size_t pp = 0; pp < np; ++pp) {
                    float g_add = probe_list[pp].first;
                    size_t l = probe_list[pp].second;
                    const auto& L = lists_[l];
                    size_t sz = L.size();
                    if (sz == 0) continue;

                    size_t n_full_tiles = sz / kTile;
                    size_t tail = sz - n_full_tiles * kTile;

                    if (force_scalar_) {
                        // Reads via the same packed layout used by the SIMD path,
                        // for use as a correctness baseline.
                        size_t tile_stride = n_blocks * kBlockBytes15;
                        for (size_t p = 0; p < sz; ++p) {
                            size_t tile = p / kTile;
                            size_t lane = p % kTile;
                            const uint8_t* tb = L.codes_bm.data() + tile * tile_stride;
                            float ip = 0.0f;
                            for (size_t b = 0; b < n_blocks; ++b) {
                                uint16_t code = read_packed_code15(tb, b, lane);
                                ip += scalar_ip_packed15(code,
                                                         rotated.data() + b * e8lib12::kBlockDim);
                            }
                            float est = g_add + L.f_add[p] + L.f_rescale[p] * ip;
                            PID id = L.ids[p];
                            if (topk.size() < k) {
                                topk.emplace(est, id);
                            } else if (est < topk.top().first) {
                                topk.pop();
                                topk.emplace(est, id);
                            }
                        }
                        continue;
                    }

                    const uint8_t* L_codes = L.codes_bm.data();
                    const float* L_fadd = L.f_add.data();
                    const float* L_fres = L.f_rescale.data();
                    const PID* L_ids = L.ids.data();
                    const size_t tile_stride = n_blocks * kBlockBytes15;

                    if (n_full_tiles > 0) {
                        for (size_t b = 0; b < n_blocks; b += 2) {
                            _mm_prefetch(reinterpret_cast<const char*>(L_codes + b * kBlockBytes15),
                                         _MM_HINT_T0);
                        }
                    }

                    for (size_t t = 0; t < n_full_tiles; ++t) {
                        const uint8_t* tile_base = L_codes + t * tile_stride;
                        if (t + 1 < n_full_tiles) {
                            const uint8_t* next_tile = L_codes + (t + 1) * tile_stride;
                            for (size_t b = 0; b < n_blocks; b += 2) {
                                _mm_prefetch(reinterpret_cast<const char*>(next_tile + b * kBlockBytes15),
                                             _MM_HINT_T0);
                            }
                        }

                        __m512 acc = _mm512_setzero_ps();
                        for (size_t b = 0; b < n_blocks; ++b) {
                            const float* q_b = rotated.data() + b * e8lib12::kBlockDim;

                            // Broadcast q_b to all 16 lanes (double the 8 floats).
                            __m256 q_b_256 = _mm256_loadu_ps(q_b);
                            __m512 q_bcast = _mm512_broadcast_f32x8(q_b_256);

                            // ----- Load 16x12-bit codes into c32 -----
                            const uint8_t* blk = tile_base + b * kBlockBytes15;
                            __m128i lo16 = _mm_loadu_si128(
                                reinterpret_cast<const __m128i*>(blk));
                            __m512i c32  = _mm512_cvtepu8_epi32(lo16);

                            uint64_t hi8u = *reinterpret_cast<const uint64_t*>(blk + 16);
                            __m128i hi8v  = _mm_cvtsi64_si128(static_cast<long long>(hi8u));
                            __m128i hi16  = _mm_unpacklo_epi8(hi8v, hi8v);
                            __m512i hi32  = _mm512_cvtepu8_epi32(hi16);
                            hi32 = _mm512_srlv_epi32(hi32, hi_nib_shifts);
                            hi32 = _mm512_and_epi32(hi32, k_nib_mask_v);
                            hi32 = _mm512_slli_epi32(hi32, 8);
                            c32  = _mm512_or_epi32(c32, hi32);

                            // pos field is shared by C and E.
                            __m512i pos_CE = _mm512_and_epi32(
                                _mm512_srli_epi32(c32, e8lib12::kPosShift), k_pos_mask_v);

                            // ----- D path -----
                            __m512i sub_id = _mm512_and_epi32(
                                _mm512_srli_epi32(c32, e8lib12::kSubsetShift), k_subset_mask_v);
                            __m512i b0 = _mm512_permutex2var_epi32(qpt_a0, sub_id, qpt_a1);
                            __m512i b1 = _mm512_permutex2var_epi32(qpt_a2, sub_id, qpt_a3);
                            __m512i b2 = _mm512_permutex2var_epi32(qpt_a4, sub_id, qpt_a5);
                            __m512i b3 = _mm512_permutex2var_epi32(qpt_a6, sub_id, qpt_a7);
                            __mmask16 mb5 = _mm512_test_epi32_mask(sub_id, k_bit5_v);
                            __mmask16 mb6 = _mm512_test_epi32_mask(sub_id, k_bit6_v);
                            __m512i lo_blend = _mm512_mask_blend_epi32(mb5, b0, b1);
                            __m512i hi_blend = _mm512_mask_blend_epi32(mb5, b2, b3);
                            __m512i quad_pos = _mm512_mask_blend_epi32(mb6, lo_blend, hi_blend);

                            __m512i d_signs = _mm512_and_epi32(c32, k_dsigns_v);
                            __m512 acc_D = _mm512_setzero_ps();
                            #pragma GCC unroll 4
                            for (int s = 0; s < 4; ++s) {
                                __m512i pos_s = _mm512_and_epi32(
                                    _mm512_srli_epi32(quad_pos, s * 8), k_byte_mask_v);
                                __m512 q_s = _mm512_permutexvar_ps(pos_s, q_bcast);
                                __mmask16 sgn = _mm512_test_epi32_mask(
                                    d_signs, _mm512_set1_epi32(1 << s));
                                q_s = _mm512_mask_sub_ps(q_s, sgn, k_zero_v, q_s);
                                acc_D = _mm512_add_ps(acc_D, q_s);
                            }
                            __m512 ip_D = acc_D;

                            // ----- C path -----
                            __m512 q_C = _mm512_permutexvar_ps(pos_CE, q_bcast);
                            __mmask16 sign_C = _mm512_test_epi32_mask(c32, k_csign_v);
                            q_C = _mm512_mask_sub_ps(q_C, sign_C, k_zero_v, q_C);
                            __m512 ip_C = _mm512_add_ps(q_C, q_C);

                            // ----- E path -----
                            __m512i mask7 = _mm512_and_epi32(c32, k_e_payload_v);
                            __m512 acc_E = _mm512_setzero_ps();
                            #pragma GCC unroll 7
                            for (int d = 0; d < 7; ++d) {
                                __mmask16 mb = _mm512_test_epi32_mask(
                                    c32, _mm512_set1_epi32(1 << d));
                                __m512 qd = _mm512_set1_ps(q_b[d]);
                                __m512 sel = _mm512_mask_sub_ps(qd, mb, k_zero_v, qd);
                                acc_E = _mm512_add_ps(acc_E, sel);
                            }
                            // s_7 = 1 ^ parity(mask7)  (full 8-bit mask has odd parity).
                            __m512i x = _mm512_xor_epi32(mask7, _mm512_srli_epi32(mask7, 4));
                            x = _mm512_xor_epi32(x, _mm512_srli_epi32(x, 2));
                            x = _mm512_xor_epi32(x, _mm512_srli_epi32(x, 1));
                            __mmask16 m_parity = _mm512_test_epi32_mask(x, k_one_v);
                            __mmask16 s7_neg = _knot_mask16(m_parity);
                            __m512 q7 = _mm512_set1_ps(q_b[7]);
                            __m512 sel7 = _mm512_mask_sub_ps(q7, s7_neg, k_zero_v, q7);
                            acc_E = _mm512_add_ps(acc_E, sel7);

                            // ±q[pos] for E: pos coord contributes an extra
                            // s_pos*q[pos] on top of the half-sum (see derivation).
                            __m512i s7_bit = _mm512_andnot_epi32(x, k_one_v);
                            __m512i mask8  = _mm512_or_epi32(
                                mask7, _mm512_slli_epi32(s7_bit, 7));
                            __m512i shifted = _mm512_srlv_epi32(mask8, pos_CE);
                            __mmask16 s_pos_neg = _mm512_test_epi32_mask(shifted, k_one_v);
                            __m512 q_pos = _mm512_permutexvar_ps(pos_CE, q_bcast);
                            q_pos = _mm512_mask_sub_ps(q_pos, s_pos_neg, k_zero_v, q_pos);

                            __m512 ip_E = _mm512_add_ps(_mm512_mul_ps(acc_E, k_half_v), q_pos);

                            // ----- Blend D / C / E -----
                            __mmask16 mCE = _mm512_test_epi32_mask(c32, k_ce_flag_v);
                            __mmask16 mE  = _mm512_test_epi32_mask(c32, k_e_flag_v);
                            __m512 ip_CE  = _mm512_mask_blend_ps(mE, ip_C, ip_E);
                            __m512 ip     = _mm512_mask_blend_ps(mCE, ip_D, ip_CE);
                            acc = _mm512_add_ps(acc, ip);
                        }
                        __m512 fadd = _mm512_loadu_ps(L_fadd + t * kTile);
                        __m512 fres = _mm512_loadu_ps(L_fres + t * kTile);
                        __m512 est  = _mm512_fmadd_ps(fres, acc,
                                        _mm512_add_ps(_mm512_set1_ps(g_add), fadd));

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

                    // Scalar tail (reads same packed layout via decode_packed).
                    if (tail > 0) {
                        const uint8_t* tile_base =
                            L.codes_bm.data() + n_full_tiles * tile_stride;
                        for (size_t v = 0; v < tail; ++v) {
                            float ip = 0.0f;
                            for (size_t b = 0; b < n_blocks; ++b) {
                                uint16_t code = read_packed_code15(tile_base, b, v);
                                ip += scalar_ip_packed15(code,
                                                         rotated.data() + b * e8lib12::kBlockDim);
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
public:
    bool force_scalar_ = false;
};

PYBIND11_MODULE(e8nolut_cpp, m) {
    m.doc() = "IVF + E8-lattice 1-bit / 1.5-bit quantization (LUT-free direct-compute)";
    py::class_<IVFE8NoLut>(m, "IVFE8NoLut")
        .def(py::init<size_t, size_t, size_t, int,
                      const std::string&, const std::string&>(),
             py::arg("n"),
             py::arg("dim"),
             py::arg("nlist"),
             py::arg("nthread") = 1,
             py::arg("metric") = "l2",
             py::arg("rotator") = "fht")
        .def("construct", &IVFE8NoLut::construct,
             py::arg("data"), py::arg("centroids"), py::arg("cluster_ids"))
        .def("search_batch", &IVFE8NoLut::search_batch,
             py::arg("queries"), py::arg("k"), py::arg("nprobe"))
        .def_readwrite("force_scalar", &IVFE8NoLut::force_scalar_);

    py::class_<IVFE8NoLut12>(m, "IVFE8NoLut12")
        .def(py::init<size_t, size_t, size_t, int,
                      const std::string&, const std::string&>(),
             py::arg("n"),
             py::arg("dim"),
             py::arg("nlist"),
             py::arg("nthread") = 1,
             py::arg("metric") = "l2",
             py::arg("rotator") = "fht")
        .def("construct", &IVFE8NoLut12::construct,
             py::arg("data"), py::arg("centroids"), py::arg("cluster_ids"))
        .def("search_batch", &IVFE8NoLut12::search_batch,
             py::arg("queries"), py::arg("k"), py::arg("nprobe"))
        .def_readwrite("force_scalar", &IVFE8NoLut12::force_scalar_);
}
