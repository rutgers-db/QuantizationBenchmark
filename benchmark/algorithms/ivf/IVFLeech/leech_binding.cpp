// pybind11 binding for IVF + Leech-lattice (Λ24) quantization.
//
// Codebook: 196,560 minimum-norm Leech vectors (squared norm 32 in the
// Conway-Sloane scaling). Block dim is 24, packed code is 18 bits stored as
// 3 bytes per block (1 bit per dimension — same budget as IVFE8 1-bit).
//
// Build path: brute-force argmax<residual_block, c_k> over all 196,560
// codewords. The codebook lives in a (24, 12285, 16) coord-broadcast int8
// layout; the encoder kernel processes 16 codewords per inner iteration with
// 24 broadcast-FMAs (the same kernel shape used at search).
//
// Search path: per (query, list, block) the kernel
//   (1) decodes 16 packed 3-byte indices into a uint32 vector via three
//       _mm_loadu_si128 + cvtepu8 + OR-shift,
//   (2) gathers 16 codewords from the flat int8 codebook and transposes them
//       into a (24, 16) lane-major scratch (24 × 16 stores),
//   (3) computes 16 inner products in parallel with 24 broadcast-FMAs,
//   (4) folds the per-tile vector into f_add, f_rescale and updates top-k.
// No per-query LUT is built; the codebook stays static and is reused across
// queries.
//
// RaBitQ factors are computed the same way as IVFE8NoLut (residual-vs-
// reconstructed codeword); the formulation is scale-independent so it works
// directly with squared-norm-32 Leech codewords.

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

#include "leech_codebook.h"
#include "leech_decoder.h"

namespace py = pybind11;

using PID = rabitqlib::PID;
using leechlib::kBlockDim;
using leechlib::kCodebookSize;
using leechlib::kNumSimdGroups;
using leechlib::kSimdGroup;
using leechlib::kCodeBytes;
using leechlib::get_leech_codebook;
using leechlib::get_leech_decoder;

namespace {

constexpr size_t kTile = 16;            // search tile width

/* AVX-512 brute-force encoder over the (24, kCodebookSize) coord-broadcast
 * int8 layout. Returns argmax<block, c_k>. */
inline uint32_t encode_block_simd(const float* block) {
    const int8_t* cb = get_leech_codebook().simd_data();

    __m512  best_ip = _mm512_set1_ps(-1e30f);
    __m512i best_k  = _mm512_setzero_si512();
    __m512i k_vec   = _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
    const __m512i k_inc = _mm512_set1_epi32(static_cast<int>(kSimdGroup));

    __m512 q[kBlockDim];
    for (size_t d = 0; d < kBlockDim; ++d) q[d] = _mm512_set1_ps(block[d]);

    for (size_t g = 0; g < kNumSimdGroups; ++g) {
        __m512 ip = _mm512_setzero_ps();
        #pragma GCC unroll 24
        for (size_t d = 0; d < kBlockDim; ++d) {
            __m128i bytes = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(
                    cb + d * kCodebookSize + g * kSimdGroup));
            __m512i ints   = _mm512_cvtepi8_epi32(bytes);
            __m512  floats = _mm512_cvtepi32_ps(ints);
            ip = _mm512_fmadd_ps(floats, q[d], ip);
        }
        __mmask16 win = _mm512_cmp_ps_mask(ip, best_ip, _CMP_GT_OQ);
        best_ip = _mm512_mask_blend_ps(win, best_ip, ip);
        best_k  = _mm512_mask_blend_epi32(win, best_k, k_vec);
        k_vec   = _mm512_add_epi32(k_vec, k_inc);
    }

    alignas(64) float   ips[kSimdGroup];
    alignas(64) int32_t ks[kSimdGroup];
    _mm512_store_ps(ips, best_ip);
    _mm512_store_si512(reinterpret_cast<__m512i*>(ks), best_k);
    float    best = ips[0];
    uint32_t k    = static_cast<uint32_t>(ks[0]);
    for (size_t i = 1; i < kSimdGroup; ++i) {
        if (ips[i] > best) { best = ips[i]; k = static_cast<uint32_t>(ks[i]); }
    }
    return k;
}

/* Scalar inner product between a 24-float query block and a packed code idx.
 * Used on the < 16-code tail. */
inline float scalar_ip(uint32_t idx, const float* q_b) {
    const int8_t* c = get_leech_codebook().codeword_int8(idx);
    float s = 0.0f;
    for (size_t d = 0; d < kBlockDim; ++d) s += q_b[d] * static_cast<float>(c[d]);
    return s;
}

/* Encode a residual block to a Shell-1 codeword index, hybrid path:
 *   1. Rescale block so ||scaled||₂ = √32 (matching Shell-1 minimum-vector
 *      norm). This puts the decoder's input on the Shell-1 sphere so it
 *      lands in Shell 1 with high probability. Quantization is direction-
 *      only — RaBitQ's f_rescale absorbs the magnitude.
 *   2. Run Conway-Sloane decoder (~1.4 µs).
 *   3. If the output is one of the 196,560 Shell-1 codewords, look up its
 *      index via the codebook hash table (~50 ns).
 *   4. Otherwise the decoder landed in Shell 2+; fall back to brute-force
 *      SIMD over Shell 1 (~38 µs).
 *
 * The data side ALWAYS stores a Shell-1 index, so the rest of the pipeline
 * (search, packed-code layout) is unchanged. The decoder turns a guaranteed
 * 38 µs/call into ~1.4 µs/call for the typical case where the residual is
 * already near a Shell-1 lattice point. */
inline uint32_t encode_block_hybrid(const float* block,
                                     uint64_t* hits, uint64_t* fallbacks) {
    // Rescale to Shell-1 sphere ||r|| = √32. If block is exactly zero, fall
    // back (no direction info; brute force will pick an arbitrary codeword).
    float n2 = 0.0f;
    for (size_t d = 0; d < kBlockDim; ++d) n2 += block[d] * block[d];
    if (n2 == 0.0f) {
        if (fallbacks) ++(*fallbacks);
        return encode_block_simd(block);
    }
    float scale = std::sqrt(32.0f / n2);
    float scaled[kBlockDim];
    for (size_t d = 0; d < kBlockDim; ++d) scaled[d] = block[d] * scale;

    int8_t v[kBlockDim];
    get_leech_decoder().decode(scaled, v);

    bool found = false;
    uint32_t idx = get_leech_codebook().index_of_shell1(v, &found);
    if (found) {
        if (hits) ++(*hits);
        return idx;
    }
    if (fallbacks) ++(*fallbacks);
    return encode_block_simd(block);
}

/* Per-list storage. Codes are packed into 3 bytes per (vector, block) and laid
 * out in tile-major form: for a tile of 16 codes, block b's 48 bytes occupy
 *   codes_bm[t * tile_stride + b * (kCodeBytes * kTile) + l * kTile + v]
 * where l ∈ {0,1,2} indexes the byte position and v ∈ {0..15} is the lane.
 * Decoding lane v's index reads three bytes at offsets l * 16 + v. */
struct ListStorage {
    std::vector<PID> ids;
    std::vector<uint8_t> codes_bm;
    std::vector<float> f_add;
    std::vector<float> f_rescale;
    size_t size() const { return ids.size(); }
};

}  // anonymous

class IVFLeech {
public:
    IVFLeech(size_t n, size_t dim, size_t nlist, int nthread,
             const std::string& metric, const std::string& rotator = "fht")
        : n_(n), dim_(dim), nlist_(nlist), nthread_(nthread) {

        (void)metric;

        // padded_dim must satisfy:
        //   * multiple of 64 (FhtKacRotator),
        //   * multiple of kBlockDim = 24 (Leech blocks).
        // LCM(64, 24) = 192.
        padded_dim_ = ((dim_ + 191) / 192) * 192;
        if (padded_dim_ % kBlockDim != 0) {
            throw std::runtime_error("padded_dim must be multiple of 24");
        }
        if (padded_dim_ % 64 != 0) {
            throw std::runtime_error("padded_dim must be multiple of 64");
        }
        n_blocks_ = padded_dim_ / kBlockDim;
        tile_stride_ = n_blocks_ * kCodeBytes * kTile;

        rabitqlib::RotatorType rtype =
            (rotator == "matrix") ? rabitqlib::RotatorType::MatrixRotator
                                  : rabitqlib::RotatorType::FhtKacRotator;
        std::srand(dim_ + padded_dim_);
        rotator_.reset(rabitqlib::choose_rotator<float>(dim_, rtype, padded_dim_));

        rotated_centroids_.assign(nlist_ * padded_dim_, 0.0f);
        centroids_.assign(nlist_ * dim_, 0.0f);
        lists_.resize(nlist_);

        // Touch the codebook so the static init runs single-threaded before
        // the OpenMP region.
        (void)get_leech_codebook();
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
            lists_[l].codes_bm.assign((padded_sz / kTile) * tile_stride_, 0);
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
        const auto& cb = get_leech_codebook();
        // Touch decoder once now (single-threaded) so its 4096-entry syndrome
        // table is initialized before the parallel region.
        (void)get_leech_decoder();

        uint64_t total_hits = 0, total_fallbacks = 0;

        #pragma omp parallel reduction(+:total_hits,total_fallbacks)
        {
            std::vector<float> rotated(padded_dim_);
            std::vector<float> residual(padded_dim_);
            std::vector<float> reconstructed(padded_dim_);
            std::vector<uint32_t> code_scratch(n_blocks_);
            uint64_t hits = 0, fallbacks = 0;

            #pragma omp for schedule(static)
            for (int64_t i = 0; i < (int64_t)nb; ++i) {
                uint32_t cid = idptr[i];
                size_t p = pos[i];

                rotator_->rotate(dptr + i * dim_, rotated.data());
                const float* cent_rot = rotated_centroids_.data() + cid * padded_dim_;
                for (size_t k = 0; k < padded_dim_; ++k) {
                    residual[k] = rotated[k] - cent_rot[k];
                }

                for (size_t b = 0; b < n_blocks_; ++b) {
                    uint32_t idx = encode_block_simd(residual.data() + b * kBlockDim);
                    code_scratch[b] = idx;
                    const int8_t* cw = cb.codeword_int8(idx);
                    float* rec = reconstructed.data() + b * kBlockDim;
                    for (size_t dd = 0; dd < kBlockDim; ++dd) {
                        rec[dd] = static_cast<float>(cw[dd]);
                    }
                }
                (void)hits; (void)fallbacks;  // decoder path disabled

                size_t tile = p / kTile;
                size_t lane = p % kTile;
                uint8_t* tile_base =
                    lists_[cid].codes_bm.data() + tile * tile_stride_;
                for (size_t b = 0; b < n_blocks_; ++b) {
                    uint32_t idx = code_scratch[b];
                    uint8_t* blk = tile_base + b * kCodeBytes * kTile;
                    blk[0 * kTile + lane] = static_cast<uint8_t>(idx & 0xFF);
                    blk[1 * kTile + lane] = static_cast<uint8_t>((idx >> 8) & 0xFF);
                    blk[2 * kTile + lane] = static_cast<uint8_t>((idx >> 16) & 0xFF);
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
            total_hits      += hits;
            total_fallbacks += fallbacks;
        }
        last_decoder_hits_      = total_hits;
        last_decoder_fallbacks_ = total_fallbacks;
    }

    /* Encoder hit/fallback stats from the most recent construct(); useful to
     * surface how often the decoder lands in Shell 1 (fast path) vs needs
     * brute-force projection (slow path). */
    uint64_t decoder_hits()      const { return last_decoder_hits_; }
    uint64_t decoder_fallbacks() const { return last_decoder_fallbacks_; }

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
        const size_t tile_stride = tile_stride_;

        const auto& cb = get_leech_codebook();
        const int8_t* cb_int8 = cb.data();

        #pragma omp parallel
        {
            std::vector<float> rotated(padded_dim);
            std::vector<std::pair<float, size_t>> probe_list(nlist_);
            alignas(64) uint32_t indices_arr[kTile];
            alignas(64) int8_t  tile_codes[kBlockDim * kTile];
            alignas(64) float   ip_tile[kTile];

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
                        for (size_t p = 0; p < sz; ++p) {
                            size_t tile = p / kTile;
                            size_t lane = p % kTile;
                            const uint8_t* tb =
                                L.codes_bm.data() + tile * tile_stride;
                            float ip = 0.0f;
                            for (size_t b = 0; b < n_blocks; ++b) {
                                const uint8_t* blk = tb + b * kCodeBytes * kTile;
                                uint32_t idx = blk[0 * kTile + lane]
                                    | (uint32_t(blk[1 * kTile + lane]) << 8)
                                    | (uint32_t(blk[2 * kTile + lane]) << 16);
                                ip += scalar_ip(idx, rotated.data() + b * kBlockDim);
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

                    for (size_t t = 0; t < n_full_tiles; ++t) {
                        const uint8_t* tile_base = L_codes + t * tile_stride;
                        if (t + 1 < n_full_tiles) {
                            const uint8_t* next_tile = L_codes + (t + 1) * tile_stride;
                            for (size_t b = 0; b < n_blocks; b += 4) {
                                _mm_prefetch(reinterpret_cast<const char*>(
                                    next_tile + b * kCodeBytes * kTile), _MM_HINT_T0);
                            }
                        }

                        __m512 acc = _mm512_setzero_ps();
                        for (size_t b = 0; b < n_blocks; ++b) {
                            const uint8_t* blk = tile_base + b * kCodeBytes * kTile;
                            const float* q_b = rotated.data() + b * kBlockDim;

                            // Decode 16 packed 3-byte indices in parallel.
                            __m128i b0 = _mm_loadu_si128(
                                reinterpret_cast<const __m128i*>(blk + 0 * kTile));
                            __m128i b1 = _mm_loadu_si128(
                                reinterpret_cast<const __m128i*>(blk + 1 * kTile));
                            __m128i b2 = _mm_loadu_si128(
                                reinterpret_cast<const __m128i*>(blk + 2 * kTile));
                            __m512i i0 = _mm512_cvtepu8_epi32(b0);
                            __m512i i1 = _mm512_cvtepu8_epi32(b1);
                            __m512i i2 = _mm512_cvtepu8_epi32(b2);
                            __m512i indices = _mm512_or_epi32(
                                _mm512_or_epi32(i0, _mm512_slli_epi32(i1, 8)),
                                _mm512_slli_epi32(i2, 16));
                            _mm512_store_si512(
                                reinterpret_cast<__m512i*>(indices_arr), indices);

                            // Gather + transpose 16 codewords into coord-broadcast
                            // layout (24, 16). Prefetching while filling helps
                            // the next tile's gather.
                            for (size_t v = 0; v < kTile; ++v) {
                                const int8_t* cw =
                                    cb_int8 + indices_arr[v] * kBlockDim;
                                #pragma GCC unroll 24
                                for (size_t d = 0; d < kBlockDim; ++d) {
                                    tile_codes[d * kTile + v] = cw[d];
                                }
                            }

                            // 16-way parallel IPs via 24 broadcast-FMAs.
                            __m512 ip = _mm512_setzero_ps();
                            #pragma GCC unroll 24
                            for (size_t d = 0; d < kBlockDim; ++d) {
                                __m128i bytes = _mm_load_si128(
                                    reinterpret_cast<const __m128i*>(
                                        tile_codes + d * kTile));
                                __m512i ints   = _mm512_cvtepi8_epi32(bytes);
                                __m512  floats = _mm512_cvtepi32_ps(ints);
                                __m512  qd = _mm512_set1_ps(q_b[d]);
                                ip = _mm512_fmadd_ps(floats, qd, ip);
                            }
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
                            _mm512_storeu_ps(ip_tile, est);
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
                            _mm512_storeu_ps(ip_tile, est);
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

                    if (tail > 0) {
                        const uint8_t* tile_base =
                            L_codes + n_full_tiles * tile_stride;
                        for (size_t v = 0; v < tail; ++v) {
                            float ip = 0.0f;
                            for (size_t b = 0; b < n_blocks; ++b) {
                                const uint8_t* blk = tile_base + b * kCodeBytes * kTile;
                                uint32_t idx = blk[0 * kTile + v]
                                    | (uint32_t(blk[1 * kTile + v]) << 8)
                                    | (uint32_t(blk[2 * kTile + v]) << 16);
                                ip += scalar_ip(idx, rotated.data() + b * kBlockDim);
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
    size_t tile_stride_;
    int nthread_;

    std::unique_ptr<rabitqlib::Rotator<float>> rotator_;
    std::vector<float> centroids_;
    std::vector<float> rotated_centroids_;

    std::vector<ListStorage> lists_;
    uint64_t last_decoder_hits_ = 0;
    uint64_t last_decoder_fallbacks_ = 0;
public:
    bool force_scalar_ = false;
};

PYBIND11_MODULE(leech_cpp, m) {
    m.doc() = "IVF + Leech-lattice (Λ24) 1-bit quantization";
    py::class_<IVFLeech>(m, "IVFLeech")
        .def(py::init<size_t, size_t, size_t, int,
                      const std::string&, const std::string&>(),
             py::arg("n"),
             py::arg("dim"),
             py::arg("nlist"),
             py::arg("nthread") = 1,
             py::arg("metric") = "l2",
             py::arg("rotator") = "fht")
        .def("construct", &IVFLeech::construct,
             py::arg("data"), py::arg("centroids"), py::arg("cluster_ids"))
        .def("search_batch", &IVFLeech::search_batch,
             py::arg("queries"), py::arg("k"), py::arg("nprobe"))
        .def("decoder_hits",      &IVFLeech::decoder_hits)
        .def("decoder_fallbacks", &IVFLeech::decoder_fallbacks)
        .def_readwrite("force_scalar", &IVFLeech::force_scalar_);
}
