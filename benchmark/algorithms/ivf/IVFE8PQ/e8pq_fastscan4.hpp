// IVFE8PQFastScan4 — IVFE8PQ with classic PQ4-fastscan kernel (nbit=4).
//
// Same training pipeline as IVFE8PQ (KMeans + FHT-Kac rotation + learned PQ
// codebook on the normalized residual + RaBitQ scoring factors), trained with
// nbit=4 → 16 codewords per subvec.
//
// Storage / search layout:
//   * 32-wide block-major tiles. Subvecs are paired (kb=0..M/2-1); within a
//     pair, byte[lane] = (code_{2kb+1} << 4) | code_{2kb}. Per-tile bytes:
//     32 * (M/2) = 16 * M  (half of nbit=8). M must be even.
//   * Per-query LUT: 16 uint8 entries per subvec (256 bytes total per pair),
//     fits easily in registers — each pair uses two 16-byte LUTs broadcast
//     into __m256i for _mm256_shuffle_epi8 (per-128b-lane pshufb).
//   * Scan inner loop: load 32 packed code bytes, split low/high nibbles,
//     pshufb on the two pair-LUTs, accumulate into a uint16 acc of 32 lanes.
//     One float dequant per tile, then RaBitQ scoring + top-k.
//
// LUT quantization (Faiss style, see quantize_lut.cpp:157):
//   For each subvec b: min_b = min_k LUT[b][k], span_b = max_k - min_b.
//   q_offset = sum_b min_b   (constant adder per query)
//   a = min(255 / max_b span_b,  65535 / sum_b span_b)   ← global scale
//   u8[b][k] = clip(((LUT[b][k] - min_b) * a) + 0.5, 0, 255)
//   Decoded ip = (1/a) * acc_u16 + q_offset.
//
// The two bounds in `a` keep both per-subvec u8 and the M-term sum within
// their respective ranges, so the uint16 accumulator never overflows for
// any M. Small M uses full uint8 precision; large M auto-tightens the scale
// (e.g. M=480 ⇒ effective u8 max ≈ 65535/480 ≈ 136).

#pragma once

#include <omp.h>
#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/ProductQuantizer.h>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"

namespace e8pqlib {

using PIDFS4 = rabitqlib::PID;

constexpr size_t kTileFS4       = 32;             // 32-wide tile
constexpr size_t kPQK4          = 16;             // 16 codewords per subvec (nbit=4)
constexpr size_t kPQNBit4       = 4;
constexpr size_t kPQTrainMaxFS4 = 256 * 1024;

struct ListStorageFS4 {
    std::vector<PIDFS4>  ids;
    // codes_bm: [n_tiles][n_pairs][32]; byte = (code_{2kb+1} << 4) | code_{2kb}
    std::vector<uint8_t> codes_bm;
    std::vector<float>   f_add;
    std::vector<float>   f_rescale;
    size_t size() const { return ids.size(); }
};

// Inline fixed-capacity max-heap on caller-supplied storage. Replaces
// std::priority_queue<std::pair<float,PID>> in the scan hot path:
//   * threshold cached in a local field — no `top()` call per check
//   * push() handles fill / replace in one branch
//   * no per-emplace allocations or pair construction
//   * dist/id stored as parallel arrays (better cache behaviour than pairs)
struct TopKMaxHeapFS4 {
    float*  dist;
    PIDFS4* id;
    size_t  cap;
    size_t  sz;
    float   threshold;   // dist[0] when sz == cap, else +inf

    inline void clear() {
        sz = 0;
        threshold = std::numeric_limits<float>::infinity();
    }

    inline void push(float d, PIDFS4 i) {
        if (sz < cap) {
            dist[sz] = d;
            id[sz]   = i;
            ++sz;
            sift_up_(sz - 1);
            if (sz == cap) threshold = dist[0];
        } else if (d < threshold) {
            dist[0] = d;
            id[0]   = i;
            sift_down_(0);
            threshold = dist[0];
        }
    }

private:
    inline void sift_up_(size_t i) {
        while (i > 0) {
            size_t p = (i - 1) >> 1;
            if (dist[p] < dist[i]) {
                std::swap(dist[p], dist[i]);
                std::swap(id[p],   id[i]);
                i = p;
            } else {
                break;
            }
        }
    }

    inline void sift_down_(size_t i) {
        const size_t n = sz;
        for (;;) {
            size_t l = 2 * i + 1;
            size_t r = l + 1;
            size_t largest = i;
            if (l < n && dist[l] > dist[largest]) largest = l;
            if (r < n && dist[r] > dist[largest]) largest = r;
            if (largest == i) break;
            std::swap(dist[i], dist[largest]);
            std::swap(id[i],   id[largest]);
            i = largest;
        }
    }
};

class IVFE8PQFastScan4 {
public:
    IVFE8PQFastScan4(size_t n, size_t dim, size_t nlist, size_t nsubvec,
                     size_t nbit, int nthread, const std::string& metric,
                     const std::string& rotator)
        : n_(n), dim_(dim), nlist_(nlist), nsubvec_(nsubvec),
          nbit_(nbit), nthread_(nthread) {
        (void)metric;

        if (nbit_ != kPQNBit4) {
            throw std::runtime_error(
                "IVFE8PQFastScan4 requires nbit=4 (use IVFE8PQFastScan for nbit=8).");
        }
        if (nsubvec_ % 2 != 0) {
            throw std::runtime_error(
                "IVFE8PQFastScan4 requires even nsubvec (subvecs are nibble-paired).");
        }
        padded_dim_ = rabitqlib::round_up_to_multiple(dim_, 64);
        if (nsubvec_ == 0 || padded_dim_ % nsubvec_ != 0) {
            throw std::runtime_error(
                "nsubvec must divide padded_dim (rounded up to multiple of 64).");
        }
        dsub_ = padded_dim_ / nsubvec_;
        n_pairs_ = nsubvec_ / 2;
        code_size_ = nsubvec_ / 2;   // bytes per vector (faiss output)

        rabitqlib::RotatorType rtype =
            (rotator == "matrix") ? rabitqlib::RotatorType::MatrixRotator
                                  : rabitqlib::RotatorType::FhtKacRotator;
        std::srand(static_cast<unsigned>(dim_ + padded_dim_ + 11u));
        rotator_.reset(rabitqlib::choose_rotator<float>(dim_, rtype, padded_dim_));

        centroids_.assign(nlist_ * dim_, 0.0f);
        rotated_centroids_.assign(nlist_ * padded_dim_, 0.0f);
        pq_centroids_.assign(nsubvec_ * kPQK4 * dsub_, 0.0f);
        pq_centroids_T_.assign(nsubvec_ * dsub_ * kPQK4, 0.0f);
        lists_.resize(nlist_);
    }

    void fit(const float* data, size_t nb) {
        if (nb == 0) return;
        omp_set_num_threads(nthread_);

        // ---------- 1. Coarse KMeans ----------
        train_coarse_(data, nb);

        // ---------- 2. Assign every vector to nearest centroid ----------
        std::vector<int64_t> assign(nb);
        {
            faiss::IndexFlatL2 coarse(static_cast<faiss::idx_t>(dim_));
            coarse.add(static_cast<faiss::idx_t>(nlist_), centroids_.data());
            std::vector<float> d_tmp(nb);
            coarse.search(static_cast<faiss::idx_t>(nb), data, 1,
                          d_tmp.data(), assign.data());
        }

        // ---------- 3. Rotate centroids once ----------
        for (size_t l = 0; l < nlist_; ++l) {
            rotator_->rotate(centroids_.data() + l * dim_,
                             rotated_centroids_.data() + l * padded_dim_);
        }

        // ---------- 4. PQ training on normalized residuals (subsample) ----------
        train_pq_(data, nb, assign);

        // Build transposed PQ centroids: [b][j][k] for vectorized LUT build.
        // pq_centroids_ layout (faiss):  [b * K*dsub + k * dsub + j]
        // pq_centroids_T_ layout (ours): [b * dsub*K + j * K + k], K=16
        for (size_t b = 0; b < nsubvec_; ++b) {
            const float* src = pq_centroids_.data() + b * kPQK4 * dsub_;
            float*       dst = pq_centroids_T_.data() + b * dsub_ * kPQK4;
            for (size_t k = 0; k < kPQK4; ++k) {
                for (size_t j = 0; j < dsub_; ++j) {
                    dst[j * kPQK4 + k] = src[k * dsub_ + j];
                }
            }
        }

        // ---------- 5. Per-list allocation ----------
        std::vector<size_t> counts(nlist_, 0);
        for (size_t i = 0; i < nb; ++i) {
            int64_t cid = assign[i];
            if (cid < 0 || static_cast<size_t>(cid) >= nlist_) {
                throw std::runtime_error("invalid cluster id from coarse search");
            }
            ++counts[cid];
        }
        // Per-list allocation. Pad f_add to +inf (and f_rescale to 0) so the
        // SIMD scan can run on full padded tiles without tail handling —
        // padded lanes always score est = +inf and never win top-k. ids are
        // also padded so update_topk_tile_'s ids32[v] reads stay in-bounds
        // even on the partial last tile (the padded entries are unused but
        // must be valid memory).
        const float kPadInf = std::numeric_limits<float>::infinity();
        for (size_t l = 0; l < nlist_; ++l) {
            size_t sz = counts[l];
            size_t padded_sz = ((sz + kTileFS4 - 1) / kTileFS4) * kTileFS4;
            lists_[l].ids.assign(padded_sz, PIDFS4{0});
            lists_[l].codes_bm.assign(padded_sz * n_pairs_, 0);
            lists_[l].f_add.assign(padded_sz, kPadInf);
            lists_[l].f_rescale.assign(padded_sz, 0.0f);
        }

        std::vector<size_t> cursor(nlist_, 0);
        std::vector<size_t> pos(nb);
        for (size_t i = 0; i < nb; ++i) {
            size_t cid = static_cast<size_t>(assign[i]);
            pos[i] = cursor[cid]++;
            lists_[cid].ids[pos[i]] = static_cast<PIDFS4>(i);
        }

        // ---------- 6. Encode (phase 1: per-vector codes + factors) ----------
        // We split encoding from layout-packing because the Faiss-style perm0
        // layout has two vectors (lanes v and v+16) sharing every output
        // byte (low/high nibbles), so per-vector parallel scatter would race.
        //
        // Phase 1 (per-vector parallel): compute PQ codes into a flat
        //   ``temp_codes`` buffer (M/2 bytes per vector) and write the RaBitQ
        //   factors directly to per-list storage (race-free, each pos unique).
        // Phase 2 (per-list parallel): for each (list, tile), single-threaded
        //   re-pack from temp_codes into the Faiss perm0-interleaved layout.
        std::vector<uint8_t> temp_codes(static_cast<size_t>(nb) * code_size_);

        faiss::ProductQuantizer pq(static_cast<size_t>(padded_dim_),
                                   nsubvec_, nbit_);
        pq.centroids = pq_centroids_;

        #pragma omp parallel
        {
            std::vector<float>   rotated(padded_dim_);
            std::vector<float>   residual(padded_dim_);
            std::vector<float>   normalized(padded_dim_);
            std::vector<float>   ohat(padded_dim_);

            #pragma omp for schedule(static)
            for (int64_t i = 0; i < (int64_t)nb; ++i) {
                size_t cid = static_cast<size_t>(assign[i]);
                size_t p = pos[i];
                const float* cent_rot =
                    rotated_centroids_.data() + cid * padded_dim_;

                rotator_->rotate(data + i * dim_, rotated.data());
                for (size_t k = 0; k < padded_dim_; ++k) {
                    residual[k] = rotated[k] - cent_rot[k];
                }

                float l2_sqr =
                    rabitqlib::l2norm_sqr<float>(residual.data(), padded_dim_);
                float norm = std::sqrt(std::max(l2_sqr, 0.0f));
                float inv_norm = (norm > 0.0f) ? (1.0f / norm) : 0.0f;
                for (size_t k = 0; k < padded_dim_; ++k) {
                    normalized[k] = residual[k] * inv_norm;
                }

                uint8_t* code_dst = temp_codes.data() + i * code_size_;
                pq.compute_code(normalized.data(), code_dst);
                pq.decode(code_dst, ohat.data());

                float ip_resi = rabitqlib::dot_product<float>(
                    residual.data(), ohat.data(), padded_dim_);
                float ip_cent = rabitqlib::dot_product<float>(
                    cent_rot, ohat.data(), padded_dim_);
                if (ip_resi == 0.0f) {
                    ip_resi = std::numeric_limits<float>::infinity();
                }
                lists_[cid].f_add[p]     = l2_sqr + 2.0f * l2_sqr * ip_cent / ip_resi;
                lists_[cid].f_rescale[p] = -2.0f * l2_sqr / ip_resi;
            }
        }

        // ---------- 7. Encode (phase 2: pack into Faiss perm0 layout) -------
        // Per-pair output (32 bytes) layout, mirroring faiss pq4_pack_codes:
        //   bytes [ 0..15] = subvec sq codes
        //   bytes [16..31] = subvec sq+1 codes
        // Within each 16-byte half, byte j packs nibbles for vectors
        // perm0[j] (low) and perm0[j]+16 (high), with
        //   perm0 = [0,8,1,9,2,10,3,11,4,12,5,13,6,14,7,15]
        //
        // This co-locates each pshufb's needed LUT in the same 128-bit lane
        // as its codes, so the LUT loads are plain 256-bit memory loads
        // (port 2/3) instead of broadcasts (port 5) — freeing port 5 for
        // pshufb itself. Inverse perm for the *write* side:
        //   perm0_inv[v] = j such that perm0[j] = v
        static constexpr uint8_t perm0_inv[16] = {
            0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15
        };
        const size_t n_pairs_local = n_pairs_;

        #pragma omp parallel for schedule(dynamic, 1)
        for (int64_t l = 0; l < (int64_t)nlist_; ++l) {
            auto& L = lists_[l];
            // L.size() returns padded_sz now (ids vector is padded). Use the
            // pre-computed real cluster count for the per-tile lane bound.
            size_t real_sz = counts[l];
            if (real_sz == 0) continue;
            size_t n_tiles_total = (real_sz + kTileFS4 - 1) / kTileFS4;

            for (size_t t = 0; t < n_tiles_total; ++t) {
                uint8_t* tile_out =
                    L.codes_bm.data() + t * kTileFS4 * n_pairs_local;
                size_t lane_count =
                    std::min<size_t>(kTileFS4, real_sz - t * kTileFS4);

                for (size_t v = 0; v < lane_count; ++v) {
                    PIDFS4 vec_id = L.ids[t * kTileFS4 + v];
                    const uint8_t* vc = temp_codes.data()
                                      + size_t(vec_id) * code_size_;

                    // v in [0, 15]: low nibble (shift=0) of byte at perm0_inv[v]
                    // v in [16,31]: high nibble (shift=4) of byte at perm0_inv[v-16]
                    size_t v_half  = v & 15u;
                    unsigned shift = (v < 16) ? 0u : 4u;
                    size_t byte_off = perm0_inv[v_half];

                    for (size_t kb = 0; kb < n_pairs_local; ++kb) {
                        uint8_t b      = vc[kb];
                        uint8_t sq_n   = b & 0x0F;
                        uint8_t sq1_n  = b >> 4;
                        uint8_t* base  = tile_out + kb * kTileFS4;
                        base[byte_off]      |= static_cast<uint8_t>(sq_n  << shift);
                        base[byte_off + 16] |= static_cast<uint8_t>(sq1_n << shift);
                    }
                }
            }
        }
    }

    // search_batch with Faiss-style query batching:
    //
    //   * Each OMP thread takes a contiguous chunk of queries; threads do
    //     not share state. (Faiss-default `search_implem_12` does the same.)
    //   * Within a thread:
    //       Phase A — for each query, rotate, coarse-search, build float LUT,
    //                  quantise to uint8 (per-subvec offset + adaptive scale),
    //                  emit nprobe (list_no, q_local, g_add) probes;
    //       Phase B — sort probes by list_no so probes hitting the same list
    //                  are contiguous;
    //       Phase C — walk the sorted probes; group up to kMaxNQ consecutive
    //                  probes hitting the same list, then dispatch a single
    //                  batched scan (codes loaded ONCE for all NQ queries —
    //                  this is the crux of the Faiss qbs2 trick);
    //       Phase D — drain each query's top-k heap into the output arrays.
    //
    // The cross-query batching turns codes traffic from O(Σ probes) into
    // O(Σ probes / NQ), which is the dominant cost at large M (where each
    // tile is several KB of codes vs. KB of LUT).
    void search_batch(const float* queries, size_t nq, size_t k, size_t nprobe,
                      int64_t* Iptr, float* Dptr) const {
        omp_set_num_threads(nthread_);

        const size_t padded_dim = padded_dim_;
        const size_t nsubvec    = nsubvec_;
        const size_t lut_per_q  = nsubvec * kPQK4;
        const size_t np         = std::min(nprobe, nlist_);

        // Initialise output to -1 / +inf (threads only overwrite indices for
        // queries they own, so partial fills stay sane).
        const float kInf = std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < nq * k; ++i) {
            Iptr[i] = -1;
            Dptr[i] = kInf;
        }

        struct Probe {
            uint32_t list_no;
            uint32_t q_local;
            float    g_add;
        };

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt  = omp_get_num_threads();
            size_t my_q0 = (size_t(tid)     * nq) / size_t(nt);
            size_t my_q1 = (size_t(tid + 1) * nq) / size_t(nt);
            size_t my_nq = my_q1 - my_q0;

            if (my_nq > 0) {
                std::vector<float>   rotated(padded_dim);
                std::vector<float>   lut_f(lut_per_q);
                std::vector<float>   subvec_min(nsubvec);
                std::vector<std::pair<float, size_t>> probe_list_buf(nlist_);
                std::vector<float>   ip_tile(kTileFS4);

                // Concatenated per-query LUTs (uint8, pair-packed) and scalars.
                std::vector<uint8_t> all_lut_u8(my_nq * lut_per_q);
                std::vector<float>   q_scales(my_nq);
                std::vector<float>   q_offsets(my_nq);

                // Per-query top-k storage (concatenated).
                std::vector<float>   topk_dist(my_nq * k, kInf);
                std::vector<PIDFS4>  topk_id(my_nq * k, PIDFS4{0});
                std::vector<TopKMaxHeapFS4> topks(my_nq);
                for (size_t q = 0; q < my_nq; ++q) {
                    topks[q] = TopKMaxHeapFS4{
                        topk_dist.data() + q * k,
                        topk_id.data()   + q * k,
                        k, 0, kInf
                    };
                }

                std::vector<Probe> probes;
                probes.reserve(my_nq * np);

                // ---------- Phase A: per-query LUT + coarse → probes ----------
                for (size_t q = 0; q < my_nq; ++q) {
                    size_t qi = my_q0 + q;
                    const float* query = queries + qi * dim_;
                    rotator_->rotate(query, rotated.data());

                    for (size_t l = 0; l < nlist_; ++l) {
                        float dsq = rabitqlib::euclidean_sqr<float>(
                            rotated.data(),
                            rotated_centroids_.data() + l * padded_dim,
                            padded_dim);
                        probe_list_buf[l] = {dsq, l};
                    }
                    std::partial_sort(probe_list_buf.begin(),
                                      probe_list_buf.begin() + np,
                                      probe_list_buf.end());

                    build_lut_(rotated.data(), lut_f.data());

                    // Adaptive Faiss-style quant scale (see quantize_lut.cpp:157).
                    float q_offset = 0.0f, max_span_LUT = 0.0f, sum_span = 0.0f;
                    for (size_t b = 0; b < nsubvec; ++b) {
                        const float* row = lut_f.data() + b * kPQK4;
                        float mn = row[0], mx = row[0];
                        for (size_t kk = 1; kk < kPQK4; ++kk) {
                            if (row[kk] < mn) mn = row[kk];
                            if (row[kk] > mx) mx = row[kk];
                        }
                        subvec_min[b] = mn;
                        q_offset += mn;
                        float span = mx - mn;
                        sum_span += span;
                        if (span > max_span_LUT) max_span_LUT = span;
                    }
                    float a;
                    if (max_span_LUT <= 0.0f) {
                        a = 1.0f;
                    } else {
                        a = 255.0f / max_span_LUT;
                        if (sum_span > 0.0f) {
                            float a_sum = 65535.0f / sum_span;
                            if (a_sum < a) a = a_sum;
                        }
                    }
                    q_scales[q]  = 1.0f / a;
                    q_offsets[q] = q_offset;

                    uint8_t* my_lut = all_lut_u8.data() + q * lut_per_q;
                    for (size_t b = 0; b < nsubvec; ++b) {
                        const float* row = lut_f.data() + b * kPQK4;
                        uint8_t*     dst = my_lut + b * kPQK4;
                        float mn = subvec_min[b];
                        for (size_t kk = 0; kk < kPQK4; ++kk) {
                            float v = (row[kk] - mn) * a;
                            int   iv = (int)(v + 0.5f);
                            if (iv < 0)   iv = 0;
                            if (iv > 255) iv = 255;
                            dst[kk] = (uint8_t)iv;
                        }
                    }

                    for (size_t pp = 0; pp < np; ++pp) {
                        probes.push_back({
                            static_cast<uint32_t>(probe_list_buf[pp].second),
                            static_cast<uint32_t>(q),
                            probe_list_buf[pp].first
                        });
                    }
                }

                // ---------- Phase B: sort probes by list_no ----------
                std::sort(probes.begin(), probes.end(),
                          [](const Probe& a, const Probe& b) {
                              return a.list_no < b.list_no;
                          });

                // ---------- Phase C: batch-scan ----------
                // Cap each batch at kMaxNQ to bound register pressure on the
                // accumulators (NQ × 4 __m256i). 4 fits comfortably alongside
                // LUT/codes/masks; bumping higher risks GP/YMM spills.
                constexpr int kMaxNQ = 4;
                const uint8_t*  batch_luts[kMaxNQ];
                float           batch_gadd[kMaxNQ];
                float           batch_qs[kMaxNQ];
                float           batch_qo[kMaxNQ];
                TopKMaxHeapFS4* batch_topks[kMaxNQ];

                size_t i = 0;
                while (i < probes.size()) {
                    uint32_t list_no = probes[i].list_no;
                    size_t j = i + 1;
                    while (j < probes.size()
                           && (j - i) < size_t(kMaxNQ)
                           && probes[j].list_no == list_no) {
                        ++j;
                    }
                    int batch_size = static_cast<int>(j - i);

                    for (int b = 0; b < batch_size; ++b) {
                        size_t q = probes[i + b].q_local;
                        batch_luts[b]  = all_lut_u8.data() + q * lut_per_q;
                        batch_gadd[b]  = probes[i + b].g_add;
                        batch_qs[b]    = q_scales[q];
                        batch_qo[b]    = q_offsets[q];
                        batch_topks[b] = &topks[q];
                    }

                    switch (batch_size) {
                        case 4:
                            scan_list_batched_<4>(list_no, batch_luts,
                                batch_gadd, batch_qs, batch_qo, batch_topks,
                                ip_tile.data());
                            break;
                        case 3: {
                            // 2 + 1: codes loaded twice for this 3-batch, but
                            // still better than 3 individual scans.
                            scan_list_batched_<2>(list_no, batch_luts,
                                batch_gadd, batch_qs, batch_qo, batch_topks,
                                ip_tile.data());
                            const uint8_t*  p[1]  = {batch_luts[2]};
                            float           g[1]  = {batch_gadd[2]};
                            float           qs[1] = {batch_qs[2]};
                            float           qo[1] = {batch_qo[2]};
                            TopKMaxHeapFS4* tk[1] = {batch_topks[2]};
                            scan_list_batched_<1>(list_no, p, g, qs, qo, tk,
                                                  ip_tile.data());
                            break;
                        }
                        case 2:
                            scan_list_batched_<2>(list_no, batch_luts,
                                batch_gadd, batch_qs, batch_qo, batch_topks,
                                ip_tile.data());
                            break;
                        case 1:
                            scan_list_batched_<1>(list_no, batch_luts,
                                batch_gadd, batch_qs, batch_qo, batch_topks,
                                ip_tile.data());
                            break;
                    }
                    i = j;
                }

                // ---------- Phase D: drain top-k heaps to output ----------
                std::vector<std::pair<float, PIDFS4>> ordered;
                for (size_t q = 0; q < my_nq; ++q) {
                    size_t qi = my_q0 + q;
                    ordered.clear();
                    ordered.reserve(topks[q].sz);
                    for (size_t s = 0; s < topks[q].sz; ++s) {
                        ordered.emplace_back(topk_dist[q * k + s],
                                             topk_id[q * k + s]);
                    }
                    std::sort(ordered.begin(), ordered.end());
                    for (size_t r = 0; r < ordered.size(); ++r) {
                        Iptr[qi * k + r] = static_cast<int64_t>(ordered[r].second);
                        Dptr[qi * k + r] = ordered[r].first;
                    }
                }
            }
        }
    }

    size_t padded_dim() const { return padded_dim_; }
    size_t nsubvec()    const { return nsubvec_; }
    size_t dsub()       const { return dsub_; }

private:
    void train_coarse_(const float* data, size_t nb) {
        if (nlist_ == 1) {
            std::vector<double> mean(dim_, 0.0);
            for (size_t i = 0; i < nb; ++i)
                for (size_t d = 0; d < dim_; ++d)
                    mean[d] += data[i * dim_ + d];
            for (size_t d = 0; d < dim_; ++d)
                centroids_[d] = static_cast<float>(mean[d] / double(nb));
            return;
        }
        faiss::ClusteringParameters cp;
        cp.niter = 25;
        cp.seed = 1234;
        cp.verbose = false;
        faiss::Clustering clus(static_cast<int>(dim_),
                               static_cast<int>(nlist_), cp);
        faiss::IndexFlatL2 quantizer(static_cast<faiss::idx_t>(dim_));
        clus.train(static_cast<faiss::idx_t>(nb), data, quantizer);
        std::memcpy(centroids_.data(), clus.centroids.data(),
                    nlist_ * dim_ * sizeof(float));
    }

    void train_pq_(const float* data, size_t nb,
                   const std::vector<int64_t>& assign) {
        size_t n_sample = std::min<size_t>(nb, kPQTrainMaxFS4);
        std::vector<size_t> idx(nb);
        std::iota(idx.begin(), idx.end(), size_t(0));
        if (n_sample < nb) {
            std::mt19937 rng(4242u);
            for (size_t i = 0; i < n_sample; ++i) {
                std::uniform_int_distribution<size_t> dist(i, nb - 1);
                std::swap(idx[i], idx[dist(rng)]);
            }
        }

        std::vector<float> sample(n_sample * padded_dim_, 0.0f);
        #pragma omp parallel
        {
            std::vector<float> rotated(padded_dim_);
            #pragma omp for schedule(static)
            for (int64_t s = 0; s < (int64_t)n_sample; ++s) {
                size_t i = idx[s];
                size_t cid = static_cast<size_t>(assign[i]);
                const float* cent_rot =
                    rotated_centroids_.data() + cid * padded_dim_;
                rotator_->rotate(data + i * dim_, rotated.data());
                float* dst = sample.data() + s * padded_dim_;
                double sq = 0.0;
                for (size_t d = 0; d < padded_dim_; ++d) {
                    float v = rotated[d] - cent_rot[d];
                    dst[d] = v;
                    sq += double(v) * double(v);
                }
                float norm = std::sqrt(std::max(float(sq), 0.0f));
                float inv = (norm > 0.0f) ? (1.0f / norm) : 0.0f;
                for (size_t d = 0; d < padded_dim_; ++d) dst[d] *= inv;
            }
        }

        faiss::ProductQuantizer pq(static_cast<size_t>(padded_dim_),
                                   nsubvec_, nbit_);
        pq.verbose = false;
        pq.cp.niter = 25;
        pq.cp.seed  = 1234;
        pq.train(static_cast<faiss::idx_t>(n_sample), sample.data());
        pq_centroids_ = pq.centroids;
    }

    // Build float LUT[b][k] = <q_b, P[b][k]> using AVX-512.
    // K = 16 fits one __m512, so a single iteration per subvec.
    void build_lut_(const float* q_rotated, float* lut) const {
        const float* PT = pq_centroids_T_.data();
        for (size_t b = 0; b < nsubvec_; ++b) {
            const float* qb = q_rotated + b * dsub_;
            const float* Pb = PT + b * dsub_ * kPQK4;
            __m512 acc = _mm512_mul_ps(
                _mm512_loadu_ps(Pb + 0 * kPQK4),
                _mm512_set1_ps(qb[0]));
            for (size_t j = 1; j < dsub_; ++j) {
                acc = _mm512_fmadd_ps(
                    _mm512_loadu_ps(Pb + j * kPQK4),
                    _mm512_set1_ps(qb[j]),
                    acc);
            }
            _mm512_storeu_ps(lut + b * kPQK4, acc);
        }
    }

    // Templated NQ-batched kernel: scan one inverted list with NQ queries
    // sharing the codes load. Per-pair codes are loaded ONCE; pshufb is run
    // NQ times against each query's LUT. This trades a small per-pair ALU
    // overhead (NQ × pshufb + 4·NQ adds) for an NQ× reduction in codes
    // bandwidth — the dominant cost at large M.
    //
    // Register accounting at NQ=4 (the cap): 4×4 acc16 = 16 YMM regs;
    // codes/clo/chi/lut/res transients reuse a few more; the compiler keeps
    // the rest free for q_scale/q_offset/g_add broadcasts during post-process.
    template <int NQ>
    void scan_list_batched_(
            size_t list_no,
            const uint8_t* const  lut_ptrs[NQ],
            const float           g_adds[NQ],
            const float           q_scales[NQ],
            const float           q_offsets[NQ],
            TopKMaxHeapFS4* const topks[NQ],
            float* ip_tile) const {
        const auto& L = lists_[list_no];
        size_t sz = L.size();
        if (sz == 0) return;

        const size_t n_pairs      = n_pairs_;
        const size_t n_full_tiles = (sz + kTileFS4 - 1) / kTileFS4;
        const size_t tile_stride  = kTileFS4 * n_pairs;

        const uint8_t* L_codes = L.codes_bm.data();
        const float*   L_fadd  = L.f_add.data();
        const float*   L_fres  = L.f_rescale.data();
        const PIDFS4*  L_ids   = L.ids.data();

        if (n_full_tiles > 0) {
            for (size_t kb = 0; kb < n_pairs; kb += 4) {
                _mm_prefetch(reinterpret_cast<const char*>(L_codes + kb * kTileFS4),
                             _MM_HINT_T0);
            }
        }

        const __m256i nibble_mask = _mm256_set1_epi8(0x0f);

        for (size_t t = 0; t < n_full_tiles; ++t) {
            const uint8_t* tile_base = L_codes + t * tile_stride;
            if (t + 1 < n_full_tiles) {
                const uint8_t* next = L_codes + (t + 1) * tile_stride;
                for (size_t kb = 0; kb < n_pairs; kb += 4) {
                    _mm_prefetch(reinterpret_cast<const char*>(next + kb * kTileFS4),
                                 _MM_HINT_T0);
                }
            }

            __m256i acc0[NQ], acc1[NQ], acc2[NQ], acc3[NQ];
            #pragma GCC unroll 8
            for (int q = 0; q < NQ; ++q) {
                acc0[q] = _mm256_setzero_si256();
                acc1[q] = _mm256_setzero_si256();
                acc2[q] = _mm256_setzero_si256();
                acc3[q] = _mm256_setzero_si256();
            }

            // Inner loop: codes loaded ONCE per pair, then NQ pshufb passes.
            for (size_t kb = 0; kb < n_pairs; ++kb) {
                __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                    tile_base + kb * kTileFS4));
                __m256i clo = _mm256_and_si256(c, nibble_mask);
                __m256i chi = _mm256_and_si256(
                    _mm256_srli_epi16(c, 4), nibble_mask);

                #pragma GCC unroll 8
                for (int q = 0; q < NQ; ++q) {
                    __m256i lut = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                        lut_ptrs[q] + kb * 32));
                    __m256i res0 = _mm256_shuffle_epi8(lut, clo);
                    __m256i res1 = _mm256_shuffle_epi8(lut, chi);
                    acc0[q] = _mm256_add_epi16(acc0[q], res0);
                    acc1[q] = _mm256_add_epi16(acc1[q], _mm256_srli_epi16(res0, 8));
                    acc2[q] = _mm256_add_epi16(acc2[q], res1);
                    acc3[q] = _mm256_add_epi16(acc3[q], _mm256_srli_epi16(res1, 8));
                }
            }

            // Per-q post-process: cancel high-byte stream, combine2x2,
            // dequant, RaBitQ score, push to query's top-k heap.
            #pragma GCC unroll 8
            for (int q = 0; q < NQ; ++q) {
                __m256i a0 = _mm256_sub_epi16(acc0[q],
                                              _mm256_slli_epi16(acc1[q], 8));
                __m256i a2 = _mm256_sub_epi16(acc2[q],
                                              _mm256_slli_epi16(acc3[q], 8));

                __m256i a1b0   = _mm256_permute2x128_si256(a0, acc1[q], 0x21);
                __m256i a0b1   = _mm256_blend_epi32(a0, acc1[q], 0xF0);
                __m256i dis_lo = _mm256_add_epi16(a1b0, a0b1);
                __m256i c1d0   = _mm256_permute2x128_si256(a2, acc3[q], 0x21);
                __m256i c0d1   = _mm256_blend_epi32(a2, acc3[q], 0xF0);
                __m256i dis_hi = _mm256_add_epi16(c1d0, c0d1);

                __m512 zmm_qs   = _mm512_set1_ps(q_scales[q]);
                __m512 zmm_qof  = _mm512_set1_ps(q_offsets[q]);
                __m512 zmm_gadd = _mm512_set1_ps(g_adds[q]);

                __m512 f_lo = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(dis_lo));
                __m512 f_hi = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(dis_hi));
                f_lo = _mm512_fmadd_ps(f_lo, zmm_qs, zmm_qof);
                f_hi = _mm512_fmadd_ps(f_hi, zmm_qs, zmm_qof);

                __m512 est_lo = _mm512_fmadd_ps(
                    _mm512_loadu_ps(L_fres + t * kTileFS4), f_lo,
                    _mm512_add_ps(zmm_gadd,
                                  _mm512_loadu_ps(L_fadd + t * kTileFS4)));
                __m512 est_hi = _mm512_fmadd_ps(
                    _mm512_loadu_ps(L_fres + t * kTileFS4 + 16), f_hi,
                    _mm512_add_ps(zmm_gadd,
                                  _mm512_loadu_ps(L_fadd + t * kTileFS4 + 16)));

                update_topk_tile_(est_lo, est_hi, L_ids + t * kTileFS4,
                                  *topks[q], ip_tile);
            }
        }
    }

    // Scan one inverted list (single-query) using Faiss-style PQ4 kernel:
    //   * Codes pre-permuted (perm0) so each per-pair 32B holds subvec sq's
    //     codes in low 16B, sq+1 in high 16B. LUT laid out the same way so a
    //     single 256-bit LOAD fetches it — no broadcast (saves 2 port-5 ops
    //     per pair vs. our prior layout).
    //   * One pshufb operates per-128-bit-lane: low 128 of `c` indexes low
    //     128 of `lut` (subvec sq), high 128 indexes high 128 (subvec sq+1).
    //     Two pshufbs per pair cover all 32 vectors × 2 subvecs.
    //   * Accumulators use the reinterpret trick to skip cvtepu8_epi16
    //     (port-5). Final per-tile combine2x2 unscrambles perm0 back to
    //     natural [v0..v31] order in 16+16 u16 lanes.
    //   * Padded tail lanes have f_add = +inf / f_rescale = 0 (set in fit())
    //     so the SIMD path runs on the full padded tile and the scalar tail
    //     is unnecessary.
    void scan_list_(size_t l, float g_add,
                    const uint8_t* lut_u8,
                    float q_scale, float q_offset,
                    TopKMaxHeapFS4& topk,
                    float* ip_tile) const {
        const auto& L = lists_[l];
        size_t sz = L.size();
        if (sz == 0) return;

        const size_t n_pairs      = n_pairs_;
        // padded_sz = ceil(sz / 32) * 32 — every padded tile is full.
        const size_t n_full_tiles = (sz + kTileFS4 - 1) / kTileFS4;
        const size_t tile_stride  = kTileFS4 * n_pairs;   // bytes per tile

        const uint8_t* L_codes = L.codes_bm.data();
        const float*   L_fadd  = L.f_add.data();
        const float*   L_fres  = L.f_rescale.data();
        const PIDFS4*  L_ids   = L.ids.data();

        if (n_full_tiles > 0) {
            for (size_t kb = 0; kb < n_pairs; kb += 4) {
                _mm_prefetch(reinterpret_cast<const char*>(L_codes + kb * kTileFS4),
                             _MM_HINT_T0);
            }
        }

        const __m256i nibble_mask = _mm256_set1_epi8(0x0f);
        const __m512  zmm_qs      = _mm512_set1_ps(q_scale);
        const __m512  zmm_qof     = _mm512_set1_ps(q_offset);
        const __m512  zmm_gadd    = _mm512_set1_ps(g_add);

        for (size_t t = 0; t < n_full_tiles; ++t) {
            const uint8_t* tile_base = L_codes + t * tile_stride;
            if (t + 1 < n_full_tiles) {
                const uint8_t* next = L_codes + (t + 1) * tile_stride;
                for (size_t kb = 0; kb < n_pairs; kb += 4) {
                    _mm_prefetch(reinterpret_cast<const char*>(next + kb * kTileFS4),
                                 _MM_HINT_T0);
                }
            }

            __m256i acc0 = _mm256_setzero_si256();
            __m256i acc1 = _mm256_setzero_si256();
            __m256i acc2 = _mm256_setzero_si256();
            __m256i acc3 = _mm256_setzero_si256();

            // 2-pair manual unroll: load LUT/codes for two adjacent pairs
            // back-to-back so the OoO engine can interleave loads, pshufbs
            // and accumulator updates around the port-5 pshufb bottleneck.
            size_t kb = 0;
            const size_t kb_pair_end = (n_pairs / 2) * 2;
            for (; kb < kb_pair_end; kb += 2) {
                __m256i lut_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                    lut_u8 + (kb + 0) * 32));
                __m256i lut_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                    lut_u8 + (kb + 1) * 32));
                __m256i ca = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                    tile_base + (kb + 0) * kTileFS4));
                __m256i cb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                    tile_base + (kb + 1) * kTileFS4));

                __m256i clo_a = _mm256_and_si256(ca, nibble_mask);
                __m256i chi_a = _mm256_and_si256(
                    _mm256_srli_epi16(ca, 4), nibble_mask);
                __m256i clo_b = _mm256_and_si256(cb, nibble_mask);
                __m256i chi_b = _mm256_and_si256(
                    _mm256_srli_epi16(cb, 4), nibble_mask);

                __m256i res0_a = _mm256_shuffle_epi8(lut_a, clo_a);
                __m256i res1_a = _mm256_shuffle_epi8(lut_a, chi_a);
                __m256i res0_b = _mm256_shuffle_epi8(lut_b, clo_b);
                __m256i res1_b = _mm256_shuffle_epi8(lut_b, chi_b);

                __m256i s0  = _mm256_add_epi16(res0_a, res0_b);
                __m256i s0h = _mm256_add_epi16(_mm256_srli_epi16(res0_a, 8),
                                                _mm256_srli_epi16(res0_b, 8));
                __m256i s1  = _mm256_add_epi16(res1_a, res1_b);
                __m256i s1h = _mm256_add_epi16(_mm256_srli_epi16(res1_a, 8),
                                                _mm256_srli_epi16(res1_b, 8));
                acc0 = _mm256_add_epi16(acc0, s0);
                acc1 = _mm256_add_epi16(acc1, s0h);
                acc2 = _mm256_add_epi16(acc2, s1);
                acc3 = _mm256_add_epi16(acc3, s1h);
            }
            // Leftover pair when n_pairs is odd.
            for (; kb < n_pairs; ++kb) {
                __m256i lut = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                    lut_u8 + kb * 32));
                __m256i c = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(
                    tile_base + kb * kTileFS4));
                __m256i clo = _mm256_and_si256(c, nibble_mask);
                __m256i chi = _mm256_and_si256(
                    _mm256_srli_epi16(c, 4), nibble_mask);
                __m256i res0 = _mm256_shuffle_epi8(lut, clo);
                __m256i res1 = _mm256_shuffle_epi8(lut, chi);
                acc0 = _mm256_add_epi16(acc0, res0);
                acc1 = _mm256_add_epi16(acc1, _mm256_srli_epi16(res0, 8));
                acc2 = _mm256_add_epi16(acc2, res1);
                acc3 = _mm256_add_epi16(acc3, _mm256_srli_epi16(res1, 8));
            }

            // Cancel the 256·byte_hi term from the packed accumulators.
            acc0 = _mm256_sub_epi16(acc0, _mm256_slli_epi16(acc1, 8));
            acc2 = _mm256_sub_epi16(acc2, _mm256_slli_epi16(acc3, 8));

            // combine2x2(acc0, acc1) → 16 u16 distances for vectors 0..15.
            __m256i a1b0 = _mm256_permute2x128_si256(acc0, acc1, 0x21);
            __m256i a0b1 = _mm256_blend_epi32(acc0, acc1, 0xF0);
            __m256i dis_lo = _mm256_add_epi16(a1b0, a0b1);
            // combine2x2(acc2, acc3) → 16 u16 distances for vectors 16..31.
            __m256i c1d0 = _mm256_permute2x128_si256(acc2, acc3, 0x21);
            __m256i c0d1 = _mm256_blend_epi32(acc2, acc3, 0xF0);
            __m256i dis_hi = _mm256_add_epi16(c1d0, c0d1);

            __m512 f_lo = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(dis_lo));
            __m512 f_hi = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(dis_hi));
            f_lo = _mm512_fmadd_ps(f_lo, zmm_qs, zmm_qof);
            f_hi = _mm512_fmadd_ps(f_hi, zmm_qs, zmm_qof);

            __m512 est_lo = _mm512_fmadd_ps(
                _mm512_loadu_ps(L_fres + t * kTileFS4), f_lo,
                _mm512_add_ps(zmm_gadd, _mm512_loadu_ps(L_fadd + t * kTileFS4)));
            __m512 est_hi = _mm512_fmadd_ps(
                _mm512_loadu_ps(L_fres + t * kTileFS4 + 16), f_hi,
                _mm512_add_ps(zmm_gadd, _mm512_loadu_ps(L_fadd + t * kTileFS4 + 16)));

            update_topk_tile_(est_lo, est_hi, L_ids + t * kTileFS4,
                              topk, ip_tile);
        }
    }

    static inline void update_topk_tile_(
            __m512 est_lo, __m512 est_hi, const PIDFS4* ids32,
            TopKMaxHeapFS4& topk, float* ip_tile) {
        // One unified branch: filter by topk.threshold (= +inf during fill,
        // so padded sentinel lanes — est = +inf — are auto-excluded).
        __m512 bound = _mm512_set1_ps(topk.threshold);
        __mmask16 wl = _mm512_cmp_ps_mask(est_lo, bound, _CMP_LT_OQ);
        __mmask16 wh = _mm512_cmp_ps_mask(est_hi, bound, _CMP_LT_OQ);
        if ((wl | wh) == 0) return;
        _mm512_storeu_ps(ip_tile,      est_lo);
        _mm512_storeu_ps(ip_tile + 16, est_hi);
        while (wl) {
            unsigned v = __builtin_ctz(wl);
            topk.push(ip_tile[v], ids32[v]);
            wl &= wl - 1;
        }
        while (wh) {
            unsigned v = __builtin_ctz(wh);
            topk.push(ip_tile[v + 16], ids32[v + 16]);
            wh &= wh - 1;
        }
    }

private:
    size_t n_;
    size_t dim_;
    size_t padded_dim_;
    size_t nlist_;
    size_t nsubvec_;
    size_t nbit_;
    size_t dsub_;
    size_t n_pairs_;
    size_t code_size_;
    int    nthread_;

    std::unique_ptr<rabitqlib::Rotator<float>> rotator_;
    std::vector<float> centroids_;            // (nlist * dim)
    std::vector<float> rotated_centroids_;    // (nlist * padded_dim)
    std::vector<float> pq_centroids_;         // [b][k][j], faiss layout (K=16)
    std::vector<float> pq_centroids_T_;       // [b][j][k], for vectorized LUT (K=16)

    std::vector<ListStorageFS4> lists_;
};

}  // namespace e8pqlib
