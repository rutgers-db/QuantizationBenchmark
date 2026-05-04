// IVFE8PQFastScan — IVFE8PQ with PQ-fastscan-style register-resident LUT.
//
// Same encoding/training pipeline as IVFE8PQ (KMeans + FHT-Kac rotation +
// learned PQ on normalized residuals + RaBitQ scoring factors). Search differs:
//   * Codes stored in 32-wide block-major tiles (vs 16 in IVFE8PQ).
//   * Per-query float LUT is quantized to uint8 with a per-subvec offset and
//     a single global scale, so it fits in 256 bytes/subvec and can be loaded
//     into 4 ZMM registers per subvec (AVX512VBMI _mm512_permutex2var_epi8).
//   * Accumulation is uint16 across the M subvecs; one float dequant per tile.
//   * Non-VBMI fallback uses 32-wide gather from the float LUT, mirroring
//     IVFE8FastScan's non-VBMI path.

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
#include <faiss/VectorTransform.h>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"

#include "e8pq_pq_train.hpp"   // init_centroids_e8_

namespace e8pqlib {

using PIDFS = rabitqlib::PID;

constexpr size_t kTileFS     = 32;            // 32-wide tile (FastScan layout)
constexpr size_t kPQKFS      = 256;           // PQ centroids per subvec (nbit=8)
constexpr size_t kPQNBitFS   = 8;
constexpr size_t kPQTrainMaxFS = 256 * 1024;

struct ListStorageFS {
    std::vector<PIDFS>   ids;
    std::vector<uint8_t> codes_bm;   // [n_tiles][b][lane] with lane in [0,32)
    std::vector<float>   f_add;
    std::vector<float>   f_rescale;
    size_t size() const { return ids.size(); }
};

class IVFE8PQFastScan {
public:
    IVFE8PQFastScan(size_t n, size_t dim, size_t nlist, size_t nsubvec,
                    size_t nbit, int nthread, const std::string& metric,
                    const std::string& rotator, int use_opq = 1)
        : n_(n), dim_(dim), nlist_(nlist), nsubvec_(nsubvec),
          nbit_(nbit), nthread_(nthread), use_opq_(use_opq) {
        (void)metric;

        if (nbit_ != kPQNBitFS) {
            throw std::runtime_error(
                "IVFE8PQFastScan only supports nbit=8 (1 byte per subvec, "
                "256-entry LUT).");
        }
        padded_dim_ = rabitqlib::round_up_to_multiple(dim_, 64);
        if (nsubvec_ == 0 || padded_dim_ % nsubvec_ != 0) {
            throw std::runtime_error(
                "nsubvec must divide padded_dim (rounded up to multiple of 64)");
        }
        dsub_ = padded_dim_ / nsubvec_;

        rabitqlib::RotatorType rtype =
            (rotator == "matrix") ? rabitqlib::RotatorType::MatrixRotator
                                  : rabitqlib::RotatorType::FhtKacRotator;
        std::srand(static_cast<unsigned>(dim_ + padded_dim_ + 7u));
        rotator_.reset(rabitqlib::choose_rotator<float>(dim_, rtype, padded_dim_));

        centroids_.assign(nlist_ * dim_, 0.0f);
        rotated_centroids_.assign(nlist_ * padded_dim_, 0.0f);
        pq_centroids_.assign(nsubvec_ * kPQKFS * dsub_, 0.0f);
        pq_centroids_T_.assign(nsubvec_ * dsub_ * kPQKFS, 0.0f);
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

        // ---------- 2b. OPQ rotation training (default mode=2 on residuals) ----------
        // See e8pq.hpp::train_opq_rotation_ for design rationale and benchmarks.
        // Skip OPQ at d_s <= 2 (no benefit, expensive training).
        if (padded_dim_ == dim_ && dsub_ > 2) {
            const char* opq_env = std::getenv("IVFE8PQ_USE_OPQ");
            int on = (opq_env != nullptr) ? std::atoi(opq_env) : use_opq_;
            if (on) train_opq_rotation_(data, nb, assign);
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
        // pq_centroids_T_ layout (ours): [b * dsub*K + j * K + k]
        for (size_t b = 0; b < nsubvec_; ++b) {
            const float* src = pq_centroids_.data() + b * kPQKFS * dsub_;
            float*       dst = pq_centroids_T_.data() + b * dsub_ * kPQKFS;
            for (size_t k = 0; k < kPQKFS; ++k) {
                for (size_t j = 0; j < dsub_; ++j) {
                    dst[j * kPQKFS + k] = src[k * dsub_ + j];
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
        for (size_t l = 0; l < nlist_; ++l) {
            size_t sz = counts[l];
            size_t padded_sz = ((sz + kTileFS - 1) / kTileFS) * kTileFS;
            lists_[l].ids.resize(sz);
            lists_[l].codes_bm.assign(padded_sz * nsubvec_, 0);
            lists_[l].f_add.assign(padded_sz, 0.0f);
            lists_[l].f_rescale.assign(padded_sz, 0.0f);
        }

        std::vector<size_t> cursor(nlist_, 0);
        std::vector<size_t> pos(nb);
        for (size_t i = 0; i < nb; ++i) {
            size_t cid = static_cast<size_t>(assign[i]);
            pos[i] = cursor[cid]++;
            lists_[cid].ids[pos[i]] = static_cast<PIDFS>(i);
        }

        // ---------- 6. Encode + RaBitQ factors ----------
        faiss::ProductQuantizer pq(static_cast<size_t>(padded_dim_),
                                   nsubvec_, nbit_);
        pq.centroids = pq_centroids_;

        #pragma omp parallel
        {
            std::vector<float>   rotated(padded_dim_);
            std::vector<float>   residual(padded_dim_);
            std::vector<float>   normalized(padded_dim_);
            std::vector<float>   ohat(padded_dim_);
            std::vector<uint8_t> code(nsubvec_);

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

                pq.compute_code(normalized.data(), code.data());
                pq.decode(code.data(), ohat.data());

                // Scatter codes into 32-wide tile.
                size_t tile = p / kTileFS;
                size_t lane = p % kTileFS;
                uint8_t* tile_base =
                    lists_[cid].codes_bm.data() + tile * kTileFS * nsubvec_;
                for (size_t b = 0; b < nsubvec_; ++b) {
                    tile_base[b * kTileFS + lane] = code[b];
                }

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
    }

    void search_batch(const float* queries, size_t nq, size_t k, size_t nprobe,
                      int64_t* Iptr, float* Dptr) const {
        omp_set_num_threads(nthread_);

        const size_t padded_dim = padded_dim_;
        const size_t nsubvec    = nsubvec_;

        #pragma omp parallel
        {
            std::vector<float>   rotated(padded_dim);
            std::vector<float>   lut_f(nsubvec * kPQKFS);
            std::vector<uint8_t> lut_u8(nsubvec * kPQKFS);
            std::vector<float>   subvec_min(nsubvec);
            std::vector<std::pair<float, size_t>> probe_list(nlist_);
            std::vector<float>   ip_tile(kTileFS);

            #pragma omp for schedule(dynamic, 8)
            for (int64_t qi = 0; qi < (int64_t)nq; ++qi) {
                const float* query = queries + qi * dim_;
                rotator_->rotate(query, rotated.data());

                // ---- Coarse: find nearest nprobe centroids ----
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

                // ---- Build float LUT (vectorized over codeword index) ----
                build_lut_(rotated.data(), lut_f.data());

                // ---- Per-subvec offset + adaptive global scale (Faiss style) ----
                // Per-subvec offset: q_offset = sum_b min_b. Global scale a is
                // the tighter of two bounds (see Faiss quantize_lut.cpp:157):
                //   a = min(255 / max_span_LUT,  65535 / sum_span)
                // where max_span_LUT = max_b (max_b - min_b) keeps each
                // quantized u8 in [0, 255], and sum_span = Σ_b (max_b - min_b)
                // bounds the worst-case sum of M u8 values, keeping the uint16
                // accumulator safe for any M (small M uses full 8-bit precision;
                // large M auto-tightens the scale).
                float q_offset      = 0.0f;
                float max_span_LUT  = 0.0f;
                float sum_span      = 0.0f;
                for (size_t b = 0; b < nsubvec; ++b) {
                    const float* row = lut_f.data() + b * kPQKFS;
                    float mn = row[0], mx = row[0];
                    for (size_t kk = 1; kk < kPQKFS; ++kk) {
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
                    a = 1.0f;          // degenerate: all LUT entries equal
                } else {
                    a = 255.0f / max_span_LUT;
                    if (sum_span > 0.0f) {
                        float a_sum = 65535.0f / sum_span;
                        if (a_sum < a) a = a_sum;
                    }
                }
                float q_scale = 1.0f / a;
                {
                    for (size_t b = 0; b < nsubvec; ++b) {
                        const float* row = lut_f.data() + b * kPQKFS;
                        uint8_t*     dst = lut_u8.data() + b * kPQKFS;
                        float mn = subvec_min[b];
                        for (size_t kk = 0; kk < kPQKFS; ++kk) {
                            float v = (row[kk] - mn) * a;
                            int   iv = (int)(v + 0.5f);
                            if (iv < 0)   iv = 0;
                            if (iv > 255) iv = 255;
                            dst[kk] = (uint8_t)iv;
                        }
                    }
                }

                // ---- Scan probed lists ----
                std::priority_queue<std::pair<float, PIDFS>> topk;
                for (size_t pp = 0; pp < np; ++pp) {
                    float  g_add = probe_list[pp].first;
                    size_t l     = probe_list[pp].second;
                    scan_list_(l, g_add, k, lut_f.data(), lut_u8.data(),
                               q_scale, q_offset, topk, ip_tile.data());
                }

                // ---- Write top-k (ascending by distance) ----
                for (size_t j = 0; j < k; ++j) {
                    Iptr[qi * k + j] = -1;
                    Dptr[qi * k + j] = std::numeric_limits<float>::infinity();
                }
                std::vector<std::pair<float, PIDFS>> ordered;
                ordered.reserve(topk.size());
                while (!topk.empty()) { ordered.push_back(topk.top()); topk.pop(); }
                std::reverse(ordered.begin(), ordered.end());
                for (size_t j = 0; j < ordered.size(); ++j) {
                    Iptr[qi * k + j] = static_cast<int64_t>(ordered[j].second);
                    Dptr[qi * k + j] = ordered[j].first;
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

    // OPQ rotation training: train faiss::OPQMatrix on a sample of post-
    // FHT-Kac normalized IVF residuals (the data PQ actually quantizes),
    // compose with the existing rotation, install the result. See e8pq.hpp
    // for full design notes.
    void train_opq_rotation_(const float* data, size_t nb,
                             const std::vector<int64_t>& assign) {
        size_t opq_n = std::min<size_t>(nb, size_t(256) * 1024);
        std::vector<size_t> idx(nb);
        std::iota(idx.begin(), idx.end(), size_t(0));
        std::mt19937 rng(2025u);
        for (size_t i = 0; i < opq_n; ++i) {
            std::uniform_int_distribution<size_t> dist(i, nb - 1);
            std::swap(idx[i], idx[dist(rng)]);
        }
        std::vector<float> opq_sample(opq_n * dim_);
        std::vector<float> rotated_cents(nlist_ * padded_dim_);
        for (size_t l = 0; l < nlist_; ++l)
            rotator_->rotate(centroids_.data() + l * dim_,
                             rotated_cents.data() + l * padded_dim_);
        #pragma omp parallel
        {
            std::vector<float> rotated(padded_dim_);
            #pragma omp for schedule(static)
            for (int64_t s = 0; s < (int64_t)opq_n; ++s) {
                size_t i = idx[s];
                size_t cid = static_cast<size_t>(assign[i]);
                rotator_->rotate(data + i * dim_, rotated.data());
                const float* cr = rotated_cents.data() + cid * padded_dim_;
                double sq = 0.0;
                for (size_t j = 0; j < padded_dim_; ++j) {
                    float v = rotated[j] - cr[j];
                    rotated[j] = v;
                    sq += double(v) * v;
                }
                float inv = (sq > 0) ? 1.0f / float(std::sqrt(sq)) : 0.0f;
                for (size_t j = 0; j < dim_; ++j)
                    opq_sample[s * dim_ + j] = rotated[j] * inv;
            }
        }

        faiss::OPQMatrix opq(static_cast<int>(dim_),
                             static_cast<int>(nsubvec_),
                             static_cast<int>(padded_dim_));
        opq.niter = 25;
        opq.niter_pq = 4;
        opq.niter_pq_0 = 25;
        opq.verbose = false;
        opq.train(static_cast<faiss::idx_t>(opq_n), opq_sample.data());

        std::vector<float> R_existing(dim_ * padded_dim_, 0.0f);
        std::vector<float> ei(dim_, 0.0f), out(padded_dim_);
        for (size_t i = 0; i < dim_; ++i) {
            std::fill(ei.begin(), ei.end(), 0.0f);
            ei[i] = 1.0f;
            rotator_->rotate(ei.data(), out.data());
            std::memcpy(R_existing.data() + i * padded_dim_, out.data(),
                        padded_dim_ * sizeof(float));
        }

        std::vector<float> rand_mat(dim_ * padded_dim_);
        for (size_t i = 0; i < dim_; ++i) {
            for (size_t j = 0; j < padded_dim_; ++j) {
                double s = 0.0;
                const float* re_row = R_existing.data() + i * padded_dim_;
                const float* opq_row = opq.A.data() + j * dim_;
                for (size_t k = 0; k < padded_dim_; ++k)
                    s += double(re_row[k]) * double(opq_row[k]);
                rand_mat[i * padded_dim_ + j] = float(s);
            }
        }

        install_rotator_(rotator_, rand_mat, dim_, padded_dim_);
    }

    void train_pq_(const float* data, size_t nb,
                   const std::vector<int64_t>& assign) {
        size_t n_sample = std::min<size_t>(nb, kPQTrainMaxFS);
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

        // Train PQ via faiss L2 k-means with E_8 hot-start when dsub==8;
        // see e8pq_pq_train.hpp.
        faiss::ProductQuantizer pq(static_cast<size_t>(padded_dim_),
                                   nsubvec_, nbit_);
        pq.verbose = false;
        pq.cp.niter = 25;
        pq.cp.seed  = 1234;
        pq_centroids_.assign(nsubvec_ * kPQKFS * dsub_, 0.0f);
        bool used_e8 = init_centroids_e8_(pq_centroids_.data(),
                                          nsubvec_, kPQKFS, dsub_);
        if (used_e8) {
            pq.centroids = pq_centroids_;
            pq.train_type = faiss::ProductQuantizer::Train_hot_start;
        }
        pq.train(static_cast<faiss::idx_t>(n_sample), sample.data());
        pq_centroids_ = pq.centroids;
    }

    // Build float LUT[b][k] = <q_b, P[b][k]> using AVX-512.
    // Iterates k in chunks of 16 codewords, summing dsub * q[b][j] * P[b][j][k].
    void build_lut_(const float* q_rotated, float* lut) const {
        const float* PT = pq_centroids_T_.data();
        for (size_t b = 0; b < nsubvec_; ++b) {
            const float* qb  = q_rotated + b * dsub_;
            const float* Pb  = PT + b * dsub_ * kPQKFS;
            float*       row = lut + b * kPQKFS;
            for (size_t ck = 0; ck < kPQKFS; ck += 16) {
                __m512 acc = _mm512_mul_ps(
                    _mm512_loadu_ps(Pb + 0 * kPQKFS + ck),
                    _mm512_set1_ps(qb[0]));
                for (size_t j = 1; j < dsub_; ++j) {
                    acc = _mm512_fmadd_ps(
                        _mm512_loadu_ps(Pb + j * kPQKFS + ck),
                        _mm512_set1_ps(qb[j]),
                        acc);
                }
                _mm512_storeu_ps(row + ck, acc);
            }
        }
    }

    void scan_list_(size_t l, float g_add, size_t k,
                    const float* lut_f, const uint8_t* lut_u8,
                    float q_scale, float q_offset,
                    std::priority_queue<std::pair<float, PIDFS>>& topk,
                    float* ip_tile) const {
        const auto& L = lists_[l];
        size_t sz = L.size();
        if (sz == 0) return;

        const size_t nsubvec     = nsubvec_;
        const size_t n_full_tiles = sz / kTileFS;
        const size_t tail         = sz - n_full_tiles * kTileFS;
        const size_t tile_stride  = kTileFS * nsubvec;

        const uint8_t* L_codes = L.codes_bm.data();
        const float*   L_fadd  = L.f_add.data();
        const float*   L_fres  = L.f_rescale.data();
        const PIDFS*   L_ids   = L.ids.data();

        if (n_full_tiles > 0) {
            for (size_t b = 0; b < nsubvec; b += 4) {
                _mm_prefetch(reinterpret_cast<const char*>(L_codes + b * kTileFS),
                             _MM_HINT_T0);
            }
        }

#ifdef __AVX512VBMI__
        // -------- Register-resident uint8 LUT path (AVX512VBMI) --------
        const __m512i zmm_7f   = _mm512_set1_epi8(0x7f);
        const __m512  zmm_qs   = _mm512_set1_ps(q_scale);
        const __m512  zmm_qof  = _mm512_set1_ps(q_offset);
        const __m512  zmm_gadd = _mm512_set1_ps(g_add);

        for (size_t t = 0; t < n_full_tiles; ++t) {
            const uint8_t* tile_base = L_codes + t * tile_stride;
            if (t + 1 < n_full_tiles) {
                const uint8_t* next = L_codes + (t + 1) * tile_stride;
                for (size_t b = 0; b < nsubvec; b += 4) {
                    _mm_prefetch(reinterpret_cast<const char*>(next + b * kTileFS),
                                 _MM_HINT_T0);
                }
            }

            __m512i acc = _mm512_setzero_si512();
            for (size_t b = 0; b < nsubvec; ++b) {
                const uint8_t* lb = lut_u8 + b * kPQKFS;
                __m512i lut0 = _mm512_loadu_si512(lb +   0);
                __m512i lut1 = _mm512_loadu_si512(lb +  64);
                __m512i lut2 = _mm512_loadu_si512(lb + 128);
                __m512i lut3 = _mm512_loadu_si512(lb + 192);
                __m256i codes32 = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(tile_base + b * kTileFS));
                __m512i idx     = _mm512_zextsi256_si512(codes32);
                __m512i d_lo    = _mm512_permutex2var_epi8(lut0, idx, lut1);
                __m512i idx_hi  = _mm512_and_si512(idx, zmm_7f);
                __m512i d_hi    = _mm512_permutex2var_epi8(lut2, idx_hi, lut3);
                __mmask64 hi_m  = _mm512_movepi8_mask(idx);
                __m512i dist    = _mm512_mask_blend_epi8(hi_m, d_lo, d_hi);
                acc = _mm512_add_epi16(acc, _mm512_cvtepu8_epi16(
                    _mm512_castsi512_si256(dist)));
            }

            __m512 f_lo = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(
                _mm512_castsi512_si256(acc)));
            __m512 f_hi = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(
                _mm512_extracti64x4_epi64(acc, 1)));
            f_lo = _mm512_fmadd_ps(f_lo, zmm_qs, zmm_qof);
            f_hi = _mm512_fmadd_ps(f_hi, zmm_qs, zmm_qof);

            __m512 est_lo = _mm512_fmadd_ps(
                _mm512_loadu_ps(L_fres + t * kTileFS), f_lo,
                _mm512_add_ps(zmm_gadd, _mm512_loadu_ps(L_fadd + t * kTileFS)));
            __m512 est_hi = _mm512_fmadd_ps(
                _mm512_loadu_ps(L_fres + t * kTileFS + 16), f_hi,
                _mm512_add_ps(zmm_gadd, _mm512_loadu_ps(L_fadd + t * kTileFS + 16)));

            update_topk_tile_(est_lo, est_hi, L_ids + t * kTileFS, k,
                              topk, ip_tile);
        }
#else
        // -------- Non-VBMI fallback: 32-wide gather from float LUT --------
        const __m512 zmm_gadd = _mm512_set1_ps(g_add);
        (void)lut_u8; (void)q_scale; (void)q_offset;

        for (size_t t = 0; t < n_full_tiles; ++t) {
            const uint8_t* tile_base = L_codes + t * tile_stride;
            if (t + 1 < n_full_tiles) {
                const uint8_t* next = L_codes + (t + 1) * tile_stride;
                for (size_t b = 0; b < nsubvec; b += 4) {
                    _mm_prefetch(reinterpret_cast<const char*>(next + b * kTileFS),
                                 _MM_HINT_T0);
                }
            }

            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();

            size_t b = 0;
            const size_t b_pair_end = (nsubvec / 2) * 2;
            for (; b < b_pair_end; b += 2) {
                const uint8_t* col0  = tile_base + (b + 0) * kTileFS;
                const uint8_t* col1  = tile_base + (b + 1) * kTileFS;
                const float*   lut0f = lut_f + (b + 0) * kPQKFS;
                const float*   lut1f = lut_f + (b + 1) * kPQKFS;
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
            for (; b < nsubvec; ++b) {
                const uint8_t* col = tile_base + b * kTileFS;
                const float*   lbf = lut_f + b * kPQKFS;
                acc0 = _mm512_add_ps(acc0, _mm512_i32gather_ps(
                    _mm512_cvtepu8_epi32(_mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(col))), lbf, 4));
                acc1 = _mm512_add_ps(acc1, _mm512_i32gather_ps(
                    _mm512_cvtepu8_epi32(_mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(col + 16))), lbf, 4));
            }

            __m512 est_lo = _mm512_fmadd_ps(
                _mm512_loadu_ps(L_fres + t * kTileFS), acc0,
                _mm512_add_ps(zmm_gadd, _mm512_loadu_ps(L_fadd + t * kTileFS)));
            __m512 est_hi = _mm512_fmadd_ps(
                _mm512_loadu_ps(L_fres + t * kTileFS + 16), acc1,
                _mm512_add_ps(zmm_gadd, _mm512_loadu_ps(L_fadd + t * kTileFS + 16)));

            update_topk_tile_(est_lo, est_hi, L_ids + t * kTileFS, k,
                              topk, ip_tile);
        }
#endif

        // ---- Scalar tail (< kTileFS vectors) ----
        if (tail > 0) {
            const uint8_t* tile_base = L_codes + n_full_tiles * tile_stride;
            for (size_t v = 0; v < tail; ++v) {
                float ip = 0.0f;
                for (size_t b = 0; b < nsubvec; ++b) {
                    uint8_t c = tile_base[b * kTileFS + v];
                    ip += lut_f[b * kPQKFS + c];
                }
                size_t p = n_full_tiles * kTileFS + v;
                float est = g_add + L.f_add[p] + L.f_rescale[p] * ip;
                PIDFS id = L.ids[p];
                if (topk.size() < k) {
                    topk.emplace(est, id);
                } else if (est < topk.top().first) {
                    topk.pop();
                    topk.emplace(est, id);
                }
            }
        }
    }

    static inline void update_topk_tile_(
            __m512 est_lo, __m512 est_hi, const PIDFS* ids32, size_t k,
            std::priority_queue<std::pair<float, PIDFS>>& topk,
            float* ip_tile) {
        if (topk.size() >= k) {
            __m512 bound = _mm512_set1_ps(topk.top().first);
            __mmask16 wl = _mm512_cmp_ps_mask(est_lo, bound, _CMP_LT_OQ);
            __mmask16 wh = _mm512_cmp_ps_mask(est_hi, bound, _CMP_LT_OQ);
            if ((wl | wh) == 0) return;
            _mm512_storeu_ps(ip_tile,      est_lo);
            _mm512_storeu_ps(ip_tile + 16, est_hi);
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
            _mm512_storeu_ps(ip_tile,      est_lo);
            _mm512_storeu_ps(ip_tile + 16, est_hi);
            for (size_t v = 0; v < kTileFS; ++v) {
                float e = ip_tile[v];
                if (topk.size() < k) topk.emplace(e, ids32[v]);
                else if (e < topk.top().first) { topk.pop(); topk.emplace(e, ids32[v]); }
            }
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
    int    nthread_;
    int    use_opq_;

    std::unique_ptr<rabitqlib::Rotator<float>> rotator_;
    std::vector<float> centroids_;            // (nlist * dim)
    std::vector<float> rotated_centroids_;    // (nlist * padded_dim)
    std::vector<float> pq_centroids_;         // [b][k][j], faiss layout
    std::vector<float> pq_centroids_T_;       // [b][j][k], for vectorized LUT

    std::vector<ListStorageFS> lists_;
};

}  // namespace e8pqlib
