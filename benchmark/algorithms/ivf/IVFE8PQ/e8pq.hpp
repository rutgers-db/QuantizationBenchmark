// IVFE8PQ — IVF + RaBitQ-style distance estimator with a LEARNED
// Product-Quantization codebook on the normalized residual.
//
// Pipeline (all training runs in this C++ core — see IVFE8PQ::fit):
//   1. KMeans coarse quantizer  (faiss::Clustering on the raw data)
//   2. FHT-Kac random rotation  (rabitqlib Rotator, same as IVFE8)
//   3. PQ codebook on normalized residuals o = r / ||r||
//        (faiss::ProductQuantizer trained on a random sample of o)
//   4. Per-vector encoding + RaBitQ factors (f_add, f_rescale) computed
//      exactly as in IVFE8; scoring kernel is identical to IVFE8.
//
// Search uses the same scoring:
//   est = g_add + f_add + f_rescale * <q_r, o_hat>
// where <q_r, o_hat> is obtained via M gathers into an (M x 256) float LUT.

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

using PID = rabitqlib::PID;

constexpr size_t kTile = 16;                // 16-wide block-major tile (gather path)
constexpr size_t kPQK  = 256;               // PQ centroids per subvector (nbit=8)
constexpr size_t kPQNBit = 8;               // only 8-bit codes supported by the SIMD scan
constexpr size_t kPQTrainMax = 256 * 1024;  // cap on PQ training sample

/* Per-list storage (mirrors IVFE8 ListStorage). */
struct ListStorage {
    std::vector<PID>     ids;
    std::vector<uint8_t> codes_bm;   // block-major, 16-wide tiles: [tile][b][lane]
    std::vector<float>   f_add;
    std::vector<float>   f_rescale;
    size_t size() const { return ids.size(); }
};

class IVFE8PQ {
public:
    IVFE8PQ(size_t n, size_t dim, size_t nlist, size_t nsubvec, size_t nbit,
            int nthread, const std::string& metric, const std::string& rotator,
            int use_opq = 1)
        : n_(n), dim_(dim), nlist_(nlist), nsubvec_(nsubvec),
          nbit_(nbit), nthread_(nthread), use_opq_(use_opq) {
        (void)metric;

        if (nbit_ != kPQNBit) {
            throw std::runtime_error(
                "IVFE8PQ currently only supports nbit=8 (SIMD scan uses "
                "1-byte-per-subvec codes and a 256-entry LUT).");
        }
        padded_dim_ = rabitqlib::round_up_to_multiple(dim_, 64);
        if (nsubvec_ == 0 || padded_dim_ % nsubvec_ != 0) {
            throw std::runtime_error(
                "nsubvec must divide padded_dim (padded to multiple of 64)");
        }
        dsub_ = padded_dim_ / nsubvec_;

        rabitqlib::RotatorType rtype =
            (rotator == "matrix") ? rabitqlib::RotatorType::MatrixRotator
                                  : rabitqlib::RotatorType::FhtKacRotator;
        std::srand(static_cast<unsigned>(dim_ + padded_dim_));
        rotator_.reset(rabitqlib::choose_rotator<float>(dim_, rtype, padded_dim_));

        centroids_.assign(nlist_ * dim_, 0.0f);
        rotated_centroids_.assign(nlist_ * padded_dim_, 0.0f);
        pq_centroids_.assign(nsubvec_ * kPQK * dsub_, 0.0f);
        lists_.resize(nlist_);
    }

    /* Full training path — KMeans -> rotation -> PQ training on normalized
     * residuals -> per-vector encoding and RaBitQ factor computation. */
    void fit(const float* data, size_t nb) {
        if (nb == 0) return;

        omp_set_num_threads(nthread_);

        // ---------- 1. Coarse KMeans ----------
        train_coarse_(data, nb);

        // ---------- 2. Assign every vector to its nearest centroid ----------
        std::vector<int64_t> assign(nb);
        {
            faiss::IndexFlatL2 coarse(static_cast<faiss::idx_t>(dim_));
            coarse.add(static_cast<faiss::idx_t>(nlist_), centroids_.data());
            std::vector<float> d_tmp(nb);
            coarse.search(static_cast<faiss::idx_t>(nb), data, 1,
                          d_tmp.data(), assign.data());
        }

        // ---------- 2b. Optional OPQ rotation training ----------
        // Train OPQ on the normalized IVF residuals — the data PQ actually
        // quantizes — and compose with the FHT-Kac rotation. Empirically
        // +3-8 pp recall@100 at d_s=8 on SIFT/GIST. Toggled by `use_opq`
        // constructor arg (env IVFE8PQ_USE_OPQ overrides for A/B). Guard:
        // dsub <= 2 always skips OPQ (no benefit, expensive training).
        // Requires padded_dim == dim (no padding pad).
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
            size_t padded_sz = ((sz + kTile - 1) / kTile) * kTile;
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
            lists_[cid].ids[pos[i]] = static_cast<PID>(i);
        }

        // ---------- 6. Encode + compute RaBitQ factors ----------
        faiss::ProductQuantizer pq(static_cast<size_t>(padded_dim_),
                                   nsubvec_, nbit_);
        pq.centroids = pq_centroids_;      // copies into faiss pq

        #pragma omp parallel
        {
            std::vector<float> rotated(padded_dim_);
            std::vector<float> residual(padded_dim_);
            std::vector<float> normalized(padded_dim_);
            std::vector<float> ohat(padded_dim_);
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

                // PQ encode the normalized residual; decode back to ô.
                pq.compute_code(normalized.data(), code.data());
                pq.decode(code.data(), ohat.data());

                // Scatter codes into block-major tile layout.
                size_t tile = p / kTile;
                size_t lane = p % kTile;
                uint8_t* tile_base =
                    lists_[cid].codes_bm.data() + tile * kTile * nsubvec_;
                for (size_t b = 0; b < nsubvec_; ++b) {
                    tile_base[b * kTile + lane] = code[b];
                }

                float ip_resi = rabitqlib::dot_product<float>(
                    residual.data(), ohat.data(), padded_dim_);
                float ip_cent = rabitqlib::dot_product<float>(
                    cent_rot, ohat.data(), padded_dim_);
                if (ip_resi == 0.0f) {
                    ip_resi = std::numeric_limits<float>::infinity();
                }
                lists_[cid].f_add[p] = l2_sqr +
                    2.0f * l2_sqr * ip_cent / ip_resi;
                lists_[cid].f_rescale[p] = -2.0f * l2_sqr / ip_resi;
            }
        }

        // Optional diagnostic: empirical kappa^2 = ||eps_perp||^2 / <ohat, o>^2
        // on a random sample. Compares to the report's spherical-uniform
        // value 0.747^2 = 0.558 for L2 PQ at d_s=8, b=8.
        if (std::getenv("IVFE8PQ_PRINT_KAPPA")) {
            faiss::ProductQuantizer kpq(static_cast<size_t>(padded_dim_),
                                        nsubvec_, nbit_);
            kpq.centroids = pq_centroids_;
            constexpr size_t kappa_n = 4096;
            std::mt19937 krng(98765u);
            std::uniform_int_distribution<size_t> kdist(0, nb - 1);
            std::vector<float> kr(padded_dim_), ko(padded_dim_),
                               koh(padded_dim_);
            std::vector<uint8_t> kc(nsubvec_);
            double sum_k2 = 0.0, sum_oho = 0.0, sum_eps_perp_sq = 0.0;
            size_t cnt = 0;
            for (size_t s = 0; s < kappa_n; ++s) {
                size_t i = kdist(krng);
                size_t cid = static_cast<size_t>(assign[i]);
                const float* cr =
                    rotated_centroids_.data() + cid * padded_dim_;
                rotator_->rotate(data + i * dim_, kr.data());
                double sq = 0.0;
                for (size_t k = 0; k < padded_dim_; ++k) {
                    float v = kr[k] - cr[k]; ko[k] = v; sq += double(v) * v;
                }
                if (sq < 1e-30) continue;
                float inv = 1.0f / std::sqrt(float(sq));
                for (size_t k = 0; k < padded_dim_; ++k) ko[k] *= inv;
                kpq.compute_code(ko.data(), kc.data());
                kpq.decode(kc.data(), koh.data());
                double oho = 0.0, eps_sq = 0.0;
                for (size_t k = 0; k < padded_dim_; ++k) {
                    double e = double(ko[k]) - double(koh[k]);
                    oho += double(koh[k]) * double(ko[k]);
                    eps_sq += e * e;
                }
                double eps_para = 1.0 - oho;          // <eps, o>
                double eps_perp_sq = eps_sq - eps_para * eps_para;
                if (oho == 0.0) continue;
                sum_k2 += eps_perp_sq / (oho * oho);
                sum_oho += oho;
                sum_eps_perp_sq += eps_perp_sq;
                ++cnt;
            }
            if (cnt > 0) {
                double mk2 = sum_k2 / double(cnt);
                std::fprintf(stderr,
                    "[IVFE8PQ_KAPPA] M=%zu dsub=%zu n=%zu  "
                    "E[kappa^2]=%.4f  kappa=%.4f  "
                    "E[<oh,o>]=%.4f  E[||eps_perp||^2]=%.4f\n",
                    nsubvec_, dsub_, cnt, mk2, std::sqrt(mk2),
                    sum_oho / double(cnt), sum_eps_perp_sq / double(cnt));
            }
        }
    }

    /* Batched k-NN search. Returns row-major (I, D) of shape (nq, k). */
    void search_batch(const float* queries, size_t nq, size_t k, size_t nprobe,
                      int64_t* Iptr, float* Dptr) const {
        omp_set_num_threads(nthread_);

        const size_t padded_dim = padded_dim_;
        const size_t nsubvec = nsubvec_;
        const size_t dsub = dsub_;

        #pragma omp parallel
        {
            std::vector<float> rotated(padded_dim);
            std::vector<float> lut(nsubvec * kPQK);   // per-subvec LUT
            std::vector<std::pair<float, size_t>> probe_list(nlist_);
            std::vector<float> ip_tile(kTile);

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

                // ---- Build LUT: LUT[b][k] = <q_b, P[b][k]> ----
                build_lut_(rotated.data(), lut.data());

                // ---- Scan probed lists with top-k heap ----
                std::priority_queue<std::pair<float, PID>> topk;
                for (size_t pp = 0; pp < np; ++pp) {
                    float g_add = probe_list[pp].first;
                    size_t l    = probe_list[pp].second;
                    scan_list_(l, g_add, k, lut.data(), topk, ip_tile.data());
                }

                // ---- Write top-k ----
                for (size_t j = 0; j < k; ++j) {
                    Iptr[qi * k + j] = -1;
                    Dptr[qi * k + j] = std::numeric_limits<float>::infinity();
                }
                std::vector<std::pair<float, PID>> ordered;
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
    // ---------- KMeans coarse quantizer ----------
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

    // OPQ rotation training. Trains faiss::OPQMatrix on a sample of
    // post-FHT-Kac normalized residuals (the data PQ actually quantizes),
    // then composes the new rotation with the existing FHT-Kac one and
    // installs the combined matrix as the active rotator.
    //
    // Empirically gives +3-8 pp recall@100 at d_s=8 on SIFT/GIST. The key
    // contrast with vanilla "OPQ-on-raw-data" is that IVF + per-cluster
    // normalization mostly absorbs raw-data anisotropy, leaving little for
    // OPQ to exploit there; training on residuals targets the actual
    // quantization input.
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

        // Sample post-FHT-Kac normalized residuals.
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

        // Sample existing rotator as a matrix to compose with OPQ.
        std::vector<float> R_existing(dim_ * padded_dim_, 0.0f);
        std::vector<float> ei(dim_, 0.0f), out(padded_dim_);
        for (size_t i = 0; i < dim_; ++i) {
            std::fill(ei.begin(), ei.end(), 0.0f);
            ei[i] = 1.0f;
            rotator_->rotate(ei.data(), out.data());
            std::memcpy(R_existing.data() + i * padded_dim_, out.data(),
                        padded_dim_ * sizeof(float));
        }

        // Compose: rand_mat[i,j] = Σ_k R_existing[i,k] * opq.A[j,k]
        // (opq.A is (pdim,dim) row-major; opq.A^T is (dim,pdim); rabitqlib
        //  MatrixRotator rotates as v * rand_mat_ in (dim,pdim) layout).
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


    // ---------- PQ training on normalized rotated residuals ----------
    void train_pq_(const float* data, size_t nb,
                   const std::vector<int64_t>& assign) {
        size_t n_sample = std::min<size_t>(nb, kPQTrainMax);
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

        // Train PQ. For dsub==8 we hot-start from the unit-norm E_8 lattice
        // (Gersho-optimal 8-D sphere packing); otherwise default init.
        //
        // Env knob IVFE8PQ_SCANN_HPAR=<float in (0,1]> swaps faiss L2 k-means
        // for a custom anisotropic Lloyd that minimizes
        //   h_par * eps_||^2 + 1.0 * eps_perp^2
        // where eps_||/eps_perp are decomposed relative to the data x's
        // direction (ScaNN-style). h_par=1 reduces to L2; smaller h_par
        // attacks the perpendicular error component the per-vector-normalized
        // estimator is sensitive to. Output layout is faiss-native either way.
        pq_centroids_.assign(nsubvec_ * kPQK * dsub_, 0.0f);
        bool used_e8 = init_centroids_e8_(pq_centroids_.data(),
                                          nsubvec_, kPQK, dsub_);
        const char* scann_env = std::getenv("IVFE8PQ_SCANN_HPAR");
        if (scann_env != nullptr && std::atof(scann_env) > 0.0) {
            double h_par = std::atof(scann_env);
            train_pq_anisotropic_(sample.data(), n_sample, padded_dim_,
                                  nsubvec_, kPQK, dsub_,
                                  used_e8 ? pq_centroids_.data() : nullptr,
                                  /*niter=*/25, /*seed=*/1234u, h_par,
                                  pq_centroids_.data());
        } else {
            faiss::ProductQuantizer pq(static_cast<size_t>(padded_dim_),
                                       nsubvec_, nbit_);
            pq.verbose = false;
            pq.cp.niter = 25;
            pq.cp.seed  = 1234;
            if (used_e8) {
                pq.centroids = pq_centroids_;
                pq.train_type = faiss::ProductQuantizer::Train_hot_start;
            }
            pq.train(static_cast<faiss::idx_t>(n_sample), sample.data());
            pq_centroids_ = pq.centroids;
        }
    }

    // ---------- LUT build: LUT[b][k] = <q_b, P[b][k]> ----------
    void build_lut_(const float* q_rotated, float* lut) const {
        // pq_centroids_ layout (faiss): for subvec b, K codewords of dsub floats:
        //   pq_centroids_[b * K*dsub + k * dsub + j] = P[b][k][j]
        // Plain dense loop; scans compactly and auto-vectorizes reasonably.
        const float* P_all = pq_centroids_.data();
        const size_t K = kPQK;
        for (size_t b = 0; b < nsubvec_; ++b) {
            const float* qb = q_rotated + b * dsub_;
            const float* Pb = P_all + b * K * dsub_;
            float* row = lut + b * K;
            for (size_t kk = 0; kk < K; ++kk) {
                const float* cw = Pb + kk * dsub_;
                float s = 0.0f;
                for (size_t j = 0; j < dsub_; ++j) s += qb[j] * cw[j];
                row[kk] = s;
            }
        }
    }

    // ---------- Scan one probed list, update top-k ----------
    void scan_list_(size_t l, float g_add, size_t k, const float* lut,
                    std::priority_queue<std::pair<float, PID>>& topk,
                    float* ip_tile) const {
        const auto& L = lists_[l];
        size_t sz = L.size();
        if (sz == 0) return;

        size_t n_full_tiles = sz / kTile;
        size_t tail = sz - n_full_tiles * kTile;
        const size_t nsubvec = nsubvec_;
        const size_t tile_stride = kTile * nsubvec;

        const uint8_t* L_codes = L.codes_bm.data();
        const float*   L_fadd  = L.f_add.data();
        const float*   L_fres  = L.f_rescale.data();
        const PID*     L_ids   = L.ids.data();

        if (n_full_tiles > 0) {
            for (size_t b = 0; b < nsubvec; b += 4) {
                _mm_prefetch(reinterpret_cast<const char*>(L_codes + b * kTile),
                             _MM_HINT_T0);
            }
        }

        for (size_t t = 0; t < n_full_tiles; ++t) {
            const uint8_t* tile_base = L_codes + t * tile_stride;
            if (t + 1 < n_full_tiles) {
                const uint8_t* next = L_codes + (t + 1) * tile_stride;
                for (size_t b = 0; b < nsubvec; b += 4) {
                    _mm_prefetch(reinterpret_cast<const char*>(next + b * kTile),
                                 _MM_HINT_T0);
                }
            }

            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            size_t b = 0;
            size_t b_pair_end = (nsubvec / 2) * 2;
            for (; b < b_pair_end; b += 2) {
                __m128i c0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                    tile_base + (b + 0) * kTile));
                __m128i c1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
                    tile_base + (b + 1) * kTile));
                __m512i i0 = _mm512_cvtepu8_epi32(c0);
                __m512i i1 = _mm512_cvtepu8_epi32(c1);
                const float* lut0 = lut + (b + 0) * kPQK;
                const float* lut1 = lut + (b + 1) * kPQK;
                __m512 v0 = _mm512_i32gather_ps(i0, lut0, 4);
                __m512 v1 = _mm512_i32gather_ps(i1, lut1, 4);
                acc0 = _mm512_add_ps(acc0, v0);
                acc1 = _mm512_add_ps(acc1, v1);
            }
            for (; b < nsubvec; ++b) {
                __m128i codes16 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(tile_base + b * kTile));
                __m512i idx = _mm512_cvtepu8_epi32(codes16);
                const float* lut_b = lut + b * kPQK;
                __m512 vals = _mm512_i32gather_ps(idx, lut_b, 4);
                acc0 = _mm512_add_ps(acc0, vals);
            }
            __m512 acc = _mm512_add_ps(acc0, acc1);
            __m512 fadd = _mm512_loadu_ps(L_fadd + t * kTile);
            __m512 fres = _mm512_loadu_ps(L_fres + t * kTile);
            __m512 est  = _mm512_fmadd_ps(fres, acc,
                            _mm512_add_ps(_mm512_set1_ps(g_add), fadd));

            if (topk.size() >= k) {
                __m512 bound = _mm512_set1_ps(topk.top().first);
                __mmask16 win = _mm512_cmp_ps_mask(est, bound, _CMP_LT_OQ);
                if (win == 0) continue;
                _mm512_storeu_ps(ip_tile, est);
                const PID* ids16 = L_ids + t * kTile;
                while (win) {
                    unsigned v = __builtin_ctz(win);
                    float e = ip_tile[v];
                    if (e < topk.top().first) {
                        topk.pop();
                        topk.emplace(e, ids16[v]);
                    }
                    win &= (win - 1);
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

        // Scalar tail
        if (tail > 0) {
            const uint8_t* tile_base =
                L_codes + n_full_tiles * tile_stride;
            for (size_t v = 0; v < tail; ++v) {
                float ip = 0.0f;
                for (size_t b = 0; b < nsubvec; ++b) {
                    uint8_t c = tile_base[b * kTile + v];
                    ip += lut[b * kPQK + c];
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

private:
    size_t n_;
    size_t dim_;
    size_t padded_dim_;
    size_t nlist_;
    size_t nsubvec_;
    size_t nbit_;
    size_t dsub_;
    int    nthread_;
    int    use_opq_;   // 0=off, 1=raw OPQ, 2=residual OPQ (default), 3=local

    std::unique_ptr<rabitqlib::Rotator<float>> rotator_;
    std::vector<float> centroids_;            // (nlist * dim)
    std::vector<float> rotated_centroids_;    // (nlist * padded_dim)
    std::vector<float> pq_centroids_;         // (nsubvec * 256 * dsub)

    std::vector<ListStorage> lists_;
};

}  // namespace e8pqlib
