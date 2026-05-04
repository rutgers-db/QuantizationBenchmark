// E_8 lattice initialization helper for IVFE8PQ's PQ codebook (used by
// e8pq.hpp, e8pq_fastscan.hpp, e8pq_fastscan4.hpp).
//
// Background: the per-vector-normalized RaBitQ-style estimator's variance is
// governed by  kappa^2 = ||eps_perp||^2 / <c, o>^2  on the GLOBAL d-dim
// vector. Per-subspace k-means must preserve sub-block magnitude (mean
// ||x_b||^2 ~ 1/M for unit-norm x), so we keep faiss::ProductQuantizer's
// L2 k-means as the workhorse and only swap the random/k-means++ init for
// the Gersho-optimal E_8 lattice when d_s = 8.
//
// Rationale for the change vs. an earlier "spherical k-means" attempt
// (commit history): forcing per-sub-block centroids to unit norm magnified
// them ~sqrt(M)x relative to the natural sub-block scale and tanked recall
// at d_s <= 4 by 5-30 pp. Lloyd's centroid condition (mean) is the right
// update; we just choose a better starting point.

#pragma once

#include <cstddef>
#include <cstdlib>
#include <vector>

#include "e8_codebook.h"   // e8lib::E8Codebook, kBlockDim=8, kCodebookSize=256
#include "rabitqlib/utils/rotator.hpp"

namespace e8pqlib {

// Install a freshly-trained rotation matrix into rotator_ via the standard
// rabitqlib MatrixRotator (dense fp32 storage).
template<typename RotatorPtr>
inline void install_rotator_(RotatorPtr& rotator_,
                             const std::vector<float>& rand_mat,
                             size_t dim, size_t padded_dim) {
    rotator_.reset(rabitqlib::choose_rotator<float>(
        dim, rabitqlib::RotatorType::MatrixRotator, padded_dim));
    rotator_->load(reinterpret_cast<const char*>(rand_mat.data()));
}

// Fill `pq_centroids` (size M*K*dsub, faiss layout
// [b * K * dsub + k * dsub + j]) with the E_8 codebook divided by sqrt(2)
// (so each codeword has unit L2 norm), replicated across all M subspaces.
// Returns true iff E_8 was applied (requires K==256 && dsub==8). On false
// the caller should fall back to faiss's default init.
//
// Wire-up at the call site:
//   bool used_e8 = init_centroids_e8_(pq_centroids_.data(),
//                                     nsubvec_, K, dsub_);
//   faiss::ProductQuantizer pq(padded_dim_, nsubvec_, nbit_);
//   if (used_e8) {
//       pq.centroids = pq_centroids_;
//       pq.train_type = faiss::ProductQuantizer::Train_hot_start;
//   }
//   pq.cp.niter = 25;  pq.cp.seed = 1234;
//   pq.train(n_sample, sample.data());
//   pq_centroids_ = pq.centroids;
inline bool init_centroids_e8_(float* pq_centroids,
                               size_t M, size_t K, size_t dsub) {
    if (K != e8lib::kCodebookSize || dsub != e8lib::kBlockDim) return false;
    // Debug kill-switch: IVFE8PQ_DISABLE_E8=1 falls back to faiss default
    // k-means++ init for A/B comparisons. Read once per training call.
    if (std::getenv("IVFE8PQ_DISABLE_E8") != nullptr) return false;
    const float* src = e8lib::get_e8_codebook().data();   // norm sqrt(2)
    constexpr float kInvSqrt2 = 0.70710678118654752440f;
    const size_t per_subvec = K * dsub;                   // 256 * 8 = 2048
    for (size_t b = 0; b < M; ++b) {
        float* dst = pq_centroids + b * per_subvec;
        for (size_t i = 0; i < per_subvec; ++i) dst[i] = src[i] * kInvSqrt2;
    }
    return true;
}

}  // namespace e8pqlib
