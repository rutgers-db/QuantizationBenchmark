#pragma once
// =====================================================================
// TurboQuantIndexV3 — centroid-normalized TurboQuant
//
// Key changes from V2:
//   1. train() computes the data centroid c = mean(x_i).
//   2. add()   normalises (x_r - c) to the unit sphere instead of x_r.
//              norms_[i] stores |x_r - c| (centred norm).
//   3. query() normalises (q_r - c) before rotation.
//   4. L2 distance uses the bridge identity:
//        ||x_r - q_r||^2 = |x_c|^2 + |q_c|^2 - 2*|x_c|*|q_c|*<x_unit,q_unit>
//      where x_c = x_r-c, q_c = q_r-c, x_unit = x_c/|x_c|.
//   5. Codebook uses hardcoded Lloyd-Max tables for N(0,1), scaled by
//      sigma = 1/sqrt(padded_dim) to match the distribution of each
//      coordinate of a unit-sphere vector after zero-padded Hadamard
//      rotation. For bitwidth > 8 the Beta codebook solver is used.
//   6. Hadamard transform zero-pads input to the next power-of-2
//      (padded_dim) and keeps ALL output components — no truncation.
//      This avoids the ~22% energy loss that killed quality on
//      non-power-of-2 dimensions.
// =====================================================================

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <immintrin.h>
#include <limits>
#include <numeric>
#include <omp.h>
#include <stdexcept>
#include <utility>
#include <vector>

namespace turboquant {

class TurboQuantIndexV3 {
 public:
  using idx_t = std::int64_t;

  enum class Mode { kMSE, kInnerProduct };
  enum class SearchMetric { kInnerProduct, kL2 };
  enum class RotationType { kDense, kHadamard };  // always Hadamard

  struct Config {
    std::size_t dim = 0;
    std::size_t bitwidth = 0;
    Mode mode = Mode::kInnerProduct;
    RotationType rotation_type = RotationType::kHadamard;
    std::uint64_t seed = 123456789ULL;
    std::size_t num_threads = 1;
  };

  explicit TurboQuantIndexV3(const Config& config)
      : dim_(config.dim),
        bitwidth_(config.bitwidth),
        mode_(config.mode),
        seed_(config.seed),
        num_threads_(std::max<std::size_t>(1, config.num_threads)) {
    if (dim_ == 0) throw std::invalid_argument("TurboQuantIndexV3: dim must be > 0");
    if (bitwidth_ == 0 || bitwidth_ > 9)
      throw std::invalid_argument("TurboQuantIndexV3: bitwidth must be in [1, 9]");
    padded_dim_ = next_pow2(dim_);
  }

  std::size_t dim() const noexcept { return dim_; }
  std::size_t ntotal() const noexcept { return ntotal_; }
  std::size_t bitwidth() const noexcept { return bitwidth_; }
  Mode mode() const noexcept { return mode_; }
  std::size_t num_threads() const noexcept { return num_threads_; }
  // Pointer to the data centroid (length = dim_, zero-padded to padded_dim_ internally)
  const float* centroid_ptr() const noexcept { return centroid_.data(); }

  // ------------------------------------------------------------------
  // train — computes centroid and builds LM (Gaussian) codebook
  // ------------------------------------------------------------------
  void train(std::size_t n, const float* x) {
    if (n == 0 || x == nullptr)
      throw std::invalid_argument("train: n must be > 0 and x must not be null");

    // --- 1. Compute centroid ---
    centroid_.assign(dim_, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
      const float* xi = x + i * dim_;
      for (std::size_t j = 0; j < dim_; ++j)
        centroid_[j] += xi[j];
    }
    const float inv_n = 1.0f / static_cast<float>(n);
    for (std::size_t j = 0; j < dim_; ++j)
      centroid_[j] *= inv_n;
    // Zero-pad centroid to padded_dim_ for internal use
    centroid_.resize(padded_dim_, 0.0f);

    // --- 2. Build codebook and thresholds from LM tables ---
    mse_bits_ = (mode_ == Mode::kInnerProduct) ? (bitwidth_ - 1) : bitwidth_;
    // Use padded_dim: codebook sigma = 1/sqrt(padded_dim) to match
    // coordinate distribution after zero-padded Hadamard rotation.
    build_codebook_and_thresholds(mse_bits_, padded_dim_);
    codebook_size_ = codebook_.size();

    use_nibble_path_ = (bitwidth_ == 4);
    if (use_nibble_path_) {
      nibble_block_stride_ = aligned_bytes(padded_dim_ * kBlockSize);
    } else {
      combined_code_sign_ = (mode_ == Mode::kInnerProduct && mse_bits_ >= 1 && mse_bits_ <= 3);
      storage_bits_ = combined_code_sign_ ? (mse_bits_ + 1) : mse_bits_;
      if (storage_bits_ >= 1 && storage_bits_ <= 4)
        lut_stride_ = 16;
      else if (storage_bits_ == 5)
        lut_stride_ = 32;
      else
        lut_stride_ = codebook_size_;
      byte_code_block_stride_ = aligned_bytes(padded_dim_ * kBlockSize);
      sign_block_stride_ =
          (mode_ == Mode::kInnerProduct && !combined_code_sign_) ? aligned_bytes(padded_dim_ * 2) : 0;
    }

    max_centroid_abs_ = 0.0f;
    for (std::size_t c = 0; c < codebook_size_; ++c)
      max_centroid_abs_ = std::max(max_centroid_abs_, std::abs(codebook_[c]));

    rotation_.generate(padded_dim_, seed_);
    if (mode_ == Mode::kInnerProduct)
      qjl_.generate(padded_dim_, seed_ ^ 0x9e3779b97f4a7c15ULL);

    nibbles_.clear();
    byte_codes_.clear();
    packed_signs_.clear();
    gammas_.clear();
    norms_.clear();
    norm_squares_.clear();
    residual_scales_.clear();
    ntotal_ = 0;
    trained_ = true;
  }

  // ------------------------------------------------------------------
  // add — centre-normalise then encode
  // ------------------------------------------------------------------
  void add(std::size_t n, const float* x) {
    require_trained();
    if (x == nullptr && n != 0) throw std::invalid_argument("add: x must not be null");

    const std::size_t old_total = ntotal_;
    ntotal_ += n;
    const std::size_t new_blocks = ceil_div(ntotal_, kBlockSize);

    if (use_nibble_path_) {
      nibbles_.resize(new_blocks * nibble_block_stride_, 0);
      gammas_.resize(new_blocks * kBlockSize, 0.0f);
    } else {
      byte_codes_.resize(new_blocks * byte_code_block_stride_, 0);
      if (sign_block_stride_ != 0) packed_signs_.resize(new_blocks * sign_block_stride_, 0);
      residual_scales_.resize(new_blocks * kBlockSize, 0.0f);
    }
    norms_.resize(new_blocks * kBlockSize, 0.0f);
    norm_squares_.resize(new_blocks * kBlockSize, 0.0f);

    const std::size_t start_block = old_total / kBlockSize;
    parallel_for(start_block, new_blocks, [&](std::size_t b0, std::size_t b1) {
      std::vector<float> centered(padded_dim_, 0.0f);
      std::vector<float> unit(padded_dim_, 0.0f);
      std::vector<float> rotated(padded_dim_);
      std::vector<float> residual(padded_dim_);
      std::vector<float> projected(mode_ == Mode::kInnerProduct ? padded_dim_ : 0);
      std::vector<float> work(padded_dim_);
      std::vector<std::uint32_t> codes(padded_dim_);

      for (std::size_t bi = b0; bi < b1; ++bi) {
        const std::size_t block_begin = bi * kBlockSize;
        const std::size_t block_end = std::min(block_begin + kBlockSize, ntotal_);
        const std::size_t first_new = std::max(block_begin, old_total);
        if (first_new >= block_end) continue;

        for (std::size_t gi = first_new; gi < block_end; ++gi) {
          const std::size_t li = gi - old_total;
          const std::size_t lane = gi - block_begin;
          const float* src = x + li * dim_;

          // Centre: x_c = x_r - c
          for (std::size_t j = 0; j < dim_; ++j)
            centered[j] = src[j] - centroid_[j];

          // Centred norm stored as norms_ (|x_c| not |x_r|)
          const float norm = l2_norm(centered.data(), dim_);
          norms_[gi] = norm;
          norm_squares_[gi] = norm * norm;

          if (norm == 0.0f) {
            if (use_nibble_path_) gammas_[gi] = 0.0f;
            else residual_scales_[gi] = 0.0f;
            continue;
          }

          const float inv_norm = 1.0f / norm;
          for (std::size_t j = 0; j < dim_; ++j) unit[j] = centered[j] * inv_norm;

          rotation_.forward(unit.data(), rotated.data(), work.data());
          encode_rotated(rotated.data(), codes.data());

          if (mode_ != Mode::kInnerProduct) {
            if (use_nibble_path_) {
              std::uint8_t* nib = nibble_dim(bi, 0);
              for (std::size_t j = 0; j < padded_dim_; ++j)
                nib[j * kBlockSize + lane] = static_cast<std::uint8_t>(codes[j]);
              gammas_[gi] = 0.0f;
            } else {
              std::uint8_t* bc = byte_code_dim(bi, 0);
              for (std::size_t j = 0; j < padded_dim_; ++j)
                bc[j * kBlockSize + lane] = static_cast<std::uint8_t>(codes[j]);
            }
            continue;
          }

          // IP mode: QJL residual on T(x_unit)
          float gamma = build_rotated_residual(rotated.data(), codes.data(), residual.data());
          qjl_.forward(residual.data(), projected.data(), work.data());

          if (use_nibble_path_) {
            gammas_[gi] = gamma;
            std::uint8_t* nib = nibble_dim(bi, 0);
            for (std::size_t j = 0; j < padded_dim_; ++j) {
              std::uint8_t sign_bit = (gamma > 0.0f && projected[j] >= 0.0f) ? 1U : 0U;
              nib[j * kBlockSize + lane] = static_cast<std::uint8_t>((sign_bit << 3) | codes[j]);
            }
          } else {
            residual_scales_[gi] =
                norm * gamma * (std::sqrt(kPi * 0.5f) / static_cast<float>(padded_dim_));
            if (combined_code_sign_) {
              std::uint8_t* bc = byte_code_dim(bi, 0);
              for (std::size_t j = 0; j < padded_dim_; ++j) {
                std::uint32_t sign_bit = (gamma > 0.0f && projected[j] >= 0.0f) ? 1U : 0U;
                bc[j * kBlockSize + lane] = static_cast<std::uint8_t>((sign_bit << mse_bits_) | codes[j]);
              }
            } else {
              std::uint8_t* bc = byte_code_dim(bi, 0);
              for (std::size_t j = 0; j < padded_dim_; ++j)
                bc[j * kBlockSize + lane] = static_cast<std::uint8_t>(codes[j]);
              for (std::size_t j = 0; j < padded_dim_; ++j) {
                std::uint8_t sign_bit = (gamma > 0.0f && projected[j] >= 0.0f) ? 1U : 0U;
                std::uint8_t* sp = sign_dim(bi, j);
                std::uint16_t w;
                std::memcpy(&w, sp, 2);
                w = static_cast<std::uint16_t>((w & ~(1U << lane)) |
                    (static_cast<std::uint16_t>(sign_bit) << lane));
                std::memcpy(sp, &w, 2);
              }
            }
          }
        }
      }
    });
  }

  // ------------------------------------------------------------------
  // reconstruct — decode quantized codes back to float vectors
  //   out: output array of shape (n, dim), caller-allocated
  // The Hadamard rotation is self-inverse (R(R(x)) = x), so applying
  // rotation_.forward() on decoded rotated-unit vectors recovers unit vectors.
  // ------------------------------------------------------------------
  void reconstruct(std::size_t n, float* out) const {
    require_trained();
    if (n > ntotal_) n = ntotal_;

    parallel_for(0, n, [&](std::size_t begin, std::size_t end) {
      std::vector<float> rotated(padded_dim_);
      std::vector<float> unit(padded_dim_);
      std::vector<float> work(padded_dim_);

      for (std::size_t i = begin; i < end; ++i) {
        const std::size_t bi   = i / kBlockSize;
        const std::size_t lane = i % kBlockSize;

        // --- decode codes into the approximate rotated-unit vector ---
        if (use_nibble_path_) {
          const std::uint8_t* nibs = nibble_dim(bi, 0);
          for (std::size_t j = 0; j < padded_dim_; ++j) {
            std::uint8_t nib = nibs[j * kBlockSize + lane];
            std::uint32_t code = (mode_ == Mode::kInnerProduct)
                                   ? (nib & 0x7u)   // 3 mse_bits + 1 sign bit
                                   : (nib & 0xFu);  // 4 mse_bits
            rotated[j] = (code < codebook_size_) ? codebook_[code] : 0.0f;
          }
        } else {
          const std::uint8_t* bc = byte_code_dim(bi, 0);
          for (std::size_t j = 0; j < padded_dim_; ++j) {
            std::uint8_t raw = bc[j * kBlockSize + lane];
            std::uint32_t code = combined_code_sign_
                                   ? (raw & ((1u << mse_bits_) - 1u))
                                   : static_cast<std::uint32_t>(raw);
            rotated[j] = (code < codebook_size_) ? codebook_[code] : 0.0f;
          }
        }

        // --- inverse rotation: R(R(x)) = x, so apply forward again ---
        rotation_.forward(rotated.data(), unit.data(), work.data());

        // --- scale by centred norm and add centroid ---
        const float norm    = norms_[i];
        float*      out_ptr = out + i * dim_;
        for (std::size_t j = 0; j < dim_; ++j)
          out_ptr[j] = unit[j] * norm + centroid_[j];
      }
    });
  }

  // ------------------------------------------------------------------
  // query
  // ------------------------------------------------------------------
  void query(std::size_t nq, const float* x, std::size_t k,
             float* distances, idx_t* labels) const {
    query(nq, x, k, SearchMetric::kL2, distances, labels);
  }

  void query(std::size_t nq, const float* x, std::size_t k,
             SearchMetric metric, float* distances, idx_t* labels) const {
    require_trained();
    if (x == nullptr && nq != 0) throw std::invalid_argument("query: x must not be null");
    if (k > ntotal_) k = ntotal_;

    if (use_nibble_path_) {
      if (mode_ == Mode::kInnerProduct) {
        if (metric == SearchMetric::kL2) query_nibble_impl<true, true>(nq, x, k, distances, labels);
        else                              query_nibble_impl<true, false>(nq, x, k, distances, labels);
      } else {
        if (metric == SearchMetric::kL2) query_nibble_impl<false, true>(nq, x, k, distances, labels);
        else                              query_nibble_impl<false, false>(nq, x, k, distances, labels);
      }
    } else {
      if (mode_ == Mode::kInnerProduct) {
        if (metric == SearchMetric::kL2) query_generic_impl<true, true>(nq, x, k, distances, labels);
        else                              query_generic_impl<true, false>(nq, x, k, distances, labels);
      } else {
        if (metric == SearchMetric::kL2) query_generic_impl<false, true>(nq, x, k, distances, labels);
        else                              query_generic_impl<false, false>(nq, x, k, distances, labels);
      }
    }
  }

 private:
  static constexpr float kPi = 3.14159265358979323846f;
  static constexpr float kQjlScale = 1.2533141373155001f;
  static constexpr std::size_t kBlockSize = 16;
  static constexpr std::size_t kRowAlignment = 64;
  static constexpr std::size_t kDrainInterval = 128;

  std::size_t dim_ = 0;
  std::size_t padded_dim_ = 0;
  std::size_t bitwidth_ = 0;
  std::size_t mse_bits_ = 0;
  Mode mode_ = Mode::kInnerProduct;
  std::uint64_t seed_ = 0;
  std::size_t num_threads_ = 1;
  std::size_t ntotal_ = 0;
  std::size_t codebook_size_ = 0;
  float max_centroid_abs_ = 0.0f;
  bool trained_ = false;

  std::vector<float> centroid_;  // V3: data centroid (padded_dim_ floats, zero-padded)

  bool use_nibble_path_ = false;
  std::size_t nibble_block_stride_ = 0;
  std::vector<std::uint8_t> nibbles_;
  std::vector<float> gammas_;

  bool combined_code_sign_ = false;
  std::size_t storage_bits_ = 0;
  std::size_t lut_stride_ = 0;
  std::size_t byte_code_block_stride_ = 0;
  std::size_t sign_block_stride_ = 0;
  std::vector<std::uint8_t> byte_codes_;
  std::vector<std::uint8_t> packed_signs_;
  std::vector<float> residual_scales_;

  std::vector<float> codebook_;
  std::vector<float> thresholds_;
  std::vector<float> norms_;        // |x_r - c|  (centred norms)
  std::vector<float> norm_squares_; // |x_r - c|^2

  // ------------------------------------------------------------------
  // Hardcoded Lloyd-Max codebooks for N(0,1)  (bits 1-8)
  // These are scaled at load time by sigma = 1/sqrt(dim) to match the
  // coordinate distribution of unit-sphere vectors after Hadamard rotation.
  // ------------------------------------------------------------------
  static const float* lm_boundaries(std::size_t bits) {
    // lm_boundaries_[b] has 2^b - 1 entries (inner boundaries, not ±∞)
    static const float b1[]  = {0.f};
    static const float b2[]  = {-0.9816f, 0.f, 0.9816f};
    static const float b3[]  = {-1.7479, -1.05  , -0.5005,  0.    ,  0.5005,  1.05  ,  1.7479};
    static const float b4[]  = {-2.4008, -1.8435, -1.4371, -1.0993, -0.7995, -0.5224, -0.2582,  0.    ,
  0.2582,  0.5224,  0.7995,  1.0993,  1.4371,  1.8435,  2.4008};
    static const float b5[]  = {
    -2.976 , -2.5045, -2.1733, -1.908 , -1.6818, -1.4813, -1.2991, -1.1303,
 -0.9717, -0.8209, -0.6761, -0.5358, -0.3991, -0.2647, -0.132 ,  0.    ,
  0.132 ,  0.2647,  0.3991,  0.5358,  0.6761,  0.8209,  0.9717,  1.1303,
  1.2991,  1.4813,  1.6818,  1.908 ,  2.1733,  2.5045,  2.976};
    static const float b6[]  = {
    -3.5169, -3.1058, -2.8233, -2.6015, -2.4159, -2.2544, -2.1103, -1.9792,
 -1.8584, -1.7457, -1.6397, -1.5392, -1.4433, -1.3515, -1.263 , -1.1774,
 -1.0943, -1.0134, -0.9344, -0.8571, -0.7812, -0.7066, -0.6331, -0.5606,
 -0.4889, -0.4178, -0.3473, -0.2773, -0.2077, -0.1383, -0.0691,  0.    ,
  0.0691,  0.1383,  0.2077,  0.2773,  0.3473,  0.4178,  0.4889,  0.5606,
  0.6331,  0.7066,  0.7812,  0.8571,  0.9344,  1.0134,  1.0943,  1.1774,
  1.263 ,  1.3515,  1.4433,  1.5392,  1.6397,  1.7457,  1.8584,  1.9792,
  2.1103,  2.2544,  2.4159,  2.6015,  2.8233,  3.1058,  3.5169};
    static const float b7[]  = {
    -4.0871, -3.7257, -3.4813, -3.292 , -3.1356, -3.0011, -2.8824, -2.7756,
 -2.6782, -2.5883, -2.5045, -2.4259, -2.3516, -2.2811, -2.2138, -2.1494,
 -2.0874, -2.0276, -1.9697, -1.9136, -1.859 , -1.8057, -1.7538, -1.7029,
 -1.6531, -1.6042, -1.5561, -1.5087, -1.4621, -1.4161, -1.3706, -1.3256,
 -1.2812, -1.2371, -1.1934, -1.1501, -1.1071, -1.0644, -1.0219, -0.9797,
 -0.9377, -0.8959, -0.8542, -0.8128, -0.7714, -0.7302, -0.6891, -0.6482,
 -0.6073, -0.5665, -0.5257, -0.4851, -0.4445, -0.4039, -0.3634, -0.3229,
 -0.2825, -0.2421, -0.2017, -0.1613, -0.121 , -0.0807, -0.0403,  0.    ,
  0.0403,  0.0807,  0.121 ,  0.1613,  0.2017,  0.2421,  0.2825,  0.3229,
  0.3634,  0.4039,  0.4445,  0.4851,  0.5257,  0.5665,  0.6073,  0.6482,
  0.6891,  0.7302,  0.7714,  0.8128,  0.8542,  0.8959,  0.9377,  0.9797,
  1.0219,  1.0644,  1.1071,  1.1501,  1.1934,  1.2371,  1.2812,  1.3256,
  1.3706,  1.4161,  1.4621,  1.5087,  1.5561,  1.6042,  1.6531,  1.7029,
  1.7538,  1.8057,  1.859 ,  1.9136,  1.9697,  2.0276,  2.0874,  2.1494,
  2.2138,  2.2811,  2.3516,  2.4259,  2.5045,  2.5883,  2.6782,  2.7756,
  2.8824,  3.0011,  3.1356,  3.292 ,  3.4813,  3.7257,  4.0871};
    static const float b8[]  = {
    -4.502 , -4.1704, -3.9485, -3.7782, -3.6388, -3.52  , -3.4161, -3.3236,
 -3.24  , -3.1636, -3.0933, -3.028 , -2.967 , -2.9098, -2.8558, -2.8047,
 -2.7562, -2.7101, -2.6659, -2.6237, -2.5831, -2.5441, -2.5065, -2.4701,
 -2.435 , -2.4009, -2.3679, -2.3357, -2.3044, -2.2738, -2.244 , -2.2148,
 -2.1862, -2.1582, -2.1307, -2.1037, -2.0771, -2.0508, -2.025 , -1.9995,
 -1.9742, -1.9493, -1.9246, -1.9002, -1.8759, -1.8519, -1.828 , -1.8042,
 -1.7806, -1.7572, -1.7338, -1.7106, -1.6875, -1.6644, -1.6414, -1.6185,
 -1.5956, -1.5728, -1.55  , -1.5273, -1.5046, -1.482 , -1.4593, -1.4367,
 -1.4142, -1.3916, -1.3691, -1.3465, -1.324 , -1.3015, -1.279 , -1.2565,
 -1.2341, -1.2116, -1.1891, -1.1667, -1.1442, -1.1218, -1.0993, -1.0769,
 -1.0544, -1.032 , -1.0095, -0.9871, -0.9646, -0.9422, -0.9198, -0.8973,
 -0.8749, -0.8525, -0.83  , -0.8076, -0.7852, -0.7627, -0.7403, -0.7179,
 -0.6954, -0.673 , -0.6506, -0.6281, -0.6057, -0.5833, -0.5608, -0.5384,
 -0.516 , -0.4935, -0.4711, -0.4487, -0.4262, -0.4038, -0.3814, -0.3589,
 -0.3365, -0.3141, -0.2916, -0.2692, -0.2468, -0.2243, -0.2019, -0.1795,
 -0.157 , -0.1346, -0.1122, -0.0897, -0.0673, -0.0449, -0.0224,  0.    ,
  0.0224,  0.0449,  0.0673,  0.0897,  0.1122,  0.1346,  0.157 ,  0.1795,
  0.2019,  0.2243,  0.2468,  0.2692,  0.2916,  0.3141,  0.3365,  0.3589,
  0.3814,  0.4038,  0.4262,  0.4487,  0.4711,  0.4935,  0.516 ,  0.5384,
  0.5608,  0.5833,  0.6057,  0.6281,  0.6506,  0.673 ,  0.6954,  0.7179,
  0.7403,  0.7627,  0.7852,  0.8076,  0.83  ,  0.8525,  0.8749,  0.8973,
  0.9198,  0.9422,  0.9646,  0.9871,  1.0095,  1.032 ,  1.0544,  1.0769,
  1.0993,  1.1218,  1.1442,  1.1667,  1.1891,  1.2116,  1.2341,  1.2565,
  1.279 ,  1.3015,  1.324 ,  1.3465,  1.3691,  1.3916,  1.4142,  1.4367,
  1.4593,  1.482 ,  1.5046,  1.5273,  1.55  ,  1.5728,  1.5956,  1.6185,
  1.6414,  1.6644,  1.6875,  1.7106,  1.7338,  1.7572,  1.7806,  1.8042,
  1.828 ,  1.8519,  1.8759,  1.9002,  1.9246,  1.9493,  1.9742,  1.9995,
  2.025 ,  2.0508,  2.0771,  2.1037,  2.1307,  2.1582,  2.1862,  2.2148,
  2.244 ,  2.2738,  2.3044,  2.3357,  2.3679,  2.4009,  2.435 ,  2.4701,
  2.5065,  2.5441,  2.5831,  2.6237,  2.6659,  2.7101,  2.7562,  2.8047,
  2.8558,  2.9098,  2.967 ,  3.028 ,  3.0933,  3.1636,  3.24  ,  3.3236,
  3.4161,  3.52  ,  3.6388,  3.7782,  3.9485,  4.1704,  4.502};
    switch (bits) {
      case 1: return b1;
      case 2: return b2;
      case 3: return b3;
      case 4: return b4;
      case 5: return b5;
      case 6: return b6;
      case 7: return b7;
      case 8: return b8;
      default: return nullptr;
    }
  }

  static const float* lm_centroids(std::size_t bits) {
    static const float c1[]  = {-0.7979f, 0.7979f};
    static const float c2[]  = {-1.5104, -0.4528,  0.4528,  1.5104};
    static const float c3[]  = {-2.1519, -1.3439, -0.756 , -0.2451,  0.2451,  0.756 ,  1.3439,  2.1519};
    static const float c4[]  = {
    -2.7326, -2.069 , -1.618 , -1.2562, -0.9423, -0.6568, -0.388 , -0.1284,
  0.1284,  0.388 ,  0.6568,  0.9423,  1.2562,  1.618 ,  2.069 ,  2.7326};
    static const float c5[]  = {
    -3.2608, -2.6912, -2.3178, -2.0288, -1.7873, -1.5763, -1.3864, -1.2118,
 -1.0488, -0.8946, -0.7472, -0.605 , -0.4667, -0.3314, -0.1981, -0.0659,
  0.0659,  0.1981,  0.3314,  0.4667,  0.605 ,  0.7472,  0.8946,  1.0488,
  1.2118,  1.3864,  1.5763,  1.7873,  2.0288,  2.3178,  2.6912,  3.2608};
    static const float c6[]  = {
    -3.7674, -3.2664, -2.9452, -2.7015, -2.5016, -2.3302, -2.1786, -2.0419,
 -1.9165, -1.8002, -1.6912, -1.5882, -1.4902, -1.3965, -1.3064, -1.2195,
 -1.1352, -1.0533, -0.9735, -0.8954, -0.8188, -0.7437, -0.6696, -0.5967,
 -0.5245, -0.4532, -0.3824, -0.3122, -0.2424, -0.1729, -0.1037, -0.0345,
  0.0345,  0.1037,  0.1729,  0.2424,  0.3122,  0.3824,  0.4532,  0.5245,
  0.5967,  0.6696,  0.7437,  0.8188,  0.8954,  0.9735,  1.0533,  1.1352,
  1.2195,  1.3064,  1.3965,  1.4902,  1.5882,  1.6912,  1.8002,  1.9165,
  2.0419,  2.1786,  2.3302,  2.5016,  2.7015,  2.9452,  3.2664,  3.7674};
    static const float c7[]  = {
    -4.3088, -3.8654, -3.5859, -3.3767, -3.2074, -3.0638, -2.9384, -2.8264,
 -2.7248, -2.6315, -2.545 , -2.464 , -2.3878, -2.3155, -2.2467, -2.1809,
 -2.1178, -2.057 , -1.9982, -1.9412, -1.8859, -1.832 , -1.7795, -1.7281,
 -1.6778, -1.6284, -1.5799, -1.5322, -1.4853, -1.4389, -1.3932, -1.348 ,
 -1.3033, -1.259 , -1.2152, -1.1717, -1.1285, -1.0857, -1.0431, -1.0008,
 -0.9586, -0.9167, -0.875 , -0.8335, -0.7921, -0.7508, -0.7097, -0.6686,
 -0.6277, -0.5868, -0.5461, -0.5054, -0.4647, -0.4242, -0.3836, -0.3432,
 -0.3027, -0.2623, -0.2219, -0.1815, -0.1412, -0.1008, -0.0605, -0.0202,
  0.0202,  0.0605,  0.1008,  0.1412,  0.1815,  0.2219,  0.2623,  0.3027,
  0.3432,  0.3836,  0.4242,  0.4647,  0.5054,  0.5461,  0.5868,  0.6277,
  0.6686,  0.7097,  0.7508,  0.7921,  0.8335,  0.875 ,  0.9167,  0.9586,
  1.0008,  1.0431,  1.0857,  1.1285,  1.1717,  1.2152,  1.259 ,  1.3033,
  1.348 ,  1.3932,  1.4389,  1.4853,  1.5322,  1.5799,  1.6284,  1.6778,
  1.7281,  1.7795,  1.832 ,  1.8859,  1.9412,  1.9982,  2.057 ,  2.1178,
  2.1809,  2.2467,  2.3155,  2.3878,  2.464 ,  2.545 ,  2.6315,  2.7248,
  2.8264,  2.9384,  3.0638,  3.2074,  3.3767,  3.5859,  3.8654,  4.3088};
    static const float c8[]  = {
    -4.7062, -4.2978, -4.043 , -3.854 , -3.7025, -3.5751, -3.4649, -3.3674,
 -3.2798, -3.2002, -3.1271, -3.0595, -2.9965, -2.9375, -2.882 , -2.8296,
 -2.7799, -2.7326, -2.6875, -2.6444, -2.603 , -2.5632, -2.5249, -2.488 ,
 -2.4523, -2.4177, -2.3842, -2.3516, -2.3198, -2.2889, -2.2587, -2.2292,
 -2.2004, -2.1721, -2.1443, -2.1171, -2.0903, -2.0639, -2.0378, -2.0121,
 -1.9868, -1.9617, -1.9369, -1.9123, -1.888 , -1.8638, -1.8399, -1.8161,
 -1.7924, -1.7689, -1.7455, -1.7222, -1.699 , -1.6759, -1.6529, -1.6299,
 -1.607 , -1.5842, -1.5614, -1.5387, -1.5159, -1.4933, -1.4706, -1.448 ,
 -1.4254, -1.4029, -1.3803, -1.3578, -1.3353, -1.3128, -1.2903, -1.2678,
 -1.2453, -1.2228, -1.2004, -1.1779, -1.1554, -1.133 , -1.1105, -1.0881,
 -1.0656, -1.0432, -1.0207, -0.9983, -0.9759, -0.9534, -0.931 , -0.9086,
 -0.8861, -0.8637, -0.8412, -0.8188, -0.7964, -0.7739, -0.7515, -0.7291,
 -0.7066, -0.6842, -0.6618, -0.6393, -0.6169, -0.5945, -0.572 , -0.5496,
 -0.5272, -0.5047, -0.4823, -0.4599, -0.4374, -0.415 , -0.3926, -0.3701,
 -0.3477, -0.3253, -0.3028, -0.2804, -0.258 , -0.2355, -0.2131, -0.1907,
 -0.1682, -0.1458, -0.1234, -0.1009, -0.0785, -0.0561, -0.0336, -0.0112,
  0.0112,  0.0336,  0.0561,  0.0785,  0.1009,  0.1234,  0.1458,  0.1682,
  0.1907,  0.2131,  0.2355,  0.258 ,  0.2804,  0.3028,  0.3253,  0.3477,
  0.3701,  0.3926,  0.415 ,  0.4374,  0.4599,  0.4823,  0.5047,  0.5272,
  0.5496,  0.572 ,  0.5945,  0.6169,  0.6393,  0.6618,  0.6842,  0.7066,
  0.7291,  0.7515,  0.7739,  0.7964,  0.8188,  0.8412,  0.8637,  0.8861,
  0.9086,  0.931 ,  0.9534,  0.9759,  0.9983,  1.0207,  1.0432,  1.0656,
  1.0881,  1.1105,  1.133 ,  1.1554,  1.1779,  1.2004,  1.2228,  1.2453,
  1.2678,  1.2903,  1.3128,  1.3353,  1.3578,  1.3803,  1.4029,  1.4254,
  1.448 ,  1.4706,  1.4933,  1.5159,  1.5387,  1.5614,  1.5842,  1.607 ,
  1.6299,  1.6529,  1.6759,  1.699 ,  1.7222,  1.7455,  1.7689,  1.7924,
  1.8161,  1.8399,  1.8638,  1.888 ,  1.9123,  1.9369,  1.9617,  1.9868,
  2.0121,  2.0378,  2.0639,  2.0903,  2.1171,  2.1443,  2.1721,  2.2004,
  2.2292,  2.2587,  2.2889,  2.3198,  2.3516,  2.3842,  2.4177,  2.4523,
  2.488 ,  2.5249,  2.5632,  2.603 ,  2.6444,  2.6875,  2.7326,  2.7799,
  2.8296,  2.882 ,  2.9375,  2.9965,  3.0595,  3.1271,  3.2002,  3.2798,
  3.3674,  3.4649,  3.5751,  3.7025,  3.854 ,  4.043 ,  4.2978,  4.7062};
    switch (bits) {
      case 1: return c1;
      case 2: return c2;
      case 3: return c3;
      case 4: return c4;
      case 5: return c5;
      case 6: return c6;
      case 7: return c7;
      case 8: return c8;
      default: return nullptr;
    }
  }

  // Build codebook: for bits 1-8 use scaled LM Gaussian tables,
  // for bits 9 fall back to Beta codebook solver.
  std::vector<float> build_lm_codebook(std::size_t bits, std::size_t d) const {
    if (bits == 0) return {};
    const float* cptr = lm_centroids(bits);
    if (cptr == nullptr) {
      // Fallback: Beta codebook (identical to V2's solve_scalar_codebook)
      return solve_beta_codebook(bits, d);
    }
    const std::size_t levels = std::size_t{1} << bits;
    const float sigma = 1.0f / std::sqrt(static_cast<float>(d));
    std::vector<float> out(levels);
    for (std::size_t i = 0; i < levels; ++i)
      out[i] = cptr[i] * sigma;
    return out;
  }

  // Lloyd-Max on Beta distribution (fallback, identical to V2)
  static std::vector<float> solve_beta_codebook(std::size_t bits, std::size_t d) {
    const std::size_t levels = std::size_t{1} << bits;
    const std::size_t grid_size = 32768;
    const double step = 2.0 / static_cast<double>(grid_size);
    const double alpha = 0.5 * (static_cast<double>(d) - 3.0);
    std::vector<double> grid(grid_size), weight(grid_size);
    std::vector<double> prefix_weight(grid_size + 1, 0.0);
    for (std::size_t i = 0; i < grid_size; ++i) {
      const double x = -1.0 + (static_cast<double>(i) + 0.5) * step;
      grid[i] = x;
      weight[i] = std::exp(alpha * std::log1p(-(x * x)));
      prefix_weight[i + 1] = prefix_weight[i] + weight[i];
    }
    std::vector<double> centroids(levels, 0.0);
    const double total_weight = prefix_weight.back();
    for (std::size_t l = 0; l < levels; ++l) {
      const double target = (static_cast<double>(l) + 0.5) * total_weight / static_cast<double>(levels);
      auto it = std::lower_bound(prefix_weight.begin(), prefix_weight.end(), target);
      auto idx = std::min<std::size_t>(
          static_cast<std::size_t>(std::distance(prefix_weight.begin(), it)), grid_size) - 1;
      centroids[l] = grid[idx];
    }
    std::vector<double> boundaries(levels + 1), masses(levels), moments(levels);
    for (std::size_t iter = 0; iter < 80; ++iter) {
      boundaries.front() = -1.0; boundaries.back() = 1.0;
      for (std::size_t i = 0; i + 1 < levels; ++i)
        boundaries[i + 1] = 0.5 * (centroids[i] + centroids[i + 1]);
      std::fill(masses.begin(), masses.end(), 0.0);
      std::fill(moments.begin(), moments.end(), 0.0);
      std::size_t bucket = 0;
      for (std::size_t i = 0; i < grid_size; ++i) {
        while (bucket + 1 < levels && grid[i] > boundaries[bucket + 1]) ++bucket;
        masses[bucket] += weight[i];
        moments[bucket] += weight[i] * grid[i];
      }
      double max_delta = 0.0;
      for (std::size_t i = 0; i < levels; ++i) {
        double updated = 0.5 * (boundaries[i] + boundaries[i + 1]);
        if (masses[i] > 0.0) updated = moments[i] / masses[i];
        updated = std::clamp(updated, boundaries[i], boundaries[i + 1]);
        max_delta = std::max(max_delta, std::abs(updated - centroids[i]));
        centroids[i] = updated;
      }
      if (max_delta < 1e-8) break;
    }
    std::vector<float> out(levels);
    for (std::size_t i = 0; i < levels; ++i) out[i] = static_cast<float>(centroids[i]);
    return out;
  }

  void rebuild_lm_thresholds(std::size_t bits, std::size_t d) {
    // Use hardcoded LM boundaries scaled by sigma, or midpoints as fallback
    const float* bptr = lm_boundaries(bits);
    const float sigma = 1.0f / std::sqrt(static_cast<float>(d));
    if (bptr != nullptr) {
      const std::size_t nb = (std::size_t{1} << bits) - 1;
      thresholds_.resize(nb);
      for (std::size_t i = 0; i < nb; ++i)
        thresholds_[i] = bptr[i] * sigma;
    } else {
      rebuild_thresholds();  // midpoint fallback
    }
  }

  void rebuild_thresholds() {
    thresholds_.clear();
    if (!codebook_.empty()) {
      thresholds_.reserve(codebook_.size() - 1);
      for (std::size_t i = 0; i + 1 < codebook_.size(); ++i)
        thresholds_.push_back(0.5f * (codebook_[i] + codebook_[i + 1]));
    }
  }

  // Override to use LM thresholds in train()
  void build_codebook_and_thresholds(std::size_t bits, std::size_t d) {
    codebook_ = build_lm_codebook(bits, d);
    const float* bptr = lm_boundaries(bits);
    const float sigma = 1.0f / std::sqrt(static_cast<float>(d));
    if (bptr != nullptr) {
      const std::size_t nb = (std::size_t{1} << bits) - 1;
      thresholds_.resize(nb);
      for (std::size_t i = 0; i < nb; ++i)
        thresholds_[i] = bptr[i] * sigma;
    } else {
      rebuild_thresholds();
    }
  }

  // ------------------------------------------------------------------
  // Structured Hadamard Transform
  //   Random signs + FWHT on padded_dim dimensions.
  //   Input vectors shorter than padded_dim are zero-padded by the caller.
  //   All padded_dim output components are kept (no truncation) to avoid
  //   the energy loss that killed quality on non-power-of-2 dims.
  // ------------------------------------------------------------------
  struct StructuredTransform {
    std::size_t padded_dim = 0;
    std::vector<float> signs;  // [padded_dim], +1 or -1

    void generate(std::size_t pd, std::uint64_t seed) {
      padded_dim = pd;
      signs.resize(padded_dim);
      std::uint64_t state = seed;
      for (std::size_t i = 0; i < padded_dim; ++i)
        signs[i] = (splitmix64(state) & 1) ? 1.0f : -1.0f;
    }

    // input/output are padded_dim floats. work is padded_dim scratch space.
    void forward(const float* input, float* output, float* work) const {
      for (std::size_t i = 0; i < padded_dim; ++i)
        work[i] = signs[i] * input[i];
      wht_butterfly_inplace(work, padded_dim);
      const float scale = 1.0f / std::sqrt(static_cast<float>(padded_dim));
      for (std::size_t i = 0; i < padded_dim; ++i)
        output[i] = work[i] * scale;
    }
  };

  StructuredTransform rotation_;
  StructuredTransform qjl_;

  static std::uint64_t splitmix64(std::uint64_t& state) {
    state += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }

  static std::size_t next_pow2(std::size_t v) {
    if (v == 0) return 1;
    v--;
    v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16; v |= v >> 32;
    return v + 1;
  }

  static std::size_t aligned_bytes(std::size_t bytes) {
    return bytes == 0 ? 0 : ((bytes + kRowAlignment - 1) / kRowAlignment) * kRowAlignment;
  }

  static std::size_t ceil_div(std::size_t a, std::size_t b) { return (a + b - 1) / b; }

  void require_trained() const {
    if (!trained_) throw std::logic_error("TurboQuantIndexV3: train() must be called first");
  }

  static void wht_butterfly_inplace(float* x, std::size_t d) {
    for (std::size_t step = 1; step < d; step <<= 1)
      for (std::size_t i = 0; i < d; i += step * 2)
        for (std::size_t j = i; j < i + step; ++j) {
          const float a = x[j], b = x[j + step];
          x[j] = a + b; x[j + step] = a - b;
        }
  }

  static float dot_product(const float* a, const float* b, std::size_t n) {
    __m512 acc = _mm512_setzero_ps();
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16)
      acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
    float sum = _mm512_reduce_add_ps(acc);
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
  }

  static float l2_norm(const float* x, std::size_t n) {
    return std::sqrt(dot_product(x, x, n));
  }

  std::uint32_t nearest_centroid(float value) const {
    auto it = std::upper_bound(thresholds_.begin(), thresholds_.end(), value);
    return static_cast<std::uint32_t>(std::distance(thresholds_.begin(), it));
  }

  void encode_rotated(const float* rotated, std::uint32_t* out) const {
    for (std::size_t j = 0; j < padded_dim_; ++j) out[j] = nearest_centroid(rotated[j]);
  }

  float build_rotated_residual(const float* rotated, const std::uint32_t* codes,
                                float* residual) const {
    float sum_sq = 0.0f;
    for (std::size_t j = 0; j < padded_dim_; ++j) {
      const float q = (mse_bits_ == 0) ? 0.0f : codebook_[codes[j]];
      const float d = rotated[j] - q;
      residual[j] = d;
      sum_sq += d * d;
    }
    return std::sqrt(sum_sq);
  }

  // ------------------------------------------------------------------
  // Storage accessors (identical to V2)
  // ------------------------------------------------------------------
  std::uint8_t* nibble_dim(std::size_t block, std::size_t dim_idx) {
    return nibbles_.data() + block * nibble_block_stride_ + dim_idx * kBlockSize;
  }
  const std::uint8_t* nibble_dim(std::size_t block, std::size_t dim_idx) const {
    return nibbles_.data() + block * nibble_block_stride_ + dim_idx * kBlockSize;
  }
  std::uint8_t* byte_code_dim(std::size_t block, std::size_t dim_idx) {
    return byte_codes_.data() + block * byte_code_block_stride_ + dim_idx * kBlockSize;
  }
  const std::uint8_t* byte_code_dim(std::size_t block, std::size_t dim_idx) const {
    return byte_codes_.data() + block * byte_code_block_stride_ + dim_idx * kBlockSize;
  }
  std::uint8_t* sign_dim(std::size_t block, std::size_t dim_idx) {
    return packed_signs_.data() + block * sign_block_stride_ + dim_idx * 2;
  }
  const std::uint8_t* sign_dim(std::size_t block, std::size_t dim_idx) const {
    return packed_signs_.data() + block * sign_block_stride_ + dim_idx * 2;
  }

  // ------------------------------------------------------------------
  // Int8 LUT16 building (identical to V2 — query is already normalised)
  // ------------------------------------------------------------------
  void build_lut16_int8(const float* rotated_query, const float* projected_query,
                        std::int8_t* base_i8, std::int8_t* qjl_i8,
                        float& base_scale, float& qjl_scale) const {
    const float max_q = [&] {
      float m = 0.0f;
      for (std::size_t d = 0; d < padded_dim_; ++d) m = std::max(m, std::abs(rotated_query[d]));
      return m;
    }();
    const float base_max = max_q * max_centroid_abs_;
    base_scale = (base_max > 1e-30f) ? (base_max / 127.0f) : 1.0f;
    const float base_inv = 1.0f / base_scale;

    if (mode_ == Mode::kInnerProduct) {
      for (std::size_t d = 0; d < padded_dim_; ++d) {
        const float q = rotated_query[d];
        for (std::size_t n = 0; n < 16; ++n)
          base_i8[d * 16 + n] = clamp_i8(std::lround(q * codebook_[n & 7u] * base_inv));
      }
    } else {
      for (std::size_t d = 0; d < padded_dim_; ++d) {
        const float q = rotated_query[d];
        for (std::size_t n = 0; n < 16; ++n)
          base_i8[d * 16 + n] = clamp_i8(std::lround(q * codebook_[n] * base_inv));
      }
    }

    if (projected_query && qjl_i8) {
      const float qjl_coeff = kQjlScale / static_cast<float>(padded_dim_);
      float max_proj = 0.0f;
      for (std::size_t d = 0; d < padded_dim_; ++d)
        max_proj = std::max(max_proj, std::abs(projected_query[d]));
      const float qjl_max = qjl_coeff * max_proj;
      qjl_scale = (qjl_max > 1e-30f) ? (qjl_max / 127.0f) : 1.0f;
      const float qjl_inv = 1.0f / qjl_scale;
      for (std::size_t d = 0; d < padded_dim_; ++d) {
        const float qjl_val = qjl_coeff * projected_query[d];
        for (std::size_t n = 0; n < 16; ++n) {
          const float sign = (n & 8u) ? 1.0f : -1.0f;
          qjl_i8[d * 16 + n] = clamp_i8(std::lround(sign * qjl_val * qjl_inv));
        }
      }
    } else {
      qjl_scale = 1.0f;
    }
  }

  static std::int8_t clamp_i8(long v) {
    return static_cast<std::int8_t>(std::max(-127L, std::min(127L, v)));
  }

  // ------------------------------------------------------------------
  // Block-16 int8 scoring (identical to V2)
  // ------------------------------------------------------------------
  void score_block16_int8(const std::uint8_t* nibbles,
                          const std::int8_t* base_i8,
                          const std::int8_t* qjl_i8,
                          const float* gammas,
                          float base_scale, float qjl_scale,
                          std::size_t block_size, float* scores) const {
    if (block_size == kBlockSize) {
      __m256i base_acc16 = _mm256_setzero_si256();
      __m512i base_acc32 = _mm512_setzero_si512();
      __m256i qjl_acc16 = _mm256_setzero_si256();
      __m512i qjl_acc32 = _mm512_setzero_si512();
      for (std::size_t d = 0; d < padded_dim_; ++d) {
        const __m128i nibs = _mm_loadu_si128(reinterpret_cast<const __m128i*>(nibbles + d * kBlockSize));
        const __m128i blut = _mm_loadu_si128(reinterpret_cast<const __m128i*>(base_i8 + d * 16));
        const __m128i bres = _mm_shuffle_epi8(blut, nibs);
        base_acc16 = _mm256_add_epi16(base_acc16, _mm256_cvtepi8_epi16(bres));
        if (qjl_i8) {
          const __m128i qlut = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qjl_i8 + d * 16));
          const __m128i qres = _mm_shuffle_epi8(qlut, nibs);
          qjl_acc16 = _mm256_add_epi16(qjl_acc16, _mm256_cvtepi8_epi16(qres));
        }
        if ((d & (kDrainInterval - 1)) == (kDrainInterval - 1)) {
          base_acc32 = _mm512_add_epi32(base_acc32, _mm512_cvtepi16_epi32(base_acc16));
          base_acc16 = _mm256_setzero_si256();
          if (qjl_i8) {
            qjl_acc32 = _mm512_add_epi32(qjl_acc32, _mm512_cvtepi16_epi32(qjl_acc16));
            qjl_acc16 = _mm256_setzero_si256();
          }
        }
      }
      base_acc32 = _mm512_add_epi32(base_acc32, _mm512_cvtepi16_epi32(base_acc16));
      __m512 base_f = _mm512_mul_ps(_mm512_cvtepi32_ps(base_acc32), _mm512_set1_ps(base_scale));
      if (qjl_i8) {
        qjl_acc32 = _mm512_add_epi32(qjl_acc32, _mm512_cvtepi16_epi32(qjl_acc16));
        __m512 qjl_f = _mm512_mul_ps(_mm512_cvtepi32_ps(qjl_acc32), _mm512_set1_ps(qjl_scale));
        __m512 gamma_v = _mm512_loadu_ps(gammas);
        _mm512_storeu_ps(scores, _mm512_fmadd_ps(gamma_v, qjl_f, base_f));
      } else {
        _mm512_storeu_ps(scores, base_f);
      }
      return;
    }
    for (std::size_t c = 0; c < block_size; ++c) {
      float base_sum = 0.0f, qjl_sum = 0.0f;
      for (std::size_t d = 0; d < padded_dim_; ++d) {
        std::uint8_t nib = nibbles[d * kBlockSize + c];
        base_sum += static_cast<float>(base_i8[d * 16 + nib]);
        if (qjl_i8) qjl_sum += static_cast<float>(qjl_i8[d * 16 + nib]);
      }
      scores[c] = base_sum * base_scale;
      if (qjl_i8) scores[c] += gammas[c] * qjl_sum * qjl_scale;
    }
  }

  // ------------------------------------------------------------------
  // V3 Postprocess — bridge formula for L2
  //   raw_score[i] ≈ <q_unit, x_unit_i>
  //   ||x_r - q_r||^2 = |x_c|^2 + |q_c|^2 - 2*|x_c|*|q_c|*<x_unit,q_unit>
  //
  // For IP output: returns |x_c|*|q_c|*raw_score as an approximation of
  // the inner product between centred vectors (ignores centroid cross-terms).
  // ------------------------------------------------------------------
  template <bool kL2>
  void postprocess_nibble(std::size_t db0, std::size_t block_size,
                          float q_cn, float q_cn_sq, const float* raw_scores,
                          float* values, float* rank_keys) const {
    if (block_size == kBlockSize) {
      __m512 score  = _mm512_loadu_ps(raw_scores);
      __m512 norms_v   = _mm512_loadu_ps(norms_.data() + db0);
      __m512 normsq_v  = _mm512_loadu_ps(norm_squares_.data() + db0);
      if constexpr (kL2) {
        // val = |x_c|^2 + q_cn^2 - 2 * |x_c| * q_cn * raw_score
        __m512 cross = _mm512_mul_ps(
            _mm512_set1_ps(2.0f * q_cn),
            _mm512_mul_ps(norms_v, score));
        __m512 val = _mm512_sub_ps(
            _mm512_add_ps(_mm512_set1_ps(q_cn_sq), normsq_v),
            cross);
        _mm512_storeu_ps(values, val);
        _mm512_storeu_ps(rank_keys, _mm512_sub_ps(_mm512_setzero_ps(), val));
      } else {
        __m512 dot = _mm512_mul_ps(_mm512_set1_ps(q_cn), _mm512_mul_ps(norms_v, score));
        _mm512_storeu_ps(values, dot);
        _mm512_storeu_ps(rank_keys, dot);
      }
      return;
    }
    for (std::size_t i = 0; i < block_size; ++i) {
      if constexpr (kL2) {
        float val = q_cn_sq + norm_squares_[db0 + i]
                    - 2.0f * q_cn * norms_[db0 + i] * raw_scores[i];
        values[i] = val;
        rank_keys[i] = -val;
      } else {
        float dot = q_cn * norms_[db0 + i] * raw_scores[i];
        values[i] = dot;
        rank_keys[i] = dot;
      }
    }
  }

  template <bool kUseResidual, bool kL2>
  void postprocess_generic(std::size_t db0, std::size_t block_size,
                           float q_cn, float q_cn_sq, float* dot_scores,
                           const float* scratch_scores,
                           float* values, float* rank_keys) const {
    if (block_size == kBlockSize) {
      __m512 score = _mm512_loadu_ps(dot_scores);
      __m512 norms_v  = _mm512_loadu_ps(norms_.data() + db0);
      __m512 normsq_v = _mm512_loadu_ps(norm_squares_.data() + db0);
      if constexpr (kUseResidual) {
        // residual_scales_[i] = |x_c| * gamma * sqrt(pi/2) / dim
        // scratch_scores[i] = QJL sign accumulation for q_unit
        // Add residual correction: residual_scales_[i] * scratch_scores[i] / |x_c|
        // = gamma * sqrt(pi/2)/dim * scratch_scores[i]
        // which corrects the IP estimate of <q_unit, x_unit>
        // Here residual_scales_ stores |x_c| * gamma * coeff;
        // to get the IP correction we divide by |x_c| (norms_v):
        __m512 res_scale = _mm512_loadu_ps(residual_scales_.data() + db0);
        // residual_scales_[i] / norms_[i] = gamma * coeff (same as V2 but normed)
        __m512 inv_norms = _mm512_div_ps(_mm512_set1_ps(1.0f), norms_v);
        __m512 gamma_coeff = _mm512_mul_ps(res_scale, inv_norms);
        score = _mm512_fmadd_ps(gamma_coeff, _mm512_loadu_ps(scratch_scores), score);
      }
      if constexpr (kL2) {
        __m512 cross = _mm512_mul_ps(
            _mm512_set1_ps(2.0f * q_cn),
            _mm512_mul_ps(norms_v, score));
        __m512 val = _mm512_sub_ps(
            _mm512_add_ps(_mm512_set1_ps(q_cn_sq), normsq_v),
            cross);
        _mm512_storeu_ps(values, val);
        _mm512_storeu_ps(rank_keys, _mm512_sub_ps(_mm512_setzero_ps(), val));
      } else {
        __m512 dot = _mm512_mul_ps(_mm512_set1_ps(q_cn), _mm512_mul_ps(norms_v, score));
        _mm512_storeu_ps(values, dot);
        _mm512_storeu_ps(rank_keys, dot);
      }
      return;
    }
    for (std::size_t i = 0; i < block_size; ++i) {
      float ip = dot_scores[i];
      if constexpr (kUseResidual) {
        // residual_scales_[i] = |x_c| * gamma * coeff → divide by |x_c|
        float gc = (norms_[db0 + i] > 0.0f)
                     ? residual_scales_[db0 + i] / norms_[db0 + i]
                     : 0.0f;
        ip += gc * scratch_scores[i];
      }
      if constexpr (kL2) {
        float val = q_cn_sq + norm_squares_[db0 + i] - 2.0f * q_cn * norms_[db0 + i] * ip;
        values[i] = val; rank_keys[i] = -val;
      } else {
        float dot = q_cn * norms_[db0 + i] * ip;
        values[i] = dot; rank_keys[i] = dot;
      }
    }
  }

  // ------------------------------------------------------------------
  // Heap (identical to V2)
  // ------------------------------------------------------------------
  struct HeapEntry { float rank_key, value; idx_t label; };

  static void heap_sift_up(HeapEntry* h, std::size_t idx) {
    while (idx > 0) {
      std::size_t p = (idx - 1) >> 1;
      if (h[p].rank_key <= h[idx].rank_key) break;
      std::swap(h[p], h[idx]); idx = p;
    }
  }
  static void heap_sift_down(HeapEntry* h, std::size_t size, std::size_t idx) {
    while (true) {
      std::size_t s = idx, l = 2 * idx + 1, r = l + 1;
      if (l < size && h[l].rank_key < h[s].rank_key) s = l;
      if (r < size && h[r].rank_key < h[s].rank_key) s = r;
      if (s == idx) break;
      std::swap(h[idx], h[s]); idx = s;
    }
  }
  static void heap_push_or_replace(std::vector<HeapEntry>& heap, std::size_t& hs,
                                   std::size_t k, float rk, float v, idx_t lab) {
    if (k == 0) return;
    if (hs < k) { heap[hs] = {rk, v, lab}; heap_sift_up(heap.data(), hs); ++hs; return; }
    if (rk <= heap[0].rank_key) return;
    heap[0] = {rk, v, lab};
    heap_sift_down(heap.data(), hs, 0);
  }

  template <typename Fn>
  void parallel_for(std::size_t begin, std::size_t end, Fn&& fn) const {
    if (end <= begin) return;
    if (num_threads_ <= 1) { fn(begin, end); return; }
    #pragma omp parallel num_threads(static_cast<int>(num_threads_))
    {
      const int tid = omp_get_thread_num(), nthr = omp_get_num_threads();
      const std::size_t total = end - begin;
      const std::size_t chunk = (total + static_cast<std::size_t>(nthr) - 1)
                                 / static_cast<std::size_t>(nthr);
      const std::size_t cb = begin + static_cast<std::size_t>(tid) * chunk;
      const std::size_t ce = std::min(end, cb + chunk);
      if (cb < ce) fn(cb, ce);
    }
  }

  // ------------------------------------------------------------------
  // Generic-path scoring helpers (identical to V2)
  // ------------------------------------------------------------------
  void score_generic_code_block(std::size_t bi, std::size_t bs, const float* lut,
                                float* out) const {
    if (mse_bits_ == 0 || bs == 0) return;
    if (bs == kBlockSize && mse_bits_ <= 4) {
      __m512 acc = _mm512_setzero_ps();
      for (std::size_t j = 0; j < padded_dim_; ++j) {
        const __m128i codes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(byte_code_dim(bi, j)));
        const __m512i idx = _mm512_cvtepu8_epi32(codes);
        const __m512 row = _mm512_loadu_ps(lut + j * lut_stride_);
        acc = _mm512_add_ps(acc, _mm512_permutexvar_ps(idx, row));
      }
      _mm512_storeu_ps(out, acc); return;
    }
    if (bs == kBlockSize && mse_bits_ == 5) {
      __m512 acc = _mm512_setzero_ps();
      const __m512i fifteen = _mm512_set1_epi32(15);
      for (std::size_t j = 0; j < padded_dim_; ++j) {
        const __m128i codes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(byte_code_dim(bi, j)));
        const __m512i idx = _mm512_cvtepu8_epi32(codes);
        const float* row = lut + j * lut_stride_;
        const __m512 lo = _mm512_loadu_ps(row), hi = _mm512_loadu_ps(row + 16);
        const __m512i mod = _mm512_and_si512(idx, fifteen);
        const __m512 v0 = _mm512_permutexvar_ps(mod, lo), v1 = _mm512_permutexvar_ps(mod, hi);
        acc = _mm512_add_ps(acc, _mm512_mask_blend_ps(_mm512_cmpgt_epi32_mask(idx, fifteen), v0, v1));
      }
      _mm512_storeu_ps(out, acc); return;
    }
    if (bs == kBlockSize) {
      __m512 acc = _mm512_setzero_ps();
      for (std::size_t j = 0; j < padded_dim_; ++j) {
        const __m128i codes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(byte_code_dim(bi, j)));
        acc = _mm512_add_ps(acc, _mm512_i32gather_ps(_mm512_cvtepu8_epi32(codes), lut + j * lut_stride_, 4));
      }
      _mm512_storeu_ps(out, acc); return;
    }
    for (std::size_t j = 0; j < padded_dim_; ++j) {
      const std::uint8_t* bc = byte_code_dim(bi, j);
      const float* row = lut + j * lut_stride_;
      for (std::size_t i = 0; i < bs; ++i) out[i] += row[bc[i]];
    }
  }

  void score_generic_code_sign_block(std::size_t bi, std::size_t bs,
                                     const float* lut, const float* proj_q,
                                     float* code_out, float* sign_out) const {
    if (bs == 0) return;
    if (bs == kBlockSize) {
      __m512 cacc = _mm512_setzero_ps(), sacc = _mm512_setzero_ps();
      if (combined_code_sign_) {
        const std::uint32_t cmask = (1u << mse_bits_) - 1u;
        const __m512i cmask_v = _mm512_set1_epi32(static_cast<int>(cmask));
        const __m512i one = _mm512_set1_epi32(1);
        const unsigned shift = static_cast<unsigned>(mse_bits_);
        for (std::size_t j = 0; j < padded_dim_; ++j) {
          const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(byte_code_dim(bi, j)));
          const __m512i combined = _mm512_cvtepu8_epi32(raw);
          const __m512i codes = _mm512_and_si512(combined, cmask_v);
          if (mse_bits_ <= 4)
            cacc = _mm512_add_ps(cacc, _mm512_permutexvar_ps(codes, _mm512_loadu_ps(lut + j * lut_stride_)));
          else
            cacc = _mm512_add_ps(cacc, _mm512_i32gather_ps(codes, lut + j * lut_stride_, 4));
          const __mmask16 smask = _mm512_test_epi32_mask(_mm512_srli_epi32(combined, shift), one);
          sacc = _mm512_add_ps(sacc, _mm512_mask_blend_ps(smask,
              _mm512_set1_ps(-proj_q[j]), _mm512_set1_ps(proj_q[j])));
        }
      } else {
        for (std::size_t j = 0; j < padded_dim_; ++j) {
          const __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(byte_code_dim(bi, j)));
          const __m512i idx = _mm512_cvtepu8_epi32(raw);
          if (mse_bits_ <= 4)
            cacc = _mm512_add_ps(cacc, _mm512_permutexvar_ps(idx, _mm512_loadu_ps(lut + j * lut_stride_)));
          else if (mse_bits_ == 5) {
            const __m512i fifteen = _mm512_set1_epi32(15);
            const float* row = lut + j * lut_stride_;
            const __m512i mod = _mm512_and_si512(idx, fifteen);
            const __m512 v0 = _mm512_permutexvar_ps(mod, _mm512_loadu_ps(row));
            const __m512 v1 = _mm512_permutexvar_ps(mod, _mm512_loadu_ps(row + 16));
            cacc = _mm512_add_ps(cacc, _mm512_mask_blend_ps(_mm512_cmpgt_epi32_mask(idx, fifteen), v0, v1));
          } else {
            cacc = _mm512_add_ps(cacc, _mm512_i32gather_ps(idx, lut + j * lut_stride_, 4));
          }
          std::uint16_t sbits; std::memcpy(&sbits, sign_dim(bi, j), 2);
          sacc = _mm512_add_ps(sacc, _mm512_mask_blend_ps(
              static_cast<__mmask16>(sbits),
              _mm512_set1_ps(-proj_q[j]), _mm512_set1_ps(proj_q[j])));
        }
      }
      _mm512_storeu_ps(code_out, cacc);
      _mm512_storeu_ps(sign_out, sacc);
      return;
    }
    const std::uint32_t cmask = mse_bits_ > 0 ? ((1u << mse_bits_) - 1u) : 0u;
    for (std::size_t j = 0; j < padded_dim_; ++j) {
      const std::uint8_t* bc = byte_code_dim(bi, j);
      const float* row = lut + j * lut_stride_;
      if (combined_code_sign_) {
        for (std::size_t i = 0; i < bs; ++i) {
          std::uint32_t val = bc[i];
          if (mse_bits_ > 0) code_out[i] += row[val & cmask];
          sign_out[i] += ((val >> mse_bits_) & 1u) ? proj_q[j] : -proj_q[j];
        }
      } else {
        if (mse_bits_ > 0)
          for (std::size_t i = 0; i < bs; ++i) code_out[i] += row[bc[i]];
        std::uint16_t sbits; std::memcpy(&sbits, sign_dim(bi, j), 2);
        for (std::size_t i = 0; i < bs; ++i)
          sign_out[i] += ((sbits >> i) & 1u) ? proj_q[j] : -proj_q[j];
      }
    }
  }

  void build_float_lut(const float* rotated_query, float* lut) const {
    const std::size_t cb = combined_code_sign_ ? (1u << storage_bits_) : codebook_size_;
    for (std::size_t j = 0; j < padded_dim_; ++j) {
      const float q = rotated_query[j];
      float* row = lut + j * lut_stride_;
      for (std::size_t c = 0; c < cb; ++c) row[c] = q * codebook_[c % codebook_size_];
      for (std::size_t c = cb; c < lut_stride_; ++c) row[c] = 0.0f;
    }
  }

  // ------------------------------------------------------------------
  // query_nibble_impl — V3: normalise (q_r - c) before rotation
  // ------------------------------------------------------------------
  template <bool kUseQjl, bool kL2>
  void query_nibble_impl(std::size_t nq, const float* x, std::size_t k,
                         float* distances, idx_t* labels) const {
    parallel_for(0, nq, [&](std::size_t q0, std::size_t q1) {
      std::vector<float> q_centered(padded_dim_, 0.0f);
      std::vector<float> q_unit(padded_dim_, 0.0f);
      std::vector<float> rotated(padded_dim_);
      std::vector<float> projected(kUseQjl ? padded_dim_ : 0);
      std::vector<std::int8_t> base_i8(padded_dim_ * 16);
      std::vector<std::int8_t> qjl_i8(kUseQjl ? padded_dim_ * 16 : 0);
      std::vector<float> work(padded_dim_);
      alignas(64) float raw_scores[kBlockSize];
      alignas(64) float cand_values[kBlockSize];
      alignas(64) float cand_rank_keys[kBlockSize];
      std::vector<HeapEntry> heap(k);
      std::size_t heap_size = 0;

      for (std::size_t qi = q0; qi < q1; ++qi) {
        const float* qptr = x + qi * dim_;

        // V3: centre and normalise the query
        for (std::size_t j = 0; j < dim_; ++j)
          q_centered[j] = qptr[j] - centroid_[j];
        const float q_cn = l2_norm(q_centered.data(), dim_);
        const float q_cn_sq = q_cn * q_cn;
        if (q_cn > 0.0f) {
          const float inv_qcn = 1.0f / q_cn;
          for (std::size_t j = 0; j < dim_; ++j) q_unit[j] = q_centered[j] * inv_qcn;
        } else {
          std::fill(q_unit.begin(), q_unit.end(), 0.0f);
        }

        rotation_.forward(q_unit.data(), rotated.data(), work.data());
        if constexpr (kUseQjl)
          qjl_.forward(rotated.data(), projected.data(), work.data());

        float base_scale = 1.0f, qjl_scale = 1.0f;
        build_lut16_int8(rotated.data(),
                         kUseQjl ? projected.data() : nullptr,
                         base_i8.data(),
                         kUseQjl ? qjl_i8.data() : nullptr,
                         base_scale, qjl_scale);
        heap_size = 0;

        for (std::size_t db0 = 0; db0 < ntotal_; db0 += kBlockSize) {
          const std::size_t bs = std::min<std::size_t>(kBlockSize, ntotal_ - db0);
          const std::size_t bi = db0 / kBlockSize;

          score_block16_int8(nibble_dim(bi, 0), base_i8.data(),
                             kUseQjl ? qjl_i8.data() : nullptr,
                             gammas_.data() + db0,
                             base_scale, qjl_scale, bs, raw_scores);

          postprocess_nibble<kL2>(db0, bs, q_cn, q_cn_sq, raw_scores,
                                  cand_values, cand_rank_keys);

          for (std::size_t i = 0; i < bs; ++i)
            heap_push_or_replace(heap, heap_size, k, cand_rank_keys[i],
                                 cand_values[i], static_cast<idx_t>(db0 + i));
        }

        std::sort(heap.begin(), heap.begin() + static_cast<std::ptrdiff_t>(heap_size),
                  [](const HeapEntry& a, const HeapEntry& b) { return a.rank_key > b.rank_key; });
        const std::size_t out_base = qi * k;
        for (std::size_t r = 0; r < heap_size; ++r) {
          distances[out_base + r] = heap[r].value;
          labels[out_base + r] = heap[r].label;
        }
      }
    });
  }

  // ------------------------------------------------------------------
  // query_generic_impl — V3: normalise (q_r - c) before rotation
  // ------------------------------------------------------------------
  template <bool kUseResidual, bool kL2>
  void query_generic_impl(std::size_t nq, const float* x, std::size_t k,
                          float* distances, idx_t* labels) const {
    parallel_for(0, nq, [&](std::size_t q0, std::size_t q1) {
      std::vector<float> q_centered(padded_dim_, 0.0f);
      std::vector<float> q_unit(padded_dim_, 0.0f);
      std::vector<float> rotated(padded_dim_);
      std::vector<float> projected(kUseResidual ? padded_dim_ : 0);
      std::vector<float> lut(padded_dim_ * lut_stride_);
      std::vector<float> work(padded_dim_);
      alignas(64) float dot_scores[kBlockSize];
      alignas(64) float scratch_scores[kBlockSize];
      alignas(64) float cand_values[kBlockSize];
      alignas(64) float cand_rank_keys[kBlockSize];
      std::vector<HeapEntry> heap(k);
      std::size_t heap_size = 0;

      for (std::size_t qi = q0; qi < q1; ++qi) {
        const float* qptr = x + qi * dim_;

        for (std::size_t j = 0; j < dim_; ++j)
          q_centered[j] = qptr[j] - centroid_[j];
        const float q_cn = l2_norm(q_centered.data(), dim_);
        const float q_cn_sq = q_cn * q_cn;
        if (q_cn > 0.0f) {
          const float inv_qcn = 1.0f / q_cn;
          for (std::size_t j = 0; j < dim_; ++j) q_unit[j] = q_centered[j] * inv_qcn;
        } else {
          std::fill(q_unit.begin(), q_unit.end(), 0.0f);
        }

        rotation_.forward(q_unit.data(), rotated.data(), work.data());
        build_float_lut(rotated.data(), lut.data());
        if constexpr (kUseResidual)
          qjl_.forward(rotated.data(), projected.data(), work.data());

        heap_size = 0;

        for (std::size_t db0 = 0; db0 < ntotal_; db0 += kBlockSize) {
          const std::size_t bs = std::min<std::size_t>(kBlockSize, ntotal_ - db0);
          const std::size_t bi = db0 / kBlockSize;

          std::fill(dot_scores, dot_scores + bs, 0.0f);
          if constexpr (kUseResidual) {
            std::fill(scratch_scores, scratch_scores + bs, 0.0f);
            score_generic_code_sign_block(bi, bs, lut.data(), projected.data(),
                                          dot_scores, scratch_scores);
          } else {
            score_generic_code_block(bi, bs, lut.data(), dot_scores);
          }

          postprocess_generic<kUseResidual, kL2>(db0, bs, q_cn, q_cn_sq,
                                                  dot_scores, scratch_scores,
                                                  cand_values, cand_rank_keys);

          for (std::size_t i = 0; i < bs; ++i)
            heap_push_or_replace(heap, heap_size, k, cand_rank_keys[i],
                                 cand_values[i], static_cast<idx_t>(db0 + i));
        }

        std::sort(heap.begin(), heap.begin() + static_cast<std::ptrdiff_t>(heap_size),
                  [](const HeapEntry& a, const HeapEntry& b) { return a.rank_key > b.rank_key; });
        const std::size_t out_base = qi * k;
        for (std::size_t r = 0; r < heap_size; ++r) {
          distances[out_base + r] = heap[r].value;
          labels[out_base + r] = heap[r].label;
        }
      }
    });
  }
};

}  // namespace turboquant
