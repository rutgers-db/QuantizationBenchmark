#pragma once
// =====================================================================
// turbo_quant_index_unified.hpp — TurboQuantIndex (unified V1 + V2 + V3)
//
// WHAT THIS FILE IS (undergraduate-friendly)
// -----------------------------------------
// We want to search a *database* of high-dimensional vectors (e.g. embeddings)
// for the nearest neighbours of a *query* vector. Storing full floats is
// expensive; TurboQuant compresses each vector to a few bits per coordinate
// after a random rotation, then scores candidates quickly.
//
// PIPELINE (same idea as the TurboQuant paper)
// --------------------------------------------
// 1) **Effective vector** x_eff:
//      If use_data_centroid: x_eff = x − c with c = mean of training data.
//      Else: x_eff = x.
//      Store ‖x_eff‖ and work with the **unit direction** u = x_eff / ‖x_eff‖.
//
// 2) **Rotation** R (orthogonal):  v = R u.
//      • Hadamard: cheap O(d log d) via FWHT; dimension is padded to a power of 2.
//      • Dense: full d×d Haar-random orthogonal matrix; O(d²) apply.
//      Rotation makes coordinates behave more like independent samples so
//      scalar quantization per coordinate is reasonable.
//
// 3) **Scalar quantization (Lloyd–Max / codebook)**:
//      Each coordinate v_j is mapped to one of K = 2^b centroids c_k
//      (b = mse_bits), minimising expected squared error for a reference
//      distribution (Beta along [-1,1] or Gaussian tables scaled by 1/√d).
//
// 4) **Inner-product mode only — QJL residual (optional)**:
//      Residual r = v − q(v) where q(v) is the quantized coordinate vector.
//      A second linear map estimates inner products from sign(r); γ stores scale.
//      (MSE mode skips this and stores codes only.)
//
// SEARCH MATH (why norms appear in query())
// -----------------------------------------
// Let q_eff, x_eff be centred (or raw) query and database vectors, with
// unit directions q_unit, x_unit. The index approximates
//   s ≈ ⟨q_unit, x_unit⟩
// from quantized data. True dot product:
//   ⟨q_eff, x_eff⟩ = ‖q_eff‖ · ‖x_eff‖ · ⟨q_unit, x_unit⟩.
// L2 distance (bridge formula, expands (q−x)²):
//   ‖q_eff − x_eff‖² = ‖q_eff‖² + ‖x_eff‖² − 2‖q_eff‖‖x_eff‖⟨q_unit, x_unit⟩.
//
// CONFIG PRESETS
// --------------
//   RotationType:
//     kHadamard  — signs + fast Walsh–Hadamard (FWHT), O(d log d); pad to power-of-2.
//     kDense     — Householder QR → Haar orthogonal matrix, O(d²); any d.
//   CodebookType:
//     kGaussianLM — Pre-tabulated Lloyd–Max for N(0,1), scaled by σ = 1/√(padded_dim).
//   Default (good recall + fast 4-bit SIMD path):
//     bitwidth=4, kHadamard, kGaussianLM, use_data_centroid=true
//     → "nibble" path: int8 LUT + VPSHUFB shuffle.
//
// CODE STRUCTURE  (bottom-up: lowest layer first, public API last)
// ----------------------------------------------------------------
//   namespace detail   — compile-time constants, SIMD math, HeapEntry + heap ops
//   struct Transform   — orthogonal rotation (Hadamard or dense); used for R and QJL
//   struct Codebook    — 1-D Lloyd–Max scalar quantizer; LUT builders for SIMD scoring
//   struct StorageLayout — buffer management, SIMD block scoring, postprocessing
//   struct KMeansIVF   — K-means++ initialisation + IVF coarse search
//   class TurboQuantIndex — public API: Config / train / add / query
//     public:  Config, constructor, getters, train(), add(),
//              query(metric), query() L2 overload, ProfileStats + profile_nibble_query()
//     private: constants, members |
//              helpers: require_trained, prepare_query, parallel_for |
//              encode: encode_rotated, build_rotated_residual, encode_vector_inplace |
//              flat query: nibble, packed, generic impls |
//              IVF: add_ivf, query_ivf_packed/nibble/generic impls |
//              debug: reconstruct
// =====================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <immintrin.h>
#include <limits>
#include <numeric>
#include <omp.h>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace turboquant {

// =====================================================================
// Shared enums (public API)
// TurboQuantIndex re-exports these via using aliases so that
// TurboQuantIndex::Mode etc. still works for callers.
// =====================================================================
enum class Mode { kMSE, kInnerProduct };
enum class SearchMetric { kInnerProduct, kL2 };

enum class RotationType {
  kHadamard,  // structured Hadamard (V2/V3): O(d log d), pads to power-of-2
  kDense,     // Haar-random orthogonal matrix (V1): O(d²), any dimension
};

enum class CodebookType {
  kGaussianLM,  // hardcoded Lloyd-Max for N(0,1), scaled by 1/sqrt(dim)
};

// =====================================================================
// detail — internal utilities (no public API contract)
// =====================================================================
namespace detail {

// SIMD block sizes and SIMD accumulator drain interval (implementation constants)
inline constexpr std::size_t kBlockSize       = 16;   // 16 lanes per block (nibble path)
inline constexpr std::size_t kPackedBlockSize = 32;   // 32 lanes per block (packed path)
inline constexpr std::size_t kRowAlignment    = 64;   // cache-line alignment for SIMD loads
inline constexpr std::size_t kDrainInterval   = 128;  // drain int16→int32 every 128 dims
inline constexpr float       kQjlScale        = 1.2533141373155001f;  // sqrt(π/2)

// ------------------------------------------------------------------
// Min-heap for top-k results
// rank_key = negated-L2 or dot-product (max-heap on rank_key = keep best)
// ------------------------------------------------------------------
struct HeapEntry { float rank_key, value; std::int64_t label; };

inline void heap_sift_up(HeapEntry* h, std::size_t idx) {
  while (idx > 0) {
    std::size_t p = (idx - 1) >> 1;
    if (h[p].rank_key <= h[idx].rank_key) break;
    std::swap(h[p], h[idx]); idx = p;
  }
}
inline void heap_sift_down(HeapEntry* h, std::size_t size, std::size_t idx) {
  while (true) {
    std::size_t s = idx, l = 2 * idx + 1, r = l + 1;
    if (l < size && h[l].rank_key < h[s].rank_key) s = l;
    if (r < size && h[r].rank_key < h[s].rank_key) s = r;
    if (s == idx) break;
    std::swap(h[idx], h[s]); idx = s;
  }
}
inline void heap_push_or_replace(std::vector<HeapEntry>& heap, std::size_t& hs,
                                  std::size_t k, float rk, float v, std::int64_t lab) {
  if (k == 0) return;
  if (hs < k) { heap[hs] = {rk, v, lab}; heap_sift_up(heap.data(), hs); ++hs; return; }
  if (rk <= heap[0].rank_key) return;
  heap[0] = {rk, v, lab}; heap_sift_down(heap.data(), hs, 0);
}

// ------------------------------------------------------------------
// RNG, alignment, integer utilities
// ------------------------------------------------------------------
// Deterministic 64-bit mixing; used to derive pseudo-random signs from seed.
inline std::uint64_t splitmix64(std::uint64_t& state) {
  state += 0x9E3779B97F4A7C15ULL;
  std::uint64_t z = state;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// Smallest power of two ≥ v (Hadamard length must be a power of two).
inline std::size_t next_pow2(std::size_t v) {
  if (v == 0) return 1;
  v--;
  v |= v >> 1; v |= v >> 2; v |= v >> 4;
  v |= v >> 8; v |= v >> 16; v |= v >> 32;
  return v + 1;
}

// Round byte count up to kRowAlignment for cache-friendly SIMD loads.
inline std::size_t aligned_bytes(std::size_t bytes) {
  return bytes == 0 ? 0 : ((bytes + kRowAlignment - 1) / kRowAlignment) * kRowAlignment;
}

// Integer ceil(a / b) for block counts.
inline std::size_t ceil_div(std::size_t a, std::size_t b) { return (a + b - 1) / b; }

// ------------------------------------------------------------------
// SIMD linear algebra
// ------------------------------------------------------------------
// In-place Walsh–Hadamard transform on length d (power of two): O(d log d) butterflies.
// Pairwise: (a,b) → (a+b, a−b); composes to orthogonal Hadamard up to scaling.
inline void wht_butterfly_inplace(float* x, std::size_t d) {
  for (std::size_t step = 1; step < d; step <<= 1)
    for (std::size_t i = 0; i < d; i += step * 2)
      for (std::size_t j = i; j < i + step; ++j) {
        const float a = x[j], b = x[j + step];
        x[j] = a + b; x[j + step] = a - b;
      }
}

// ⟨a,b⟩ with AVX-512 FMA when n is large; Θ(n) time, O(1) extra space.
inline float dot_product(const float* a, const float* b, std::size_t n) {
  __m512 acc = _mm512_setzero_ps();
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16)
    acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
  float sum = _mm512_reduce_add_ps(acc);
  for (; i < n; ++i) sum += a[i] * b[i];
  return sum;
}

// Euclidean ‖x‖₂ = √(x·x).
inline float l2_norm(const float* x, std::size_t n) {
  return std::sqrt(dot_product(x, x, n));
}

// Squared Euclidean ‖a−b‖₂² between two vectors of length n.
inline float l2_sq_distance(const float* a, const float* b, std::size_t n) {
  __m512 vacc = _mm512_setzero_ps();
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    __m512 dv = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
    vacc = _mm512_fmadd_ps(dv, dv, vacc);
  }
  float dsq = _mm512_reduce_add_ps(vacc);
  for (; i < n; ++i) { float dj = a[i] - b[i]; dsq += dj * dj; }
  return dsq;
}

// Saturating int8 clamp for LUT quantization.
inline std::int8_t clamp_i8(long v) {
  return static_cast<std::int8_t>(std::max(-127L, std::min(127L, v)));
}

}  // namespace detail

// =====================================================================
// Transform — orthogonal "rotation" R (main path) or Gaussian matrix (QJL dense path)
//
// Hadamard: v = (1/√d) · H · (s ⊙ u) with random sign vector s and FWHT for H.
// Dense: v = M u with M from QR of a Gaussian matrix (columns orthonormal → MᵀM = I).
// =====================================================================
struct Transform {
  RotationType type        = RotationType::kHadamard;
  std::size_t  dim         = 0;
  bool         is_gaussian = false;  // true for Dense QJL (random Gaussian, not orthogonal)

  std::vector<float> matrix;  // Dense: dim×dim, row-major
  std::vector<float> signs;   // Hadamard: dim random ±1

  // Main rotation for TurboQuant: either Hadamard pipeline or dense orthogonal M.
  void generate(RotationType t, std::size_t d, std::uint64_t seed) {
    type = t; dim = d; is_gaussian = false;
    if (t == RotationType::kHadamard) {
      signs.resize(d);
      std::uint64_t state = seed;
      for (std::size_t i = 0; i < d; ++i)
        signs[i] = (detail::splitmix64(state) & 1) ? 1.0f : -1.0f;
      matrix.clear();
    } else {
      generate_dense_orthogonal(d, seed);
      signs.clear();
    }
  }

  // QJL second linear map: Hadamard+signs (structured) or full Gaussian matrix (dense IP).
  void generate_qjl(RotationType t, std::size_t d, std::uint64_t seed) {
    type = t; dim = d;
    if (t == RotationType::kHadamard) {
      is_gaussian = false;
      signs.resize(d);
      std::uint64_t state = seed;
      for (std::size_t i = 0; i < d; ++i)
        signs[i] = (detail::splitmix64(state) & 1) ? 1.0f : -1.0f;
      matrix.clear();
    } else {
      is_gaussian = true;
      matrix.resize(d * d);
      std::mt19937_64 rng(seed);
      std::normal_distribution<float> gauss(0.0f, 1.0f);
      for (float& v : matrix) v = gauss(rng);
      signs.clear();
    }
  }

  // forward: y = R x (Hadamard) or y = M x (dense).  `work` is scratch for FWHT.
  void forward(const float* in, float* out, float* work) const {
    if (type == RotationType::kHadamard) {
      for (std::size_t i = 0; i < dim; ++i) work[i] = signs[i] * in[i];
      detail::wht_butterfly_inplace(work, dim);
      const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
      for (std::size_t i = 0; i < dim; ++i) out[i] = work[i] * scale;
    } else {
      const float* mat = matrix.data();
      for (std::size_t i = 0; i < dim; ++i)
        out[i] = detail::dot_product(mat + i * dim, in, dim);
    }
  }

  // backward: x = Rᵀ y.  For Hadamard with same scaling, forward is its own inverse.
  // Dense: multiply by Mᵀ (matrix stored row-wise, so accumulate along columns).
  void backward(const float* in, float* out, float* work) const {
    if (type == RotationType::kHadamard) {
      forward(in, out, work);
    } else {
      const float* mat = matrix.data();
      for (std::size_t i = 0; i < dim; ++i) {
        float s = 0.0f;
        for (std::size_t j = 0; j < dim; ++j) s += mat[j * dim + i] * in[j];
        out[i] = s;
      }
    }
  }

 private:
  // Haar-random orthogonal d×d matrix: QR decomposition of i.i.d. Gaussian A = Q R,
  // then column sign fixes so the Haar measure is correct (see standard random matrix refs).
  void generate_dense_orthogonal(std::size_t d, std::uint64_t seed) {
    std::vector<double> A(d * d);
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    for (auto& v : A) v = gauss(rng);

    std::vector<double> Q(d * d, 0.0);
    for (std::size_t i = 0; i < d; ++i) Q[i * d + i] = 1.0;

    std::vector<double> tau(d, 0.0);
    for (std::size_t k = 0; k < d; ++k) {
      double col_norm_sq = 0.0;
      for (std::size_t i = k; i < d; ++i) { const double v = A[i * d + k]; col_norm_sq += v * v; }
      const double col_norm = std::sqrt(col_norm_sq);
      if (col_norm < 1e-15) continue;

      const double alpha = -std::copysign(col_norm, A[k * d + k]);
      A[k * d + k] -= alpha;

      double v_norm_sq = 0.0;
      for (std::size_t i = k; i < d; ++i) { const double v = A[i * d + k]; v_norm_sq += v * v; }
      if (v_norm_sq < 1e-30) continue;
      tau[k] = 2.0 / v_norm_sq;

      for (std::size_t j = k + 1; j < d; ++j) {
        double vTa = 0.0;
        for (std::size_t i = k; i < d; ++i) vTa += A[i * d + k] * A[i * d + j];
        vTa *= tau[k];
        for (std::size_t i = k; i < d; ++i) A[i * d + j] -= A[i * d + k] * vTa;
      }
      for (std::size_t i = 0; i < d; ++i) {
        double qv = 0.0;
        for (std::size_t j = k; j < d; ++j) qv += Q[i * d + j] * A[j * d + k];
        qv *= tau[k];
        for (std::size_t j = k; j < d; ++j) Q[i * d + j] -= qv * A[j * d + k];
      }
      A[k * d + k] = alpha;
    }
    // Sign correction: ensure R diagonal is positive (Haar distribution)
    for (std::size_t k = 0; k < d; ++k)
      if (A[k * d + k] < 0.0)
        for (std::size_t i = 0; i < d; ++i) Q[i * d + k] = -Q[i * d + k];

    matrix.resize(d * d);
    for (std::size_t i = 0; i < d * d; ++i) matrix[i] = static_cast<float>(Q[i]);
  }
};

// =====================================================================
// Codebook — 1-D Lloyd–Max scalar quantizer for each rotated coordinate
//
// Builds centroids + decision thresholds using pre-tabulated N(0,1) Lloyd–Max,
// scaled by σ = 1/√padded_dim. Also provides LUT builders for SIMD scoring paths.
// =====================================================================
struct Codebook {
  std::vector<float> centroids;   // 2^mse_bits reconstruction levels
  std::vector<float> thresholds;  // 2^mse_bits − 1 decision boundaries
  std::size_t size      = 0;      // == centroids.size()
  float       max_abs   = 0.f;    // max |centroid| — used to scale int8 LUT
  std::size_t mse_bits  = 0;      // bits used for MSE codes
  std::size_t padded_dim = 0;     // padded dimension (needed for LUT scaling)
  Mode        mode      = Mode::kMSE;

  // Build centroids + thresholds from configuration.
  void build(std::size_t bits, std::size_t d, Mode m) {
    mse_bits   = bits;
    padded_dim = d;
    mode       = m;

    centroids = build_lm_codebook(bits, d);
    build_lm_thresholds(bits, d);
    size    = centroids.size();
    max_abs = 0.f;
    for (float c : centroids) max_abs = std::max(max_abs, std::abs(c));
  }

  // Nearest centroid: binary search on thresholds (O(bits) comparisons).
  std::uint32_t nearest(float v) const {
    auto it = std::upper_bound(thresholds.begin(), thresholds.end(), v);
    return static_cast<std::uint32_t>(std::distance(thresholds.begin(), it));
  }

  // ------------------------------------------------------------------
  // Int8 LUT16 — per-dimension lookup tables for fast nibble scoring
  //
  // For each dimension d and nibble index n∈[0,16), stores int8 ≈
  //   round(rotated_q[d] · centroid[n & (cb−1)] / base_scale)
  // so Σ_d LUT[d,n] approximates a dot-product proxy via VPSHUFB.
  // IP mode also builds qjl_i8 for the residual sign path.
  // base_scale / qjl_scale map int8 accumulators back to float.
  // ------------------------------------------------------------------
  void build_lut16_int8(const float* rotated_q, const float* projected_q,
                        std::int8_t* base_i8, std::int8_t* qjl_i8,
                        float& base_scale, float& qjl_scale) const {
    float max_q = 0.f;
    for (std::size_t d = 0; d < padded_dim; ++d)
      max_q = std::max(max_q, std::abs(rotated_q[d]));
    const float base_max = max_q * max_abs;
    base_scale = (base_max > 1e-30f) ? (base_max / 127.f) : 1.f;
    const float base_inv = 1.f / base_scale;

    const std::size_t cb        = std::size_t{1} << mse_bits;
    const std::size_t sign_mask = cb;  // bit mse_bits of nibble encodes QJL sign

    for (std::size_t d = 0; d < padded_dim; ++d) {
      const float q = rotated_q[d];
      for (std::size_t n = 0; n < 16; ++n)
        base_i8[d * 16 + n] = detail::clamp_i8(std::lround(q * centroids[n & (cb - 1)] * base_inv));
    }

    if (projected_q && qjl_i8) {
      const float qjl_coeff = detail::kQjlScale / static_cast<float>(padded_dim);
      float max_proj = 0.f;
      for (std::size_t d = 0; d < padded_dim; ++d)
        max_proj = std::max(max_proj, std::abs(projected_q[d]));
      const float qjl_max = qjl_coeff * max_proj;
      qjl_scale = (qjl_max > 1e-30f) ? (qjl_max / 127.f) : 1.f;
      const float qjl_inv = 1.f / qjl_scale;
      for (std::size_t d = 0; d < padded_dim; ++d) {
        const float val = qjl_coeff * projected_q[d];
        for (std::size_t n = 0; n < 16; ++n) {
          const float sign = (n & sign_mask) ? 1.f : -1.f;
          qjl_i8[d * 16 + n] = detail::clamp_i8(std::lround(sign * val * qjl_inv));
        }
      }
    } else {
      qjl_scale = 1.f;
    }
  }

  // ------------------------------------------------------------------
  // Static table accessors and codebook builders
  // ------------------------------------------------------------------

  // Precomputed optimal partition boundaries for standard Gaussian (offline Lloyd–Max).
  static const float* lm_boundaries_table(std::size_t bits) {
    static const float b1[] = {0.f};
    static const float b2[] = {-0.9816f, 0.f, 0.9816f};
    static const float b3[] = {-1.7479f,-1.05f,-0.5005f,0.f,0.5005f,1.05f,1.7479f};
    static const float b4[] = {
      -2.4008f,-1.8435f,-1.4371f,-1.0993f,-0.7995f,-0.5224f,-0.2582f,0.f,
       0.2582f, 0.5224f, 0.7995f, 1.0993f, 1.4371f, 1.8435f, 2.4008f};
    static const float b5[] = {
      -2.976f,-2.5045f,-2.1733f,-1.908f,-1.6818f,-1.4813f,-1.2991f,-1.1303f,
      -0.9717f,-0.8209f,-0.6761f,-0.5358f,-0.3991f,-0.2647f,-0.132f,0.f,
       0.132f,0.2647f,0.3991f,0.5358f,0.6761f,0.8209f,0.9717f,1.1303f,
       1.2991f,1.4813f,1.6818f,1.908f,2.1733f,2.5045f,2.976f};
    static const float b6[] = {
      -3.5169f,-3.1058f,-2.8233f,-2.6015f,-2.4159f,-2.2544f,-2.1103f,-1.9792f,
      -1.8584f,-1.7457f,-1.6397f,-1.5392f,-1.4433f,-1.3515f,-1.263f,-1.1774f,
      -1.0943f,-1.0134f,-0.9344f,-0.8571f,-0.7812f,-0.7066f,-0.6331f,-0.5606f,
      -0.4889f,-0.4178f,-0.3473f,-0.2773f,-0.2077f,-0.1383f,-0.0691f,0.f,
       0.0691f,0.1383f,0.2077f,0.2773f,0.3473f,0.4178f,0.4889f,0.5606f,
       0.6331f,0.7066f,0.7812f,0.8571f,0.9344f,1.0134f,1.0943f,1.1774f,
       1.263f,1.3515f,1.4433f,1.5392f,1.6397f,1.7457f,1.8584f,1.9792f,
       2.1103f,2.2544f,2.4159f,2.6015f,2.8233f,3.1058f,3.5169f};
    static const float b7[] = {
      -4.0871f,-3.7257f,-3.4813f,-3.292f,-3.1356f,-3.0011f,-2.8824f,-2.7756f,
      -2.6782f,-2.5883f,-2.5045f,-2.4259f,-2.3516f,-2.2811f,-2.2138f,-2.1494f,
      -2.0874f,-2.0276f,-1.9697f,-1.9136f,-1.859f,-1.8057f,-1.7538f,-1.7029f,
      -1.6531f,-1.6042f,-1.5561f,-1.5087f,-1.4621f,-1.4161f,-1.3706f,-1.3256f,
      -1.2812f,-1.2371f,-1.1934f,-1.1501f,-1.1071f,-1.0644f,-1.0219f,-0.9797f,
      -0.9377f,-0.8959f,-0.8542f,-0.8128f,-0.7714f,-0.7302f,-0.6891f,-0.6482f,
      -0.6073f,-0.5665f,-0.5257f,-0.4851f,-0.4445f,-0.4039f,-0.3634f,-0.3229f,
      -0.2825f,-0.2421f,-0.2017f,-0.1613f,-0.121f,-0.0807f,-0.0403f,0.f,
       0.0403f,0.0807f,0.121f,0.1613f,0.2017f,0.2421f,0.2825f,0.3229f,
       0.3634f,0.4039f,0.4445f,0.4851f,0.5257f,0.5665f,0.6073f,0.6482f,
       0.6891f,0.7302f,0.7714f,0.8128f,0.8542f,0.8959f,0.9377f,0.9797f,
       1.0219f,1.0644f,1.1071f,1.1501f,1.1934f,1.2371f,1.2812f,1.3256f,
       1.3706f,1.4161f,1.4621f,1.5087f,1.5561f,1.6042f,1.6531f,1.7029f,
       1.7538f,1.8057f,1.859f,1.9136f,1.9697f,2.0276f,2.0874f,2.1494f,
       2.2138f,2.2811f,2.3516f,2.4259f,2.5045f,2.5883f,2.6782f,2.7756f,
       2.8824f,3.0011f,3.1356f,3.292f,3.4813f,3.7257f,4.0871f};
    static const float b8[] = {
      -4.502f,-4.1704f,-3.9485f,-3.7782f,-3.6388f,-3.52f,-3.4161f,-3.3236f,
      -3.24f,-3.1636f,-3.0933f,-3.028f,-2.967f,-2.9098f,-2.8558f,-2.8047f,
      -2.7562f,-2.7101f,-2.6659f,-2.6237f,-2.5831f,-2.5441f,-2.5065f,-2.4701f,
      -2.435f,-2.4009f,-2.3679f,-2.3357f,-2.3044f,-2.2738f,-2.244f,-2.2148f,
      -2.1862f,-2.1582f,-2.1307f,-2.1037f,-2.0771f,-2.0508f,-2.025f,-1.9995f,
      -1.9742f,-1.9493f,-1.9246f,-1.9002f,-1.8759f,-1.8519f,-1.828f,-1.8042f,
      -1.7806f,-1.7572f,-1.7338f,-1.7106f,-1.6875f,-1.6644f,-1.6414f,-1.6185f,
      -1.5956f,-1.5728f,-1.55f,-1.5273f,-1.5046f,-1.482f,-1.4593f,-1.4367f,
      -1.4142f,-1.3916f,-1.3691f,-1.3465f,-1.324f,-1.3015f,-1.279f,-1.2565f,
      -1.2341f,-1.2116f,-1.1891f,-1.1667f,-1.1442f,-1.1218f,-1.0993f,-1.0769f,
      -1.0544f,-1.032f,-1.0095f,-0.9871f,-0.9646f,-0.9422f,-0.9198f,-0.8973f,
      -0.8749f,-0.8525f,-0.83f,-0.8076f,-0.7852f,-0.7627f,-0.7403f,-0.7179f,
      -0.6954f,-0.673f,-0.6506f,-0.6281f,-0.6057f,-0.5833f,-0.5608f,-0.5384f,
      -0.516f,-0.4935f,-0.4711f,-0.4487f,-0.4262f,-0.4038f,-0.3814f,-0.3589f,
      -0.3365f,-0.3141f,-0.2916f,-0.2692f,-0.2468f,-0.2243f,-0.2019f,-0.1795f,
      -0.157f,-0.1346f,-0.1122f,-0.0897f,-0.0673f,-0.0449f,-0.0224f,0.f,
       0.0224f,0.0449f,0.0673f,0.0897f,0.1122f,0.1346f,0.157f,0.1795f,
       0.2019f,0.2243f,0.2468f,0.2692f,0.2916f,0.3141f,0.3365f,0.3589f,
       0.3814f,0.4038f,0.4262f,0.4487f,0.4711f,0.4935f,0.516f,0.5384f,
       0.5608f,0.5833f,0.6057f,0.6281f,0.6506f,0.673f,0.6954f,0.7179f,
       0.7403f,0.7627f,0.7852f,0.8076f,0.83f,0.8525f,0.8749f,0.8973f,
       0.9198f,0.9422f,0.9646f,0.9871f,1.0095f,1.032f,1.0544f,1.0769f,
       1.0993f,1.1218f,1.1442f,1.1667f,1.1891f,1.2116f,1.2341f,1.2565f,
       1.279f,1.3015f,1.324f,1.3465f,1.3691f,1.3916f,1.4142f,1.4367f,
       1.4593f,1.482f,1.5046f,1.5273f,1.55f,1.5728f,1.5956f,1.6185f,
       1.6414f,1.6644f,1.6875f,1.7106f,1.7338f,1.7572f,1.7806f,1.8042f,
       1.828f,1.8519f,1.8759f,1.9002f,1.9246f,1.9493f,1.9742f,1.9995f,
       2.025f,2.0508f,2.0771f,2.1037f,2.1307f,2.1582f,2.1862f,2.2148f,
       2.244f,2.2738f,2.3044f,2.3357f,2.3679f,2.4009f,2.435f,2.4701f,
       2.5065f,2.5441f,2.5831f,2.6237f,2.6659f,2.7101f,2.7562f,2.8047f,
       2.8558f,2.9098f,2.967f,3.028f,3.0933f,3.1636f,3.24f,3.3236f,
       3.4161f,3.52f,3.6388f,3.7782f,3.9485f,4.1704f,4.502f};
    switch (bits) {
      case 1: return b1; case 2: return b2; case 3: return b3;
      case 4: return b4; case 5: return b5; case 6: return b6;
      case 7: return b7; case 8: return b8; default: return nullptr;
    }
  }

  // Precomputed optimal reconstruction levels for N(0,1) (offline Lloyd–Max).
  static const float* lm_centroids_table(std::size_t bits) {
    static const float c1[] = {-0.7979f, 0.7979f};
    static const float c2[] = {-1.5104f,-0.4528f,0.4528f,1.5104f};
    static const float c3[] = {-2.1519f,-1.3439f,-0.756f,-0.2451f,0.2451f,0.756f,1.3439f,2.1519f};
    static const float c4[] = {
      -2.7326f,-2.069f,-1.618f,-1.2562f,-0.9423f,-0.6568f,-0.388f,-0.1284f,
       0.1284f,0.388f,0.6568f,0.9423f,1.2562f,1.618f,2.069f,2.7326f};
    static const float c5[] = {
      -3.2608f,-2.6912f,-2.3178f,-2.0288f,-1.7873f,-1.5763f,-1.3864f,-1.2118f,
      -1.0488f,-0.8946f,-0.7472f,-0.605f,-0.4667f,-0.3314f,-0.1981f,-0.0659f,
       0.0659f,0.1981f,0.3314f,0.4667f,0.605f,0.7472f,0.8946f,1.0488f,
       1.2118f,1.3864f,1.5763f,1.7873f,2.0288f,2.3178f,2.6912f,3.2608f};
    static const float c6[] = {
      -3.7674f,-3.2664f,-2.9452f,-2.7015f,-2.5016f,-2.3302f,-2.1786f,-2.0419f,
      -1.9165f,-1.8002f,-1.6912f,-1.5882f,-1.4902f,-1.3965f,-1.3064f,-1.2195f,
      -1.1352f,-1.0533f,-0.9735f,-0.8954f,-0.8188f,-0.7437f,-0.6696f,-0.5967f,
      -0.5245f,-0.4532f,-0.3824f,-0.3122f,-0.2424f,-0.1729f,-0.1037f,-0.0345f,
       0.0345f,0.1037f,0.1729f,0.2424f,0.3122f,0.3824f,0.4532f,0.5245f,
       0.5967f,0.6696f,0.7437f,0.8188f,0.8954f,0.9735f,1.0533f,1.1352f,
       1.2195f,1.3064f,1.3965f,1.4902f,1.5882f,1.6912f,1.8002f,1.9165f,
       2.0419f,2.1786f,2.3302f,2.5016f,2.7015f,2.9452f,3.2664f,3.7674f};
    static const float c7[] = {
      -4.3088f,-3.8654f,-3.5859f,-3.3767f,-3.2074f,-3.0638f,-2.9384f,-2.8264f,
      -2.7248f,-2.6315f,-2.545f,-2.464f,-2.3878f,-2.3155f,-2.2467f,-2.1809f,
      -2.1178f,-2.057f,-1.9982f,-1.9412f,-1.859f,-1.832f,-1.7795f,-1.7281f,
      -1.6778f,-1.6284f,-1.5799f,-1.5322f,-1.4853f,-1.4389f,-1.3932f,-1.348f,
      -1.3033f,-1.259f,-1.2152f,-1.1717f,-1.1285f,-1.0857f,-1.0431f,-1.0008f,
      -0.9586f,-0.9167f,-0.875f,-0.8335f,-0.7921f,-0.7508f,-0.7097f,-0.6686f,
      -0.6277f,-0.5868f,-0.5461f,-0.5054f,-0.4647f,-0.4242f,-0.3836f,-0.3432f,
      -0.3027f,-0.2623f,-0.2219f,-0.1815f,-0.1412f,-0.1008f,-0.0605f,-0.0202f,
       0.0202f,0.0605f,0.1008f,0.1412f,0.1815f,0.2219f,0.2623f,0.3027f,
       0.3432f,0.3836f,0.4242f,0.4647f,0.5054f,0.5461f,0.5868f,0.6277f,
       0.6686f,0.7097f,0.7508f,0.7921f,0.8335f,0.875f,0.9167f,0.9586f,
       1.0008f,1.0431f,1.0857f,1.1285f,1.1717f,1.2152f,1.259f,1.3033f,
       1.348f,1.3932f,1.4389f,1.4853f,1.5322f,1.5799f,1.6284f,1.6778f,
       1.7281f,1.7795f,1.832f,1.859f,1.9412f,1.9982f,2.057f,2.1178f,
       2.1809f,2.2467f,2.3155f,2.3878f,2.464f,2.545f,2.6315f,2.7248f,
       2.8264f,2.9384f,3.0638f,3.2074f,3.3767f,3.5859f,3.8654f,4.3088f};
    static const float c8[] = {
      -4.7062f,-4.2978f,-4.043f,-3.854f,-3.7025f,-3.5751f,-3.4649f,-3.3674f,
      -3.2798f,-3.2002f,-3.1271f,-3.0595f,-2.9965f,-2.9375f,-2.882f,-2.8296f,
      -2.7799f,-2.7326f,-2.6875f,-2.6444f,-2.603f,-2.5632f,-2.5249f,-2.488f,
      -2.4523f,-2.4177f,-2.3842f,-2.3516f,-2.3198f,-2.2889f,-2.2587f,-2.2292f,
      -2.2004f,-2.1721f,-2.1443f,-2.1171f,-2.0903f,-2.0639f,-2.0378f,-2.0121f,
      -1.9868f,-1.9617f,-1.9369f,-1.9123f,-1.888f,-1.8638f,-1.8399f,-1.8161f,
      -1.7924f,-1.7689f,-1.7455f,-1.7222f,-1.699f,-1.6759f,-1.6529f,-1.6299f,
      -1.607f,-1.5842f,-1.5614f,-1.5387f,-1.5159f,-1.4933f,-1.4706f,-1.448f,
      -1.4254f,-1.4029f,-1.3803f,-1.3578f,-1.3353f,-1.3128f,-1.2903f,-1.2678f,
      -1.2453f,-1.2228f,-1.2004f,-1.1779f,-1.1554f,-1.133f,-1.1105f,-1.0881f,
      -1.0656f,-1.032f,-1.0207f,-0.9983f,-0.9759f,-0.9534f,-0.931f,-0.9086f,
      -0.8861f,-0.8637f,-0.8412f,-0.8188f,-0.7964f,-0.7739f,-0.7515f,-0.7291f,
      -0.7066f,-0.6842f,-0.6618f,-0.6393f,-0.6169f,-0.5945f,-0.572f,-0.5496f,
      -0.5272f,-0.5047f,-0.4823f,-0.4599f,-0.4374f,-0.415f,-0.3926f,-0.3701f,
      -0.3477f,-0.3253f,-0.3028f,-0.2804f,-0.258f,-0.2355f,-0.2131f,-0.1907f,
      -0.1682f,-0.1458f,-0.1234f,-0.1009f,-0.0785f,-0.0561f,-0.0336f,-0.0112f,
       0.0112f,0.0336f,0.0561f,0.0785f,0.1009f,0.1234f,0.1458f,0.1682f,
       0.1907f,0.2131f,0.2355f,0.258f,0.2804f,0.3028f,0.3253f,0.3477f,
       0.3701f,0.3926f,0.415f,0.4374f,0.4599f,0.4823f,0.5047f,0.5272f,
       0.5496f,0.572f,0.5945f,0.6169f,0.6393f,0.6618f,0.6842f,0.7066f,
       0.7291f,0.7515f,0.7739f,0.7964f,0.8188f,0.8412f,0.8637f,0.8861f,
       0.9086f,0.931f,0.9534f,0.9759f,0.9983f,1.0207f,1.032f,1.0544f,
       1.0769f,1.0993f,1.1218f,1.1442f,1.1667f,1.1891f,1.2116f,1.2341f,
       1.2565f,1.279f,1.3015f,1.324f,1.3465f,1.3691f,1.3916f,1.4142f,
       1.448f,1.4706f,1.4933f,1.5159f,1.5387f,1.5614f,1.5842f,1.607f,
       1.6299f,1.6529f,1.6759f,1.699f,1.7222f,1.7455f,1.7689f,1.7924f,
       1.8161f,1.8399f,1.8638f,1.888f,1.9123f,1.9369f,1.9617f,1.9868f,
       2.0121f,2.0378f,2.0639f,2.0903f,2.1171f,2.1443f,2.1721f,2.2004f,
       2.2292f,2.2587f,2.2889f,2.3198f,2.3516f,2.3842f,2.4177f,2.4523f,
       2.488f,2.5249f,2.5632f,2.603f,2.6444f,2.6875f,2.7326f,2.7799f,
       2.8296f,2.882f,2.9375f,2.967f,3.028f,3.0933f,3.1636f,3.24f,3.3236f,
       3.4161f,3.52f,3.6388f,3.7782f,3.9485f,4.1704f,4.7062f};
    switch (bits) {
      case 1: return c1; case 2: return c2; case 3: return c3;
      case 4: return c4; case 5: return c5; case 6: return c6;
      case 7: return c7; case 8: return c8; default: return nullptr;
    }
  }

  // Scale N(0,1) centroids by σ = 1/√d so rotated coordinates match ~N(0,1/d).
  static std::vector<float> build_lm_codebook(std::size_t bits, std::size_t d) {
    if (bits == 0) return {};
    const float* cptr = lm_centroids_table(bits);
    const std::size_t levels = std::size_t{1} << bits;
    const float sigma = 1.0f / std::sqrt(static_cast<float>(d));
    std::vector<float> out(levels);
    if (cptr != nullptr) {
      for (std::size_t i = 0; i < levels; ++i) out[i] = cptr[i] * sigma;
    } else {
      // Fallback: use uniform codebook as-is (bits > 8, no precomputed table).
      for (std::size_t i = 0; i < levels; ++i)
        out[i] = -1.0f + 2.0f * static_cast<float>(i + 0.5f) / static_cast<float>(levels);
    }
    return out;
  }

 private:

  // Gaussian LM: thresholds = boundary_table × σ.
  void build_lm_thresholds(std::size_t bits, std::size_t d) {
    const float* bptr = lm_boundaries_table(bits);
    const std::size_t nb = (std::size_t{1} << bits) - 1;
    const float sigma = 1.0f / std::sqrt(static_cast<float>(d));
    thresholds.resize(nb);
    if (bptr != nullptr) {
      for (std::size_t i = 0; i < nb; ++i) thresholds[i] = bptr[i] * sigma;
    } else {
      // Fallback: compute midpoint thresholds from centroids.
      for (std::size_t i = 0; i < nb; ++i)
        thresholds[i] = 0.5f * (centroids[i] + centroids[i + 1]);
    }
  }
};

// =====================================================================
// StorageLayout — buffer management, SIMD scoring, and postprocessing
//
// Owns all compressed database buffers and per-vector metadata (norms,
// gammas, residual scales).  Also implements the hot-path scoring and
// postprocessing methods that operate on those buffers.
//
// Three storage paths (decided at train() time):
//   kPackedNibble — 2 nibbles/byte, block-32 VPSHUFB (fastest, bitwidth ≤ 4)
//   kNibble       — 1 nibble/byte,  block-16 VPSHUFB (bitwidth ≤ 4, fallback)
//   kGeneric      — full byte codes + optional sign bits (wider bitwidths)
// =====================================================================
struct StorageLayout {
  enum class Path { kPackedNibble, kNibble, kGeneric };

  Path        path                  = Path::kGeneric;
  std::size_t padded_dim            = 0;
  std::size_t mse_bits              = 0;
  std::size_t codebook_size         = 0;

  // Strides (bytes per block for each buffer type; 0 if not active)
  std::size_t packed_block_stride   = 0;
  std::size_t nibble_block_stride   = 0;
  std::size_t byte_code_block_stride = 0;
  std::size_t sign_block_stride     = 0;
  std::size_t lut_stride            = 0;  // generic path only

  // Generic-path flags
  bool        combined_code_sign    = false;
  std::size_t storage_bits          = 0;

  // Compressed DB buffers
  std::vector<std::uint8_t> nibbles;          // kNibble path
  std::vector<std::uint8_t> packed_nibbles;   // kPackedNibble path
  std::vector<std::uint8_t> byte_codes;       // kGeneric path
  std::vector<std::uint8_t> packed_signs;     // kGeneric IP path (separate sign bits)

  // Per-vector metadata (indexed by global slot index)
  std::vector<float> gammas;           // QJL residual norms (nibble + packed paths)
  std::vector<float> residual_scales;  // γ × coeff × ‖x_eff‖ (generic IP path)
  std::vector<float> norms;            // ‖x_eff‖
  std::vector<float> norm_squares;     // ‖x_eff‖²

  // Effective block size for this path.
  std::size_t eff_block_size() const noexcept {
    return (path == Path::kPackedNibble) ? detail::kPackedBlockSize : detail::kBlockSize;
  }

  // ------------------------------------------------------------------
  // configure — select path and compute strides from codebook/mode info.
  // Called during train(); must precede any add() or query().
  // ------------------------------------------------------------------
  void configure(std::size_t mse_bits_, std::size_t bitwidth, bool force_generic,
                 bool use_packed_nibbles_, std::size_t padded_dim_, Mode mode,
                 std::size_t codebook_size_) {
    padded_dim    = padded_dim_;
    mse_bits      = mse_bits_;
    codebook_size = codebook_size_;

    // Nibble path: bitwidth 1–4 all fit in VPSHUFB's 16-entry table.
    // mse_bits == 0 (1-bit IP, no MSE codes) is degenerate — skip.
    const bool nibble_ok = (!force_generic && mse_bits_ >= 1 && bitwidth <= 4);
    const bool packed_ok = (nibble_ok && use_packed_nibbles_);

    if (packed_ok) {
      path                  = Path::kPackedNibble;
      packed_block_stride   = detail::aligned_bytes(padded_dim_ * detail::kPackedBlockSize / 2);
      nibble_block_stride   = 0;
      byte_code_block_stride = 0;
      sign_block_stride     = 0;
      lut_stride            = 0;
      combined_code_sign    = false;
      storage_bits          = 0;
    } else if (nibble_ok) {
      path                  = Path::kNibble;
      nibble_block_stride   = detail::aligned_bytes(padded_dim_ * detail::kBlockSize);
      packed_block_stride   = 0;
      byte_code_block_stride = 0;
      sign_block_stride     = 0;
      lut_stride            = 0;
      combined_code_sign    = false;
      storage_bits          = 0;
    } else {
      path               = Path::kGeneric;
      combined_code_sign = (mode == Mode::kInnerProduct && mse_bits_ >= 1 && mse_bits_ <= 3);
      storage_bits       = combined_code_sign ? (mse_bits_ + 1) : mse_bits_;
      if (storage_bits >= 1 && storage_bits <= 4)      lut_stride = 16;
      else if (storage_bits == 5)                      lut_stride = 32;
      else                                             lut_stride = codebook_size_;
      byte_code_block_stride = detail::aligned_bytes(padded_dim_ * detail::kBlockSize);
      sign_block_stride  = (mode == Mode::kInnerProduct && !combined_code_sign)
                           ? detail::aligned_bytes(padded_dim_ * 2) : 0;
      packed_block_stride   = 0;
      nibble_block_stride   = 0;
    }
  }

  void clear_buffers() {
    nibbles.clear(); packed_nibbles.clear();
    byte_codes.clear(); packed_signs.clear();
    gammas.clear(); residual_scales.clear();
    norms.clear(); norm_squares.clear();
  }

  // ------------------------------------------------------------------
  // Buffer accessors — inline wrappers: zero runtime overhead
  // ------------------------------------------------------------------
  std::uint8_t* nibble_dim_ptr(std::size_t block, std::size_t d) {
    return nibbles.data() + block * nibble_block_stride + d * detail::kBlockSize;
  }
  const std::uint8_t* nibble_dim_ptr(std::size_t block, std::size_t d) const {
    return nibbles.data() + block * nibble_block_stride + d * detail::kBlockSize;
  }

  std::uint8_t* packed_nibble_block_ptr(std::size_t block) {
    return packed_nibbles.data() + block * packed_block_stride;
  }
  const std::uint8_t* packed_nibble_block_ptr(std::size_t block) const {
    return packed_nibbles.data() + block * packed_block_stride;
  }

  std::uint8_t* byte_code_dim_ptr(std::size_t block, std::size_t d) {
    return byte_codes.data() + block * byte_code_block_stride + d * detail::kBlockSize;
  }
  const std::uint8_t* byte_code_dim_ptr(std::size_t block, std::size_t d) const {
    return byte_codes.data() + block * byte_code_block_stride + d * detail::kBlockSize;
  }

  std::uint8_t* sign_dim_ptr(std::size_t block, std::size_t d) {
    return packed_signs.data() + block * sign_block_stride + d * 2;
  }
  const std::uint8_t* sign_dim_ptr(std::size_t block, std::size_t d) const {
    return packed_signs.data() + block * sign_block_stride + d * 2;
  }

  // ------------------------------------------------------------------
  // build_float_lut — generic-path float LUT rows: lut[j,k] = q_j · codebook[k]
  // One row per dimension, lut_stride entries (padded) for gather / permute SIMD.
  // ------------------------------------------------------------------
  void build_float_lut(const float* rotated_q, float* lut, const Codebook& cb) const {
    const std::size_t n_codes = combined_code_sign ? (1u << storage_bits) : cb.size;
    for (std::size_t j = 0; j < padded_dim; ++j) {
      const float q = rotated_q[j];
      float* row = lut + j * lut_stride;
      for (std::size_t c = 0; c < n_codes; ++c) row[c] = q * cb.centroids[c % cb.size];
      for (std::size_t c = n_codes; c < lut_stride; ++c) row[c] = 0.f;
    }
  }

  // ------------------------------------------------------------------
  // score_block16_int8 — block-16 VPSHUFB int8 scoring
  // For each of up to 16 DB vectors in parallel:
  //   raw[c] ≈ Σ_d base_i8[d, nib[d,c]] · scale  (+ γ·QJL term if IP)
  // Θ(padded_dim · 16) work per block.
  // ------------------------------------------------------------------
  void score_block16_int8(const std::uint8_t* nib_ptr,
                          const std::int8_t* base_i8,
                          const std::int8_t* qjl_i8,
                          const float* gammas_ptr,
                          float base_scale, float qjl_scale_v,
                          std::size_t bs, float* scores) const {
    if (bs == detail::kBlockSize) {
      __m256i base_acc16 = _mm256_setzero_si256();
      __m512i base_acc32 = _mm512_setzero_si512();
      __m256i qjl_acc16  = _mm256_setzero_si256();
      __m512i qjl_acc32  = _mm512_setzero_si512();
      for (std::size_t d = 0; d < padded_dim; ++d) {
        const __m128i nibs = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(nib_ptr + d * detail::kBlockSize));
        const __m128i blut = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(base_i8 + d * 16));
        base_acc16 = _mm256_add_epi16(base_acc16,
            _mm256_cvtepi8_epi16(_mm_shuffle_epi8(blut, nibs)));
        if (qjl_i8) {
          const __m128i qlut = _mm_loadu_si128(
              reinterpret_cast<const __m128i*>(qjl_i8 + d * 16));
          qjl_acc16 = _mm256_add_epi16(qjl_acc16,
              _mm256_cvtepi8_epi16(_mm_shuffle_epi8(qlut, nibs)));
        }
        if ((d & (detail::kDrainInterval - 1)) == (detail::kDrainInterval - 1)) {
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
        __m512 qjl_f = _mm512_mul_ps(_mm512_cvtepi32_ps(qjl_acc32), _mm512_set1_ps(qjl_scale_v));
        _mm512_storeu_ps(scores, _mm512_fmadd_ps(_mm512_loadu_ps(gammas_ptr), qjl_f, base_f));
      } else {
        _mm512_storeu_ps(scores, base_f);
      }
      return;
    }
    // Scalar tail
    for (std::size_t c = 0; c < bs; ++c) {
      float bsum = 0.f, qsum = 0.f;
      for (std::size_t d = 0; d < padded_dim; ++d) {
        const std::uint8_t nib = nib_ptr[d * detail::kBlockSize + c];
        bsum += static_cast<float>(base_i8[d * 16 + nib]);
        if (qjl_i8) qsum += static_cast<float>(qjl_i8[d * 16 + nib]);
      }
      scores[c] = bsum * base_scale;
      if (qjl_i8) scores[c] += gammas_ptr[c] * qsum * qjl_scale_v;
    }
  }

  // ------------------------------------------------------------------
  // score_block32_packed — block-32 packed scoring: 2 nibbles/byte, 32 lanes
  // Same math as score_block16_int8 but processes lo and hi nibbles separately.
  // ------------------------------------------------------------------
  void score_block32_packed(const std::uint8_t* packed,
                            const std::int8_t* base_i8,
                            const std::int8_t* qjl_i8,
                            const float* gammas_ptr,
                            float base_scale, float qjl_scale_v,
                            std::size_t bs, float* scores) const {
    if (bs == detail::kPackedBlockSize) {
      const __m128i lo_mask = _mm_set1_epi8(0x0f);
      __m256i b_lo_acc16 = _mm256_setzero_si256();
      __m256i b_hi_acc16 = _mm256_setzero_si256();
      __m512i b_lo_acc32 = _mm512_setzero_si512();
      __m512i b_hi_acc32 = _mm512_setzero_si512();
      __m256i q_lo_acc16 = _mm256_setzero_si256();
      __m256i q_hi_acc16 = _mm256_setzero_si256();
      __m512i q_lo_acc32 = _mm512_setzero_si512();
      __m512i q_hi_acc32 = _mm512_setzero_si512();
      for (std::size_t d = 0; d < padded_dim; ++d) {
        const __m128i pk = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(packed + d * 16));
        const __m128i lo = _mm_and_si128(pk, lo_mask);
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(pk, 4), lo_mask);
        const __m128i blut = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(base_i8 + d * 16));
        b_lo_acc16 = _mm256_add_epi16(b_lo_acc16,
            _mm256_cvtepi8_epi16(_mm_shuffle_epi8(blut, lo)));
        b_hi_acc16 = _mm256_add_epi16(b_hi_acc16,
            _mm256_cvtepi8_epi16(_mm_shuffle_epi8(blut, hi)));
        if (qjl_i8) {
          const __m128i qlut = _mm_loadu_si128(
              reinterpret_cast<const __m128i*>(qjl_i8 + d * 16));
          q_lo_acc16 = _mm256_add_epi16(q_lo_acc16,
              _mm256_cvtepi8_epi16(_mm_shuffle_epi8(qlut, lo)));
          q_hi_acc16 = _mm256_add_epi16(q_hi_acc16,
              _mm256_cvtepi8_epi16(_mm_shuffle_epi8(qlut, hi)));
        }
        if ((d & (detail::kDrainInterval - 1)) == (detail::kDrainInterval - 1)) {
          b_lo_acc32 = _mm512_add_epi32(b_lo_acc32, _mm512_cvtepi16_epi32(b_lo_acc16));
          b_hi_acc32 = _mm512_add_epi32(b_hi_acc32, _mm512_cvtepi16_epi32(b_hi_acc16));
          b_lo_acc16 = _mm256_setzero_si256();
          b_hi_acc16 = _mm256_setzero_si256();
          if (qjl_i8) {
            q_lo_acc32 = _mm512_add_epi32(q_lo_acc32, _mm512_cvtepi16_epi32(q_lo_acc16));
            q_hi_acc32 = _mm512_add_epi32(q_hi_acc32, _mm512_cvtepi16_epi32(q_hi_acc16));
            q_lo_acc16 = _mm256_setzero_si256();
            q_hi_acc16 = _mm256_setzero_si256();
          }
        }
      }
      b_lo_acc32 = _mm512_add_epi32(b_lo_acc32, _mm512_cvtepi16_epi32(b_lo_acc16));
      b_hi_acc32 = _mm512_add_epi32(b_hi_acc32, _mm512_cvtepi16_epi32(b_hi_acc16));
      __m512 blo_f = _mm512_mul_ps(_mm512_cvtepi32_ps(b_lo_acc32), _mm512_set1_ps(base_scale));
      __m512 bhi_f = _mm512_mul_ps(_mm512_cvtepi32_ps(b_hi_acc32), _mm512_set1_ps(base_scale));
      if (qjl_i8) {
        q_lo_acc32 = _mm512_add_epi32(q_lo_acc32, _mm512_cvtepi16_epi32(q_lo_acc16));
        q_hi_acc32 = _mm512_add_epi32(q_hi_acc32, _mm512_cvtepi16_epi32(q_hi_acc16));
        __m512 qlo_f = _mm512_mul_ps(_mm512_cvtepi32_ps(q_lo_acc32), _mm512_set1_ps(qjl_scale_v));
        __m512 qhi_f = _mm512_mul_ps(_mm512_cvtepi32_ps(q_hi_acc32), _mm512_set1_ps(qjl_scale_v));
        _mm512_storeu_ps(scores,      _mm512_fmadd_ps(_mm512_loadu_ps(gammas_ptr),      qlo_f, blo_f));
        _mm512_storeu_ps(scores + 16, _mm512_fmadd_ps(_mm512_loadu_ps(gammas_ptr + 16), qhi_f, bhi_f));
      } else {
        _mm512_storeu_ps(scores,      blo_f);
        _mm512_storeu_ps(scores + 16, bhi_f);
      }
      return;
    }
    // Scalar tail for partial block (bs < 32)
    for (std::size_t c = 0; c < bs; ++c) {
      float bsum = 0.f, qsum = 0.f;
      for (std::size_t d = 0; d < padded_dim; ++d) {
        const std::size_t byte_idx = d * 16 + (c < 16 ? c : c - 16);
        const std::uint8_t raw = packed[byte_idx];
        const std::uint8_t nib = (c < 16) ? (raw & 0x0f) : (raw >> 4);
        bsum += static_cast<float>(base_i8[d * 16 + nib]);
        if (qjl_i8) qsum += static_cast<float>(qjl_i8[d * 16 + nib]);
      }
      scores[c] = bsum * base_scale;
      if (qjl_i8) scores[c] += gammas_ptr[c] * qsum * qjl_scale_v;
    }
  }

  // ------------------------------------------------------------------
  // score_generic_code_block — float LUT gather for MSE codes only.
  // out[i] += Σ_j lut[j, code_{j,i}] for bs vectors in block bi.
  // ------------------------------------------------------------------
  void score_generic_code_block(std::size_t bi, std::size_t bs,
                                const float* lut, float* out) const {
    if (mse_bits == 0 || bs == 0) return;
    if (bs == detail::kBlockSize && mse_bits <= 4) {
      __m512 acc = _mm512_setzero_ps();
      for (std::size_t j = 0; j < padded_dim; ++j) {
        const __m128i c = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(byte_code_dim_ptr(bi, j)));
        acc = _mm512_add_ps(acc, _mm512_permutexvar_ps(
            _mm512_cvtepu8_epi32(c), _mm512_loadu_ps(lut + j * lut_stride)));
      }
      _mm512_storeu_ps(out, acc); return;
    }
    if (bs == detail::kBlockSize && mse_bits == 5) {
      __m512 acc = _mm512_setzero_ps();
      const __m512i fifteen = _mm512_set1_epi32(15);
      for (std::size_t j = 0; j < padded_dim; ++j) {
        const __m128i c = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(byte_code_dim_ptr(bi, j)));
        const __m512i idx = _mm512_cvtepu8_epi32(c);
        const float* row = lut + j * lut_stride;
        const __m512i mod = _mm512_and_si512(idx, fifteen);
        const __mmask16 hm = _mm512_cmpgt_epi32_mask(idx, fifteen);
        acc = _mm512_add_ps(acc, _mm512_mask_blend_ps(hm,
            _mm512_permutexvar_ps(mod, _mm512_loadu_ps(row)),
            _mm512_permutexvar_ps(mod, _mm512_loadu_ps(row + 16))));
      }
      _mm512_storeu_ps(out, acc); return;
    }
    if (bs == detail::kBlockSize) {
      __m512 acc = _mm512_setzero_ps();
      for (std::size_t j = 0; j < padded_dim; ++j) {
        const __m128i c = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(byte_code_dim_ptr(bi, j)));
        acc = _mm512_add_ps(acc, _mm512_i32gather_ps(
            _mm512_cvtepu8_epi32(c), lut + j * lut_stride, 4));
      }
      _mm512_storeu_ps(out, acc); return;
    }
    for (std::size_t j = 0; j < padded_dim; ++j) {
      const std::uint8_t* bc = byte_code_dim_ptr(bi, j);
      const float* row = lut + j * lut_stride;
      for (std::size_t i = 0; i < bs; ++i) out[i] += row[bc[i]];
    }
  }

  // ------------------------------------------------------------------
  // score_generic_code_sign_block — IP generic path:
  // splits contribution from MSE codes (code_out) vs QJL sign term (sign_out).
  // ------------------------------------------------------------------
  void score_generic_code_sign_block(std::size_t bi, std::size_t bs, const float* lut,
                                     const float* proj_q,
                                     float* code_out, float* sign_out) const {
    if (bs == 0) return;
    if (bs == detail::kBlockSize) {
      __m512 cacc = _mm512_setzero_ps(), sacc = _mm512_setzero_ps();
      if (combined_code_sign) {
        const std::uint32_t cmask_val = (1u << mse_bits) - 1u;
        const __m512i cmask_v = _mm512_set1_epi32(static_cast<int>(cmask_val));
        const unsigned shift = static_cast<unsigned>(mse_bits);
        for (std::size_t j = 0; j < padded_dim; ++j) {
          const __m512i combined = _mm512_cvtepu8_epi32(
              _mm_loadu_si128(reinterpret_cast<const __m128i*>(byte_code_dim_ptr(bi, j))));
          const __m512i codes = _mm512_and_si512(combined, cmask_v);
          if (mse_bits <= 4)
            cacc = _mm512_add_ps(cacc,
                _mm512_permutexvar_ps(codes, _mm512_loadu_ps(lut + j * lut_stride)));
          else
            cacc = _mm512_add_ps(cacc,
                _mm512_i32gather_ps(codes, lut + j * lut_stride, 4));
          const __mmask16 smask = _mm512_test_epi32_mask(
              _mm512_srli_epi32(combined, shift), _mm512_set1_epi32(1));
          sacc = _mm512_add_ps(sacc, _mm512_mask_blend_ps(smask,
              _mm512_set1_ps(-proj_q[j]), _mm512_set1_ps(proj_q[j])));
        }
      } else {
        for (std::size_t j = 0; j < padded_dim; ++j) {
          if (mse_bits > 0) {
            const __m512i idx = _mm512_cvtepu8_epi32(
                _mm_loadu_si128(reinterpret_cast<const __m128i*>(byte_code_dim_ptr(bi, j))));
            if (mse_bits <= 4)
              cacc = _mm512_add_ps(cacc,
                  _mm512_permutexvar_ps(idx, _mm512_loadu_ps(lut + j * lut_stride)));
            else if (mse_bits == 5) {
              const __m512i mod = _mm512_and_si512(idx, _mm512_set1_epi32(15));
              const float* row = lut + j * lut_stride;
              cacc = _mm512_add_ps(cacc, _mm512_mask_blend_ps(
                  _mm512_cmpgt_epi32_mask(idx, _mm512_set1_epi32(15)),
                  _mm512_permutexvar_ps(mod, _mm512_loadu_ps(row)),
                  _mm512_permutexvar_ps(mod, _mm512_loadu_ps(row + 16))));
            } else {
              cacc = _mm512_add_ps(cacc,
                  _mm512_i32gather_ps(idx, lut + j * lut_stride, 4));
            }
          }
          std::uint16_t sbits;
          std::memcpy(&sbits, sign_dim_ptr(bi, j), 2);
          sacc = _mm512_add_ps(sacc, _mm512_mask_blend_ps(
              static_cast<__mmask16>(sbits),
              _mm512_set1_ps(-proj_q[j]), _mm512_set1_ps(proj_q[j])));
        }
      }
      _mm512_storeu_ps(code_out, cacc);
      _mm512_storeu_ps(sign_out, sacc);
      return;
    }
    // Scalar tail
    const std::uint32_t cmask_val = mse_bits > 0 ? ((1u << mse_bits) - 1u) : 0u;
    for (std::size_t j = 0; j < padded_dim; ++j) {
      const std::uint8_t* bc = byte_code_dim_ptr(bi, j);
      const float* row = lut + j * lut_stride;
      if (combined_code_sign) {
        for (std::size_t i = 0; i < bs; ++i) {
          if (mse_bits > 0) code_out[i] += row[bc[i] & cmask_val];
          sign_out[i] += ((bc[i] >> mse_bits) & 1u) ? proj_q[j] : -proj_q[j];
        }
      } else {
        if (mse_bits > 0) for (std::size_t i = 0; i < bs; ++i) code_out[i] += row[bc[i]];
        std::uint16_t sbits; std::memcpy(&sbits, sign_dim_ptr(bi, j), 2);
        for (std::size_t i = 0; i < bs; ++i)
          sign_out[i] += ((sbits >> i) & 1u) ? proj_q[j] : -proj_q[j];
      }
    }
  }

  // ------------------------------------------------------------------
  // Postprocessing — convert raw inner-product proxy to final score/distance.
  //
  // raw[i] ≈ ⟨q_unit, x_unit_i⟩ (from quantized score / scales).
  // dot ≈ ‖q_eff‖ · ‖x_eff_i‖ · raw  ≈ ⟨q_eff, x_eff_i⟩.
  // L2: ‖q_eff − x_eff_i‖² = ‖q_eff‖² + ‖x_eff_i‖² − 2·dot.
  // rank_keys: negate L2 for max-heap (smaller distance → better rank).
  // ------------------------------------------------------------------

  // Nibble / packed path postprocess (shared formula).
  template <bool kL2>
  void postprocess_nibble(std::size_t db0, std::size_t bs,
                          float q_eff_norm, float q_eff_norm_sq,
                          const float* raw, float* values, float* rank_keys) const {
    if (bs == detail::kBlockSize) {
      const __m512 norms_v  = _mm512_loadu_ps(norms.data() + db0);
      const __m512 normsq_v = _mm512_loadu_ps(norm_squares.data() + db0);
      const __m512 score    = _mm512_loadu_ps(raw);
      if constexpr (kL2) {
        const __m512 cross = _mm512_mul_ps(_mm512_set1_ps(2.0f * q_eff_norm),
                                           _mm512_mul_ps(norms_v, score));
        const __m512 val = _mm512_sub_ps(
            _mm512_add_ps(_mm512_set1_ps(q_eff_norm_sq), normsq_v), cross);
        _mm512_storeu_ps(values, val);
        _mm512_storeu_ps(rank_keys, _mm512_sub_ps(_mm512_setzero_ps(), val));
      } else {
        const __m512 dot = _mm512_mul_ps(_mm512_set1_ps(q_eff_norm),
                                         _mm512_mul_ps(norms_v, score));
        _mm512_storeu_ps(values, dot);
        _mm512_storeu_ps(rank_keys, dot);
      }
      return;
    }
    for (std::size_t i = 0; i < bs; ++i) {
      if constexpr (kL2) {
        const float val = q_eff_norm_sq + norm_squares[db0 + i]
                          - 2.0f * q_eff_norm * norms[db0 + i] * raw[i];
        values[i] = val; rank_keys[i] = -val;
      } else {
        const float dot = q_eff_norm * norms[db0 + i] * raw[i];
        values[i] = dot; rank_keys[i] = dot;
      }
    }
  }

  // Packed path postprocess: same formulas, two AVX-512 halves of 32 vectors.
  template <bool kL2>
  void postprocess_packed(std::size_t db0, std::size_t bs,
                          float q_eff_norm, float q_eff_norm_sq,
                          const float* raw, float* values, float* rank_keys) const {
    if (bs == detail::kPackedBlockSize) {
      for (std::size_t half = 0; half < 2; ++half) {
        const std::size_t off = half * 16;
        const __m512 norms_v  = _mm512_loadu_ps(norms.data() + db0 + off);
        const __m512 normsq_v = _mm512_loadu_ps(norm_squares.data() + db0 + off);
        const __m512 score    = _mm512_loadu_ps(raw + off);
        if constexpr (kL2) {
          const __m512 cross = _mm512_mul_ps(_mm512_set1_ps(2.0f * q_eff_norm),
                                             _mm512_mul_ps(norms_v, score));
          const __m512 val = _mm512_sub_ps(
              _mm512_add_ps(_mm512_set1_ps(q_eff_norm_sq), normsq_v), cross);
          _mm512_storeu_ps(values + off, val);
          _mm512_storeu_ps(rank_keys + off, _mm512_sub_ps(_mm512_setzero_ps(), val));
        } else {
          const __m512 dot = _mm512_mul_ps(_mm512_set1_ps(q_eff_norm),
                                           _mm512_mul_ps(norms_v, score));
          _mm512_storeu_ps(values + off, dot);
          _mm512_storeu_ps(rank_keys + off, dot);
        }
      }
      return;
    }
    for (std::size_t i = 0; i < bs; ++i) {
      if constexpr (kL2) {
        const float val = q_eff_norm_sq + norm_squares[db0 + i]
                          - 2.0f * q_eff_norm * norms[db0 + i] * raw[i];
        values[i] = val; rank_keys[i] = -val;
      } else {
        const float dot = q_eff_norm * norms[db0 + i] * raw[i];
        values[i] = dot; rank_keys[i] = dot;
      }
    }
  }

  // Generic path postprocess: optional QJL residual correction from scratch[].
  template <bool kUseResidual, bool kL2>
  void postprocess_generic(std::size_t db0, std::size_t bs,
                           float q_eff_norm, float q_eff_norm_sq,
                           float* dot_scores, const float* scratch,
                           float* values, float* rank_keys) const {
    if (bs == detail::kBlockSize) {
      const __m512 norms_v  = _mm512_loadu_ps(norms.data() + db0);
      const __m512 normsq_v = _mm512_loadu_ps(norm_squares.data() + db0);
      __m512 score = _mm512_loadu_ps(dot_scores);
      if constexpr (kUseResidual) {
        const __m512 inv_norms = _mm512_div_ps(_mm512_set1_ps(1.0f), norms_v);
        const __m512 gc = _mm512_mul_ps(
            _mm512_loadu_ps(residual_scales.data() + db0), inv_norms);
        score = _mm512_fmadd_ps(gc, _mm512_loadu_ps(scratch), score);
      }
      if constexpr (kL2) {
        const __m512 cross = _mm512_mul_ps(_mm512_set1_ps(2.0f * q_eff_norm),
                                           _mm512_mul_ps(norms_v, score));
        const __m512 val = _mm512_sub_ps(
            _mm512_add_ps(_mm512_set1_ps(q_eff_norm_sq), normsq_v), cross);
        _mm512_storeu_ps(values, val);
        _mm512_storeu_ps(rank_keys, _mm512_sub_ps(_mm512_setzero_ps(), val));
      } else {
        const __m512 dot = _mm512_mul_ps(_mm512_set1_ps(q_eff_norm),
                                         _mm512_mul_ps(norms_v, score));
        _mm512_storeu_ps(values, dot);
        _mm512_storeu_ps(rank_keys, dot);
      }
      return;
    }
    for (std::size_t i = 0; i < bs; ++i) {
      float ip = dot_scores[i];
      if constexpr (kUseResidual) {
        const float gc = (norms[db0 + i] > 0.0f)
                          ? residual_scales[db0 + i] / norms[db0 + i] : 0.0f;
        ip += gc * scratch[i];
      }
      if constexpr (kL2) {
        const float val = q_eff_norm_sq + norm_squares[db0 + i]
                          - 2.0f * q_eff_norm * norms[db0 + i] * ip;
        values[i] = val; rank_keys[i] = -val;
      } else {
        const float dot = q_eff_norm * norms[db0 + i] * ip;
        values[i] = dot; rank_keys[i] = dot;
      }
    }
  }
};

// =====================================================================
// KMeansIVF — K-means++ initialization + IVF coarse search
//
// Used when nlist > 1; stores cluster centroids and per-cluster slot layout.
// train() runs K-means++ init + Lloyd iterations.
// coarse_search() returns top-nprobe cluster indices for one query.
// =====================================================================
struct KMeansIVF {
  std::size_t nlist  = 1;  // number of K-means clusters
  std::size_t nprobe = 1;  // clusters to probe per query

  std::vector<float>       centroids;   // [nlist * padded_dim] cluster centroids
  std::vector<std::size_t> list_start;  // [nlist] block-aligned slot start per cluster
  std::vector<std::size_t> list_size;   // [nlist] actual vector count per cluster
  std::vector<std::int64_t> ids;        // [total_slots] storage-slot → original add-order index

  bool active() const noexcept { return nlist > 1; }

  void setup(std::size_t nlist_, std::size_t nprobe_) {
    nlist  = std::max<std::size_t>(1, nlist_);
    nprobe = std::min(std::max<std::size_t>(1, nprobe_), nlist);
  }

  // Run K-means++ on `data` (n × dim) and initialise cluster bookkeeping.
  void train(const float* data, std::size_t n, std::size_t dim, std::size_t padded_dim,
             std::uint64_t seed, std::size_t num_threads) {
    centroids.resize(nlist * padded_dim, 0.f);
    run_kmeans(data, n, dim, padded_dim, seed, num_threads);
    list_start.assign(nlist, 0);
    list_size.assign(nlist, 0);
    ids.clear();
  }

  // SIMD L2 distance to all cluster centroids → rank top-nprobe, sorted closest-first.
  void coarse_search(const float* qptr, std::size_t dim, std::size_t padded_dim,
                     std::vector<float>& cdists, std::vector<std::size_t>& probe_order) const {
    for (std::size_t ci = 0; ci < nlist; ++ci) {
      if (ci + 2 < nlist)
        _mm_prefetch(reinterpret_cast<const char*>(
            centroids.data() + (ci + 2) * padded_dim), _MM_HINT_T0);
      cdists[ci]      = detail::l2_sq_distance(qptr, centroids.data() + ci * padded_dim, dim);
      probe_order[ci] = ci;
    }
    if (nprobe < nlist)
      std::nth_element(probe_order.begin(), probe_order.begin() + nprobe, probe_order.end(),
          [&](std::size_t a, std::size_t b){ return cdists[a] < cdists[b]; });
    std::sort(probe_order.begin(), probe_order.begin() + nprobe,
        [&](std::size_t a, std::size_t b){ return cdists[a] < cdists[b]; });
  }

 private:
  // K-means++ init + Lloyd iterations with AVX-512 FMA assignment.
  // Subsamples to min(n, 256*nlist) training vectors to cap cost.
  void run_kmeans(const float* data, std::size_t n, std::size_t dim, std::size_t padded_dim,
                  std::uint64_t seed, std::size_t num_threads) {
    const std::size_t k     = nlist;
    const std::size_t d     = dim;
    const std::size_t max_n = std::min(n, 256 * k);

    // Partial Fisher-Yates: pick max_n indices uniformly at random from [0, n)
    std::vector<std::size_t> sample(n);
    std::iota(sample.begin(), sample.end(), 0);
    if (max_n < n) {
      std::mt19937_64 rng(seed ^ 0xabcdef0123456789ULL);
      for (std::size_t i = 0; i < max_n; ++i) {
        std::size_t j = i + rng() % (n - i);
        std::swap(sample[i], sample[j]);
      }
      sample.resize(max_n);
    }

    std::vector<float> cents(k * d, 0.f);

    // K-means++ initialization
    {
      std::mt19937_64 rng(seed ^ 0x9e3779b97f4a7c15ULL);
      std::uniform_int_distribution<std::size_t> uni(0, max_n - 1);
      std::size_t first = sample[uni(rng)];
      std::copy(data + first * d, data + first * d + d, cents.data());

      std::vector<float> min_sq(max_n, std::numeric_limits<float>::max());
      for (std::size_t ci = 1; ci < k; ++ci) {
        const float* last = cents.data() + (ci - 1) * d;
        for (std::size_t i = 0; i < max_n; ++i) {
          const float* xi = data + sample[i] * d;
          float dsq = detail::l2_sq_distance(xi, last, d);
          if (dsq < min_sq[i]) min_sq[i] = dsq;
        }
        float total = 0.f;
        for (float v : min_sq) total += v;
        std::uniform_real_distribution<float> ureal(0.f, total > 0.f ? total : 1.f);
        float tgt = ureal(rng), acc = 0.f;
        std::size_t chosen = max_n - 1;
        for (std::size_t i = 0; i < max_n; ++i) {
          acc += min_sq[i];
          if (acc >= tgt) { chosen = i; break; }
        }
        std::copy(data + sample[chosen] * d, data + sample[chosen] * d + d,
                  cents.data() + ci * d);
      }
    }

    // Lloyd iterations
    std::vector<std::size_t> asgn(max_n, 0);
    for (std::size_t iter = 0; iter < 25; ++iter) {
      std::size_t changed = 0;

      #pragma omp parallel for num_threads(static_cast<int>(num_threads)) \
          schedule(static) reduction(+:changed)
      for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(max_n); ++ii) {
        const std::size_t i = static_cast<std::size_t>(ii);
        const float* xi = data + sample[i] * d;
        float best_dsq = std::numeric_limits<float>::max();
        std::size_t best_k = asgn[i];
        for (std::size_t c = 0; c < k; ++c) {
          float dsq = detail::l2_sq_distance(xi, cents.data() + c * d, d);
          if (dsq < best_dsq) { best_dsq = dsq; best_k = c; }
        }
        if (asgn[i] != best_k) { asgn[i] = best_k; ++changed; }
      }

      if (iter > 0 && changed == 0) break;

      // Recompute centroids as cluster means
      std::fill(cents.begin(), cents.end(), 0.f);
      std::vector<std::size_t> cnt(k, 0);
      for (std::size_t i = 0; i < max_n; ++i) {
        const std::size_t c = asgn[i];
        const float* xi = data + sample[i] * d;
        float* cc = cents.data() + c * d;
        for (std::size_t j = 0; j < d; ++j) cc[j] += xi[j];
        ++cnt[c];
      }
      for (std::size_t c = 0; c < k; ++c) {
        if (cnt[c] > 0) {
          const float inv = 1.f / static_cast<float>(cnt[c]);
          float* cc = cents.data() + c * d;
          for (std::size_t j = 0; j < d; ++j) cc[j] *= inv;
        } else {
          // Empty cluster: copy from a non-empty neighbor
          std::size_t src = (c + 1) % k;
          while (cnt[src] == 0) src = (src + 1) % k;
          std::copy(cents.data() + src * d, cents.data() + src * d + d, cents.data() + c * d);
        }
      }
    }

    // Store into centroids (padded to padded_dim, zeros beyond dim)
    std::fill(centroids.begin(), centroids.end(), 0.f);
    for (std::size_t c = 0; c < k; ++c)
      std::copy(cents.data() + c * d, cents.data() + (c + 1) * d,
                centroids.data() + c * padded_dim);
  }
};

// =====================================================================
// TurboQuantIndex: train once on a sample, then add() vectors and
// query() for top-k by inner product or L2.
// Time: query is linear in database size n (exhaustive scan over blocks).
// Space: O(n · compressed size + codebook).
// =====================================================================
class TurboQuantIndex {
 public:
  using idx_t = std::int64_t;

  // Re-export enums — TurboQuantIndex::Mode etc. still works (backward compatible).
  using Mode         = turboquant::Mode;
  using SearchMetric = turboquant::SearchMetric;
  using RotationType = turboquant::RotationType;
  using CodebookType = turboquant::CodebookType;

  // All hyperparameters in one struct.  After construction, call train(), then add().
  struct Config {
    std::size_t dim = 0;
    std::size_t bitwidth = 4;                               // 4-bit → nibble path (int8 LUT + VPSHUFB, ~2× faster)
    Mode mode = Mode::kInnerProduct;
    RotationType rotation_type = RotationType::kHadamard;  // O(d log d), zero-pads to power-of-2
    CodebookType codebook_type = CodebookType::kGaussianLM; // hardcoded LM tables, faster train()
    bool use_data_centroid = true;                          // centre-normalise: best recall on L2 search
    bool force_generic_path = false;                        // disable nibble path (for benchmarking)
    bool use_packed_nibbles = true;                          // pack 2 nibbles/byte, block-32 VPSHUFB
    std::uint64_t seed = 123456789ULL;
    std::size_t num_threads = 1;
    std::size_t nlist  = 1;   // IVF clusters; 1 = flat (current behavior preserved)
    std::size_t nprobe = 1;   // Clusters to probe per query (clamped to nlist)
  };

  // Validates dim and bitwidth; sets padded_dim_ = next power of 2 for Hadamard else dim.
  explicit TurboQuantIndex(const Config& cfg)
      : dim_(cfg.dim),
        bitwidth_(cfg.bitwidth),
        mode_(cfg.mode),
        rotation_type_(cfg.rotation_type),
        use_data_centroid_(cfg.use_data_centroid),
        force_generic_path_(cfg.force_generic_path),
        use_packed_nibbles_(cfg.use_packed_nibbles),
        seed_(cfg.seed),
        num_threads_(std::max<std::size_t>(1, cfg.num_threads)) {
    if (dim_ == 0)
      throw std::invalid_argument("TurboQuantIndex: dim must be > 0");
    if (bitwidth_ == 0 || bitwidth_ > 9)
      throw std::invalid_argument("TurboQuantIndex: bitwidth must be in [1, 9]");
    padded_dim_ = (rotation_type_ == RotationType::kHadamard)
                  ? detail::next_pow2(dim_) : dim_;
    ivf_.setup(cfg.nlist, cfg.nprobe);
  }

  // Original vector dimension d (before zero-padding for Hadamard).
  std::size_t dim()         const noexcept { return dim_; }
  // Number of database vectors currently stored (after add()).
  std::size_t ntotal()      const noexcept { return ntotal_; }
  // Bits per coordinate in the *stored* MSE code.
  std::size_t bitwidth()    const noexcept { return bitwidth_; }
  Mode        mode()        const noexcept { return mode_; }
  std::size_t num_threads() const noexcept { return num_threads_; }

  // Sets OpenMP thread count for parallel_for in add(), query(), reconstruct().
  void set_num_threads(std::size_t t) noexcept {
    num_threads_ = std::max<std::size_t>(1, t);
  }

  // ------------------------------------------------------------------
  // train — prepare centroid, codebook, rotations, and storage layout
  //
  // If use_data_centroid: c = (1/n) Σ_i x_i (component-wise mean).
  // Builds Lloyd–Max centroids for mse_bits = bitwidth (MSE) or bitwidth−1 (IP).
  // Picks packed-nibble, nibble, or generic byte layout for add().
  // Must be called before add().
  // ------------------------------------------------------------------
  void train(std::size_t n, const float* x) {
    // 1. Optionally compute data centroid
    if (use_data_centroid_) {
      if (n == 0 || x == nullptr)
        throw std::invalid_argument("train: n > 0 and x != null required for use_data_centroid");
      centroid_.assign(dim_, 0.0f);
      for (std::size_t i = 0; i < n; ++i) {
        const float* xi = x + i * dim_;
        for (std::size_t j = 0; j < dim_; ++j) centroid_[j] += xi[j];
      }
      const float inv_n = 1.0f / static_cast<float>(n);
      for (std::size_t j = 0; j < dim_; ++j) centroid_[j] *= inv_n;
      centroid_.resize(padded_dim_, 0.0f);  // zero-pad for internal use
    } else {
      centroid_.clear();
    }

    // 2. Build codebook (centroids + decision thresholds)
    mse_bits_ = (mode_ == Mode::kInnerProduct) ? (bitwidth_ - 1) : bitwidth_;
    codebook_.build(mse_bits_, padded_dim_, mode_);

    // 3. Decide storage path and compute strides
    storage_.configure(mse_bits_, bitwidth_, force_generic_path_, use_packed_nibbles_,
                       padded_dim_, mode_, codebook_.size);

    // 4. Generate rotation and (for IP) QJL transform
    rotation_.generate(rotation_type_, padded_dim_, seed_);
    if (mode_ == Mode::kInnerProduct)
      qjl_.generate_qjl(rotation_type_, padded_dim_, seed_ ^ 0x9e3779b97f4a7c15ULL);

    // 5. Clear storage buffers
    storage_.clear_buffers();
    ntotal_  = 0;
    trained_ = true;

    // 6. IVF: run K-means on training data to produce nlist_ cluster centroids
    if (ivf_.active()) {
      if (n == 0 || x == nullptr)
        throw std::invalid_argument("train: IVF requires n > 0 and x != null");
      ivf_.train(x, n, dim_, padded_dim_, seed_, num_threads_);
    }
  }

  // ------------------------------------------------------------------
  // add — encode n row-major float vectors x (shape n×dim) and append.
  //
  // For each vector: form x_eff, store ‖x_eff‖, normalize to u, compute v = R u,
  // quantize v coordinate-wise to codes, optionally QJL on residual.
  // Amortized O(n·d·cost(R)) per batch.
  // ------------------------------------------------------------------
  void add(std::size_t n, const float* x) {
    require_trained();
    if (x == nullptr && n != 0)
      throw std::invalid_argument("add: x must not be null");
    if (ivf_.active()) { add_ivf(n, x); return; }

    const std::size_t old_total = ntotal_;
    ntotal_ += n;
    const std::size_t eff_bs    = storage_.eff_block_size();
    const std::size_t new_blocks = detail::ceil_div(ntotal_, eff_bs);

    if (storage_.path == StorageLayout::Path::kPackedNibble) {
      storage_.packed_nibbles.resize(new_blocks * storage_.packed_block_stride, 0);
      storage_.gammas.resize(new_blocks * kPackedBlockSize, 0.f);
    } else if (storage_.path == StorageLayout::Path::kNibble) {
      storage_.nibbles.resize(new_blocks * storage_.nibble_block_stride, 0);
      storage_.gammas.resize(new_blocks * kBlockSize, 0.f);
    } else {
      storage_.byte_codes.resize(new_blocks * storage_.byte_code_block_stride, 0);
      if (storage_.sign_block_stride != 0)
        storage_.packed_signs.resize(new_blocks * storage_.sign_block_stride, 0);
      storage_.residual_scales.resize(new_blocks * kBlockSize, 0.f);
    }
    storage_.norms.resize(new_blocks * eff_bs, 0.f);
    storage_.norm_squares.resize(new_blocks * eff_bs, 0.f);

    const std::size_t start_block = old_total / eff_bs;
    parallel_for(start_block, new_blocks, [&](std::size_t b0, std::size_t b1) {
      std::vector<float> x_eff(padded_dim_, 0.0f);
      std::vector<float> unit(padded_dim_, 0.0f);
      std::vector<float> rotated(padded_dim_);
      std::vector<float> residual(padded_dim_);
      std::vector<float> projected(mode_ == Mode::kInnerProduct ? padded_dim_ : 0);
      std::vector<float> work(padded_dim_);
      std::vector<std::uint32_t> codes(padded_dim_);

      for (std::size_t bi = b0; bi < b1; ++bi) {
        const std::size_t block_begin = bi * eff_bs;
        const std::size_t block_end   = std::min(block_begin + eff_bs, ntotal_);
        const std::size_t first_new   = std::max(block_begin, old_total);
        if (first_new >= block_end) continue;

        for (std::size_t gi = first_new; gi < block_end; ++gi) {
          const std::size_t li   = gi - old_total;
          const std::size_t lane = gi - block_begin;
          const float* src = x + li * dim_;

          // Compute x_eff = (x - centroid) or x; store norm of x_eff
          std::fill(x_eff.begin(), x_eff.end(), 0.0f);
          if (use_data_centroid_) {
            for (std::size_t j = 0; j < dim_; ++j)
              x_eff[j] = src[j] - centroid_[j];
          } else {
            for (std::size_t j = 0; j < dim_; ++j)
              x_eff[j] = src[j];
          }
          const float norm = detail::l2_norm(x_eff.data(), dim_);
          storage_.norms[gi]        = norm;
          storage_.norm_squares[gi] = norm * norm;

          if (norm == 0.0f) {
            if (storage_.path != StorageLayout::Path::kGeneric) storage_.gammas[gi] = 0.0f;
            else                                                 storage_.residual_scales[gi] = 0.0f;
            continue;
          }

          // Normalise and rotate
          const float inv_norm = 1.0f / norm;
          std::fill(unit.begin(), unit.end(), 0.0f);
          for (std::size_t j = 0; j < dim_; ++j) unit[j] = x_eff[j] * inv_norm;
          rotation_.forward(unit.data(), rotated.data(), work.data());

          encode_rotated(rotated.data(), codes.data());

          if (mode_ != Mode::kInnerProduct) {
            // MSE path: store codes only
            if (storage_.path == StorageLayout::Path::kPackedNibble) {
              std::uint8_t* pk = storage_.packed_nibble_block_ptr(bi);
              for (std::size_t j = 0; j < padded_dim_; ++j) {
                const std::uint8_t code = static_cast<std::uint8_t>(codes[j]) & 0x0f;
                if (lane < 16)
                  pk[j * 16 + lane] |= code;
                else
                  pk[j * 16 + (lane - 16)] |= (code << 4);
              }
              storage_.gammas[gi] = 0.0f;
            } else if (storage_.path == StorageLayout::Path::kNibble) {
              std::uint8_t* nib = storage_.nibble_dim_ptr(bi, 0);
              for (std::size_t j = 0; j < padded_dim_; ++j)
                nib[j * kBlockSize + lane] = static_cast<std::uint8_t>(codes[j]);
              storage_.gammas[gi] = 0.0f;
            } else {
              std::uint8_t* bc = storage_.byte_code_dim_ptr(bi, 0);
              for (std::size_t j = 0; j < padded_dim_; ++j)
                bc[j * kBlockSize + lane] = static_cast<std::uint8_t>(codes[j]);
            }
            continue;
          }

          // IP path: compute QJL residual
          const float gamma = build_rotated_residual(rotated.data(), codes.data(), residual.data());
          qjl_.forward(residual.data(), projected.data(), work.data());

          if (storage_.path == StorageLayout::Path::kPackedNibble) {
            storage_.gammas[gi] = gamma;
            std::uint8_t* pk = storage_.packed_nibble_block_ptr(bi);
            for (std::size_t j = 0; j < padded_dim_; ++j) {
              const std::uint8_t sign_bit = (gamma > 0.0f && projected[j] >= 0.0f) ? 1U : 0U;
              const std::uint8_t code = static_cast<std::uint8_t>((sign_bit << mse_bits_) | codes[j]) & 0x0f;
              if (lane < 16) pk[j * 16 + lane] |= code;
              else           pk[j * 16 + (lane - 16)] |= (code << 4);
            }
          } else if (storage_.path == StorageLayout::Path::kNibble) {
            storage_.gammas[gi] = gamma;
            std::uint8_t* nib = storage_.nibble_dim_ptr(bi, 0);
            for (std::size_t j = 0; j < padded_dim_; ++j) {
              const std::uint8_t sign_bit = (gamma > 0.0f && projected[j] >= 0.0f) ? 1U : 0U;
              nib[j * kBlockSize + lane] = static_cast<std::uint8_t>((sign_bit << mse_bits_) | codes[j]);
            }
          } else {
            storage_.residual_scales[gi] = norm * gamma * (kQjlScale / static_cast<float>(padded_dim_));
            if (storage_.combined_code_sign) {
              std::uint8_t* bc = storage_.byte_code_dim_ptr(bi, 0);
              for (std::size_t j = 0; j < padded_dim_; ++j) {
                const std::uint32_t sign_bit = (gamma > 0.0f && projected[j] >= 0.0f) ? 1U : 0U;
                bc[j * kBlockSize + lane] = static_cast<std::uint8_t>((sign_bit << mse_bits_) | codes[j]);
              }
            } else {
              std::uint8_t* bc = storage_.byte_code_dim_ptr(bi, 0);
              for (std::size_t j = 0; j < padded_dim_; ++j)
                bc[j * kBlockSize + lane] = static_cast<std::uint8_t>(codes[j]);
              for (std::size_t j = 0; j < padded_dim_; ++j) {
                const std::uint8_t sign_bit = (gamma > 0.0f && projected[j] >= 0.0f) ? 1U : 0U;
                std::uint8_t* sp = storage_.sign_dim_ptr(bi, j);
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
  // query — exhaustive k-NN with explicit metric selection.
  // Dispatches to IVF or flat path, then to nibble / packed / generic kernel.
  // ------------------------------------------------------------------
  void query(std::size_t nq, const float* x, std::size_t k,
             SearchMetric metric, float* distances, idx_t* labels) const {
    require_trained();
    if (x == nullptr && nq != 0)
      throw std::invalid_argument("query: x must not be null");
    if (k > ntotal_) k = ntotal_;

    // IVF path: coarse cluster search + per-cluster fine scan
    if (ivf_.active()) {
      if (storage_.path == StorageLayout::Path::kPackedNibble) {
        if (mode_ == Mode::kInnerProduct) {
          if (metric == SearchMetric::kL2) query_ivf_packed_impl<true, true>(nq, x, k, distances, labels);
          else                              query_ivf_packed_impl<true, false>(nq, x, k, distances, labels);
        } else {
          if (metric == SearchMetric::kL2) query_ivf_packed_impl<false, true>(nq, x, k, distances, labels);
          else                              query_ivf_packed_impl<false, false>(nq, x, k, distances, labels);
        }
      } else if (storage_.path == StorageLayout::Path::kNibble) {
        if (mode_ == Mode::kInnerProduct) {
          if (metric == SearchMetric::kL2) query_ivf_nibble_impl<true, true>(nq, x, k, distances, labels);
          else                              query_ivf_nibble_impl<true, false>(nq, x, k, distances, labels);
        } else {
          if (metric == SearchMetric::kL2) query_ivf_nibble_impl<false, true>(nq, x, k, distances, labels);
          else                              query_ivf_nibble_impl<false, false>(nq, x, k, distances, labels);
        }
      } else {
        if (mode_ == Mode::kInnerProduct) {
          if (metric == SearchMetric::kL2) query_ivf_generic_impl<true, true>(nq, x, k, distances, labels);
          else                              query_ivf_generic_impl<true, false>(nq, x, k, distances, labels);
        } else {
          if (metric == SearchMetric::kL2) query_ivf_generic_impl<false, true>(nq, x, k, distances, labels);
          else                              query_ivf_generic_impl<false, false>(nq, x, k, distances, labels);
        }
      }
      return;
    }

    if (storage_.path == StorageLayout::Path::kPackedNibble) {
      if (mode_ == Mode::kInnerProduct) {
        if (metric == SearchMetric::kL2) query_packed_impl<true, true>(nq, x, k, distances, labels);
        else                              query_packed_impl<true, false>(nq, x, k, distances, labels);
      } else {
        if (metric == SearchMetric::kL2) query_packed_impl<false, true>(nq, x, k, distances, labels);
        else                              query_packed_impl<false, false>(nq, x, k, distances, labels);
      }
    } else if (storage_.path == StorageLayout::Path::kNibble) {
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

  // query — k nearest neighbours under L2 distance (convenience overload).
  void query(std::size_t nq, const float* x, std::size_t k,
             float* distances, idx_t* labels) const {
    query(nq, x, k, SearchMetric::kL2, distances, labels);
  }

  // ------------------------------------------------------------------
  // ProfileStats — per-phase timing breakdown for the nibble/packed query path.
  // ------------------------------------------------------------------
  struct ProfileStats {
    double prepare_pct   = 0;
    double rotate_pct    = 0;
    double lut_pct       = 0;
    double score_pct     = 0;
    double postproc_pct  = 0;
    double heap_pct      = 0;
    double total_ms_per_query = 0;
  };

  ProfileStats profile_nibble_query(std::size_t nq, const float* x, std::size_t k) const {
    require_trained();
    if (storage_.path == StorageLayout::Path::kGeneric)
      throw std::logic_error("profile_nibble_query: nibble/packed path not active");

    using Clock = std::chrono::steady_clock;
    using Dur   = std::chrono::duration<double, std::micro>;

    const std::size_t eff_bs  = storage_.eff_block_size();
    const std::size_t n_blocks = detail::ceil_div(ntotal_, eff_bs);
    if (k > ntotal_) k = ntotal_;

    std::vector<float>   raw_all(n_blocks * eff_bs);
    std::vector<float>   val_all(n_blocks * eff_bs);
    std::vector<float>   key_all(n_blocks * eff_bs);

    std::vector<float>       q_work(padded_dim_, 0.0f);
    std::vector<float>       q_unit(padded_dim_, 0.0f);
    std::vector<float>       rotated(padded_dim_);
    std::vector<float>       work(padded_dim_);
    std::vector<std::int8_t> base_i8(padded_dim_ * 16);
    std::vector<detail::HeapEntry> heap(k);

    double t_prepare = 0, t_rotate = 0, t_lut = 0;
    double t_score = 0, t_postproc = 0, t_heap = 0;

    for (std::size_t qi = 0; qi < nq; ++qi) {
      const float* qptr = x + qi * dim_;

      auto tp0 = Clock::now();
      const float q_en = prepare_query(qptr, q_work, q_unit);
      const float q_en_sq = q_en * q_en;
      auto tp1 = Clock::now();
      t_prepare += Dur(tp1 - tp0).count();

      rotation_.forward(q_unit.data(), rotated.data(), work.data());
      auto tp2 = Clock::now();
      t_rotate += Dur(tp2 - tp1).count();

      float base_scale = 1.0f, dummy_qjl = 1.0f;
      codebook_.build_lut16_int8(rotated.data(), nullptr,
                                 base_i8.data(), nullptr,
                                 base_scale, dummy_qjl);
      auto tp3 = Clock::now();
      t_lut += Dur(tp3 - tp2).count();

      if (storage_.path == StorageLayout::Path::kPackedNibble) {
        for (std::size_t bi = 0; bi < n_blocks; ++bi) {
          const std::size_t db0 = bi * kPackedBlockSize;
          const std::size_t bs = std::min<std::size_t>(kPackedBlockSize, ntotal_ - db0);
          storage_.score_block32_packed(storage_.packed_nibble_block_ptr(bi),
                                        base_i8.data(), nullptr,
                                        storage_.gammas.data() + db0,
                                        base_scale, 1.0f, bs, raw_all.data() + db0);
        }
      } else {
        for (std::size_t db0 = 0; db0 < ntotal_; db0 += kBlockSize) {
          const std::size_t bs = std::min<std::size_t>(kBlockSize, ntotal_ - db0);
          const std::size_t bi = db0 / kBlockSize;
          storage_.score_block16_int8(storage_.nibble_dim_ptr(bi, 0),
                                      base_i8.data(), nullptr,
                                      storage_.gammas.data() + db0,
                                      base_scale, 1.0f, bs, raw_all.data() + db0);
        }
      }
      auto tp4 = Clock::now();
      t_score += Dur(tp4 - tp3).count();

      for (std::size_t db0 = 0; db0 < ntotal_; db0 += eff_bs) {
        const std::size_t bs = std::min<std::size_t>(eff_bs, ntotal_ - db0);
        if (storage_.path == StorageLayout::Path::kPackedNibble) {
          storage_.postprocess_packed<true>(db0, bs, q_en, q_en_sq,
                                            raw_all.data() + db0,
                                            val_all.data() + db0,
                                            key_all.data() + db0);
        } else {
          storage_.postprocess_nibble<true>(db0, bs, q_en, q_en_sq,
                                            raw_all.data() + db0,
                                            val_all.data() + db0,
                                            key_all.data() + db0);
        }
      }
      auto tp5 = Clock::now();
      t_postproc += Dur(tp5 - tp4).count();

      std::size_t heap_size = 0;
      for (std::size_t i = 0; i < ntotal_; ++i)
        detail::heap_push_or_replace(heap, heap_size, k, key_all[i], val_all[i],
                                     static_cast<idx_t>(i));
      std::sort(heap.begin(), heap.begin() + static_cast<std::ptrdiff_t>(heap_size),
                [](const detail::HeapEntry& a, const detail::HeapEntry& b) {
                  return a.rank_key > b.rank_key; });
      auto tp6 = Clock::now();
      t_heap += Dur(tp6 - tp5).count();
    }

    const double total = t_prepare + t_rotate + t_lut + t_score + t_postproc + t_heap;
    ProfileStats s;
    s.prepare_pct  = 100.0 * t_prepare  / total;
    s.rotate_pct   = 100.0 * t_rotate   / total;
    s.lut_pct      = 100.0 * t_lut      / total;
    s.score_pct    = 100.0 * t_score    / total;
    s.postproc_pct = 100.0 * t_postproc / total;
    s.heap_pct     = 100.0 * t_heap     / total;
    s.total_ms_per_query = total / static_cast<double>(nq) / 1000.0;
    return s;
  }

 private:
  // ------------------------------------------------------------------
  // Constants — mirror detail:: values as local aliases for convenience
  // ------------------------------------------------------------------
  static constexpr float       kQjlScale       = detail::kQjlScale;
  static constexpr std::size_t kBlockSize       = detail::kBlockSize;
  static constexpr std::size_t kPackedBlockSize = detail::kPackedBlockSize;

  // ------------------------------------------------------------------
  // Members
  // ------------------------------------------------------------------
  std::size_t   dim_         = 0;
  std::size_t   padded_dim_  = 0;
  std::size_t   bitwidth_    = 0;
  std::size_t   mse_bits_    = 0;
  Mode          mode_        = Mode::kInnerProduct;
  RotationType  rotation_type_  = RotationType::kHadamard;
  bool          use_data_centroid_  = false;
  bool          force_generic_path_ = false;
  bool          use_packed_nibbles_ = true;
  std::uint64_t seed_        = 0;
  std::size_t   num_threads_ = 1;
  std::size_t   ntotal_      = 0;
  bool          trained_     = false;

  std::vector<float> centroid_;  // padded_dim_ floats when use_data_centroid_ is active

  // Sub-components
  Codebook      codebook_;
  Transform     rotation_;
  Transform     qjl_;
  StorageLayout storage_;
  KMeansIVF     ivf_;

  // ------------------------------------------------------------------
  // Small private helpers — guard, query prep, threading
  // ------------------------------------------------------------------
  void require_trained() const {
    if (!trained_) throw std::logic_error("TurboQuantIndex: train() must be called first");
  }

  // Writes q_unit[j] = q_eff[j] / ‖q_eff‖; returns ‖q_eff‖.
  float prepare_query(const float* qptr,
                      std::vector<float>& q_work,
                      std::vector<float>& q_unit) const {
    float q_eff_norm;
    if (use_data_centroid_) {
      for (std::size_t j = 0; j < dim_; ++j) q_work[j] = qptr[j] - centroid_[j];
      q_eff_norm = detail::l2_norm(q_work.data(), dim_);
    } else {
      q_eff_norm = detail::l2_norm(qptr, dim_);
    }
    std::fill(q_unit.begin(), q_unit.end(), 0.0f);
    if (q_eff_norm > 0.0f) {
      const float inv = 1.0f / q_eff_norm;
      const float* src = use_data_centroid_ ? q_work.data() : qptr;
      for (std::size_t j = 0; j < dim_; ++j) q_unit[j] = src[j] * inv;
    }
    return q_eff_norm;
  }

  // parallel_for — split [begin,end) into chunks across OpenMP threads.
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
  // Encode helpers — used by add() / add_ivf()
  // ------------------------------------------------------------------

  // Per-coordinate quantization of `rotated` (length padded_dim_) → code indices.
  void encode_rotated(const float* rotated, std::uint32_t* out) const {
    for (std::size_t j = 0; j < padded_dim_; ++j) out[j] = codebook_.nearest(rotated[j]);
  }

  // Residual r_j = v_j − codebook_[code_j]; returns ‖r‖₂ (QJL uses direction of r).
  float build_rotated_residual(const float* rotated, const std::uint32_t* codes,
                                float* residual) const {
    float sum_sq = 0.0f;
    for (std::size_t j = 0; j < padded_dim_; ++j) {
      const float q = (mse_bits_ == 0) ? 0.0f : codebook_.centroids[codes[j]];
      const float d = rotated[j] - q;
      residual[j] = d;
      sum_sq += d * d;
    }
    return std::sqrt(sum_sq);
  }

  // ------------------------------------------------------------------
  // encode_vector_inplace — encode a single vector into block storage.
  // Factored from add() and add_ivf(); local_centroid overrides centroid_ (IVF).
  // ------------------------------------------------------------------
  void encode_vector_inplace(
      std::size_t gi, std::size_t bi, std::size_t lane,
      const float* src, const float* local_centroid,
      std::vector<float>& x_eff, std::vector<float>& unit,
      std::vector<float>& rotated, std::vector<float>& residual,
      std::vector<float>& projected, std::vector<float>& work,
      std::vector<std::uint32_t>& codes)
  {
    std::fill(x_eff.begin(), x_eff.end(), 0.0f);
    const float* cp = local_centroid ? local_centroid
                    : (use_data_centroid_ ? centroid_.data() : nullptr);
    if (cp) {
      for (std::size_t j = 0; j < dim_; ++j) x_eff[j] = src[j] - cp[j];
    } else {
      for (std::size_t j = 0; j < dim_; ++j) x_eff[j] = src[j];
    }

    const float norm = detail::l2_norm(x_eff.data(), dim_);
    storage_.norms[gi]        = norm;
    storage_.norm_squares[gi] = norm * norm;

    if (norm == 0.0f) {
      if (storage_.path != StorageLayout::Path::kGeneric) storage_.gammas[gi] = 0.0f;
      else                                                 storage_.residual_scales[gi] = 0.0f;
      return;
    }

    const float inv = 1.0f / norm;
    std::fill(unit.begin(), unit.end(), 0.0f);
    for (std::size_t j = 0; j < dim_; ++j) unit[j] = x_eff[j] * inv;
    rotation_.forward(unit.data(), rotated.data(), work.data());
    encode_rotated(rotated.data(), codes.data());

    if (mode_ != Mode::kInnerProduct) {
      if (storage_.path == StorageLayout::Path::kPackedNibble) {
        std::uint8_t* pk = storage_.packed_nibble_block_ptr(bi);
        for (std::size_t j = 0; j < padded_dim_; ++j) {
          const std::uint8_t code = static_cast<std::uint8_t>(codes[j]) & 0x0f;
          if (lane < 16) pk[j * 16 + lane] |= code;
          else           pk[j * 16 + (lane - 16)] |= (code << 4);
        }
        storage_.gammas[gi] = 0.0f;
      } else if (storage_.path == StorageLayout::Path::kNibble) {
        std::uint8_t* nib = storage_.nibble_dim_ptr(bi, 0);
        for (std::size_t j = 0; j < padded_dim_; ++j)
          nib[j * kBlockSize + lane] = static_cast<std::uint8_t>(codes[j]);
        storage_.gammas[gi] = 0.0f;
      } else {
        std::uint8_t* bc = storage_.byte_code_dim_ptr(bi, 0);
        for (std::size_t j = 0; j < padded_dim_; ++j)
          bc[j * kBlockSize + lane] = static_cast<std::uint8_t>(codes[j]);
      }
      return;
    }

    // IP path: QJL residual
    const float gamma = build_rotated_residual(rotated.data(), codes.data(), residual.data());
    qjl_.forward(residual.data(), projected.data(), work.data());

    if (storage_.path == StorageLayout::Path::kPackedNibble) {
      storage_.gammas[gi] = gamma;
      std::uint8_t* pk = storage_.packed_nibble_block_ptr(bi);
      for (std::size_t j = 0; j < padded_dim_; ++j) {
        const std::uint8_t sb   = (gamma > 0.0f && projected[j] >= 0.0f) ? 1U : 0U;
        const std::uint8_t code = static_cast<std::uint8_t>((sb << mse_bits_) | codes[j]) & 0x0f;
        if (lane < 16) pk[j * 16 + lane] |= code;
        else           pk[j * 16 + (lane - 16)] |= (code << 4);
      }
    } else if (storage_.path == StorageLayout::Path::kNibble) {
      storage_.gammas[gi] = gamma;
      std::uint8_t* nib = storage_.nibble_dim_ptr(bi, 0);
      for (std::size_t j = 0; j < padded_dim_; ++j) {
        const std::uint8_t sb = (gamma > 0.0f && projected[j] >= 0.0f) ? 1U : 0U;
        nib[j * kBlockSize + lane] = static_cast<std::uint8_t>((sb << mse_bits_) | codes[j]);
      }
    } else {
      storage_.residual_scales[gi] = norm * gamma * (kQjlScale / static_cast<float>(padded_dim_));
      if (storage_.combined_code_sign) {
        std::uint8_t* bc = storage_.byte_code_dim_ptr(bi, 0);
        for (std::size_t j = 0; j < padded_dim_; ++j) {
          const std::uint32_t sb = (gamma > 0.0f && projected[j] >= 0.0f) ? 1U : 0U;
          bc[j * kBlockSize + lane] = static_cast<std::uint8_t>((sb << mse_bits_) | codes[j]);
        }
      } else {
        std::uint8_t* bc = storage_.byte_code_dim_ptr(bi, 0);
        for (std::size_t j = 0; j < padded_dim_; ++j)
          bc[j * kBlockSize + lane] = static_cast<std::uint8_t>(codes[j]);
        for (std::size_t j = 0; j < padded_dim_; ++j) {
          const std::uint8_t sb = (gamma > 0.0f && projected[j] >= 0.0f) ? 1U : 0U;
          std::uint8_t* sp = storage_.sign_dim_ptr(bi, j);
          std::uint16_t w; std::memcpy(&w, sp, 2);
          w = static_cast<std::uint16_t>((w & ~(1U << lane)) | (static_cast<std::uint16_t>(sb) << lane));
          std::memcpy(sp, &w, 2);
        }
      }
    }
  }

  // ------------------------------------------------------------------
  // query_nibble_impl — 16-lane nibble path: LUT16 + score_block16_int8 + heap
  // kUseQjl: inner-product mode with QJL residual; kL2: output L2² else IP.
  // ------------------------------------------------------------------
  template <bool kUseQjl, bool kL2>
  void query_nibble_impl(std::size_t nq, const float* x, std::size_t k,
                         float* distances, idx_t* labels) const {
    parallel_for(0, nq, [&](std::size_t q0, std::size_t q1) {
      std::vector<float> q_work(padded_dim_, 0.0f);
      std::vector<float> q_unit(padded_dim_, 0.0f);
      std::vector<float> rotated(padded_dim_);
      std::vector<float> projected(kUseQjl ? padded_dim_ : 0);
      std::vector<float> work(padded_dim_);
      std::vector<std::int8_t> base_i8(padded_dim_ * 16);
      std::vector<std::int8_t> qjl_i8(kUseQjl ? padded_dim_ * 16 : 0);
      alignas(64) float raw_scores[kBlockSize];
      alignas(64) float cand_values[kBlockSize];
      alignas(64) float cand_rank_keys[kBlockSize];
      std::vector<detail::HeapEntry> heap(k);
      std::size_t heap_size = 0;

      for (std::size_t qi = q0; qi < q1; ++qi) {
        const float* qptr = x + qi * dim_;
        const float q_en = prepare_query(qptr, q_work, q_unit);
        const float q_en_sq = q_en * q_en;

        rotation_.forward(q_unit.data(), rotated.data(), work.data());
        if constexpr (kUseQjl)
          qjl_.forward(rotated.data(), projected.data(), work.data());

        float base_scale = 1.0f, qjl_scale_v = 1.0f;
        codebook_.build_lut16_int8(rotated.data(),
                                   kUseQjl ? projected.data() : nullptr,
                                   base_i8.data(),
                                   kUseQjl ? qjl_i8.data() : nullptr,
                                   base_scale, qjl_scale_v);
        heap_size = 0;

        for (std::size_t db0 = 0; db0 < ntotal_; db0 += kBlockSize) {
          const std::size_t bs = std::min<std::size_t>(kBlockSize, ntotal_ - db0);
          const std::size_t bi = db0 / kBlockSize;

          storage_.score_block16_int8(storage_.nibble_dim_ptr(bi, 0),
                                      base_i8.data(),
                                      kUseQjl ? qjl_i8.data() : nullptr,
                                      storage_.gammas.data() + db0,
                                      base_scale, qjl_scale_v, bs, raw_scores);

          storage_.postprocess_nibble<kL2>(db0, bs, q_en, q_en_sq,
                                           raw_scores, cand_values, cand_rank_keys);

          for (std::size_t i = 0; i < bs; ++i)
            detail::heap_push_or_replace(heap, heap_size, k, cand_rank_keys[i],
                                         cand_values[i], static_cast<idx_t>(db0 + i));
        }

        std::sort(heap.begin(), heap.begin() + static_cast<std::ptrdiff_t>(heap_size),
                  [](const detail::HeapEntry& a, const detail::HeapEntry& b) {
                    return a.rank_key > b.rank_key; });
        const std::size_t out_base = qi * k;
        for (std::size_t r = 0; r < heap_size; ++r) {
          distances[out_base + r] = heap[r].value;
          labels[out_base + r]    = heap[r].label;
        }
      }
    });
  }

  // ------------------------------------------------------------------
  // query_packed_impl — 32-lane packed nibble path (faster memory layout)
  // ------------------------------------------------------------------
  template <bool kUseQjl, bool kL2>
  void query_packed_impl(std::size_t nq, const float* x, std::size_t k,
                         float* distances, idx_t* labels) const {
    parallel_for(0, nq, [&](std::size_t q0, std::size_t q1) {
      std::vector<float> q_work(padded_dim_, 0.0f);
      std::vector<float> q_unit(padded_dim_, 0.0f);
      std::vector<float> rotated(padded_dim_);
      std::vector<float> projected(kUseQjl ? padded_dim_ : 0);
      std::vector<float> work(padded_dim_);
      std::vector<std::int8_t> base_i8(padded_dim_ * 16);
      std::vector<std::int8_t> qjl_i8(kUseQjl ? padded_dim_ * 16 : 0);
      alignas(64) float raw_scores[kPackedBlockSize];
      alignas(64) float cand_values[kPackedBlockSize];
      alignas(64) float cand_rank_keys[kPackedBlockSize];
      std::vector<detail::HeapEntry> heap(k);
      std::size_t heap_size = 0;

      for (std::size_t qi = q0; qi < q1; ++qi) {
        const float* qptr = x + qi * dim_;
        const float q_en = prepare_query(qptr, q_work, q_unit);
        const float q_en_sq = q_en * q_en;

        rotation_.forward(q_unit.data(), rotated.data(), work.data());
        if constexpr (kUseQjl)
          qjl_.forward(rotated.data(), projected.data(), work.data());

        float base_scale = 1.0f, qjl_scale_v = 1.0f;
        codebook_.build_lut16_int8(rotated.data(),
                                   kUseQjl ? projected.data() : nullptr,
                                   base_i8.data(),
                                   kUseQjl ? qjl_i8.data() : nullptr,
                                   base_scale, qjl_scale_v);
        heap_size = 0;

        const std::size_t num_blocks = detail::ceil_div(ntotal_, kPackedBlockSize);
        for (std::size_t bi = 0; bi < num_blocks; ++bi) {
          const std::size_t db0 = bi * kPackedBlockSize;
          const std::size_t bs  = std::min<std::size_t>(kPackedBlockSize, ntotal_ - db0);

          // Prefetch next block
          if (bi + 1 < num_blocks) {
            const std::uint8_t* next = storage_.packed_nibble_block_ptr(bi + 1);
            for (std::size_t off = 0; off < storage_.packed_block_stride; off += 64)
              _mm_prefetch(reinterpret_cast<const char*>(next + off), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(storage_.norms.data() + db0 + kPackedBlockSize), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(storage_.norm_squares.data() + db0 + kPackedBlockSize), _MM_HINT_T0);
          }

          storage_.score_block32_packed(storage_.packed_nibble_block_ptr(bi),
                                        base_i8.data(),
                                        kUseQjl ? qjl_i8.data() : nullptr,
                                        storage_.gammas.data() + db0,
                                        base_scale, qjl_scale_v, bs, raw_scores);

          storage_.postprocess_packed<kL2>(db0, bs, q_en, q_en_sq,
                                           raw_scores, cand_values, cand_rank_keys);

          for (std::size_t i = 0; i < bs; ++i)
            detail::heap_push_or_replace(heap, heap_size, k, cand_rank_keys[i],
                                         cand_values[i], static_cast<idx_t>(db0 + i));
        }

        std::sort(heap.begin(), heap.begin() + static_cast<std::ptrdiff_t>(heap_size),
                  [](const detail::HeapEntry& a, const detail::HeapEntry& b) {
                    return a.rank_key > b.rank_key; });
        const std::size_t out_base = qi * k;
        for (std::size_t r = 0; r < heap_size; ++r) {
          distances[out_base + r] = heap[r].value;
          labels[out_base + r]    = heap[r].label;
        }
      }
    });
  }

  // ------------------------------------------------------------------
  // query_generic_impl — float LUT + byte gathers; kUseResidual → QJL sign split
  // ------------------------------------------------------------------
  template <bool kUseResidual, bool kL2>
  void query_generic_impl(std::size_t nq, const float* x, std::size_t k,
                          float* distances, idx_t* labels) const {
    parallel_for(0, nq, [&](std::size_t q0, std::size_t q1) {
      std::vector<float> q_work(padded_dim_, 0.0f);
      std::vector<float> q_unit(padded_dim_, 0.0f);
      std::vector<float> rotated(padded_dim_);
      std::vector<float> projected(kUseResidual ? padded_dim_ : 0);
      std::vector<float> lut(padded_dim_ * storage_.lut_stride);
      std::vector<float> work(padded_dim_);
      alignas(64) float dot_scores[kBlockSize];
      alignas(64) float scratch_scores[kBlockSize];
      alignas(64) float cand_values[kBlockSize];
      alignas(64) float cand_rank_keys[kBlockSize];
      std::vector<detail::HeapEntry> heap(k);
      std::size_t heap_size = 0;

      for (std::size_t qi = q0; qi < q1; ++qi) {
        const float* qptr = x + qi * dim_;
        const float q_en = prepare_query(qptr, q_work, q_unit);
        const float q_en_sq = q_en * q_en;

        rotation_.forward(q_unit.data(), rotated.data(), work.data());
        if constexpr (kUseResidual)
          qjl_.forward(rotated.data(), projected.data(), work.data());

        storage_.build_float_lut(rotated.data(), lut.data(), codebook_);
        heap_size = 0;

        for (std::size_t db0 = 0; db0 < ntotal_; db0 += kBlockSize) {
          const std::size_t bs = std::min<std::size_t>(kBlockSize, ntotal_ - db0);
          const std::size_t bi = db0 / kBlockSize;

          if constexpr (kUseResidual) {
            std::fill(dot_scores, dot_scores + bs, 0.0f);
            std::fill(scratch_scores, scratch_scores + bs, 0.0f);
            storage_.score_generic_code_sign_block(bi, bs, lut.data(), projected.data(),
                                                   dot_scores, scratch_scores);
            storage_.postprocess_generic<true, kL2>(db0, bs, q_en, q_en_sq,
                                                    dot_scores, scratch_scores,
                                                    cand_values, cand_rank_keys);
          } else {
            std::fill(dot_scores, dot_scores + bs, 0.0f);
            storage_.score_generic_code_block(bi, bs, lut.data(), dot_scores);
            storage_.postprocess_generic<false, kL2>(db0, bs, q_en, q_en_sq,
                                                     dot_scores, nullptr,
                                                     cand_values, cand_rank_keys);
          }

          for (std::size_t i = 0; i < bs; ++i)
            detail::heap_push_or_replace(heap, heap_size, k, cand_rank_keys[i],
                                         cand_values[i], static_cast<idx_t>(db0 + i));
        }

        std::sort(heap.begin(), heap.begin() + static_cast<std::ptrdiff_t>(heap_size),
                  [](const detail::HeapEntry& a, const detail::HeapEntry& b) {
                    return a.rank_key > b.rank_key; });
        const std::size_t out_base = qi * k;
        for (std::size_t r = 0; r < heap_size; ++r) {
          distances[out_base + r] = heap[r].value;
          labels[out_base + r]    = heap[r].label;
        }
      }
    });
  }

  // ==================================================================
  // IVF private methods
  // ==================================================================

  // ------------------------------------------------------------------
  // add_ivf — IVF-aware encoding: assign vectors to clusters, reorder by
  // cluster (block-aligned), encode with cluster-relative residuals.
  // ------------------------------------------------------------------
  void add_ivf(std::size_t n, const float* x) {
    const std::size_t eff_bs = storage_.eff_block_size();

    // Assign each vector to its nearest cluster
    std::vector<std::size_t> asgn(n);
    #pragma omp parallel for num_threads(static_cast<int>(num_threads_)) schedule(static)
    for (std::ptrdiff_t ii = 0; ii < static_cast<std::ptrdiff_t>(n); ++ii) {
      const std::size_t i = static_cast<std::size_t>(ii);
      const float* xi = x + i * dim_;
      float best = std::numeric_limits<float>::max();
      std::size_t best_k = 0;
      for (std::size_t c = 0; c < ivf_.nlist; ++c) {
        if (c + 2 < ivf_.nlist)
          _mm_prefetch(reinterpret_cast<const char*>(
              ivf_.centroids.data() + (c + 2) * padded_dim_), _MM_HINT_T0);
        float dsq = detail::l2_sq_distance(xi, ivf_.centroids.data() + c * padded_dim_, dim_);
        if (dsq < best) { best = dsq; best_k = c; }
      }
      asgn[i] = best_k;
    }

    // Compute cluster sizes and block-aligned slot layout
    ivf_.list_size.assign(ivf_.nlist, 0);
    for (std::size_t i = 0; i < n; ++i) ++ivf_.list_size[asgn[i]];

    ivf_.list_start.resize(ivf_.nlist);
    std::size_t offset = 0;
    for (std::size_t c = 0; c < ivf_.nlist; ++c) {
      ivf_.list_start[c] = offset;
      offset += detail::ceil_div(ivf_.list_size[c], eff_bs) * eff_bs;
    }
    const std::size_t total_slots  = offset;
    const std::size_t total_blocks = total_slots / eff_bs;

    ntotal_ = n;
    if (storage_.path == StorageLayout::Path::kPackedNibble) {
      storage_.packed_nibbles.assign(total_blocks * storage_.packed_block_stride, 0);
      storage_.gammas.assign(total_slots, 0.f);
    } else if (storage_.path == StorageLayout::Path::kNibble) {
      storage_.nibbles.assign(total_blocks * storage_.nibble_block_stride, 0);
      storage_.gammas.assign(total_slots, 0.f);
    } else {
      storage_.byte_codes.assign(total_blocks * storage_.byte_code_block_stride, 0);
      if (storage_.sign_block_stride != 0)
        storage_.packed_signs.assign(total_blocks * storage_.sign_block_stride, 0);
      storage_.residual_scales.assign(total_slots, 0.f);
    }
    storage_.norms.assign(total_slots, 0.f);
    storage_.norm_squares.assign(total_slots, 0.f);

    // Fill ivf_.ids: storage-slot → original add-order index
    ivf_.ids.assign(total_slots, -1);
    {
      std::vector<std::size_t> cursor(ivf_.nlist);
      for (std::size_t c = 0; c < ivf_.nlist; ++c) cursor[c] = ivf_.list_start[c];
      for (std::size_t i = 0; i < n; ++i)
        ivf_.ids[cursor[asgn[i]]++] = static_cast<idx_t>(i);
    }

    // Encode each cluster's vectors (parallelised over clusters)
    parallel_for(0, ivf_.nlist, [&](std::size_t c0, std::size_t c1) {
      std::vector<float> x_eff(padded_dim_, 0.0f);
      std::vector<float> unit(padded_dim_, 0.0f);
      std::vector<float> rotated(padded_dim_);
      std::vector<float> residual(padded_dim_);
      std::vector<float> projected(mode_ == Mode::kInnerProduct ? padded_dim_ : 0);
      std::vector<float> work(padded_dim_);
      std::vector<std::uint32_t> codes(padded_dim_);

      for (std::size_t c = c0; c < c1; ++c) {
        const float* local_c = ivf_.centroids.data() + c * padded_dim_;
        for (std::size_t r = 0; r < ivf_.list_size[c]; ++r) {
          const std::size_t gi   = ivf_.list_start[c] + r;
          const std::size_t bi   = gi / eff_bs;
          const std::size_t lane = gi % eff_bs;
          const float* src = x + static_cast<std::size_t>(ivf_.ids[gi]) * dim_;
          encode_vector_inplace(gi, bi, lane, src, local_c,
                                x_eff, unit, rotated, residual, projected, work, codes);
        }
      }
    });
  }

  // ------------------------------------------------------------------
  // query_ivf_packed_impl — IVF query for packed-nibble (block-32) path.
  // ------------------------------------------------------------------
  template <bool kUseQjl, bool kL2>
  void query_ivf_packed_impl(std::size_t nq, const float* x, std::size_t k,
                              float* distances, idx_t* labels) const {
    parallel_for(0, nq, [&](std::size_t q0, std::size_t q1) {
      std::vector<float> q_r(padded_dim_, 0.0f);
      std::vector<float> q_r_unit(padded_dim_, 0.0f);
      std::vector<float> q_r_rot(padded_dim_);
      std::vector<float> projected(kUseQjl ? padded_dim_ : 0);
      std::vector<float> work(padded_dim_);
      std::vector<std::int8_t> base_i8(padded_dim_ * 16);
      std::vector<std::int8_t> qjl_i8(kUseQjl ? padded_dim_ * 16 : 0);
      alignas(64) float raw_scores[kPackedBlockSize];
      alignas(64) float cand_values[kPackedBlockSize];
      alignas(64) float cand_rank_keys[kPackedBlockSize];
      std::vector<detail::HeapEntry> heap(k);
      std::vector<float>       cdists(ivf_.nlist);
      std::vector<std::size_t> probe_order(ivf_.nlist);

      for (std::size_t qi = q0; qi < q1; ++qi) {
        const float* qptr = x + qi * dim_;

        ivf_.coarse_search(qptr, dim_, padded_dim_, cdists, probe_order);

        std::size_t heap_size = 0;
        for (std::size_t pi = 0; pi < ivf_.nprobe; ++pi) {
          const std::size_t ck = probe_order[pi];
          if (ivf_.list_size[ck] == 0) continue;

          const float* ck_c = ivf_.centroids.data() + ck * padded_dim_;
          float q_r_norm_sq = 0.0f;
          for (std::size_t j = 0; j < dim_; ++j) {
            const float v = qptr[j] - ck_c[j];
            q_r[j] = v; q_r_norm_sq += v * v;
          }
          const float q_r_norm = std::sqrt(q_r_norm_sq);
          if (q_r_norm > 0.0f) {
            const float inv = 1.0f / q_r_norm;
            for (std::size_t j = 0; j < dim_; ++j) q_r_unit[j] = q_r[j] * inv;
          } else {
            std::fill(q_r_unit.begin(), q_r_unit.begin() + dim_, 0.0f);
          }

          rotation_.forward(q_r_unit.data(), q_r_rot.data(), work.data());
          if constexpr (kUseQjl)
            qjl_.forward(q_r_rot.data(), projected.data(), work.data());

          float base_scale = 1.0f, qjl_scale_v = 1.0f;
          codebook_.build_lut16_int8(q_r_rot.data(),
                                     kUseQjl ? projected.data() : nullptr,
                                     base_i8.data(),
                                     kUseQjl ? qjl_i8.data() : nullptr,
                                     base_scale, qjl_scale_v);

          // Prefetch next cluster
          if (pi + 1 < ivf_.nprobe) {
            const std::size_t nck = probe_order[pi + 1];
            if (ivf_.list_size[nck] > 0) {
              const std::size_t nbi = ivf_.list_start[nck] / kPackedBlockSize;
              _mm_prefetch(reinterpret_cast<const char*>(storage_.packed_nibble_block_ptr(nbi)), _MM_HINT_T1);
              _mm_prefetch(reinterpret_cast<const char*>(storage_.norms.data() + ivf_.list_start[nck]), _MM_HINT_T1);
              _mm_prefetch(reinterpret_cast<const char*>(ck_c + padded_dim_), _MM_HINT_T1);
            }
          }

          const std::size_t cl_start = ivf_.list_start[ck];
          const std::size_t cl_end   = cl_start + ivf_.list_size[ck];
          const std::size_t start_bi = cl_start / kPackedBlockSize;
          const std::size_t end_bi   = detail::ceil_div(cl_end, kPackedBlockSize);

          for (std::size_t bi = start_bi; bi < end_bi; ++bi) {
            const std::size_t db0 = bi * kPackedBlockSize;
            const std::size_t bs  = std::min<std::size_t>(kPackedBlockSize, cl_end - db0);

            if (bi + 1 < end_bi) {
              const std::uint8_t* nxt = storage_.packed_nibble_block_ptr(bi + 1);
              for (std::size_t off = 0; off < storage_.packed_block_stride; off += 64)
                _mm_prefetch(reinterpret_cast<const char*>(nxt + off), _MM_HINT_T0);
              _mm_prefetch(reinterpret_cast<const char*>(storage_.norms.data() + db0 + kPackedBlockSize), _MM_HINT_T0);
              _mm_prefetch(reinterpret_cast<const char*>(storage_.norm_squares.data() + db0 + kPackedBlockSize), _MM_HINT_T0);
            }

            storage_.score_block32_packed(storage_.packed_nibble_block_ptr(bi),
                                          base_i8.data(),
                                          kUseQjl ? qjl_i8.data() : nullptr,
                                          storage_.gammas.data() + db0,
                                          base_scale, qjl_scale_v, bs, raw_scores);

            storage_.postprocess_packed<kL2>(db0, bs, q_r_norm, q_r_norm_sq,
                                             raw_scores, cand_values, cand_rank_keys);

            for (std::size_t i = 0; i < bs; ++i)
              detail::heap_push_or_replace(heap, heap_size, k, cand_rank_keys[i],
                                           cand_values[i], static_cast<idx_t>(db0 + i));
          }
        }

        std::sort(heap.begin(), heap.begin() + static_cast<std::ptrdiff_t>(heap_size),
                  [](const detail::HeapEntry& a, const detail::HeapEntry& b){ return a.rank_key > b.rank_key; });
        const std::size_t out_base = qi * k;
        for (std::size_t r = 0; r < heap_size; ++r) {
          distances[out_base + r] = heap[r].value;
          labels[out_base + r]    = ivf_.ids[heap[r].label];
        }
      }
    });
  }

  // ------------------------------------------------------------------
  // query_ivf_nibble_impl — IVF query for 16-lane nibble path.
  // ------------------------------------------------------------------
  template <bool kUseQjl, bool kL2>
  void query_ivf_nibble_impl(std::size_t nq, const float* x, std::size_t k,
                              float* distances, idx_t* labels) const {
    parallel_for(0, nq, [&](std::size_t q0, std::size_t q1) {
      std::vector<float> q_r(padded_dim_, 0.0f);
      std::vector<float> q_r_unit(padded_dim_, 0.0f);
      std::vector<float> q_r_rot(padded_dim_);
      std::vector<float> projected(kUseQjl ? padded_dim_ : 0);
      std::vector<float> work(padded_dim_);
      std::vector<std::int8_t> base_i8(padded_dim_ * 16);
      std::vector<std::int8_t> qjl_i8(kUseQjl ? padded_dim_ * 16 : 0);
      alignas(64) float raw_scores[kBlockSize];
      alignas(64) float cand_values[kBlockSize];
      alignas(64) float cand_rank_keys[kBlockSize];
      std::vector<detail::HeapEntry> heap(k);
      std::vector<float>       cdists(ivf_.nlist);
      std::vector<std::size_t> probe_order(ivf_.nlist);

      for (std::size_t qi = q0; qi < q1; ++qi) {
        const float* qptr = x + qi * dim_;

        ivf_.coarse_search(qptr, dim_, padded_dim_, cdists, probe_order);

        std::size_t heap_size = 0;
        for (std::size_t pi = 0; pi < ivf_.nprobe; ++pi) {
          const std::size_t ck = probe_order[pi];
          if (ivf_.list_size[ck] == 0) continue;

          const float* ck_c = ivf_.centroids.data() + ck * padded_dim_;
          float q_r_norm_sq = 0.0f;
          for (std::size_t j = 0; j < dim_; ++j) {
            const float v = qptr[j] - ck_c[j];
            q_r[j] = v; q_r_norm_sq += v * v;
          }
          const float q_r_norm = std::sqrt(q_r_norm_sq);
          if (q_r_norm > 0.0f) {
            const float inv = 1.0f / q_r_norm;
            for (std::size_t j = 0; j < dim_; ++j) q_r_unit[j] = q_r[j] * inv;
          } else {
            std::fill(q_r_unit.begin(), q_r_unit.begin() + dim_, 0.0f);
          }
          rotation_.forward(q_r_unit.data(), q_r_rot.data(), work.data());
          if constexpr (kUseQjl)
            qjl_.forward(q_r_rot.data(), projected.data(), work.data());

          float base_scale = 1.0f, qjl_scale_v = 1.0f;
          codebook_.build_lut16_int8(q_r_rot.data(),
                                     kUseQjl ? projected.data() : nullptr,
                                     base_i8.data(),
                                     kUseQjl ? qjl_i8.data() : nullptr,
                                     base_scale, qjl_scale_v);

          const std::size_t cl_start = ivf_.list_start[ck];
          const std::size_t cl_end   = cl_start + ivf_.list_size[ck];
          for (std::size_t db0 = cl_start; db0 < cl_end; db0 += kBlockSize) {
            const std::size_t bs = std::min<std::size_t>(kBlockSize, cl_end - db0);
            const std::size_t bi = db0 / kBlockSize;
            storage_.score_block16_int8(storage_.nibble_dim_ptr(bi, 0),
                                        base_i8.data(),
                                        kUseQjl ? qjl_i8.data() : nullptr,
                                        storage_.gammas.data() + db0,
                                        base_scale, qjl_scale_v, bs, raw_scores);
            storage_.postprocess_nibble<kL2>(db0, bs, q_r_norm, q_r_norm_sq,
                                             raw_scores, cand_values, cand_rank_keys);
            for (std::size_t i = 0; i < bs; ++i)
              detail::heap_push_or_replace(heap, heap_size, k, cand_rank_keys[i],
                                           cand_values[i], static_cast<idx_t>(db0 + i));
          }
        }

        std::sort(heap.begin(), heap.begin() + static_cast<std::ptrdiff_t>(heap_size),
                  [](const detail::HeapEntry& a, const detail::HeapEntry& b){ return a.rank_key > b.rank_key; });
        const std::size_t out_base = qi * k;
        for (std::size_t r = 0; r < heap_size; ++r) {
          distances[out_base + r] = heap[r].value;
          labels[out_base + r]    = ivf_.ids[heap[r].label];
        }
      }
    });
  }

  // ------------------------------------------------------------------
  // query_ivf_generic_impl — IVF query for generic (wide-bitwidth) path.
  // ------------------------------------------------------------------
  template <bool kUseResidual, bool kL2>
  void query_ivf_generic_impl(std::size_t nq, const float* x, std::size_t k,
                               float* distances, idx_t* labels) const {
    parallel_for(0, nq, [&](std::size_t q0, std::size_t q1) {
      std::vector<float> q_r(padded_dim_, 0.0f);
      std::vector<float> q_r_unit(padded_dim_, 0.0f);
      std::vector<float> q_r_rot(padded_dim_);
      std::vector<float> projected(kUseResidual ? padded_dim_ : 0);
      std::vector<float> lut(padded_dim_ * storage_.lut_stride);
      std::vector<float> work(padded_dim_);
      alignas(64) float dot_scores[kBlockSize];
      alignas(64) float scratch_scores[kBlockSize];
      alignas(64) float cand_values[kBlockSize];
      alignas(64) float cand_rank_keys[kBlockSize];
      std::vector<detail::HeapEntry> heap(k);
      std::vector<float>       cdists(ivf_.nlist);
      std::vector<std::size_t> probe_order(ivf_.nlist);

      for (std::size_t qi = q0; qi < q1; ++qi) {
        const float* qptr = x + qi * dim_;

        ivf_.coarse_search(qptr, dim_, padded_dim_, cdists, probe_order);

        std::size_t heap_size = 0;
        for (std::size_t pi = 0; pi < ivf_.nprobe; ++pi) {
          const std::size_t ck = probe_order[pi];
          if (ivf_.list_size[ck] == 0) continue;

          const float* ck_c = ivf_.centroids.data() + ck * padded_dim_;
          float q_r_norm_sq = 0.0f;
          for (std::size_t j = 0; j < dim_; ++j) {
            const float v = qptr[j] - ck_c[j];
            q_r[j] = v; q_r_norm_sq += v * v;
          }
          const float q_r_norm = std::sqrt(q_r_norm_sq);
          if (q_r_norm > 0.0f) {
            const float inv = 1.0f / q_r_norm;
            for (std::size_t j = 0; j < dim_; ++j) q_r_unit[j] = q_r[j] * inv;
          } else {
            std::fill(q_r_unit.begin(), q_r_unit.begin() + dim_, 0.0f);
          }
          rotation_.forward(q_r_unit.data(), q_r_rot.data(), work.data());
          if constexpr (kUseResidual)
            qjl_.forward(q_r_rot.data(), projected.data(), work.data());
          storage_.build_float_lut(q_r_rot.data(), lut.data(), codebook_);

          const std::size_t cl_start = ivf_.list_start[ck];
          const std::size_t cl_end   = cl_start + ivf_.list_size[ck];
          for (std::size_t db0 = cl_start; db0 < cl_end; db0 += kBlockSize) {
            const std::size_t bs = std::min<std::size_t>(kBlockSize, cl_end - db0);
            const std::size_t bi = db0 / kBlockSize;
            if constexpr (kUseResidual) {
              std::fill(dot_scores, dot_scores + bs, 0.0f);
              std::fill(scratch_scores, scratch_scores + bs, 0.0f);
              storage_.score_generic_code_sign_block(bi, bs, lut.data(), projected.data(),
                                                     dot_scores, scratch_scores);
              storage_.postprocess_generic<true, kL2>(db0, bs, q_r_norm, q_r_norm_sq,
                                                      dot_scores, scratch_scores,
                                                      cand_values, cand_rank_keys);
            } else {
              std::fill(dot_scores, dot_scores + bs, 0.0f);
              storage_.score_generic_code_block(bi, bs, lut.data(), dot_scores);
              storage_.postprocess_generic<false, kL2>(db0, bs, q_r_norm, q_r_norm_sq,
                                                       dot_scores, nullptr,
                                                       cand_values, cand_rank_keys);
            }
            for (std::size_t i = 0; i < bs; ++i)
              detail::heap_push_or_replace(heap, heap_size, k, cand_rank_keys[i],
                                           cand_values[i], static_cast<idx_t>(db0 + i));
          }
        }

        std::sort(heap.begin(), heap.begin() + static_cast<std::ptrdiff_t>(heap_size),
                  [](const detail::HeapEntry& a, const detail::HeapEntry& b){ return a.rank_key > b.rank_key; });
        const std::size_t out_base = qi * k;
        for (std::size_t r = 0; r < heap_size; ++r) {
          distances[out_base + r] = heap[r].value;
          labels[out_base + r]    = ivf_.ids[heap[r].label];
        }
      }
    });
  }

 public:
  // ------------------------------------------------------------------
  // reconstruct — approximate inverse of add() (debug / analysis only)
  //
  // Exposed for the Python binding, which uses it to provide reconstruction
  // inspection, MSE reporting, and graph-distance estimation helpers.
  // ------------------------------------------------------------------
  void reconstruct(std::size_t n, float* out) const {
    require_trained();
    if (n > ntotal_) n = ntotal_;

    parallel_for(0, n, [&](std::size_t begin, std::size_t end) {
      std::vector<float> rotated(padded_dim_, 0.0f);
      std::vector<float> unit(padded_dim_, 0.0f);
      std::vector<float> work(padded_dim_, 0.0f);
      const std::uint32_t mse_mask = (1u << mse_bits_) - 1u;

      for (std::size_t i = begin; i < end; ++i) {
        if (storage_.path == StorageLayout::Path::kPackedNibble) {
          const std::size_t bi   = i / kPackedBlockSize;
          const std::size_t lane = i % kPackedBlockSize;
          const std::uint8_t* pk = storage_.packed_nibble_block_ptr(bi);
          for (std::size_t j = 0; j < padded_dim_; ++j) {
            const std::uint8_t byte = pk[j * 16 + (lane < 16 ? lane : lane - 16)];
            const std::uint8_t nib  = (lane < 16) ? (byte & 0x0f) : (byte >> 4);
            const std::uint32_t code = static_cast<std::uint32_t>(nib) & mse_mask;
            rotated[j] = (code < codebook_.size) ? codebook_.centroids[code] : 0.0f;
          }
        } else if (storage_.path == StorageLayout::Path::kNibble) {
          const std::size_t bi   = i / kBlockSize;
          const std::size_t lane = i % kBlockSize;
          const std::uint8_t* nibs = storage_.nibble_dim_ptr(bi, 0);
          for (std::size_t j = 0; j < padded_dim_; ++j) {
            const std::uint8_t nib  = nibs[j * kBlockSize + lane];
            const std::uint32_t code = static_cast<std::uint32_t>(nib) & mse_mask;
            rotated[j] = (code < codebook_.size) ? codebook_.centroids[code] : 0.0f;
          }
        } else {
          const std::size_t bi   = i / kBlockSize;
          const std::size_t lane = i % kBlockSize;
          const std::uint8_t* bc = storage_.byte_code_dim_ptr(bi, 0);
          for (std::size_t j = 0; j < padded_dim_; ++j) {
            const std::uint8_t raw  = bc[j * kBlockSize + lane];
            const std::uint32_t code = storage_.combined_code_sign
                                     ? (static_cast<std::uint32_t>(raw) & mse_mask)
                                     : static_cast<std::uint32_t>(raw);
            rotated[j] = (code < codebook_.size) ? codebook_.centroids[code] : 0.0f;
          }
        }

        rotation_.backward(rotated.data(), unit.data(), work.data());

        const float norm = storage_.norms[i];
        float* out_ptr = out + i * dim_;
        if (use_data_centroid_) {
          for (std::size_t j = 0; j < dim_; ++j)
            out_ptr[j] = unit[j] * norm + centroid_[j];
        } else {
          for (std::size_t j = 0; j < dim_; ++j)
            out_ptr[j] = unit[j] * norm;
        }
      }
    });
  }
};

}  // namespace turboquant
