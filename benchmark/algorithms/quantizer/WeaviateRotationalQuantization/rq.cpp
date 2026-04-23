// Rotational Quantization (RQ) — C++ implementation.
// Matches Weaviate's layout: bits=1 is BRQ (1-bit data, 5-bit bit-sliced
// asymmetric query); bits=2/4/8 is uniform RQ with 1 byte per dim.

#include "rq.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace rq {

// ---------- PRNG: splitmix64 ----------
namespace {
inline uint64_t splitmix64(uint64_t& s) {
  s += 0x9E3779B97F4A7C15ULL;
  uint64_t z = s;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}
}  // namespace

// ---------- FWHT (length must be a power of 2, normalized to orthonormal) ----------
static void fwht_inplace(float* x, int n) {
  for (int h = 1; h < n; h <<= 1) {
    for (int i = 0; i < n; i += 2 * h) {
#if defined(__AVX2__)
      if (h >= 8) {
        for (int j = 0; j < h; j += 8) {
          __m256 a = _mm256_loadu_ps(x + i + j);
          __m256 b = _mm256_loadu_ps(x + i + j + h);
          _mm256_storeu_ps(x + i + j,     _mm256_add_ps(a, b));
          _mm256_storeu_ps(x + i + j + h, _mm256_sub_ps(a, b));
        }
        continue;
      }
#endif
      for (int j = 0; j < h; j++) {
        float u = x[i + j];
        float v = x[i + j + h];
        x[i + j]     = u + v;
        x[i + j + h] = u - v;
      }
    }
  }
  float norm = 1.0f / std::sqrt(static_cast<float>(n));
#if defined(__AVX2__)
  __m256 vn = _mm256_set1_ps(norm);
  int j = 0;
  for (; j + 8 <= n; j += 8) {
    _mm256_storeu_ps(x + j, _mm256_mul_ps(_mm256_loadu_ps(x + j), vn));
  }
  for (; j < n; j++) x[j] *= norm;
#else
  for (int j = 0; j < n; j++) x[j] *= norm;
#endif
}

// ---------- SIMD helpers ----------
#if defined(__AVX2__)
// Dot product Σ a[i]*b[i] over n bytes (both unsigned 0..255).
// Two int32 accumulators hide vpmaddwd latency.
static inline int32_t dot_u8_avx2(const uint8_t* a, const uint8_t* b, size_t n) {
  const __m256i zero = _mm256_setzero_si256();
  __m256i acc0 = _mm256_setzero_si256();
  __m256i acc1 = _mm256_setzero_si256();
  size_t i = 0;
  for (; i + 64 <= n; i += 64) {
    __m256i va0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
    __m256i vb0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
    __m256i va1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i + 32));
    __m256i vb1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i + 32));
    __m256i a0_lo = _mm256_unpacklo_epi8(va0, zero);
    __m256i a0_hi = _mm256_unpackhi_epi8(va0, zero);
    __m256i b0_lo = _mm256_unpacklo_epi8(vb0, zero);
    __m256i b0_hi = _mm256_unpackhi_epi8(vb0, zero);
    acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(a0_lo, b0_lo));
    acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(a0_hi, b0_hi));
    __m256i a1_lo = _mm256_unpacklo_epi8(va1, zero);
    __m256i a1_hi = _mm256_unpackhi_epi8(va1, zero);
    __m256i b1_lo = _mm256_unpacklo_epi8(vb1, zero);
    __m256i b1_hi = _mm256_unpackhi_epi8(vb1, zero);
    acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(a1_lo, b1_lo));
    acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(a1_hi, b1_hi));
  }
  for (; i + 32 <= n; i += 32) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
    __m256i a_lo = _mm256_unpacklo_epi8(va, zero);
    __m256i a_hi = _mm256_unpackhi_epi8(va, zero);
    __m256i b_lo = _mm256_unpacklo_epi8(vb, zero);
    __m256i b_hi = _mm256_unpackhi_epi8(vb, zero);
    acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(a_lo, b_lo));
    acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(a_hi, b_hi));
  }
  __m256i acc = _mm256_add_epi32(acc0, acc1);
  __m128i s = _mm_add_epi32(_mm256_castsi256_si128(acc),
                            _mm256_extracti128_si256(acc, 1));
  s = _mm_hadd_epi32(s, s);
  s = _mm_hadd_epi32(s, s);
  int32_t r = _mm_cvtsi128_si32(s);
  for (; i < n; i++) r += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
  return r;
}
#endif

static inline int32_t dot_u8(const uint8_t* a, const uint8_t* b, size_t n) {
#if defined(__AVX2__)
  return dot_u8_avx2(a, b, n);
#else
  int32_t r = 0;
  for (size_t i = 0; i < n; i++) r += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
  return r;
#endif
}

// Hamming distances between x_sign and 5 query bit-planes (bits0..bits4),
// each of length `nwords` uint64 words. 5 independent popcount chains → ILP.
static inline void hamming_5planes(const uint64_t* x_sign,
                                   const uint64_t* qbits, int nwords,
                                   int& h0, int& h1, int& h2, int& h3, int& h4) {
  const uint64_t* p0 = qbits + 0 * nwords;
  const uint64_t* p1 = qbits + 1 * nwords;
  const uint64_t* p2 = qbits + 2 * nwords;
  const uint64_t* p3 = qbits + 3 * nwords;
  const uint64_t* p4 = qbits + 4 * nwords;
  int a = 0, b = 0, c = 0, d = 0, e = 0;
  for (int i = 0; i < nwords; i++) {
    uint64_t xw = x_sign[i];
    a += __builtin_popcountll(xw ^ p0[i]);
    b += __builtin_popcountll(xw ^ p1[i]);
    c += __builtin_popcountll(xw ^ p2[i]);
    d += __builtin_popcountll(xw ^ p3[i]);
    e += __builtin_popcountll(xw ^ p4[i]);
  }
  h0 = a; h1 = b; h2 = c; h3 = d; h4 = e;
}

// Single pass over rx[0..n): compute min, max, and Σ rx[i]^2.
static inline void scan_stats(const float* rx, int n,
                              float& lo_out, float& hi_out, float& n2_out) {
  float lo = rx[0];
  float hi = rx[0];
#if defined(__AVX2__)
  __m256 vlo = _mm256_set1_ps(lo);
  __m256 vhi = _mm256_set1_ps(hi);
  __m256 vn2 = _mm256_setzero_ps();
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    __m256 v = _mm256_loadu_ps(rx + i);
    vlo = _mm256_min_ps(vlo, v);
    vhi = _mm256_max_ps(vhi, v);
    vn2 = _mm256_fmadd_ps(v, v, vn2);
  }
  alignas(32) float lo_buf[8], hi_buf[8], n2_buf[8];
  _mm256_store_ps(lo_buf, vlo);
  _mm256_store_ps(hi_buf, vhi);
  _mm256_store_ps(n2_buf, vn2);
  for (int k = 0; k < 8; k++) {
    if (lo_buf[k] < lo) lo = lo_buf[k];
    if (hi_buf[k] > hi) hi = hi_buf[k];
  }
  float norm2 = 0.f;
  for (int k = 0; k < 8; k++) norm2 += n2_buf[k];
  for (; i < n; i++) {
    float v = rx[i];
    if (v < lo) lo = v;
    if (v > hi) hi = v;
    norm2 += v * v;
  }
#else
  float norm2 = rx[0] * rx[0];
  for (int i = 1; i < n; i++) {
    float v = rx[i];
    if (v < lo) lo = v;
    if (v > hi) hi = v;
    norm2 += v * v;
  }
#endif
  lo_out = lo; hi_out = hi; n2_out = norm2;
}

// ---------- RotationalQuantizer ----------
RotationalQuantizer::RotationalQuantizer(int d, int bits, Metric metric, uint64_t seed)
    : d_(d), bits_(bits), metric_(metric), rounds_(3) {
  if (bits != 1 && bits != 2 && bits != 4 && bits != 8) {
    throw std::invalid_argument("RotationalQuantizer: only bits in {1, 2, 4, 8} are supported");
  }
  if (d <= 0) throw std::invalid_argument("RotationalQuantizer: d must be > 0");

  int padded = d;
  if (bits == 1 && padded < kBrqMinDim) padded = kBrqMinDim;
  out_d_ = ((padded + kBlock - 1) / kBlock) * kBlock;

  uint64_t st = seed;
  swap_a_.resize(rounds_);
  swap_b_.resize(rounds_);
  signs_.resize(rounds_);
  for (int r = 0; r < rounds_; r++) {
    std::vector<int> perm(out_d_);
    for (int i = 0; i < out_d_; i++) perm[i] = i;
    for (int i = out_d_ - 1; i > 0; i--) {
      int j = static_cast<int>(splitmix64(st) % static_cast<uint64_t>(i + 1));
      std::swap(perm[i], perm[j]);
    }
    int npairs = out_d_ / 2;
    swap_a_[r].resize(npairs);
    swap_b_[r].resize(npairs);
    for (int i = 0; i < npairs; i++) {
      swap_a_[r][i] = perm[2 * i];
      swap_b_[r][i] = perm[2 * i + 1];
    }
    signs_[r].resize(out_d_);
    for (int i = 0; i < out_d_; i++) {
      signs_[r][i] = (splitmix64(st) & 1ULL) ? -1.0f : 1.0f;
    }
  }

  if (bits == 1) {
    // Randomized rounding in [0, 1) per dim — matches Weaviate's
    // BRQ encodeQuery, makes the 5-bit query estimator unbiased.
    rounding_.resize(out_d_);
    for (int i = 0; i < out_d_; i++) {
      uint32_t r32 = static_cast<uint32_t>(splitmix64(st) >> 32);
      rounding_[i] = static_cast<float>(r32) * (1.0f / 4294967296.0f);
    }
  }

  // Per-data-vector code size.
  code_bytes_ = (bits == 1) ? static_cast<size_t>(out_d_ / 8)
                            : static_cast<size_t>(out_d_);
}

void RotationalQuantizer::reset() {
  ntotal_ = 0;
  codes_.clear();
  meta_.clear();
}

void RotationalQuantizer::train(size_t, const float*) {}

void RotationalQuantizer::rotate(const float* x, float* out) const {
  std::memcpy(out, x, sizeof(float) * static_cast<size_t>(d_));
  if (out_d_ > d_) {
    std::memset(out + d_, 0, sizeof(float) * static_cast<size_t>(out_d_ - d_));
  }

  for (int r = 0; r < rounds_; r++) {
    const int* sa = swap_a_[r].data();
    const int* sb = swap_b_[r].data();
    size_t npairs = swap_a_[r].size();
    for (size_t s = 0; s < npairs; s++) {
      float tmp = out[sa[s]];
      out[sa[s]] = out[sb[s]];
      out[sb[s]] = tmp;
    }
    const float* sg = signs_[r].data();
#if defined(__AVX2__)
    for (int i = 0; i < out_d_; i += 8) {
      __m256 v = _mm256_loadu_ps(out + i);
      __m256 s = _mm256_loadu_ps(sg + i);
      _mm256_storeu_ps(out + i, _mm256_mul_ps(v, s));
    }
#else
    for (int i = 0; i < out_d_; i++) out[i] *= sg[i];
#endif
    for (int off = 0; off < out_d_; off += kBlock) {
      fwht_inplace(out + off, kBlock);
    }
  }
}

// ---------- Uniform RQ (bits=2, 4, 8) ----------
void RotationalQuantizer::encode_uniform(const float* rx, uint8_t* code, float* meta) const {
  float lo, hi, norm2;
  scan_stats(rx, out_d_, lo, hi, norm2);

  const int max_code = (1 << bits_) - 1;
  const float step = (hi - lo) / static_cast<float>(max_code);
  const float inv  = (step > 0.f) ? 1.0f / step : 0.0f;
  int64_t csum = 0;
  for (int i = 0; i < out_d_; i++) {
    int c = static_cast<int>((rx[i] - lo) * inv + 0.5f);
    if (c < 0) c = 0;
    else if (c > max_code) c = max_code;
    code[i] = static_cast<uint8_t>(c);
    csum += c;
  }
  meta[0] = lo;
  meta[1] = step;
  meta[2] = step * static_cast<float>(csum);  // "CodeSum" pre-multiplied by step
  meta[3] = norm2;
}

float RotationalQuantizer::dist_uniform(const uint8_t* cx, const float* mx,
                                        const uint8_t* cq, const float* mq) const {
  float lo_x = mx[0], step_x = mx[1], cs_x = mx[2], n2_x = mx[3];
  float lo_q = mq[0], step_q = mq[1], cs_q = mq[2], n2_q = mq[3];
  int32_t d_bytes = dot_u8(cx, cq, static_cast<size_t>(out_d_));
  float a = static_cast<float>(out_d_) * lo_x * lo_q;
  float b = lo_x * cs_q;
  float c = lo_q * cs_x;
  float dterm = step_x * step_q * static_cast<float>(d_bytes);
  float dot_est = a + b + c + dterm;
  if (metric_ == Metric::L2) return n2_x + n2_q - 2.0f * dot_est;
  return dot_est;  // IP
}

// ---------- BRQ (bits=1) ----------
void RotationalQuantizer::encode_brq_data(const float* rx, uint8_t* code, float* meta) const {
  uint64_t* words = reinterpret_cast<uint64_t*>(code);
  const int nwords = out_d_ / 64;
  float norm2 = 0.f;
  float l1    = 0.f;
  int i = 0;
  for (int b = 0; b < nwords; b++) {
    uint64_t w = 0;
    for (uint64_t bit = 1; bit != 0; bit <<= 1) {
      float v = rx[i];
      if (v > 0.f) { w |= bit; l1 += v; }
      else         {           l1 -= v; }
      norm2 += v * v;
      i++;
    }
    words[b] = w;
  }
  // Weaviate: step = ‖rx‖² / ‖rx‖₁. Baked-in normalization for the RaBitQ estimator.
  float step = (l1 > 0.f) ? norm2 / l1 : 0.f;
  meta[0] = step;
  meta[1] = norm2;
  meta[2] = 0.f;
  meta[3] = 0.f;
}

void RotationalQuantizer::encode_brq_query(const float* rx,
                                           uint64_t* qbits_out,
                                           float& step_out, float& norm2_out) const {
  const int nwords = out_d_ / 64;
  // Pass 1: ‖rx‖² and max|rx|.
  float abs_max = 0.f;
  float norm2   = 0.f;
  for (int i = 0; i < out_d_; i++) {
    float v = rx[i];
    norm2 += v * v;
    float av = std::fabs(v);
    if (av > abs_max) abs_max = av;
  }
  std::memset(qbits_out, 0, sizeof(uint64_t) * 5 * nwords);
  if (abs_max == 0.f) { step_out = 0.f; norm2_out = norm2; return; }

  const float step     = abs_max / 31.0f;
  const float inv_2st  = 1.0f / (2.0f * step);

  uint64_t* p0 = qbits_out + 0 * nwords;
  uint64_t* p1 = qbits_out + 1 * nwords;
  uint64_t* p2 = qbits_out + 2 * nwords;
  uint64_t* p3 = qbits_out + 3 * nwords;
  uint64_t* p4 = qbits_out + 4 * nwords;

  int i = 0;
  for (int b = 0; b < nwords; b++) {
    uint64_t b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0;
    for (uint64_t bit = 1; bit != 0; bit <<= 1) {
      // Matches Weaviate: c = uint64((rx+abs_max)/(2*step) + rounding[i])
      float f = (rx[i] + abs_max) * inv_2st + rounding_[i];
      int c = static_cast<int>(f);          // truncation (f >= 0 by construction)
      if (c < 0) c = 0;                     // defensive, shouldn't happen
      else if (c > 31) c = 31;
      if (c & 1)  b0 |= bit;
      if (c & 2)  b1 |= bit;
      if (c & 4)  b2 |= bit;
      if (c & 8)  b3 |= bit;
      if (c & 16) b4 |= bit;
      i++;
    }
    p0[b] = b0; p1[b] = b1; p2[b] = b2; p3[b] = b3; p4[b] = b4;
  }
  step_out  = step;
  norm2_out = norm2;
}

float RotationalQuantizer::dist_brq(const uint8_t* cx, const float* mx,
                                    const uint64_t* qbits,
                                    float q_step, float q_norm2) const {
  const float x_step  = mx[0];
  const float x_norm2 = mx[1];
  const uint64_t* x_sign = reinterpret_cast<const uint64_t*>(cx);
  const int nwords = out_d_ / 64;
  int h0, h1, h2, h3, h4;
  hamming_5planes(x_sign, qbits, nwords, h0, h1, h2, h3, h4);
  // dot_w = 31*D − 2 H0 − 4 H1 − 8 H2 − 16 H3 − 32 H4
  // (see derivation in the Go BRQ Distance implementation).
  int32_t dot_w = 31 * out_d_
                - (h0 << 1) - (h1 << 2) - (h2 << 3) - (h3 << 4) - (h4 << 5);
  float dot_est = q_step * x_step * static_cast<float>(dot_w);
  if (metric_ == Metric::L2) return x_norm2 + q_norm2 - 2.0f * dot_est;
  return dot_est;  // IP
}

// ---------- add / search ----------
void RotationalQuantizer::add(size_t n, const float* x) {
  size_t base = ntotal_;
  codes_.resize((base + n) * code_bytes_);
  meta_.resize((base + n) * 4);

  const int _nt_add = (num_threads_ > 0) ? num_threads_ :
#if defined(_OPENMP)
                      omp_get_max_threads();
#else
                      1;
#endif
#pragma omp parallel num_threads(_nt_add)
{
  std::vector<float> rx(static_cast<size_t>(out_d_));
#pragma omp for schedule(static)
  for (size_t i = 0; i < n; i++) {
    rotate(x + i * static_cast<size_t>(d_), rx.data());
    uint8_t* code = codes_.data() + (base + i) * code_bytes_;
    float*   m    = meta_.data()  + (base + i) * 4;
    if (bits_ == 1) encode_brq_data(rx.data(), code, m);
    else            encode_uniform(rx.data(), code, m);
  }
}
  ntotal_ += n;
}

void RotationalQuantizer::search(size_t nq, const float* queries, size_t k,
                                 float* distances, int64_t* labels) const {
  if (k == 0) return;

  struct Pair { float d; int64_t i; };
  auto cmp_max = [](const Pair& a, const Pair& b) { return a.d < b.d; };  // L2: max-heap
  auto cmp_min = [](const Pair& a, const Pair& b) { return a.d > b.d; };  // IP: min-heap

  auto scan = [&](auto dist_fn, auto cmp, std::vector<Pair>& heap, size_t ntotal) {
    for (size_t i = 0; i < ntotal; i++) {
      float dist = dist_fn(i);
      Pair p{dist, static_cast<int64_t>(i)};
      if (heap.size() < k) {
        heap.push_back(p);
        std::push_heap(heap.begin(), heap.end(), cmp);
      } else if (cmp(p, heap.front())) {
        std::pop_heap(heap.begin(), heap.end(), cmp);
        heap.back() = p;
        std::push_heap(heap.begin(), heap.end(), cmp);
      }
    }
  };

  const bool is_l2 = (metric_ == Metric::L2);
  const size_t ntotal = ntotal_;

  const int _nt_search = (num_threads_ > 0) ? num_threads_ :
#if defined(_OPENMP)
                         omp_get_max_threads();
#else
                         1;
#endif
#pragma omp parallel num_threads(_nt_search)
{
  // Per-thread scratch (allocated once, reused across queries this thread handles).
  std::vector<float>    rq_buf(static_cast<size_t>(out_d_));
  std::vector<Pair>     heap;
  heap.reserve(k);

  // Uniform query scratch (unused for bits=1).
  std::vector<uint8_t>  qcode_u(bits_ != 1 ? code_bytes_ : 0);
  float qmeta_u[4] = {0, 0, 0, 0};

  // BRQ query scratch (unused for bits≥2).
  const int nwords = out_d_ / 64;
  std::vector<uint64_t> qbits_b(bits_ == 1 ? static_cast<size_t>(5 * nwords) : 0);
  float q_step_b = 0.f;
  float q_norm2_b = 0.f;

  const uint8_t* codes_base = codes_.data();
  const float*   meta_base  = meta_.data();
  const size_t   cb         = code_bytes_;

#pragma omp for schedule(dynamic, 1) nowait
  for (size_t qi = 0; qi < nq; qi++) {
    heap.clear();
    rotate(queries + qi * static_cast<size_t>(d_), rq_buf.data());

    if (bits_ == 1) {
      encode_brq_query(rq_buf.data(), qbits_b.data(), q_step_b, q_norm2_b);
      auto df = [&](size_t i) {
        return dist_brq(codes_base + i * cb, meta_base + i * 4,
                        qbits_b.data(), q_step_b, q_norm2_b);
      };
      if (is_l2) scan(df, cmp_max, heap, ntotal);
      else       scan(df, cmp_min, heap, ntotal);
    } else {
      encode_uniform(rq_buf.data(), qcode_u.data(), qmeta_u);
      auto df = [&](size_t i) {
        return dist_uniform(codes_base + i * cb, meta_base + i * 4,
                            qcode_u.data(), qmeta_u);
      };
      if (is_l2) scan(df, cmp_max, heap, ntotal);
      else       scan(df, cmp_min, heap, ntotal);
    }

    if (is_l2) std::sort_heap(heap.begin(), heap.end(), cmp_max);
    else       std::sort_heap(heap.begin(), heap.end(), cmp_min);

    for (size_t j = 0; j < k; j++) {
      if (j < heap.size()) {
        distances[qi * k + j] = heap[j].d;
        labels[qi * k + j]    = heap[j].i;
      } else {
        distances[qi * k + j] = is_l2 ?  std::numeric_limits<float>::infinity()
                                      : -std::numeric_limits<float>::infinity();
        labels[qi * k + j] = -1;
      }
    }
  }
}  // omp parallel
}

}  // namespace rq
