#include "OSQIndex.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <immintrin.h>
#endif
#include <limits>
#include <queue>
#include <stdexcept>
#include <thread>

namespace osq {

namespace {

inline int popcnt8(uint8_t x) {
  return __builtin_popcount(static_cast<unsigned>(x));
}

inline int popcnt64(uint64_t x) {
  return __builtin_popcountll(static_cast<unsigned long long>(x));
}

inline uint64_t load_u64_unaligned(const uint8_t* p) {
  uint64_t v;
  std::memcpy(&v, p, sizeof(v));
  return v;
}

inline int popcnt_and_bytes(const uint8_t* a, const uint8_t* b, size_t n) {
  size_t i = 0;
  uint64_t sum = 0;

#if defined(__AVX512VPOPCNTDQ__) && defined(__AVX512BW__)
  for (; i + 64 <= n; i += 64) {
    const __m512i va = _mm512_loadu_si512(reinterpret_cast<const void*>(a + i));
    const __m512i vb = _mm512_loadu_si512(reinterpret_cast<const void*>(b + i));
    const __m512i v_and = _mm512_and_si512(va, vb);
    const __m512i v_cnt = _mm512_popcnt_epi64(v_and);
    sum += static_cast<uint64_t>(_mm512_reduce_add_epi64(v_cnt));
  }
#endif

#if defined(__AVX2__)
  const __m256i lut = _mm256_setr_epi8(
      0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
      0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
  const __m256i low_mask = _mm256_set1_epi8(0x0F);
  const __m256i zero = _mm256_setzero_si256();

  for (; i + 32 <= n; i += 32) {
    const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
    const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
    const __m256i v_and = _mm256_and_si256(va, vb);
    const __m256i lo = _mm256_and_si256(v_and, low_mask);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v_and, 4), low_mask);
    const __m256i cnt = _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo), _mm256_shuffle_epi8(lut, hi));
    const __m256i sad = _mm256_sad_epu8(cnt, zero);
    sum += static_cast<uint64_t>(_mm256_extract_epi64(sad, 0));
    sum += static_cast<uint64_t>(_mm256_extract_epi64(sad, 1));
    sum += static_cast<uint64_t>(_mm256_extract_epi64(sad, 2));
    sum += static_cast<uint64_t>(_mm256_extract_epi64(sad, 3));
  }
#endif

  for (; i < n; ++i) {
    sum += static_cast<uint64_t>(popcnt8(static_cast<uint8_t>(a[i] & b[i])));
  }

  return static_cast<int>(sum);
}

inline float scale_lut(uint8_t bits) {
  switch (bits) {
    case 1:
      return 1.0f;
    case 2:
      return 1.0f / 3.0f;
    case 3:
      return 1.0f / 7.0f;
    case 4:
      return 1.0f / 15.0f;
    case 5:
      return 1.0f / 31.0f;
    case 6:
      return 1.0f / 63.0f;
    case 7:
      return 1.0f / 127.0f;
    case 8:
      return 1.0f / 255.0f;
    default:
      throw std::invalid_argument("bits must be in [1,8]");
  }
}

template <typename Fn>
void parallel_for(size_t begin, size_t end, int num_threads, Fn fn) {
  if (end <= begin) return;
  if (num_threads <= 1 || (end - begin) < 2) {
    fn(begin, end, 0);
    return;
  }
  const size_t total = end - begin;
  const int t = std::max(1, std::min<int>(num_threads, static_cast<int>(total)));
  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(t));
  for (int tid = 0; tid < t; ++tid) {
    const size_t b = begin + (total * static_cast<size_t>(tid)) / static_cast<size_t>(t);
    const size_t e = begin + (total * static_cast<size_t>(tid + 1)) / static_cast<size_t>(t);
    workers.emplace_back([=, &fn]() { fn(b, e, tid); });
  }
  for (auto& th : workers) th.join();
}

} // namespace

OSQIndex::OSQIndex(size_t dims, Similarity similarity, ScalarEncoding encoding)
    : dims_(dims),
      discrete_dims_(EncodingConfig::discrete_dims(encoding, dims)),
      doc_packed_len_(EncodingConfig::doc_packed_len(encoding, dims)),
      similarity_(similarity),
      encoding_(encoding),
      trained_(false),
      num_threads_(std::max(1u, std::thread::hardware_concurrency())),
      query_scale_(scale_lut(EncodingConfig::query_bits(encoding))),
      doc_scale_(scale_lut(EncodingConfig::doc_bits(encoding))),
      centroid_(dims, 0.0f),
      centroid_dp_(0.0f),
      quantizer_(similarity) {}

void OSQIndex::set_num_threads(int n) {
  num_threads_ = std::max(1, n);
}

void OSQIndex::l2_normalize(std::vector<float>& v) {
  double s = 0.0;
  for (float x : v) s += static_cast<double>(x) * static_cast<double>(x);
  if (s <= 0.0) return;
  const float inv = static_cast<float>(1.0 / std::sqrt(s));
  for (float& x : v) x *= inv;
}

float OSQIndex::scale_max_inner_product_score(float s) {
  // Matches Lucene's monotonic scaling behavior.
  return (s < 0.0f) ? 1.0f / (1.0f - s) : (s + 1.0f);
}

void OSQIndex::train(size_t n, const float* x) {
  if (n == 0) {
    trained_ = true;
    centroid_dp_ = 0.0f;
    return;
  }

  std::fill(centroid_.begin(), centroid_.end(), 0.0f);
  const int t = std::max(1, std::min<int>(num_threads_, static_cast<int>(n)));
  std::vector<std::vector<float>> partial(static_cast<size_t>(t), std::vector<float>(dims_, 0.0f));

  parallel_for(0, n, t, [&](size_t b, size_t e, int tid) {
    auto& local = partial[static_cast<size_t>(tid)];
    for (size_t i = b; i < e; ++i) {
      const float* v = x + i * dims_;
      for (size_t j = 0; j < dims_; ++j) local[j] += v[j];
    }
  });

  for (int tid = 0; tid < t; ++tid) {
    const auto& local = partial[static_cast<size_t>(tid)];
    for (size_t j = 0; j < dims_; ++j) centroid_[j] += local[j];
  }

  const float inv_n = 1.0f / static_cast<float>(n);
  for (float& c : centroid_) c *= inv_n;

  if (similarity_ == Similarity::COSINE) {
    l2_normalize(centroid_);
  }

  centroid_dp_ = 0.0f;
  for (float c : centroid_) centroid_dp_ += c * c;
  trained_ = true;
}

EncodedVector OSQIndex::encode_doc(const float* x) const {
  const uint8_t dbits = EncodingConfig::doc_bits(encoding_);

  std::vector<float> v(x, x + dims_);
  if (similarity_ == Similarity::COSINE) l2_normalize(v);

  std::vector<uint8_t> scratch(discrete_dims_, 0);
  const QuantizationResult corr =
      quantizer_.scalar_quantize(v.data(), dims_, dbits, centroid_.data(), scratch.data());

  EncodedVector out;
  out.correction = corr;

  switch (encoding_) {
    case ScalarEncoding::UNSIGNED_BYTE:
    case ScalarEncoding::SEVEN_BIT:
      out.packed.assign(scratch.begin(), scratch.end());
      break;
    case ScalarEncoding::PACKED_NIBBLE: {
      out.packed.resize(EncodingConfig::doc_packed_len(encoding_, dims_), 0);
      for (size_t i = 0; i < discrete_dims_; i += 2) {
        const uint8_t lo = scratch[i] & 0x0F;
        const uint8_t hi = (i + 1 < discrete_dims_) ? (scratch[i + 1] & 0x0F) : 0;
        out.packed[i / 2] = static_cast<uint8_t>(lo | (hi << 4));
      }
      break;
    }
    case ScalarEncoding::SINGLE_BIT_QUERY_NIBBLE:
      out.packed.resize(EncodingConfig::doc_packed_len(encoding_, dims_), 0);
      OptimizedScalarQuantizer::pack_as_binary(
          scratch.data(), discrete_dims_, out.packed.data());
      break;
    case ScalarEncoding::DIBIT_QUERY_NIBBLE:
      out.packed.resize(EncodingConfig::doc_packed_len(encoding_, dims_), 0);
      OptimizedScalarQuantizer::transpose_dibit(
          scratch.data(), discrete_dims_, out.packed.data());
      break;
  }

  return out;
}

OSQIndex::EncodedQuery OSQIndex::encode_query(const float* q) const {
  const uint8_t qbits = EncodingConfig::query_bits(encoding_);

  std::vector<float> v(q, q + dims_);
  if (similarity_ == Similarity::COSINE) l2_normalize(v);

  std::vector<uint8_t> scratch(discrete_dims_, 0);
  QuantizationResult corr =
      quantizer_.scalar_quantize(v.data(), dims_, qbits, centroid_.data(), scratch.data());

  EncodedQuery out;
  out.corr = corr;
  out.ay = corr.lower_interval;
  out.ly = (corr.upper_interval - corr.lower_interval) * query_scale_;
  out.sy = static_cast<float>(corr.quantized_component_sum);
  out.additional_correction = corr.additional_correction;

  if (!EncodingConfig::is_asymmetric(encoding_)) {
    out.q.assign(scratch.begin(), scratch.end());
    return out;
  }

  out.q.resize(EncodingConfig::query_packed_len(encoding_, dims_), 0);
  OptimizedScalarQuantizer::transpose_half_byte(
      scratch.data(), discrete_dims_, out.q.data());
  return out;
}

int OSQIndex::dot_uint8(const uint8_t* a, const uint8_t* b, size_t n) {
#if defined(__AVX2__)
  size_t i = 0;
  __m256i acc = _mm256_setzero_si256();
  const __m256i ones = _mm256_set1_epi16(1);

  for (; i + 32 <= n; i += 32) {
    const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
    const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));

    const __m128i va_lo_128 = _mm256_castsi256_si128(va);
    const __m128i va_hi_128 = _mm256_extracti128_si256(va, 1);
    const __m128i vb_lo_128 = _mm256_castsi256_si128(vb);
    const __m128i vb_hi_128 = _mm256_extracti128_si256(vb, 1);

    const __m256i va_lo = _mm256_cvtepu8_epi16(va_lo_128);
    const __m256i va_hi = _mm256_cvtepu8_epi16(va_hi_128);
    const __m256i vb_lo = _mm256_cvtepu8_epi16(vb_lo_128);
    const __m256i vb_hi = _mm256_cvtepu8_epi16(vb_hi_128);

    const __m256i prod_lo = _mm256_mullo_epi16(va_lo, vb_lo);
    const __m256i prod_hi = _mm256_mullo_epi16(va_hi, vb_hi);

    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(prod_lo, ones));
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(prod_hi, ones));
  }

  alignas(32) int32_t lanes[8];
  _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), acc);
  int s = lanes[0] + lanes[1] + lanes[2] + lanes[3] + lanes[4] + lanes[5] + lanes[6] + lanes[7];
  for (; i < n; ++i) s += static_cast<int>(a[i]) * static_cast<int>(b[i]);
  return s;
#else
  int s = 0;
  for (size_t i = 0; i < n; ++i) s += static_cast<int>(a[i]) * static_cast<int>(b[i]);
  return s;
#endif
}

int OSQIndex::dot_int4_with_packed_doc(const uint8_t* q, const uint8_t* packed_doc, size_t dims) {
  size_t i = 0;
  int s = 0;

#if defined(__AVX512F__) && defined(__AVX512BW__)
  {
    __m512i acc512 = _mm512_setzero_si512();
    const __m512i ones512 = _mm512_set1_epi16(1);
    const __m128i nibble_mask128 = _mm_set1_epi8(0x0F);
    for (; i + 32 <= dims; i += 32) {
      // Load 16 packed bytes = 32 nibbles
      const __m128i packed = _mm_loadu_si128(
          reinterpret_cast<const __m128i*>(packed_doc + i / 2));
      const __m128i lo = _mm_and_si128(packed, nibble_mask128);
      const __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask128);
      // Interleave: [lo0,hi0,lo1,hi1,...] = dims[i..i+31] as uint8
      const __m128i unp_lo = _mm_unpacklo_epi8(lo, hi);
      const __m128i unp_hi = _mm_unpackhi_epi8(lo, hi);
      const __m256i unpacked = _mm256_set_m128i(unp_hi, unp_lo);
      // Load 32 query bytes
      const __m256i qv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q + i));
      // Zero-extend 32 uint8 → 32 int16 and multiply-accumulate
      const __m512i q16 = _mm512_cvtepu8_epi16(qv);
      const __m512i d16 = _mm512_cvtepu8_epi16(unpacked);
      const __m512i prod = _mm512_mullo_epi16(q16, d16);
      acc512 = _mm512_add_epi32(acc512, _mm512_madd_epi16(prod, ones512));
    }
    s = _mm512_reduce_add_epi32(acc512);
  }
#endif

#if defined(__AVX2__)
  {
    __m256i acc = _mm256_setzero_si256();
    const __m128i nibble_mask = _mm_set1_epi8(0x0F);
    const __m256i ones = _mm256_set1_epi16(1);
    for (; i + 16 <= dims; i += 16) {
      const __m128i packed = _mm_loadl_epi64(
          reinterpret_cast<const __m128i*>(packed_doc + i / 2));
      const __m128i lo = _mm_and_si128(packed, nibble_mask);
      const __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask);
      const __m128i unpacked = _mm_unpacklo_epi8(lo, hi);
      const __m128i q16_128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q + i));
      const __m256i q16_wide = _mm256_cvtepu8_epi16(q16_128);
      const __m256i d16_wide = _mm256_cvtepu8_epi16(unpacked);
      const __m256i prod = _mm256_mullo_epi16(q16_wide, d16_wide);
      acc = _mm256_add_epi32(acc, _mm256_madd_epi16(prod, ones));
    }
    alignas(32) int32_t lanes[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), acc);
    s += lanes[0] + lanes[1] + lanes[2] + lanes[3] +
         lanes[4] + lanes[5] + lanes[6] + lanes[7];
  }
#endif

  for (; i < dims; ++i) {
    const uint8_t p = packed_doc[i / 2];
    const uint8_t d = (i % 2 == 0) ? (p & 0x0F) : ((p >> 4) & 0x0F);
    s += static_cast<int>(q[i]) * static_cast<int>(d);
  }
  return s;
}

int OSQIndex::dot_int4_with_binary_doc_transposed(
    const uint8_t* q_transposed,
    const uint8_t* binary_doc,
    size_t dims) {
  const size_t stripe = (dims + 7) / 8;
  const uint8_t* q0 = q_transposed;
  const uint8_t* q1 = q_transposed + stripe;
  const uint8_t* q2 = q_transposed + 2 * stripe;
  const uint8_t* q3 = q_transposed + 3 * stripe;
  return popcnt_and_bytes(q0, binary_doc, stripe) +
         2 * popcnt_and_bytes(q1, binary_doc, stripe) +
         4 * popcnt_and_bytes(q2, binary_doc, stripe) +
         8 * popcnt_and_bytes(q3, binary_doc, stripe);
}

int OSQIndex::dot_int4_with_dibit_doc_transposed(
    const uint8_t* q_transposed,
    const uint8_t* dibit_doc,
    size_t dims) {
  const size_t stripe = (dims + 7) / 8;
  const uint8_t* low = dibit_doc;
  const uint8_t* high = dibit_doc + stripe;
  const int low_score = dot_int4_with_binary_doc_transposed(q_transposed, low, dims);
  const int high_score = dot_int4_with_binary_doc_transposed(q_transposed, high, dims);
  return low_score + 2 * high_score;
}

const uint8_t* OSQIndex::doc_packed_ptr(size_t idx) const {
  return packed_codes_.data() + idx * doc_packed_len_;
}

const QuantizationResult& OSQIndex::doc_correction(size_t idx) const {
  return corrections_[idx];
}

float OSQIndex::score_query_doc(
    const EncodedQuery& q,
    const uint8_t* packed,
    const QuantizationResult& correction) const {
  float qc_dist = 0.0f;
  switch (encoding_) {
    case ScalarEncoding::UNSIGNED_BYTE:
    case ScalarEncoding::SEVEN_BIT:
      qc_dist = static_cast<float>(dot_uint8(q.q.data(), packed, discrete_dims_));
      break;
    case ScalarEncoding::PACKED_NIBBLE:
      qc_dist = static_cast<float>(dot_int4_with_packed_doc(q.q.data(), packed, discrete_dims_));
      break;
    case ScalarEncoding::SINGLE_BIT_QUERY_NIBBLE:
      qc_dist = static_cast<float>(
          dot_int4_with_binary_doc_transposed(q.q.data(), packed, discrete_dims_));
      break;
    case ScalarEncoding::DIBIT_QUERY_NIBBLE:
      qc_dist = static_cast<float>(
          dot_int4_with_dibit_doc_transposed(q.q.data(), packed, discrete_dims_));
      break;
  }

  const float ax = correction.lower_interval;
  const float ay = q.ay;
  const float lx = (correction.upper_interval - ax) * doc_scale_;
  const float ly = q.ly;
  const float sx = static_cast<float>(correction.quantized_component_sum);
  const float sy = q.sy;

  float score =
      ax * ay * static_cast<float>(dims_) + ay * lx * sx + ax * ly * sy + lx * ly * qc_dist;

  if (similarity_ == Similarity::EUCLIDEAN) {
    score = q.additional_correction + correction.additional_correction - 2.0f * score;
    if (score < 0.0f) score = 0.0f;
    return 1.0f / (1.0f + score);
  }

  score += q.additional_correction + correction.additional_correction - centroid_dp_;

  if (similarity_ == Similarity::MAX_INNER_PRODUCT) {
    return scale_max_inner_product_score(score);
  }

  // dot/cosine style mapped to [0,1]
  score = std::max(-1.0f, std::min(1.0f, score));
  return (1.0f + score) / 2.0f;
}

void OSQIndex::add(size_t n, const float* x) {
  if (!trained_) train(n, x);
  std::vector<EncodedVector> encoded(n);
  parallel_for(0, n, num_threads_, [&](size_t b, size_t e, int /*tid*/) {
    for (size_t i = b; i < e; ++i) {
      encoded[i] = encode_doc(x + i * dims_);
    }
  });
  const size_t old_size = corrections_.size();
  corrections_.resize(old_size + n);
  packed_codes_.resize((old_size + n) * doc_packed_len_);
  for (size_t i = 0; i < n; ++i) {
    corrections_[old_size + i] = encoded[i].correction;
    std::memcpy(
        packed_codes_.data() + (old_size + i) * doc_packed_len_,
        encoded[i].packed.data(),
        doc_packed_len_);
  }
}

void OSQIndex::prefetch_doc(idx_t id) const {
  if (id < 0 || static_cast<size_t>(id) >= corrections_.size()) {
    return;
  }
  const size_t idx = static_cast<size_t>(id);
  __builtin_prefetch(doc_packed_ptr(idx), 0, 1);
  __builtin_prefetch(&doc_correction(idx), 0, 1);
}

void OSQIndex::search(size_t nq, const float* queries, size_t k, float* distances, idx_t* labels) const {
  if (!trained_) throw std::runtime_error("index is not trained");
  if (k == 0) return;
  constexpr size_t kPrefetchDistance = 16;

  parallel_for(0, nq, num_threads_, [&](size_t qb, size_t qe, int /*tid*/) {
    for (size_t qi = qb; qi < qe; ++qi) {
      EncodedQuery eq = encode_query(queries + qi * dims_);
      const size_t dd = EncodingConfig::discrete_dims(encoding_, dims_);
      const float qscale = scale_lut(EncodingConfig::query_bits(encoding_));
      const float dscale = scale_lut(EncodingConfig::doc_bits(encoding_));
      const float ay = eq.corr.lower_interval;
      const float ly = (eq.corr.upper_interval - ay) * qscale;
      const float sy = static_cast<float>(eq.corr.quantized_component_sum);
      const float q_add = eq.corr.additional_correction;

      // Min-heap: top() is always the worst (smallest) score in current top-k.
      // O(log k) insertion vs O(k) linear scan.
      auto pq_cmp = [](const SearchResult& a, const SearchResult& b) {
        return a.score > b.score;
      };
      std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(pq_cmp)> pq(pq_cmp);

      for (size_t i = 0; i < corrections_.size(); ++i) {
        if (i + kPrefetchDistance < corrections_.size()) {
          __builtin_prefetch(doc_packed_ptr(i + kPrefetchDistance), 0, 1);
          __builtin_prefetch(&doc_correction(i + kPrefetchDistance), 0, 1);
        }

        const uint8_t* packed = doc_packed_ptr(i);
        const QuantizationResult& correction = doc_correction(i);
        float qc_dist = 0.0f;
        switch (encoding_) {
          case ScalarEncoding::UNSIGNED_BYTE:
          case ScalarEncoding::SEVEN_BIT:
            qc_dist = static_cast<float>(dot_uint8(eq.q.data(), packed, dd));
            break;
          case ScalarEncoding::PACKED_NIBBLE:
            qc_dist = static_cast<float>(dot_int4_with_packed_doc(eq.q.data(), packed, dd));
            break;
          case ScalarEncoding::SINGLE_BIT_QUERY_NIBBLE:
            qc_dist = static_cast<float>(dot_int4_with_binary_doc_transposed(
                eq.q.data(), packed, dd));
            break;
          case ScalarEncoding::DIBIT_QUERY_NIBBLE:
            qc_dist = static_cast<float>(dot_int4_with_dibit_doc_transposed(
                eq.q.data(), packed, dd));
            break;
        }

        const float ax = correction.lower_interval;
        const float lx = (correction.upper_interval - ax) * dscale;
        const float sx = static_cast<float>(correction.quantized_component_sum);
        float score =
            ax * ay * static_cast<float>(dims_) + ay * lx * sx + ax * ly * sy + lx * ly * qc_dist;

        if (similarity_ == Similarity::EUCLIDEAN) {
          score = q_add + correction.additional_correction - 2.0f * score;
          if (score < 0.0f) score = 0.0f;
          score = 1.0f / (1.0f + score);
        } else {
          score += q_add + correction.additional_correction - centroid_dp_;
          if (similarity_ == Similarity::MAX_INNER_PRODUCT) {
            score = scale_max_inner_product_score(score);
          } else {
            score = std::max(-1.0f, std::min(1.0f, score));
            score = (1.0f + score) / 2.0f;
          }
        }

        if (pq.size() < k) {
          pq.push(SearchResult{score, static_cast<idx_t>(i)});
        } else if (score > pq.top().score) {
          pq.pop();
          pq.push(SearchResult{score, static_cast<idx_t>(i)});
        }
      }

      std::vector<SearchResult> topk;
      topk.reserve(pq.size());
      while (!pq.empty()) {
        topk.push_back(pq.top());
        pq.pop();
      }

      std::sort(topk.begin(), topk.end(), [](const SearchResult& a, const SearchResult& b) {
        return a.score > b.score;
      });

      for (size_t i = 0; i < k; ++i) {
        const size_t at = qi * k + i;
        if (i < topk.size()) {
          distances[at] = topk[i].score;
          labels[at] = topk[i].id;
        } else {
          distances[at] = -std::numeric_limits<float>::infinity();
          labels[at] = -1;
        }
      }
    }
  });
}

float OSQIndex::score(const float* query, idx_t id) const {
  return score(encode_query(query), id);
}

float OSQIndex::score(const EncodedQuery& query, idx_t id) const {
  if (!trained_) throw std::runtime_error("index is not trained");
  if (id < 0 || static_cast<size_t>(id) >= corrections_.size()) {
    throw std::out_of_range("OSQIndex::score id out of range");
  }
  const size_t idx = static_cast<size_t>(id);
  return score_query_doc(query, doc_packed_ptr(idx), doc_correction(idx));
}

void OSQIndex::reconstruct(idx_t id, float* output) const {
  if (!trained_) throw std::runtime_error("index is not trained");
  if (id < 0 || static_cast<size_t>(id) >= corrections_.size()) {
    throw std::out_of_range("OSQIndex::reconstruct id out of range");
  }

  const size_t idx = static_cast<size_t>(id);
  const auto& correction = doc_correction(idx);
  const uint8_t* packed = doc_packed_ptr(idx);
  std::vector<uint8_t> unpacked(discrete_dims_, 0);

  switch (encoding_) {
    case ScalarEncoding::UNSIGNED_BYTE:
    case ScalarEncoding::SEVEN_BIT:
      std::memcpy(unpacked.data(), packed, discrete_dims_ * sizeof(uint8_t));
      break;
    case ScalarEncoding::PACKED_NIBBLE:
      for (size_t i = 0; i < discrete_dims_; ++i) {
        const uint8_t p = packed[i / 2];
        unpacked[i] = (i % 2 == 0) ? (p & 0x0F) : ((p >> 4) & 0x0F);
      }
      break;
    case ScalarEncoding::SINGLE_BIT_QUERY_NIBBLE:
      OptimizedScalarQuantizer::unpack_binary(packed, discrete_dims_, unpacked.data());
      break;
    case ScalarEncoding::DIBIT_QUERY_NIBBLE:
      OptimizedScalarQuantizer::untranspose_dibit(packed, discrete_dims_, unpacked.data());
      break;
  }

  OptimizedScalarQuantizer::dequantize(
      unpacked.data(),
      dims_,
      EncodingConfig::doc_bits(encoding_),
      correction.lower_interval,
      correction.upper_interval,
      centroid_.data(),
      output);
}

void OSQIndex::score_batch_4(
    const EncodedQuery& query,
    idx_t id0,
    idx_t id1,
    idx_t id2,
    idx_t id3,
    float& s0,
    float& s1,
    float& s2,
    float& s3) const {
  if (!trained_) throw std::runtime_error("index is not trained");
  prefetch_doc(id0);
  prefetch_doc(id1);
  prefetch_doc(id2);
  prefetch_doc(id3);
  s0 = score(query, id0);
  s1 = score(query, id1);
  s2 = score(query, id2);
  s3 = score(query, id3);
}

bool OSQIndex::save(const std::string& path) const {
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs.is_open()) return false;

  const uint64_t magic = 0x4f5351494e445831ULL;  // OSQINDX1
  const uint64_t version = 1;
  ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
  ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));

  auto write_u64 = [&](uint64_t v) {
    ofs.write(reinterpret_cast<const char*>(&v), sizeof(v));
  };
  auto write_u8 = [&](uint8_t v) {
    ofs.write(reinterpret_cast<const char*>(&v), sizeof(v));
  };

  write_u64(static_cast<uint64_t>(dims_));
  write_u64(static_cast<uint64_t>(similarity_));
  write_u64(static_cast<uint64_t>(encoding_));
  write_u8(trained_ ? 1 : 0);
  write_u64(static_cast<uint64_t>(num_threads_));
  ofs.write(reinterpret_cast<const char*>(&centroid_dp_), sizeof(centroid_dp_));

  const uint64_t centroid_size = static_cast<uint64_t>(centroid_.size());
  write_u64(centroid_size);
  ofs.write(reinterpret_cast<const char*>(centroid_.data()), centroid_size * sizeof(float));

  const uint64_t docs_size = static_cast<uint64_t>(corrections_.size());
  write_u64(docs_size);
  for (size_t i = 0; i < corrections_.size(); ++i) {
    const auto& correction = corrections_[i];
    const uint64_t packed_size = static_cast<uint64_t>(doc_packed_len_);
    write_u64(packed_size);
    ofs.write(reinterpret_cast<const char*>(&correction.lower_interval), sizeof(float));
    ofs.write(reinterpret_cast<const char*>(&correction.upper_interval), sizeof(float));
    ofs.write(reinterpret_cast<const char*>(&correction.additional_correction), sizeof(float));
    ofs.write(reinterpret_cast<const char*>(&correction.quantized_component_sum), sizeof(int));
    if (packed_size > 0) {
      ofs.write(
          reinterpret_cast<const char*>(doc_packed_ptr(i)),
          packed_size * sizeof(uint8_t));
    }
  }

  ofs.close();
  return ofs.good();
}

bool OSQIndex::load(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs.is_open()) return false;

  uint64_t magic = 0;
  uint64_t version = 0;
  ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
  ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
  if (magic != 0x4f5351494e445831ULL || version != 1) return false;

  auto read_u64 = [&]() -> uint64_t {
    uint64_t v = 0;
    ifs.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
  };
  auto read_u8 = [&]() -> uint8_t {
    uint8_t v = 0;
    ifs.read(reinterpret_cast<char*>(&v), sizeof(v));
    return v;
  };

  const uint64_t dims = read_u64();
  const uint64_t similarity = read_u64();
  const uint64_t encoding = read_u64();
  const bool trained = read_u8() != 0;
  const uint64_t num_threads = read_u64();
  float centroid_dp = 0.0f;
  ifs.read(reinterpret_cast<char*>(&centroid_dp), sizeof(centroid_dp));

  if (dims != dims_ ||
      similarity != static_cast<uint64_t>(similarity_) ||
      encoding != static_cast<uint64_t>(encoding_)) {
    return false;
  }

  const uint64_t centroid_size = read_u64();
  if (centroid_size != dims_) {
    return false;
  }
  std::vector<float> centroid(centroid_size);
  if (centroid_size > 0) {
    ifs.read(reinterpret_cast<char*>(centroid.data()), centroid_size * sizeof(float));
  }

  const uint64_t docs_size = read_u64();
  const size_t expected_packed_size = doc_packed_len_;
  std::vector<QuantizationResult> corrections(static_cast<size_t>(docs_size));
  std::vector<uint8_t> packed_codes(static_cast<size_t>(docs_size) * expected_packed_size);
  for (uint64_t i = 0; i < docs_size; ++i) {
    const uint64_t packed_size = read_u64();
    if (packed_size != expected_packed_size) {
      return false;
    }
    ifs.read(reinterpret_cast<char*>(&corrections[i].lower_interval), sizeof(float));
    ifs.read(reinterpret_cast<char*>(&corrections[i].upper_interval), sizeof(float));
    ifs.read(reinterpret_cast<char*>(&corrections[i].additional_correction), sizeof(float));
    ifs.read(reinterpret_cast<char*>(&corrections[i].quantized_component_sum), sizeof(int));
    if (packed_size > 0) {
      ifs.read(
          reinterpret_cast<char*>(packed_codes.data() + i * expected_packed_size),
          packed_size * sizeof(uint8_t));
    }
  }

  if (!ifs.good()) {
    return false;
  }

  trained_ = trained;
  num_threads_ = static_cast<int>(num_threads);
  centroid_dp_ = centroid_dp;
  centroid_ = std::move(centroid);
  corrections_ = std::move(corrections);
  packed_codes_ = std::move(packed_codes);
  return true;
}

} // namespace osq
