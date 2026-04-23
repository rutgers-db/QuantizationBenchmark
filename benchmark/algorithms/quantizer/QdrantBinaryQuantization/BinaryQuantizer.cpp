#include "BinaryQuantizer.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <queue>
#include <stdexcept>

#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
#endif

#ifdef _OPENMP
    #include <omp.h>
#endif

namespace bq {

// ============================================================
// Aligned allocation helpers
// ============================================================
namespace {

constexpr size_t kAlign = 64;

uint64_t* aligned_alloc_words(size_t n_words) {
    if (n_words == 0) return nullptr;
    size_t bytes = n_words * sizeof(uint64_t);
    bytes = (bytes + kAlign - 1) & ~(kAlign - 1);
    void* p = nullptr;
#if defined(_MSC_VER)
    p = _aligned_malloc(bytes, kAlign);
    if (!p) throw std::bad_alloc();
#else
    if (posix_memalign(&p, kAlign, bytes) != 0) throw std::bad_alloc();
#endif
    return static_cast<uint64_t*>(p);
}

void aligned_free_words(uint64_t* p) {
    if (!p) return;
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

inline size_t round_up(size_t x, size_t m) { return (x + m - 1) / m * m; }

} // anonymous

// ============================================================
// Runtime kernel selection
// ============================================================
namespace {

enum class KernelChoice { Scalar, AVX2, AVX512VPopcnt };

KernelChoice detect_kernel() {
#if defined(__x86_64__) || defined(_M_X64)
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512vpopcntdq")
        && __builtin_cpu_supports("avx512f")
        && __builtin_cpu_supports("avx512vl"))  // needed for 256-bit popcnt_epi64
    {
        return KernelChoice::AVX512VPopcnt;
    }
    if (__builtin_cpu_supports("avx2")) {
        return KernelChoice::AVX2;
    }
#endif
    return KernelChoice::Scalar;
}

const KernelChoice g_kernel = detect_kernel();

} // anonymous

Kernel active_kernel() {
    switch (g_kernel) {
        case KernelChoice::AVX512VPopcnt: return Kernel::AVX512VPopcnt;
        case KernelChoice::AVX2:          return Kernel::AVX2;
        default:                          return Kernel::Scalar;
    }
}

// ============================================================
// Symmetric kernel: plain XOR + popcount over n_words (multiple of 8).
// Used when query_encoding == SameAsStorage.
// ============================================================
namespace {

inline int32_t xor_popcnt_scalar(const uint64_t* __restrict a,
                                 const uint64_t* __restrict b,
                                 size_t n_words) {
    uint64_t acc = 0;
    for (size_t i = 0; i < n_words; ++i) {
        acc += __builtin_popcountll(a[i] ^ b[i]);
    }
    return static_cast<int32_t>(acc);
}

#if defined(__x86_64__) || defined(_M_X64)

__attribute__((target("avx2")))
inline __m256i avx2_popcnt_epi64(__m256i v) {
    const __m256i lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    const __m256i low_mask = _mm256_set1_epi8(0x0f);
    __m256i lo = _mm256_and_si256(v, low_mask);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
    __m256i byte_pop = _mm256_add_epi8(_mm256_shuffle_epi8(lookup, lo),
                                        _mm256_shuffle_epi8(lookup, hi));
    return _mm256_sad_epu8(byte_pop, _mm256_setzero_si256());
}

__attribute__((target("avx2")))
int32_t xor_popcnt_avx2(const uint64_t* __restrict a,
                        const uint64_t* __restrict b,
                        size_t n_words) {
    __m256i acc = _mm256_setzero_si256();
    const __m256i* pa = reinterpret_cast<const __m256i*>(a);
    const __m256i* pb = reinterpret_cast<const __m256i*>(b);
    size_t n_blocks = n_words / 8;
    for (size_t i = 0; i < n_blocks; ++i) {
        __m256i a0 = _mm256_load_si256(pa + 2*i + 0);
        __m256i a1 = _mm256_load_si256(pa + 2*i + 1);
        __m256i b0 = _mm256_load_si256(pb + 2*i + 0);
        __m256i b1 = _mm256_load_si256(pb + 2*i + 1);
        acc = _mm256_add_epi64(acc, avx2_popcnt_epi64(_mm256_xor_si256(a0, b0)));
        acc = _mm256_add_epi64(acc, avx2_popcnt_epi64(_mm256_xor_si256(a1, b1)));
    }
    alignas(32) uint64_t t[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(t), acc);
    return static_cast<int32_t>(t[0] + t[1] + t[2] + t[3]);
}

__attribute__((target("avx512f,avx512vpopcntdq")))
int32_t xor_popcnt_avx512(const uint64_t* __restrict a,
                          const uint64_t* __restrict b,
                          size_t n_words) {
    __m512i acc = _mm512_setzero_si512();
    const __m512i* pa = reinterpret_cast<const __m512i*>(a);
    const __m512i* pb = reinterpret_cast<const __m512i*>(b);
    size_t n_blocks = n_words / 8;
    for (size_t i = 0; i < n_blocks; ++i) {
        __m512i x = _mm512_xor_si512(_mm512_load_si512(pa + i),
                                     _mm512_load_si512(pb + i));
        acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(x));
    }
    return static_cast<int32_t>(_mm512_reduce_add_epi64(acc));
}

#endif // x86_64

inline int32_t xor_popcnt(const uint64_t* a, const uint64_t* b, size_t n_words) {
#if defined(__x86_64__) || defined(_M_X64)
    switch (g_kernel) {
        case KernelChoice::AVX512VPopcnt: return xor_popcnt_avx512(a, b, n_words);
        case KernelChoice::AVX2:          return xor_popcnt_avx2(a, b, n_words);
        default:                          return xor_popcnt_scalar(a, b, n_words);
    }
#else
    return xor_popcnt_scalar(a, b, n_words);
#endif
}

// ============================================================
// Asymmetric kernel: xor_popcnt_scalar (qdrant-style).
//
//   Given per-DB-word interleaved query codes where q[w*K_q + b] is the
//   b-th bit-plane of the K_q-bit scalar-quantized query for the dims
//   packed into db[w], computes:
//       H = Σ_w Σ_{b=0..K_q-1} popcount(db[w] ⊕ q[w*K_q+b]) << b
//
//   For K_q = 8 on AVX-512: broadcast db[w] into all 8 lanes of a ZMM,
//   XOR with a single 512-bit load of the 8 query bit-plane words, then
//   one _mm512_popcnt_epi64 gives 8 lane-wise popcounts. Accumulate into
//   a per-lane ZMM; at the end extract and do sum_{b} lane[b] << b.
//
//   For K_q = 4 on AVX-512VL: the analogous 256-bit version with
//   _mm256_popcnt_epi64.
// ============================================================

template <int Kq>
inline int64_t xor_popcnt_scalar_scalar(const uint64_t* __restrict db,
                                        const uint64_t* __restrict q,
                                        size_t n_words) {
    // Per-bit-plane accumulators so adds stay 64-bit until the end.
    uint64_t acc[Kq] = {};
    for (size_t w = 0; w < n_words; ++w) {
        uint64_t dbv = db[w];
        const uint64_t* qw = q + w * Kq;
        for (int b = 0; b < Kq; ++b) {
            acc[b] += __builtin_popcountll(dbv ^ qw[b]);
        }
    }
    int64_t score = 0;
    for (int b = 0; b < Kq; ++b) score += static_cast<int64_t>(acc[b]) << b;
    return score;
}

#if defined(__x86_64__) || defined(_M_X64)

__attribute__((target("avx512f,avx512vpopcntdq")))
int64_t xor_popcnt_scalar8_avx512(const uint64_t* __restrict db,
                                  const uint64_t* __restrict q,
                                  size_t n_words) {
    // Two interleaved accumulators for ILP. n_words is a multiple of 8,
    // so always >= 2 unless the code is trivially small; handle odd tail.
    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();
    size_t w = 0;
    for (; w + 2 <= n_words; w += 2) {
        __m512i dbv0 = _mm512_set1_epi64(static_cast<long long>(db[w + 0]));
        __m512i dbv1 = _mm512_set1_epi64(static_cast<long long>(db[w + 1]));
        __m512i qv0 = _mm512_load_si512(
            reinterpret_cast<const __m512i*>(q + (w + 0) * 8));
        __m512i qv1 = _mm512_load_si512(
            reinterpret_cast<const __m512i*>(q + (w + 1) * 8));
        acc0 = _mm512_add_epi64(acc0, _mm512_popcnt_epi64(_mm512_xor_si512(dbv0, qv0)));
        acc1 = _mm512_add_epi64(acc1, _mm512_popcnt_epi64(_mm512_xor_si512(dbv1, qv1)));
    }
    for (; w < n_words; ++w) {
        __m512i dbv = _mm512_set1_epi64(static_cast<long long>(db[w]));
        __m512i qv  = _mm512_load_si512(
            reinterpret_cast<const __m512i*>(q + w * 8));
        acc0 = _mm512_add_epi64(acc0, _mm512_popcnt_epi64(_mm512_xor_si512(dbv, qv)));
    }
    __m512i acc = _mm512_add_epi64(acc0, acc1);
    alignas(64) uint64_t lanes[8];
    _mm512_store_si512(reinterpret_cast<__m512i*>(lanes), acc);
    // Lane b holds the total popcount that should be weighted by 2^b.
    int64_t score = 0;
    for (int b = 0; b < 8; ++b) score += static_cast<int64_t>(lanes[b]) << b;
    return score;
}

__attribute__((target("avx512f,avx512vl,avx512vpopcntdq")))
int64_t xor_popcnt_scalar4_avx512(const uint64_t* __restrict db,
                                  const uint64_t* __restrict q,
                                  size_t n_words) {
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    size_t w = 0;
    for (; w + 2 <= n_words; w += 2) {
        __m256i dbv0 = _mm256_set1_epi64x(static_cast<long long>(db[w + 0]));
        __m256i dbv1 = _mm256_set1_epi64x(static_cast<long long>(db[w + 1]));
        __m256i qv0 = _mm256_load_si256(
            reinterpret_cast<const __m256i*>(q + (w + 0) * 4));
        __m256i qv1 = _mm256_load_si256(
            reinterpret_cast<const __m256i*>(q + (w + 1) * 4));
        acc0 = _mm256_add_epi64(acc0, _mm256_popcnt_epi64(_mm256_xor_si256(dbv0, qv0)));
        acc1 = _mm256_add_epi64(acc1, _mm256_popcnt_epi64(_mm256_xor_si256(dbv1, qv1)));
    }
    for (; w < n_words; ++w) {
        __m256i dbv = _mm256_set1_epi64x(static_cast<long long>(db[w]));
        __m256i qv  = _mm256_load_si256(
            reinterpret_cast<const __m256i*>(q + w * 4));
        acc0 = _mm256_add_epi64(acc0, _mm256_popcnt_epi64(_mm256_xor_si256(dbv, qv)));
    }
    __m256i acc = _mm256_add_epi64(acc0, acc1);
    alignas(32) uint64_t lanes[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), acc);
    int64_t score = 0;
    for (int b = 0; b < 4; ++b) score += static_cast<int64_t>(lanes[b]) << b;
    return score;
}

// AVX-2 fallback: uses the shuffle-LUT popcount. Same broadcast pattern,
// but popcount is implemented via byte-LUT + _mm256_sad_epu8.
__attribute__((target("avx2")))
int64_t xor_popcnt_scalar4_avx2(const uint64_t* __restrict db,
                                const uint64_t* __restrict q,
                                size_t n_words) {
    __m256i acc = _mm256_setzero_si256();
    for (size_t w = 0; w < n_words; ++w) {
        __m256i dbv = _mm256_set1_epi64x(static_cast<long long>(db[w]));
        __m256i qv  = _mm256_load_si256(
            reinterpret_cast<const __m256i*>(q + w * 4));
        acc = _mm256_add_epi64(acc, avx2_popcnt_epi64(_mm256_xor_si256(dbv, qv)));
    }
    alignas(32) uint64_t lanes[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), acc);
    int64_t score = 0;
    for (int b = 0; b < 4; ++b) score += static_cast<int64_t>(lanes[b]) << b;
    return score;
}

// K_q=8 on AVX-2: no native 8-lane popcount; split into low/high halves.
__attribute__((target("avx2")))
int64_t xor_popcnt_scalar8_avx2(const uint64_t* __restrict db,
                                const uint64_t* __restrict q,
                                size_t n_words) {
    __m256i acc_lo = _mm256_setzero_si256();
    __m256i acc_hi = _mm256_setzero_si256();
    for (size_t w = 0; w < n_words; ++w) {
        __m256i dbv = _mm256_set1_epi64x(static_cast<long long>(db[w]));
        __m256i qvl = _mm256_load_si256(
            reinterpret_cast<const __m256i*>(q + w * 8));
        __m256i qvh = _mm256_load_si256(
            reinterpret_cast<const __m256i*>(q + w * 8 + 4));
        acc_lo = _mm256_add_epi64(acc_lo, avx2_popcnt_epi64(_mm256_xor_si256(dbv, qvl)));
        acc_hi = _mm256_add_epi64(acc_hi, avx2_popcnt_epi64(_mm256_xor_si256(dbv, qvh)));
    }
    alignas(32) uint64_t lo[4], hi[4];
    _mm256_store_si256(reinterpret_cast<__m256i*>(lo), acc_lo);
    _mm256_store_si256(reinterpret_cast<__m256i*>(hi), acc_hi);
    int64_t score = 0;
    for (int b = 0; b < 4; ++b) score += static_cast<int64_t>(lo[b]) << b;
    for (int b = 0; b < 4; ++b) score += static_cast<int64_t>(hi[b]) << (b + 4);
    return score;
}

#endif // x86_64

inline int64_t xor_popcnt_scalar_kq(const uint64_t* db, const uint64_t* q,
                                    size_t n_words, int Kq) {
#if defined(__x86_64__) || defined(_M_X64)
    if (g_kernel == KernelChoice::AVX512VPopcnt) {
        if (Kq == 8) return xor_popcnt_scalar8_avx512(db, q, n_words);
        if (Kq == 4) return xor_popcnt_scalar4_avx512(db, q, n_words);
    } else if (g_kernel == KernelChoice::AVX2) {
        if (Kq == 8) return xor_popcnt_scalar8_avx2(db, q, n_words);
        if (Kq == 4) return xor_popcnt_scalar4_avx2(db, q, n_words);
    }
#endif
    if (Kq == 8) return xor_popcnt_scalar_scalar<8>(db, q, n_words);
    if (Kq == 4) return xor_popcnt_scalar_scalar<4>(db, q, n_words);
    // Shouldn't happen; we only support K_q in {4, 8}.
    return 0;
}

} // anonymous

// ============================================================
// BinaryQuantizer
// ============================================================

int BinaryQuantizer::k_q() const {
    switch (query_encoding) {
        case QueryEncoding::SameAsStorage: return 1;
        case QueryEncoding::Scalar4Bits:   return 4;
        case QueryEncoding::Scalar8Bits:   return 8;
    }
    return 1;
}

size_t BinaryQuantizer::query_code_words() const {
    return (query_encoding == QueryEncoding::SameAsStorage)
               ? n_words
               : n_words * static_cast<size_t>(k_q());
}

BinaryQuantizer::BinaryQuantizer(size_t d_,
                                 Encoding encoding_,
                                 QueryEncoding query_encoding_,
                                 Metric metric_)
    : d(d_),
      encoding(encoding_),
      query_encoding(query_encoding_),
      metric(metric_) {
    if (d == 0) throw std::invalid_argument("BinaryQuantizer: d must be > 0");
    const size_t n_bits = static_cast<size_t>(k_db()) * d;
    size_t raw_words = (n_bits + 63) / 64;
    n_words   = round_up(raw_words, 8);      // 8 words = 64 B
    code_size = n_words * sizeof(uint64_t);
    thresholds.assign(static_cast<size_t>(k_db()) * d, 0.0f);
    center.assign(d, 0.0f);
}

BinaryQuantizer::~BinaryQuantizer() { aligned_free_words(codes); }

void BinaryQuantizer::reset() {
    aligned_free_words(codes);
    codes = nullptr;
    codes_capacity = 0;
    ntotal = 0;
}

void BinaryQuantizer::reserve(size_t n_vectors) {
    if (n_vectors <= codes_capacity) return;
    size_t new_cap = std::max<size_t>(codes_capacity * 2, n_vectors);
    new_cap = std::max<size_t>(new_cap, 64);
    uint64_t* new_codes = aligned_alloc_words(new_cap * n_words);
    if (codes) {
        std::memcpy(new_codes, codes, ntotal * n_words * sizeof(uint64_t));
        aligned_free_words(codes);
    }
    codes = new_codes;
    codes_capacity = new_cap;
}

void BinaryQuantizer::train(size_t n, const float* x) {
    const int K = k_db();
    thresholds.assign(static_cast<size_t>(K) * d, 0.0f);
    center.assign(d, 0.0f);
    if (n == 0) {
        is_trained = true;
        return;
    }

    // Per-dim mean and stddev in one pass using Welford-ish two-pass (simpler).
    std::vector<double> mean(d, 0.0), m2(d, 0.0);
    for (size_t i = 0; i < n; ++i) {
        const float* v = x + i * d;
        for (size_t j = 0; j < d; ++j) mean[j] += v[j];
    }
    const double inv_n = 1.0 / static_cast<double>(n);
    for (size_t j = 0; j < d; ++j) mean[j] *= inv_n;
    for (size_t i = 0; i < n; ++i) {
        const float* v = x + i * d;
        for (size_t j = 0; j < d; ++j) {
            double t = v[j] - mean[j];
            m2[j] += t * t;
        }
    }

    // Thermometer thresholds. For OneBit: { mean }. For TwoBits: { mean ± (2/3)σ }.
    //   (Qdrant uses SIGMAS = 2/3 in encoded_vectors_binary.rs:L647.)
    static constexpr double SIGMAS_TWO_BITS = 2.0 / 3.0;
    for (size_t j = 0; j < d; ++j) {
        const double mu = mean[j];
        const double sd = std::sqrt(m2[j] * inv_n);
        center[j] = static_cast<float>(mu);
        if (K == 1) {
            thresholds[0 * d + j] = static_cast<float>(mu);
        } else { // K == 2
            thresholds[0 * d + j] = static_cast<float>(mu - SIGMAS_TWO_BITS * sd);
            thresholds[1 * d + j] = static_cast<float>(mu + SIGMAS_TWO_BITS * sd);
        }
    }
    is_trained = true;
}

void BinaryQuantizer::encode_db(const float* x, uint64_t* code) const {
    std::memset(code, 0, code_size);
    const int K = k_db();
    // Virtual bit position of plane p, dim i is p*d + i.
    // Because the planes are laid out one after another, we can split by plane
    // and encode each plane's bit positions contiguously — this makes the
    // inner loop branchless and cache-linear.
    for (int p = 0; p < K; ++p) {
        const float* thr = thresholds.data() + static_cast<size_t>(p) * d;
        const size_t base_bit = static_cast<size_t>(p) * d;
        for (size_t i = 0; i < d; ++i) {
            const uint64_t bit = static_cast<uint64_t>(x[i] > thr[i]);
            const size_t v = base_bit + i;
            code[v >> 6] |= (bit << (v & 63));
        }
    }
}

void BinaryQuantizer::encode_query(const float* x, uint64_t* code) const {
    if (query_encoding == QueryEncoding::SameAsStorage) {
        encode_db(x, code);
        return;
    }

    const int Kq = k_q();
    const int Kdb = k_db();
    const size_t n_bits = static_cast<size_t>(Kdb) * d;

    // Zero the whole query code (includes padding words past n_bits).
    std::memset(code, 0, query_code_words() * sizeof(uint64_t));

    // Per-query scalar-quantization range (qdrant-style max|.|, but taken
    // on the *centered* query so the quantized bit that represents "0" of
    // the query aligns with the DB's threshold split).
    float max_abs = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        float a = std::fabs(x[i] - center[i]);
        if (a > max_abs) max_abs = a;
    }
    const float vmin = -max_abs;
    const int ranges = (1 << Kq) - 1;                // 15 or 255
    const float delta = (max_abs <= 0.0f)
                            ? 0.0f
                            : (2.0f * max_abs) / static_cast<float>(ranges);

    // Emit bits for every virtual dim v in [0, n_bits).
    // Virtual dim v maps to original dim (v mod d); the same query value
    // is replicated across K_db planes (qdrant's extend_from_slice trick).
    // Per-word interleaved layout: the K_q output words for DB word w are
    // at code[w*K_q .. (w+1)*K_q).
    for (size_t v = 0; v < n_bits; ++v) {
        const size_t i = v % d;                       // original dim
        const float val = x[i] - center[i];           // centered query value
        // Uniform K_q-bit scalar quant onto [-max_abs, +max_abs].
        int q_int;
        if (delta <= 0.0f) {
            q_int = 0;
        } else {
            float shifted = val - vmin;               // in [0, 2*max_abs]
            int r = static_cast<int>(std::lrintf(shifted / delta));
            if (r < 0) r = 0;
            if (r > ranges) r = ranges;
            q_int = r;
        }
        const size_t w  = v >> 6;                     // DB-word index
        const size_t sh = v & 63;                     // bit position within word
        uint64_t* out_word = code + w * static_cast<size_t>(Kq);
        // Scatter the K_q bits of q_int into K_q output words.
        for (int b = 0; b < Kq; ++b) {
            const uint64_t bit = static_cast<uint64_t>((q_int >> b) & 1);
            out_word[b] |= (bit << sh);
        }
    }
}

void BinaryQuantizer::add(size_t n, const float* x) {
    if (!is_trained) throw std::runtime_error("BinaryQuantizer::add called before train");
    if (n == 0) return;
    reserve(ntotal + n);
#ifdef _OPENMP
    const int _nt_add = (num_threads > 0) ? num_threads : omp_get_max_threads();
    #pragma omp parallel for schedule(static) num_threads(_nt_add) if (n > 256)
#endif
    for (long long i = 0; i < static_cast<long long>(n); ++i) {
        encode_db(x + static_cast<size_t>(i) * d,
                  codes + (ntotal + static_cast<size_t>(i)) * n_words);
    }
    ntotal += n;
}

float BinaryQuantizer::h_to_metric(int64_t h) const {
    const int Kq = k_q();
    const float h_scale = (Kq == 1)
        ? static_cast<float>(h)
        : static_cast<float>(h) / static_cast<float>((1 << Kq) - 1);
    const int64_t n_eff = static_cast<int64_t>(k_db()) * static_cast<int64_t>(d);
    switch (metric) {
        case Metric::Hamming: return h_scale;
        case Metric::L2:      return 4.0f * h_scale;
        case Metric::IP:      return static_cast<float>(n_eff) - 2.0f * h_scale;
    }
    return h_scale;
}

namespace {
// Max-heap keyed by raw weighted Hamming — smallest H = closest for every metric.
struct Cand {
    int64_t h;
    int64_t id;
    bool operator<(const Cand& o) const { return h < o.h; }
};
} // anonymous

void BinaryQuantizer::search(size_t nq,
                             const float* x,
                             size_t k,
                             float* distances,
                             int64_t* labels) const {
    if (k == 0) return;
    if (!is_trained) throw std::runtime_error("BinaryQuantizer::search called before train");

    const size_t qwords = query_code_words();
    uint64_t* qcodes = aligned_alloc_words(nq * qwords);
    struct G { uint64_t* p; ~G() { aligned_free_words(p); } } guard{qcodes};

#ifdef _OPENMP
    const int _nt_search = (num_threads > 0) ? num_threads : omp_get_max_threads();
    #pragma omp parallel for schedule(static) num_threads(_nt_search) if (nq > 8)
#endif
    for (long long i = 0; i < static_cast<long long>(nq); ++i) {
        encode_query(x + static_cast<size_t>(i) * d,
                     qcodes + static_cast<size_t>(i) * qwords);
    }

    const size_t eff_k = std::min<size_t>(k, ntotal);
    const float pad_dist = (metric == Metric::IP)
                               ? -std::numeric_limits<float>::infinity()
                               : std::numeric_limits<float>::infinity();
    const bool symmetric = (query_encoding == QueryEncoding::SameAsStorage);
    const int Kq = k_q();

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) num_threads(_nt_search) if (nq > 1)
#endif
    for (long long qi = 0; qi < static_cast<long long>(nq); ++qi) {
        const uint64_t* q = qcodes + static_cast<size_t>(qi) * qwords;
        std::priority_queue<Cand> heap;

        for (size_t di = 0; di < ntotal; ++di) {
            int64_t h;
            if (symmetric) {
                h = static_cast<int64_t>(xor_popcnt(codes + di * n_words, q, n_words));
            } else {
                h = xor_popcnt_scalar_kq(codes + di * n_words, q, n_words, Kq);
            }
            if (heap.size() < eff_k) {
                heap.push({h, static_cast<int64_t>(di)});
            } else if (h < heap.top().h) {
                heap.pop();
                heap.push({h, static_cast<int64_t>(di)});
            }
        }

        const size_t out_base = static_cast<size_t>(qi) * k;
        const size_t count = heap.size();
        for (size_t j = 0; j < count; ++j) {
            const Cand& c = heap.top();
            const size_t idx = out_base + (count - 1 - j);
            labels[idx]    = c.id;
            distances[idx] = h_to_metric(c.h);
            heap.pop();
        }
        for (size_t j = count; j < k; ++j) {
            labels[out_base + j]    = -1;
            distances[out_base + j] = pad_dist;
        }
    }
}

} // namespace bq
