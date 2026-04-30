#include "BinaryQuantizer.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

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
constexpr size_t kBlock = 8;   // B: vectors per lane-interleaved storage block

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
        && __builtin_cpu_supports("avx512vl")) {
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
// AVX2 popcount helper (64-bit-lane popcount via byte LUT + psadbw).
// ============================================================
#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx2")))
static inline __m256i avx2_popcnt_epi64(__m256i v) {
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
#endif

// ============================================================
// Block-8 symmetric kernel.
// Inputs:
//   block_codes : lane-interleaved block, n_words * kBlock uint64s
//   q           : query code, n_words uint64s
// Output: out[i] = popcount over the XOR of query vs vector i in the block.
// ============================================================
namespace {

static inline void xor_popcnt_block8_sym_scalar(const uint64_t* __restrict block_codes,
                                                const uint64_t* __restrict q,
                                                size_t n_words,
                                                int64_t out[kBlock]) {
    int64_t acc[kBlock] = {};
    for (size_t w = 0; w < n_words; ++w) {
        const uint64_t qw = q[w];
        const uint64_t* bw = block_codes + w * kBlock;
        for (size_t i = 0; i < kBlock; ++i) {
            acc[i] += __builtin_popcountll(bw[i] ^ qw);
        }
    }
    for (size_t i = 0; i < kBlock; ++i) out[i] = acc[i];
}

#if defined(__x86_64__) || defined(_M_X64)

__attribute__((target("avx512f,avx512vpopcntdq")))
static void xor_popcnt_block8_sym_avx512(const uint64_t* __restrict block_codes,
                                         const uint64_t* __restrict q,
                                         size_t n_words,
                                         int64_t out[kBlock]) {
    __m512i acc = _mm512_setzero_si512();
    for (size_t w = 0; w < n_words; ++w) {
        __m512i dbv = _mm512_load_si512(
            reinterpret_cast<const __m512i*>(block_codes + w * kBlock));
        __m512i qv  = _mm512_set1_epi64(static_cast<long long>(q[w]));
        acc = _mm512_add_epi64(
            acc, _mm512_popcnt_epi64(_mm512_xor_si512(dbv, qv)));
    }
    _mm512_store_si512(reinterpret_cast<__m512i*>(out), acc);
}

__attribute__((target("avx2")))
static void xor_popcnt_block8_sym_avx2(const uint64_t* __restrict block_codes,
                                       const uint64_t* __restrict q,
                                       size_t n_words,
                                       int64_t out[kBlock]) {
    __m256i acc_lo = _mm256_setzero_si256();
    __m256i acc_hi = _mm256_setzero_si256();
    for (size_t w = 0; w < n_words; ++w) {
        __m256i qv = _mm256_set1_epi64x(static_cast<long long>(q[w]));
        __m256i dblo = _mm256_load_si256(
            reinterpret_cast<const __m256i*>(block_codes + w * kBlock));
        __m256i dbhi = _mm256_load_si256(
            reinterpret_cast<const __m256i*>(block_codes + w * kBlock + 4));
        acc_lo = _mm256_add_epi64(
            acc_lo, avx2_popcnt_epi64(_mm256_xor_si256(dblo, qv)));
        acc_hi = _mm256_add_epi64(
            acc_hi, avx2_popcnt_epi64(_mm256_xor_si256(dbhi, qv)));
    }
    _mm256_store_si256(reinterpret_cast<__m256i*>(out), acc_lo);
    _mm256_store_si256(reinterpret_cast<__m256i*>(out + 4), acc_hi);
}

#endif // x86_64

inline void xor_popcnt_block8_sym(const uint64_t* block_codes,
                                  const uint64_t* q,
                                  size_t n_words,
                                  int64_t out[kBlock]) {
#if defined(__x86_64__) || defined(_M_X64)
    switch (g_kernel) {
        case KernelChoice::AVX512VPopcnt:
            xor_popcnt_block8_sym_avx512(block_codes, q, n_words, out); return;
        case KernelChoice::AVX2:
            xor_popcnt_block8_sym_avx2(block_codes, q, n_words, out); return;
        default: break;
    }
#endif
    xor_popcnt_block8_sym_scalar(block_codes, q, n_words, out);
}

// ============================================================
// Block-8 asymmetric kernel (K_q in {4, 8}).
// Weighted Hamming per lane:
//   H_i = Σ_w Σ_{b=0..Kq-1} popcount(block_codes[w][i] ⊕ q[w*Kq + b]) << b
// Query layout (unchanged from qdrant-style): for DB word w, the Kq bit-plane
// words live contiguously at q[w*Kq .. (w+1)*Kq).
// ============================================================

template <int Kq>
static inline void xor_popcnt_block8_asym_scalar(const uint64_t* __restrict block_codes,
                                                 const uint64_t* __restrict q,
                                                 size_t n_words,
                                                 int64_t out[kBlock]) {
    int64_t acc[Kq][kBlock] = {};
    for (size_t w = 0; w < n_words; ++w) {
        const uint64_t* bw = block_codes + w * kBlock;
        const uint64_t* qw = q + w * Kq;
        for (int b = 0; b < Kq; ++b) {
            const uint64_t qv = qw[b];
            for (size_t i = 0; i < kBlock; ++i) {
                acc[b][i] += __builtin_popcountll(bw[i] ^ qv);
            }
        }
    }
    for (size_t i = 0; i < kBlock; ++i) {
        int64_t s = 0;
        for (int b = 0; b < Kq; ++b) s += acc[b][i] << b;
        out[i] = s;
    }
}

#if defined(__x86_64__) || defined(_M_X64)

__attribute__((target("avx512f,avx512vpopcntdq")))
static void xor_popcnt_block8_asym8_avx512(const uint64_t* __restrict block_codes,
                                           const uint64_t* __restrict q,
                                           size_t n_words,
                                           int64_t out[kBlock]) {
    // 8 per-plane accumulators; per-lane popcount stays in int64 until the
    // final shift+reduce. Register pressure: 8 ZMM accs + 1 db + 1 qv scratch.
    __m512i a0 = _mm512_setzero_si512();
    __m512i a1 = _mm512_setzero_si512();
    __m512i a2 = _mm512_setzero_si512();
    __m512i a3 = _mm512_setzero_si512();
    __m512i a4 = _mm512_setzero_si512();
    __m512i a5 = _mm512_setzero_si512();
    __m512i a6 = _mm512_setzero_si512();
    __m512i a7 = _mm512_setzero_si512();
    for (size_t w = 0; w < n_words; ++w) {
        __m512i dbv = _mm512_load_si512(
            reinterpret_cast<const __m512i*>(block_codes + w * kBlock));
        const uint64_t* qw = q + w * 8;
        a0 = _mm512_add_epi64(a0, _mm512_popcnt_epi64(
            _mm512_xor_si512(dbv, _mm512_set1_epi64((long long)qw[0]))));
        a1 = _mm512_add_epi64(a1, _mm512_popcnt_epi64(
            _mm512_xor_si512(dbv, _mm512_set1_epi64((long long)qw[1]))));
        a2 = _mm512_add_epi64(a2, _mm512_popcnt_epi64(
            _mm512_xor_si512(dbv, _mm512_set1_epi64((long long)qw[2]))));
        a3 = _mm512_add_epi64(a3, _mm512_popcnt_epi64(
            _mm512_xor_si512(dbv, _mm512_set1_epi64((long long)qw[3]))));
        a4 = _mm512_add_epi64(a4, _mm512_popcnt_epi64(
            _mm512_xor_si512(dbv, _mm512_set1_epi64((long long)qw[4]))));
        a5 = _mm512_add_epi64(a5, _mm512_popcnt_epi64(
            _mm512_xor_si512(dbv, _mm512_set1_epi64((long long)qw[5]))));
        a6 = _mm512_add_epi64(a6, _mm512_popcnt_epi64(
            _mm512_xor_si512(dbv, _mm512_set1_epi64((long long)qw[6]))));
        a7 = _mm512_add_epi64(a7, _mm512_popcnt_epi64(
            _mm512_xor_si512(dbv, _mm512_set1_epi64((long long)qw[7]))));
    }
    __m512i s = a0;
    s = _mm512_add_epi64(s, _mm512_slli_epi64(a1, 1));
    s = _mm512_add_epi64(s, _mm512_slli_epi64(a2, 2));
    s = _mm512_add_epi64(s, _mm512_slli_epi64(a3, 3));
    s = _mm512_add_epi64(s, _mm512_slli_epi64(a4, 4));
    s = _mm512_add_epi64(s, _mm512_slli_epi64(a5, 5));
    s = _mm512_add_epi64(s, _mm512_slli_epi64(a6, 6));
    s = _mm512_add_epi64(s, _mm512_slli_epi64(a7, 7));
    _mm512_store_si512(reinterpret_cast<__m512i*>(out), s);
}

__attribute__((target("avx512f,avx512vpopcntdq")))
static void xor_popcnt_block8_asym4_avx512(const uint64_t* __restrict block_codes,
                                           const uint64_t* __restrict q,
                                           size_t n_words,
                                           int64_t out[kBlock]) {
    __m512i a0 = _mm512_setzero_si512();
    __m512i a1 = _mm512_setzero_si512();
    __m512i a2 = _mm512_setzero_si512();
    __m512i a3 = _mm512_setzero_si512();
    for (size_t w = 0; w < n_words; ++w) {
        __m512i dbv = _mm512_load_si512(
            reinterpret_cast<const __m512i*>(block_codes + w * kBlock));
        const uint64_t* qw = q + w * 4;
        a0 = _mm512_add_epi64(a0, _mm512_popcnt_epi64(
            _mm512_xor_si512(dbv, _mm512_set1_epi64((long long)qw[0]))));
        a1 = _mm512_add_epi64(a1, _mm512_popcnt_epi64(
            _mm512_xor_si512(dbv, _mm512_set1_epi64((long long)qw[1]))));
        a2 = _mm512_add_epi64(a2, _mm512_popcnt_epi64(
            _mm512_xor_si512(dbv, _mm512_set1_epi64((long long)qw[2]))));
        a3 = _mm512_add_epi64(a3, _mm512_popcnt_epi64(
            _mm512_xor_si512(dbv, _mm512_set1_epi64((long long)qw[3]))));
    }
    __m512i s = a0;
    s = _mm512_add_epi64(s, _mm512_slli_epi64(a1, 1));
    s = _mm512_add_epi64(s, _mm512_slli_epi64(a2, 2));
    s = _mm512_add_epi64(s, _mm512_slli_epi64(a3, 3));
    _mm512_store_si512(reinterpret_cast<__m512i*>(out), s);
}

#endif // x86_64

inline void xor_popcnt_block8_asym(const uint64_t* block_codes,
                                   const uint64_t* q,
                                   size_t n_words,
                                   int Kq,
                                   int64_t out[kBlock]) {
#if defined(__x86_64__) || defined(_M_X64)
    if (g_kernel == KernelChoice::AVX512VPopcnt) {
        if (Kq == 8) { xor_popcnt_block8_asym8_avx512(block_codes, q, n_words, out); return; }
        if (Kq == 4) { xor_popcnt_block8_asym4_avx512(block_codes, q, n_words, out); return; }
    }
#endif
    if (Kq == 8) { xor_popcnt_block8_asym_scalar<8>(block_codes, q, n_words, out); return; }
    if (Kq == 4) { xor_popcnt_block8_asym_scalar<4>(block_codes, q, n_words, out); return; }
}

// ============================================================
// Single-vector scoring (for score_one). Scalar is fine here — not hot.
// ============================================================

inline int64_t xor_popcnt_one_sym(const uint64_t* a, const uint64_t* b,
                                  size_t n_words) {
    uint64_t acc = 0;
    for (size_t i = 0; i < n_words; ++i) {
        acc += __builtin_popcountll(a[i] ^ b[i]);
    }
    return static_cast<int64_t>(acc);
}

inline int64_t xor_popcnt_one_asym(const uint64_t* db, const uint64_t* q,
                                   size_t n_words, int Kq) {
    int64_t score = 0;
    for (int b = 0; b < Kq; ++b) {
        uint64_t acc = 0;
        for (size_t w = 0; w < n_words; ++w) {
            acc += __builtin_popcountll(db[w] ^ q[w * Kq + b]);
        }
        score += static_cast<int64_t>(acc) << b;
    }
    return score;
}

// Scatter / gather between per-vector code and lane-interleaved block storage.
inline void scatter_to_block(uint64_t* codes, size_t n_words, size_t id,
                             const uint64_t* src) {
    const size_t block = id / kBlock;
    const size_t lane  = id % kBlock;
    uint64_t* base = codes + block * n_words * kBlock;
    for (size_t w = 0; w < n_words; ++w) {
        base[w * kBlock + lane] = src[w];
    }
}

inline void gather_from_block(const uint64_t* codes, size_t n_words, size_t id,
                              uint64_t* dst) {
    const size_t block = id / kBlock;
    const size_t lane  = id % kBlock;
    const uint64_t* base = codes + block * n_words * kBlock;
    for (size_t w = 0; w < n_words; ++w) {
        dst[w] = base[w * kBlock + lane];
    }
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
    n_words   = (n_bits + 63) / 64;              // real words, no pad-up
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
    new_cap = std::max<size_t>(new_cap, kBlock * 8);                    // 64-vector floor
    new_cap = ((new_cap + kBlock - 1) / kBlock) * kBlock;               // multiple of B
    const size_t new_blocks = new_cap / kBlock;
    const size_t new_words_total = new_blocks * n_words * kBlock;
    uint64_t* new_codes = aligned_alloc_words(new_words_total);
    // Zero whole allocation: (a) tail-block unused lanes must be zero so the
    // SIMD kernel produces deterministic "large" distances that we still mask
    // out at the top-k layer, (b) future partial writes preserve sibling lanes.
    std::memset(new_codes, 0, new_words_total * sizeof(uint64_t));
    if (codes) {
        const size_t blocks_in_use = (ntotal + kBlock - 1) / kBlock;
        std::memcpy(new_codes, codes,
                    blocks_in_use * n_words * kBlock * sizeof(uint64_t));
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
    std::memset(code, 0, n_words * sizeof(uint64_t));
    const int K = k_db();
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

    std::memset(code, 0, query_code_words() * sizeof(uint64_t));

    float max_abs = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        float a = std::fabs(x[i] - center[i]);
        if (a > max_abs) max_abs = a;
    }
    const float vmin = -max_abs;
    const int ranges = (1 << Kq) - 1;
    const float delta = (max_abs <= 0.0f)
                            ? 0.0f
                            : (2.0f * max_abs) / static_cast<float>(ranges);

    for (size_t v = 0; v < n_bits; ++v) {
        const size_t i = v % d;
        const float val = x[i] - center[i];
        int q_int;
        if (delta <= 0.0f) {
            q_int = 0;
        } else {
            float shifted = val - vmin;
            int r = static_cast<int>(std::lrintf(shifted / delta));
            if (r < 0) r = 0;
            if (r > ranges) r = ranges;
            q_int = r;
        }
        const size_t w  = v >> 6;
        const size_t sh = v & 63;
        uint64_t* out_word = code + w * static_cast<size_t>(Kq);
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
    const size_t base_id = ntotal;
#ifdef _OPENMP
    const int _nt_add = (num_threads > 0) ? num_threads : omp_get_max_threads();
    #pragma omp parallel num_threads(_nt_add) if (n > 256)
#endif
    {
        std::vector<uint64_t> tmp(n_words);
#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            encode_db(x + static_cast<size_t>(i) * d, tmp.data());
            scatter_to_block(codes, n_words,
                             base_id + static_cast<size_t>(i), tmp.data());
        }
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

float BinaryQuantizer::score_one(const uint64_t* qcode, size_t db_id) const {
    if (!is_trained) throw std::runtime_error("BinaryQuantizer::score_one called before train");
    if (db_id >= ntotal) throw std::out_of_range("BinaryQuantizer::score_one: db_id out of range");
    std::vector<uint64_t> tmp(n_words);
    gather_from_block(codes, n_words, db_id, tmp.data());
    int64_t h;
    if (query_encoding == QueryEncoding::SameAsStorage) {
        h = xor_popcnt_one_sym(tmp.data(), qcode, n_words);
    } else {
        h = xor_popcnt_one_asym(tmp.data(), qcode, n_words, k_q());
    }
    return h_to_metric(h);
}

void BinaryQuantizer::score_range(const uint64_t* qcode,
                                  size_t begin, size_t end,
                                  float* out) const {
    if (!is_trained) throw std::runtime_error("BinaryQuantizer::score_range called before train");
    if (begin > end || end > ntotal)
        throw std::out_of_range("BinaryQuantizer::score_range: bad [begin, end)");
    if (begin == end) return;

    const bool symmetric = (query_encoding == QueryEncoding::SameAsStorage);
    const int Kq = k_q();
    const size_t first_block = begin / kBlock;
    const size_t last_block  = (end - 1) / kBlock;   // inclusive
    alignas(64) int64_t block_out[kBlock];

    for (size_t bi = first_block; bi <= last_block; ++bi) {
        const uint64_t* block_codes_ = codes + bi * n_words * kBlock;
        if (symmetric) {
            xor_popcnt_block8_sym(block_codes_, qcode, n_words, block_out);
        } else {
            xor_popcnt_block8_asym(block_codes_, qcode, n_words, Kq, block_out);
        }
        const size_t id_base = bi * kBlock;
        const size_t lane_lo = (begin > id_base) ? (begin - id_base) : 0;
        const size_t lane_hi = (end   < id_base + kBlock) ? (end - id_base) : kBlock;
        for (size_t li = lane_lo; li < lane_hi; ++li) {
            out[(id_base + li) - begin] = h_to_metric(block_out[li]);
        }
    }
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
    uint64_t* qcodes = aligned_alloc_words(std::max<size_t>(nq * qwords, 1));
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
    const size_t tail = ntotal % kBlock;
    const size_t total_blocks = (ntotal + kBlock - 1) / kBlock;

#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1) num_threads(_nt_search) if (nq > 1)
#endif
    for (long long qi = 0; qi < static_cast<long long>(nq); ++qi) {
        const uint64_t* q = qcodes + static_cast<size_t>(qi) * qwords;
        std::priority_queue<Cand> heap;
        alignas(64) int64_t block_out[kBlock];

        for (size_t bi = 0; bi < total_blocks; ++bi) {
            const uint64_t* block_codes_ = codes + bi * n_words * kBlock;
            if (symmetric) {
                xor_popcnt_block8_sym(block_codes_, q, n_words, block_out);
            } else {
                xor_popcnt_block8_asym(block_codes_, q, n_words, Kq, block_out);
            }
            // Last block may be partially populated; pad lanes are zero-filled
            // code vs arbitrary query → spurious small/large H, exclude them.
            const size_t lanes_valid =
                (bi + 1 < total_blocks || tail == 0) ? kBlock : tail;
            const size_t id_base = bi * kBlock;
            for (size_t li = 0; li < lanes_valid; ++li) {
                const int64_t h = block_out[li];
                const int64_t id = static_cast<int64_t>(id_base + li);
                if (heap.size() < eff_k) {
                    heap.push({h, id});
                } else if (h < heap.top().h) {
                    heap.pop();
                    heap.push({h, id});
                }
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
