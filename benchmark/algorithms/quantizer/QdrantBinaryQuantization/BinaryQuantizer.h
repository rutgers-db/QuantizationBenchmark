// Binary Quantization index — FAISS-like interface, reimplementation of
// Qdrant's BQ (lib/quantization/src/encoded_vectors_binary.rs).
//
// DB encoding (thermometer / unary code):
//   K_db thresholds per input dim, sorted ascending. Bit p of dim i is set
//   iff x[i] > thresholds[p, i]. That makes the per-dim code monotonic,
//   so Hamming distance between two per-dim codes counts how many
//   thresholds fall between the underlying real values — a good proxy
//   for |delta|. Qdrant's TwoBits scheme is exactly this with two
//   thresholds at mean ± (2/3)·stddev. We support:
//     OneBit  : { mean }
//     TwoBits : { mean - (2/3)·sd, mean + (2/3)·sd }   [matches qdrant]
//
//   Bit layout in the packed code: plane p occupies virtual bit positions
//   [p*d, (p+1)*d). Every code is zero-padded up to a multiple of 64 B
//   so the whole code is a sequence of aligned AVX-512 blocks.
//
// Query encoding (per-word interleaved, exactly as qdrant):
//   - SameAsStorage : encode the query with the same thermometer as db.
//                     Scoring = one XOR + popcount over the whole code.
//   - Scalar4Bits / Scalar8Bits : uniform K_q-bit scalar quantization of
//                     the raw query (range ± max|q|), then bit-transpose
//                     PER DB WORD. For each DB word w in [0, n_words), the
//                     K_q output words at positions [w*K_q, (w+1)*K_q)
//                     are the K_q bit-planes of the scalar-quantized
//                     values whose DB bits sit in word w.
//                     Scoring (qdrant's xor_popcnt_scalar):
//                       H = Σ_w Σ_b popcount(db[w] ⊕ q[w*K_q+b]) << b
//                     With K_q=8 this layout is exactly one ZMM per DB
//                     word (broadcast db → 8 lane-wise XORs → one
//                     _mm512_popcnt_epi64 → lane-weighted reduce at end).
//
// Metrics:
//   Raw score from the kernel is a weighted Hamming H. For ranking,
//   smallest H is closest for all three metrics (L2, IP, Hamming). The
//   per-metric final scalar is:
//     - Hamming : H_scale
//     - L2      : 4 * H_scale                 (≈ ||q−db||² on ±1 view)
//     - IP      : (K_db * d) - 2 * H_scale    (≈ ⟨q,db⟩ on ±1 view)
//   where H_scale = H / (2^K_q − 1) when K_q > 1, else H. Rankings are
//   preserved; only the numeric scale differs.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace bq {

enum class Encoding : uint8_t {
    OneBit   = 1,
    TwoBits  = 2,
};

enum class QueryEncoding : uint8_t {
    SameAsStorage,   // query encoded with the same thermometer as db
    Scalar4Bits,     // 4-bit asymmetric scalar query (qdrant-style)
    Scalar8Bits,     // 8-bit asymmetric scalar query (qdrant-style)
};

enum class Metric {
    L2, IP, Hamming,
};

class BinaryQuantizer {
public:
    // --- configuration ---
    size_t d;
    Encoding encoding;
    QueryEncoding query_encoding;
    Metric metric;

    // Per-dim thresholds for the thermometer DB encoding.
    // Shape (K_db, d), thresholds[p*d + i] = threshold p for dim i,
    // ascending in p for every i.
    std::vector<float> thresholds;

    // Per-dim reference value used to center the query before scalar
    // quantization (asymmetric modes). Always the midpoint of the DB
    // thresholds, which equals the per-dim mean for OneBit and TwoBits
    // (since TwoBits thresholds are symmetric around mean). Centering
    // matters whenever the data is not zero-centered (e.g. SIFT uint8
    // vectors) — without it the scalar query's ± max|q| range and the
    // DB's threshold split live in different spaces and the XOR+popcount
    // scoring becomes meaningless.
    std::vector<float> center;

    // --- code layout ---
    // n_bits  = K_db * d
    // n_words = ceil(n_bits / 64), rounded up to a multiple of 8 (= 64 B).
    size_t n_words;
    size_t code_size;   // n_words * 8

    // --- db state ---
    bool   is_trained = false;
    size_t ntotal = 0;
    size_t codes_capacity = 0;
    uint64_t* codes = nullptr;   // 64-B aligned; ntotal * n_words uint64s in use

    // Number of OpenMP threads used by add() / search(). 0 or negative
    // means "OpenMP default" (OMP_NUM_THREADS). Settable via set_num_threads.
    int num_threads = 0;

    BinaryQuantizer(size_t d,
                    Encoding encoding = Encoding::OneBit,
                    QueryEncoding query_encoding = QueryEncoding::SameAsStorage,
                    Metric metric = Metric::L2);
    ~BinaryQuantizer();

    void set_num_threads(int n) { num_threads = n; }

    BinaryQuantizer(const BinaryQuantizer&) = delete;
    BinaryQuantizer& operator=(const BinaryQuantizer&) = delete;

    // Learn per-dim mean + stddev, then pick thresholds per the encoding.
    // n == 0 degenerates to thresholds == 0 (pure sign quantization for
    // OneBit; degenerate but consistent for TwoBits).
    void train(size_t n, const float* x);

    // Encode n vectors of size d and append them to the index.
    void add(size_t n, const float* x);

    // Top-k search. `distances` / `labels` each have length nq*k.
    // L2 / Hamming : smaller = closer.  IP : larger = closer.
    // Trailing slots when ntotal < k get id = -1 and a sentinel distance.
    void search(size_t nq, const float* x, size_t k,
                float* distances, int64_t* labels) const;

    // Encode helpers. Caller-provided buffers must have the right size:
    //   encode_db    : n_words uint64 words
    //   encode_query : query_code_words() uint64 words, 64-B aligned
    void encode_db(const float* x, uint64_t* code) const;
    void encode_query(const float* x, uint64_t* code) const;

    // Size of one encoded query, in uint64 words (depends on query_encoding).
    size_t query_code_words() const;

    void reset();

    int k_db() const { return static_cast<int>(encoding); }
    int k_q()  const; // 1 for SameAsStorage, 4 or 8 for scalar modes

private:
    void reserve(size_t n_vectors);
    float h_to_metric(int64_t weighted_h) const;
};

enum class Kernel { Scalar, AVX2, AVX512VPopcnt };
Kernel active_kernel();

} // namespace bq
