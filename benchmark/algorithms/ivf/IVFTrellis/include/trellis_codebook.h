#pragma once
//
// Trellis-coded scalar quantizer (TCQ-style) used by IVFTrellis.
//
// Layout:
//   * K bits / step, S states.  This file fixes K=2, S=64 ⇒ IDX = S * 2^K = 256
//     so the in-register vpermi2b kernel applies (4 ZMM = 256B output table).
//   * One step per (rotated/residual) dimension.  Code length per vector
//     = ceil(D*K / 8) bytes, where D = padded_dim.
//   * Output table sampled from N(0, 1) with deterministic seed → quantized
//     to int8 with scale 1/64.
//   * State transitions are a CANONICAL CONVOLUTIONAL-CODE SHIFT REGISTER:
//        next_state(s, u) = ((s << K) | u) & (S - 1)
//     i.e. shift K new input bits into the low position, drop the K oldest
//     bits.  Equivalently, next_state = (state * B + u) & kStateMask.  No
//     stored transition table; the SIMD kernel skips 2 of its 4 vpermi2b
//     lookups per step (output table only).  This is the QTIP-style
//     trellis structure (computed transitions, not random).
//   * Codebook is GLOBAL (independent of dataset).  Per-vector scale is
//     absorbed by the same RaBitQ-style f_add / f_rescale used by IVFE8NoLut.
//
// All output values use the same fp32 = int8 / 64.0f convention.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace trellislib {

constexpr int kK     = 2;            // bits per step
constexpr int kS     = 64;           // number of states  (must be power of two)
constexpr int kBr    = 1 << kK;      // = 4
constexpr int kIdx   = kS * kBr;     // = 256
constexpr uint8_t kStateMask = static_cast<uint8_t>(kS - 1);   // 0x3F
constexpr uint8_t kS0 = 0;           // start state
constexpr float kScale = 1.0f / 64.f;

// Shift-register transition: next_state = (state * B + u) & kStateMask.
//   For S=64, K=2 this drops the top 2 bits of state and shifts in u.
inline uint8_t next_state_of(uint32_t idx) {
    return static_cast<uint8_t>(idx & kStateMask);
}

struct TrellisCodebook {
    alignas(64) int8_t  out_i8[kIdx];     // reconstruction symbol (signed)
                float   out_f [kIdx];     // float mirror

    // Build a deterministic codebook so encoder + decoder always agree across
    // build() calls.  Seed kept fixed; we can sweep later if needed.
    void build(uint32_t seed = 42) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (int i = 0; i < kIdx; ++i) {
            float c = nd(rng);
            int   v = std::max(-127, std::min(127, int(std::round(c * 64.f))));
            out_i8[i] = static_cast<int8_t>(v);
            out_f [i] = static_cast<float>(v) * kScale;
        }
    }
};

inline const TrellisCodebook& get_trellis_codebook() {
    static TrellisCodebook cb = []{
        TrellisCodebook c;
        c.build(42);
        return c;
    }();
    return cb;
}

// ---------------- bit-stream packing ----------------
// Per-vector code stored as the contiguous bit stream of input symbols (LSB
// first), step t at bit positions [t*K, t*K+K-1].  K=2 so every 2-bit input
// fits inside a single byte.

inline uint32_t bs_get(const uint8_t* code, size_t t) {
    size_t p = t * kK;
    size_t by = p >> 3, off = p & 7;
    return uint32_t((code[by] >> off) & (kBr - 1));
}
inline void bs_set(uint8_t* code, size_t t, uint32_t u) {
    size_t p = t * kK;
    size_t by = p >> 3, off = p & 7;
    uint8_t mask = static_cast<uint8_t>((kBr - 1) << off);
    code[by] = static_cast<uint8_t>(
        (code[by] & ~mask) | static_cast<uint8_t>((u & (kBr - 1)) << off));
}

// ---------------- scalar Viterbi encoder ----------------
// Encodes x[0..D-1] into a code minimizing sum_t (x[t] - out_f[idx_t])^2.
// Also returns the reconstructed sequence (length D) for the caller's factor
// computation, and the achieved squared error.
//
// Allocates two double-buffered cost arrays and a D*S backtrack table; caller
// is expected to reuse `scratch` between vectors via a thread-local buffer.

struct ViterbiScratch {
    std::vector<float>   cost_a;
    std::vector<float>   cost_b;
    std::vector<uint8_t> back;  // size D * S, holds incoming idx for each (t, s')

    void reset(size_t D) {
        cost_a.assign(kS, 0.f);
        cost_b.assign(kS, 0.f);
        back.assign(D * kS, 0);
    }
};

inline float viterbi_encode(const float* x, size_t D, uint8_t* code,
                            float* reconstructed, ViterbiScratch& scratch) {
    constexpr float INF = std::numeric_limits<float>::infinity();
    const auto& cb = get_trellis_codebook();
    if (scratch.back.size() < D * kS) scratch.reset(D);
    float* cur = scratch.cost_a.data();
    float* nxt = scratch.cost_b.data();
    uint8_t* back = scratch.back.data();

    for (int s = 0; s < kS; ++s) cur[s] = (s == kS0 ? 0.f : INF);

    for (size_t t = 0; t < D; ++t) {
        for (int s2 = 0; s2 < kS; ++s2) nxt[s2] = INF;
        for (int s = 0; s < kS; ++s) {
            float cs = cur[s];
            if (cs == INF) continue;
            for (int u = 0; u < kBr; ++u) {
                int idx = s * kBr + u;
                float d = x[t] - cb.out_f[idx];
                float cn = cs + d * d;
                int s2 = next_state_of(uint32_t(idx));
                if (cn < nxt[s2]) {
                    nxt[s2] = cn;
                    back[t * kS + s2] = static_cast<uint8_t>(idx);
                }
            }
        }
        std::swap(cur, nxt);
    }

    int best_s = 0; float best_c = INF;
    for (int s = 0; s < kS; ++s) if (cur[s] < best_c) { best_c = cur[s]; best_s = s; }

    // Backtrack & also fill in `reconstructed` from the visited (state,input).
    size_t code_bytes = (D * kK + 7) / 8;
    std::fill(code, code + code_bytes, uint8_t(0));
    int s = best_s;
    for (size_t t = D; t-- > 0; ) {
        int idx = back[t * kS + s];
        bs_set(code, t, uint32_t(idx & (kBr - 1)));
        if (reconstructed) reconstructed[t] = cb.out_f[idx];
        s = idx / kBr;
    }
    return best_c;
}

// scalar IP <q, c(code)>; reference for SIMD kernel.
inline float ip_scalar(const float* q, size_t D, const uint8_t* code) {
    const auto& cb = get_trellis_codebook();
    uint32_t s = kS0;
    float acc = 0.f;
    for (size_t t = 0; t < D; ++t) {
        uint32_t u   = bs_get(code, t);
        uint32_t idx = (s * kBr) + u;
        acc += q[t] * cb.out_f[idx];
        s = next_state_of(idx);
    }
    return acc;
}

}  // namespace trellislib
