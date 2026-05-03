/*
 * E8 lattice 15-bit (per 8-dim block) quantization, LUT-free path is scalar.
 *
 * Uses E8's fifth shell (norm^2 = 10), which has 30240 lattice points,
 * partitioned into five families:
 *   alpha:   (+/-3, +/-1, 0^6)                                     224
 *   beta:    (+/-2, +/-2, +/-1, +/-1, 0^4)                        6720
 *   gamma:   (+/-2, +/-1^6, 0)                                    7168
 *   delta:   (+/-1/2)^4 + (+/-3/2)^4, 8-bit sign mask even parity 8960
 *   epsilon: (+/-1/2)^6 + +/-3/2 + +/-5/2, 8-bit mask odd parity  7168
 * Total = 30240 <= 32768 = 2^15.
 *
 * For now, the packed code is simply the raw codebook index (0..30239) stored
 * in 15 bits. Decoding looks the codeword up directly in cb_; the search path
 * is scalar (codebook is ~1 MB, fits in L2). SIMD per-type evaluation can be
 * added later without changing the wire format.
 *
 * Codes are stored two-byte-per-lane (low byte + high byte, top bit unused),
 * 32 bytes per (tile, block).
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace e8lib15 {

constexpr size_t kBlockDim = 8;

constexpr size_t kNumAlpha   = 224;     // (+-3, +-1, 0^6): 8*7*2*2
constexpr size_t kNumBeta    = 6720;    // (+-2^2, +-1^2, 0^4): C(8,2)*C(6,2)*16
constexpr size_t kNumGamma   = 7168;    // (+-2, +-1^6, 0): 8*7*128
constexpr size_t kNumDelta   = 8960;    // 4x +-1/2 + 4x +-3/2 + even parity: C(8,4)*128
constexpr size_t kNumEpsilon = 7168;    // 6x +-1/2 + +-3/2 + +-5/2 + odd parity: 8*7*128
constexpr size_t kNumRoots   = 30240;

constexpr size_t kCodeBitsPerBlock = 15;
constexpr size_t kPackedBytesPerTileBlock = 32;  // 2 bytes/lane * 16 lanes

class E8Shell5Codebook {
public:
    E8Shell5Codebook() {
        cb_.resize(kNumRoots * kBlockDim, 0.0f);
        fill_codebook();
    }

    const float* data() const { return cb_.data(); }
    const float* codeword(size_t k) const { return cb_.data() + k * kBlockDim; }

    /* Brute-force nearest in shell 5 by max IP. Returns raw index 0..30239. */
    uint16_t encode_block_raw(const float* __restrict__ block) const {
        float best_ip = -1e30f;
        uint16_t best_k = 0;
        const float* c = cb_.data();
        for (size_t k = 0; k < kNumRoots; ++k, c += kBlockDim) {
            float ip = block[0]*c[0] + block[1]*c[1] + block[2]*c[2] + block[3]*c[3]
                     + block[4]*c[4] + block[5]*c[5] + block[6]*c[6] + block[7]*c[7];
            if (ip > best_ip) {
                best_ip = ip;
                best_k = static_cast<uint16_t>(k);
            }
        }
        return best_k;
    }

    /* For shell 5, packed == raw (15 bits suffice since 30240 < 32768). */
    uint16_t raw_to_packed(uint16_t k) const { return k; }
    uint16_t encode_block(const float* block) const { return encode_block_raw(block); }

    void decode_packed(uint16_t packed, float* out) const {
        const float* c = cb_.data() + static_cast<size_t>(packed) * kBlockDim;
        for (size_t d = 0; d < kBlockDim; ++d) out[d] = c[d];
    }

private:
    void fill_codebook() {
        size_t idx = 0;
        // alpha: (+-3, +-1, 0^6)
        for (size_t pos3 = 0; pos3 < kBlockDim; ++pos3) {
            for (size_t pos1 = 0; pos1 < kBlockDim; ++pos1) {
                if (pos1 == pos3) continue;
                for (int s3 = 0; s3 < 2; ++s3) {
                    for (int s1 = 0; s1 < 2; ++s1) {
                        float* c = cb_.data() + idx * kBlockDim;
                        for (size_t d = 0; d < kBlockDim; ++d) c[d] = 0.0f;
                        c[pos3] = s3 ? -3.0f : 3.0f;
                        c[pos1] = s1 ? -1.0f : 1.0f;
                        ++idx;
                    }
                }
            }
        }
        // beta: (+-2^2, +-1^2, 0^4) -- choose 2 positions for +-2, 2 of remaining for +-1
        for (uint8_t i = 0; i < kBlockDim; ++i) {
            for (uint8_t j = static_cast<uint8_t>(i + 1); j < kBlockDim; ++j) {
                // Two ±2 positions: i, j
                for (uint8_t a = 0; a < kBlockDim; ++a) {
                    if (a == i || a == j) continue;
                    for (uint8_t b = static_cast<uint8_t>(a + 1); b < kBlockDim; ++b) {
                        if (b == i || b == j) continue;
                        for (uint16_t s = 0; s < 16; ++s) {
                            float* c = cb_.data() + idx * kBlockDim;
                            for (size_t d = 0; d < kBlockDim; ++d) c[d] = 0.0f;
                            c[i] = (s & 1u) ? -2.0f : 2.0f;
                            c[j] = (s & 2u) ? -2.0f : 2.0f;
                            c[a] = (s & 4u) ? -1.0f : 1.0f;
                            c[b] = (s & 8u) ? -1.0f : 1.0f;
                            ++idx;
                        }
                    }
                }
            }
        }
        // gamma: (+-2, +-1^6, 0)  -- 8 positions for +-2, 7 for the zero
        for (uint8_t pos2 = 0; pos2 < kBlockDim; ++pos2) {
            for (uint8_t z = 0; z < kBlockDim; ++z) {
                if (z == pos2) continue;
                // remaining 6 positions take +-1
                std::array<uint8_t, 6> ones_pos{};
                size_t op = 0;
                for (uint8_t d = 0; d < kBlockDim; ++d) {
                    if (d == pos2 || d == z) continue;
                    ones_pos[op++] = d;
                }
                for (uint16_t s = 0; s < 128; ++s) {  // 1 sign of +-2 + 6 signs of +-1
                    float* c = cb_.data() + idx * kBlockDim;
                    for (size_t d = 0; d < kBlockDim; ++d) c[d] = 0.0f;
                    c[pos2] = (s & 1u) ? -2.0f : 2.0f;
                    for (size_t k = 0; k < 6; ++k) {
                        bool neg = ((s >> (k + 1)) & 1u) != 0;
                        c[ones_pos[k]] = neg ? -1.0f : 1.0f;
                    }
                    ++idx;
                }
            }
        }
        // delta: 4x +-1/2, 4x +-3/2; full 8-bit sign mask M has even parity.
        // (Derivation: integer-part sum required even => popcount(M) even.)
        // Enumerate by 4-subset for the +-3/2 positions, then 7-bit sign payload
        // (s_0..s_6); s_7 = parity(s_0..s_6) so parity(M) = even.
        for (uint8_t i0 = 0; i0 < kBlockDim; ++i0) {
            for (uint8_t i1 = static_cast<uint8_t>(i0 + 1); i1 < kBlockDim; ++i1) {
                for (uint8_t i2 = static_cast<uint8_t>(i1 + 1); i2 < kBlockDim; ++i2) {
                    for (uint8_t i3 = static_cast<uint8_t>(i2 + 1); i3 < kBlockDim; ++i3) {
                        bool is_3half[8] = {false, false, false, false,
                                            false, false, false, false};
                        is_3half[i0] = is_3half[i1] = is_3half[i2] = is_3half[i3] = true;
                        for (uint16_t mask7 = 0; mask7 < 128; ++mask7) {
                            uint8_t parity = static_cast<uint8_t>(__builtin_popcount(mask7) & 1u);
                            uint8_t mask8  = static_cast<uint8_t>(mask7 | (parity << 7));
                            float* c = cb_.data() + idx * kBlockDim;
                            for (size_t d = 0; d < kBlockDim; ++d) {
                                bool neg = ((mask8 >> d) & 1u) != 0;
                                float mag = is_3half[d] ? 1.5f : 0.5f;
                                c[d] = neg ? -mag : mag;
                            }
                            ++idx;
                        }
                    }
                }
            }
        }
        // epsilon: 6x +-1/2 + 1x +-3/2 + 1x +-5/2; full 8-bit mask M has odd parity.
        for (uint8_t p3 = 0; p3 < kBlockDim; ++p3) {
            for (uint8_t p5 = 0; p5 < kBlockDim; ++p5) {
                if (p5 == p3) continue;
                for (uint16_t mask7 = 0; mask7 < 128; ++mask7) {
                    uint8_t parity = static_cast<uint8_t>(__builtin_popcount(mask7) & 1u);
                    // odd parity: s_7 = 1 ^ parity
                    uint8_t s7    = static_cast<uint8_t>(1u ^ parity);
                    uint8_t mask8 = static_cast<uint8_t>(mask7 | (s7 << 7));
                    float* c = cb_.data() + idx * kBlockDim;
                    for (size_t d = 0; d < kBlockDim; ++d) {
                        bool neg = ((mask8 >> d) & 1u) != 0;
                        float mag;
                        if (d == p3) mag = 1.5f;
                        else if (d == p5) mag = 2.5f;
                        else mag = 0.5f;
                        c[d] = neg ? -mag : mag;
                    }
                    ++idx;
                }
            }
        }
        // assert idx == kNumRoots
    }

    std::vector<float> cb_;
};

inline const E8Shell5Codebook& get_e8_shell5_codebook() {
    static const E8Shell5Codebook cb;
    return cb;
}

}  // namespace e8lib15
