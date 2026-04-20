/*
 * E8 lattice 1-bit quantization with a LUT-free packed code format.
 *
 * Physical codebook is the same 240 minimum-norm E8 vectors as the LUT
 * variant (Type A: (+/-1,+/-1,0^6); Type B: (+/-1/2)^8, even number of
 * minus signs), but the per-block byte is repacked so search can evaluate
 * <q_b, codeword> directly from the code bits:
 *
 *   bit 7 = flag (0 = Type A, 1 = Type B)
 *   Type A (flag=0): bits 6..2 = pair index (0..27),
 *                    bits 1..0 = sign bits (bit=1 => coord is negative),
 *                                bit 1 is sign of coord i, bit 0 is sign of coord j.
 *   Type B (flag=1): bits 6..0 = signs of coords 0..6 (bit=1 => -1/2);
 *                    sign of coord 7 is s_0 ^ s_1 ^ ... ^ s_6 (even parity).
 *
 * The layout uses 0 extra bits: Type B has 128 codewords (7 bits) and Type A
 * has 112 (5+2 bits), both strictly < 128, so bit 7 is free.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace e8lib {

constexpr size_t kBlockDim = 8;
constexpr size_t kCodebookSize = 256;  // physical codebook has 256 slots (240 real + 16 unused)
constexpr size_t kNumRoots = 240;

constexpr size_t kNumPairs = 28;       // C(8,2)
constexpr size_t kNumTypeA = 112;      // 28 pairs * 4 sign combos
constexpr size_t kNumTypeB = 128;      // even-parity 8-bit sign masks

/* Packed-code field layout */
constexpr uint8_t kFlagMask     = 0x80;
constexpr uint8_t kPayloadMask  = 0x7F;
constexpr uint8_t kFlagTypeB    = 0x80;
constexpr uint8_t kPairShift    = 2;
constexpr uint8_t kPairMask     = 0x1F;   // 5 bits
constexpr uint8_t kSignIBit     = 0x02;
constexpr uint8_t kSignJBit     = 0x01;

class E8Codebook {
public:
    E8Codebook() {
        fill_codebook();
        fill_pair_tables();
        fill_type_b_tables();
    }

    const float* data() const { return cb_.data(); }
    const float* codeword(size_t k) const { return cb_.data() + k * kBlockDim; }

    /* Original 0..239 scan (cheap for build; correctness reference). */
    uint8_t encode_block_raw(const float* __restrict__ block) const {
        float best_ip = -1e30f;
        uint8_t best_k = 0;
        for (size_t k = 0; k < kNumRoots; ++k) {
            const float* c = cb_.data() + k * kBlockDim;
            float ip = block[0]*c[0] + block[1]*c[1] + block[2]*c[2] + block[3]*c[3]
                     + block[4]*c[4] + block[5]*c[5] + block[6]*c[6] + block[7]*c[7];
            if (ip > best_ip) {
                best_ip = ip;
                best_k = static_cast<uint8_t>(k);
            }
        }
        return best_k;
    }

    /* Nearest codeword, returned in packed [flag|payload] format. */
    uint8_t encode_block(const float* __restrict__ block) const {
        uint8_t k = encode_block_raw(block);
        return raw_to_packed(k);
    }

    uint8_t raw_to_packed(uint8_t k) const {
        if (k < kNumTypeA) {
            uint8_t pair = k >> 2;                   // 0..27
            uint8_t sub  = k & 0x3;                  // 2*si + sj, si=1 -> c_i = +1
            uint8_t si_neg = (sub >> 1) & 1 ? 0u : 1u;
            uint8_t sj_neg = (sub & 1)     ? 0u : 1u;
            return static_cast<uint8_t>((pair << kPairShift)
                                        | (si_neg << 1) | sj_neg);
        } else {
            uint8_t off = static_cast<uint8_t>(k - kNumTypeA);   // 0..127
            uint8_t mask8 = b_offset_to_mask_[off];
            return static_cast<uint8_t>(kFlagTypeB | (mask8 & kPayloadMask));
        }
    }

    /* Reconstruct a codeword from a packed byte (scalar, for tail path). */
    void decode_packed(uint8_t packed, float* out) const {
        for (size_t d = 0; d < kBlockDim; ++d) out[d] = 0.0f;
        if ((packed & kFlagMask) == 0) {
            uint8_t pair = (packed >> kPairShift) & kPairMask;
            uint8_t i = pair_to_i_[pair];
            uint8_t j = pair_to_j_[pair];
            out[i] = (packed & kSignIBit) ? -1.0f : 1.0f;
            out[j] = (packed & kSignJBit) ? -1.0f : 1.0f;
        } else {
            uint8_t mask7 = packed & kPayloadMask;
            uint8_t parity = __builtin_popcount(mask7) & 1u;
            uint8_t mask8 = static_cast<uint8_t>(mask7 | (parity << 7));
            for (size_t d = 0; d < kBlockDim; ++d) {
                out[d] = ((mask8 >> d) & 1u) ? -0.5f : 0.5f;
            }
        }
    }

    /* Arrays exposed to the SIMD search kernel. Padded to 32 entries so
     * vpermi2d can look them up with a single instruction. */
    const int32_t* pair_to_i_arr() const { return pair_to_i32_.data(); }
    const int32_t* pair_to_j_arr() const { return pair_to_j32_.data(); }

private:
    void fill_codebook() {
        size_t idx = 0;
        for (size_t i = 0; i < kBlockDim; ++i) {
            for (size_t j = i + 1; j < kBlockDim; ++j) {
                for (int si = 0; si < 2; ++si) {
                    for (int sj = 0; sj < 2; ++sj) {
                        float* c = cb_.data() + idx * kBlockDim;
                        for (size_t d = 0; d < kBlockDim; ++d) c[d] = 0.0f;
                        c[i] = si ? 1.0f : -1.0f;
                        c[j] = sj ? 1.0f : -1.0f;
                        ++idx;
                    }
                }
            }
        }
        for (unsigned mask = 0; mask < 256u; ++mask) {
            if (__builtin_popcount(mask) & 1) continue;
            float* c = cb_.data() + idx * kBlockDim;
            for (size_t d = 0; d < kBlockDim; ++d) {
                c[d] = (mask >> d) & 1u ? -0.5f : 0.5f;
            }
            ++idx;
        }
        for (size_t k = kNumRoots; k < kCodebookSize; ++k) {
            float* dst = cb_.data() + k * kBlockDim;
            const float* src = cb_.data() + (k - kNumRoots) * kBlockDim;
            for (size_t d = 0; d < kBlockDim; ++d) dst[d] = src[d];
        }
    }

    void fill_pair_tables() {
        size_t p = 0;
        for (size_t i = 0; i < kBlockDim; ++i) {
            for (size_t j = i + 1; j < kBlockDim; ++j) {
                pair_to_i_[p] = static_cast<uint8_t>(i);
                pair_to_j_[p] = static_cast<uint8_t>(j);
                pair_to_i32_[p] = static_cast<int32_t>(i);
                pair_to_j32_[p] = static_cast<int32_t>(j);
                ++p;
            }
        }
        for (; p < 32; ++p) {
            pair_to_i_[p] = 0;
            pair_to_j_[p] = 0;
            pair_to_i32_[p] = 0;
            pair_to_j32_[p] = 0;
        }
    }

    void fill_type_b_tables() {
        size_t off = 0;
        for (unsigned mask = 0; mask < 256u; ++mask) {
            if (__builtin_popcount(mask) & 1) continue;
            b_offset_to_mask_[off] = static_cast<uint8_t>(mask);
            ++off;
        }
    }

    alignas(64) std::array<float, kCodebookSize * kBlockDim> cb_{};
    std::array<uint8_t, 32> pair_to_i_{};
    std::array<uint8_t, 32> pair_to_j_{};
    alignas(64) std::array<int32_t, 32> pair_to_i32_{};
    alignas(64) std::array<int32_t, 32> pair_to_j32_{};
    std::array<uint8_t, kNumTypeB> b_offset_to_mask_{};
};

inline const E8Codebook& get_e8_codebook() {
    static const E8Codebook cb;
    return cb;
}

}  // namespace e8lib
