/*
 * E8 lattice 1.5-bit (12-bit per 8-dim block) quantization, LUT-free format.
 *
 * Uses E8's second shell (norm^2 = 4), which has 2160 lattice points,
 * partitioned into three families:
 *   Type C: (+/-2, 0^7)                          16 = 8 pos x 2 signs
 *   Type D: (+/-1^4, 0^4)                      1120 = 70 subsets x 16 signs
 *   Type E: one coord at +/-3/2, others +/-1/2 1024 = 8 pos x 128 sign masks
 *
 * 12-bit packed code layout:
 *   bit 11  : type major flag (0 = D, 1 = C/E)
 *   D     (bit 11 = 0):
 *     bits 10..4 : subset_id (0..69)
 *     bits  3..0 : signs of the 4 subset elements (1 = negative)
 *   C/E   (bit 11 = 1):
 *     bit 10     : 0 = C, 1 = E
 *     bits 9..7  : pos (0..7) -- which coord carries the +/-2 (C) or +/-3/2 (E)
 *     bits 6..0  : sign payload
 *       C: bit 0 = sign at pos
 *       E: signs s_0..s_6 of coords 0..6 (1 = negative).
 *          Sign of coord 7 is recovered from the lattice constraint
 *          (full 8-bit sign mask has odd parity), i.e. s_7 = 1 ^ parity(s_0..6).
 *
 * Codes are stored bit-packed in tiles of 16 lanes per (tile, block), as
 *   16 bytes of low 8 bits   (lane v at offset v)
 * + 8 bytes of high 4 bits    (lane v at offset 16 + v/2, low nibble for even v).
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace e8lib12 {

constexpr size_t kBlockDim       = 8;
constexpr size_t kNumTypeC       = 16;
constexpr size_t kNumTypeD       = 1120;
constexpr size_t kNumTypeE       = 1024;
constexpr size_t kNumRoots       = 2160;
constexpr size_t kNumDSubsets    = 70;        // C(8,4)
constexpr size_t kDSubsetTabSize = 128;       // padded to 2^7 for SIMD lookup

constexpr size_t kCodeBitsPerBlock = 12;
constexpr size_t kPackedBytesPerTileBlock = 24;  // 16 + 8

/* Packed-code field layout (uint16_t, low 12 bits used) */
constexpr uint16_t kFlagCEMask   = 0x0800;    // bit 11
constexpr uint16_t kFlagETypeBit = 0x0400;    // bit 10
constexpr uint16_t kSubsetShift  = 4;
constexpr uint16_t kSubsetMask   = 0x7F;
constexpr uint16_t kDSignsMask   = 0x0F;
constexpr uint16_t kPosShift     = 7;
constexpr uint16_t kPosMask      = 0x07;
constexpr uint16_t kEPayloadMask = 0x7F;
constexpr uint16_t kCSignBit     = 0x0001;

class E8Shell2Codebook {
public:
    E8Shell2Codebook() {
        fill_d_subsets();
        fill_codebook();
    }

    const float* data() const { return cb_.data(); }
    const float* codeword(size_t k) const { return cb_.data() + k * kBlockDim; }

    /* Brute-force nearest in shell 2 by max IP. Returns raw index 0..2159. */
    uint16_t encode_block_raw(const float* __restrict__ block) const {
        float best_ip = -1e30f;
        uint16_t best_k = 0;
        for (size_t k = 0; k < kNumRoots; ++k) {
            const float* c = cb_.data() + k * kBlockDim;
            float ip = block[0]*c[0] + block[1]*c[1] + block[2]*c[2] + block[3]*c[3]
                     + block[4]*c[4] + block[5]*c[5] + block[6]*c[6] + block[7]*c[7];
            if (ip > best_ip) {
                best_ip = ip;
                best_k = static_cast<uint16_t>(k);
            }
        }
        return best_k;
    }

    uint16_t encode_block(const float* __restrict__ block) const {
        return raw_to_packed(encode_block_raw(block));
    }

    /* Codebook order: [0..15] = C, [16..1135] = D, [1136..2159] = E. */
    uint16_t raw_to_packed(uint16_t k) const {
        if (k < kNumTypeC) {
            uint16_t pos  = static_cast<uint16_t>(k >> 1);
            uint16_t sign = static_cast<uint16_t>(k & 1u);
            return static_cast<uint16_t>(kFlagCEMask | (pos << kPosShift) | sign);
        }
        if (k < kNumTypeC + kNumTypeD) {
            uint16_t off       = static_cast<uint16_t>(k - kNumTypeC);
            uint16_t subset_id = static_cast<uint16_t>(off >> 4);          // /16
            uint16_t signs     = static_cast<uint16_t>(off & 0xFu);
            return static_cast<uint16_t>((subset_id << kSubsetShift) | signs);
        }
        uint16_t off    = static_cast<uint16_t>(k - kNumTypeC - kNumTypeD);
        uint16_t pos    = static_cast<uint16_t>(off >> 7);                 // /128
        uint16_t mask7  = static_cast<uint16_t>(off & 0x7Fu);
        return static_cast<uint16_t>(kFlagCEMask | kFlagETypeBit
                                     | (pos << kPosShift) | mask7);
    }

    /* Reconstruct a codeword from a packed code (scalar, for tail). */
    void decode_packed(uint16_t packed, float* out) const {
        for (size_t d = 0; d < kBlockDim; ++d) out[d] = 0.0f;
        if ((packed & kFlagCEMask) == 0) {
            // Type D
            uint16_t subset_id = static_cast<uint16_t>((packed >> kSubsetShift) & kSubsetMask);
            uint16_t signs     = static_cast<uint16_t>(packed & kDSignsMask);
            uint8_t d0 = d_subset_pos_[subset_id][0];
            uint8_t d1 = d_subset_pos_[subset_id][1];
            uint8_t d2 = d_subset_pos_[subset_id][2];
            uint8_t d3 = d_subset_pos_[subset_id][3];
            out[d0] = (signs & 1u) ? -1.0f : 1.0f;
            out[d1] = (signs & 2u) ? -1.0f : 1.0f;
            out[d2] = (signs & 4u) ? -1.0f : 1.0f;
            out[d3] = (signs & 8u) ? -1.0f : 1.0f;
        } else if ((packed & kFlagETypeBit) == 0) {
            // Type C
            uint16_t pos  = static_cast<uint16_t>((packed >> kPosShift) & kPosMask);
            uint16_t sign = static_cast<uint16_t>(packed & kCSignBit);
            out[pos] = sign ? -2.0f : 2.0f;
        } else {
            // Type E
            uint16_t pos   = static_cast<uint16_t>((packed >> kPosShift) & kPosMask);
            uint16_t mask7 = static_cast<uint16_t>(packed & kEPayloadMask);
            uint8_t parity = static_cast<uint8_t>(__builtin_popcount(mask7) & 1u);
            uint8_t mask8  = static_cast<uint8_t>(mask7 | ((1u ^ parity) << 7));
            for (size_t d = 0; d < kBlockDim; ++d) {
                bool neg = ((mask8 >> d) & 1u) != 0;
                float mag = (d == pos) ? 1.5f : 0.5f;
                out[d] = neg ? -mag : mag;
            }
        }
    }

    /* SIMD D-path lookup table: 128 entries, each int32 packing
     * (d0 | d1<<8 | d2<<16 | d3<<24). Padded with 0 for subset_id 70..127. */
    const int32_t* d_quad_pos_arr() const { return quad_pos_table_.data(); }

    uint8_t d_subset_pos(size_t subset_id, size_t slot) const {
        return d_subset_pos_[subset_id][slot];
    }

private:
    void fill_d_subsets() {
        size_t s = 0;
        for (uint8_t i0 = 0; i0 < 8; ++i0) {
            for (uint8_t i1 = static_cast<uint8_t>(i0 + 1); i1 < 8; ++i1) {
                for (uint8_t i2 = static_cast<uint8_t>(i1 + 1); i2 < 8; ++i2) {
                    for (uint8_t i3 = static_cast<uint8_t>(i2 + 1); i3 < 8; ++i3) {
                        d_subset_pos_[s][0] = i0;
                        d_subset_pos_[s][1] = i1;
                        d_subset_pos_[s][2] = i2;
                        d_subset_pos_[s][3] = i3;
                        quad_pos_table_[s] =
                            static_cast<int32_t>(i0)
                          | (static_cast<int32_t>(i1) << 8)
                          | (static_cast<int32_t>(i2) << 16)
                          | (static_cast<int32_t>(i3) << 24);
                        ++s;
                    }
                }
            }
        }
        for (; s < kDSubsetTabSize; ++s) {
            d_subset_pos_[s][0] = d_subset_pos_[s][1] =
            d_subset_pos_[s][2] = d_subset_pos_[s][3] = 0;
            quad_pos_table_[s] = 0;
        }
    }

    void fill_codebook() {
        size_t idx = 0;
        // Type C: (+-2, 0^7), pos=0..7, sign=0..1.
        for (size_t pos = 0; pos < kBlockDim; ++pos) {
            for (int sign = 0; sign < 2; ++sign) {
                float* c = cb_.data() + idx * kBlockDim;
                for (size_t d = 0; d < kBlockDim; ++d) c[d] = 0.0f;
                c[pos] = sign ? -2.0f : 2.0f;
                ++idx;
            }
        }
        // Type D: 4 ones (signed) at the 4 positions of subset_id (lex order).
        for (size_t s = 0; s < kNumDSubsets; ++s) {
            uint8_t d0 = d_subset_pos_[s][0];
            uint8_t d1 = d_subset_pos_[s][1];
            uint8_t d2 = d_subset_pos_[s][2];
            uint8_t d3 = d_subset_pos_[s][3];
            for (uint16_t signs = 0; signs < 16; ++signs) {
                float* c = cb_.data() + idx * kBlockDim;
                for (size_t d = 0; d < kBlockDim; ++d) c[d] = 0.0f;
                c[d0] = (signs & 1u) ? -1.0f : 1.0f;
                c[d1] = (signs & 2u) ? -1.0f : 1.0f;
                c[d2] = (signs & 4u) ? -1.0f : 1.0f;
                c[d3] = (signs & 8u) ? -1.0f : 1.0f;
                ++idx;
            }
        }
        // Type E: pos at +-3/2, others +-1/2; full 8-bit sign mask has odd parity.
        for (size_t pos = 0; pos < kBlockDim; ++pos) {
            for (uint16_t mask7 = 0; mask7 < 128; ++mask7) {
                float* c = cb_.data() + idx * kBlockDim;
                uint8_t parity = static_cast<uint8_t>(__builtin_popcount(mask7) & 1u);
                uint8_t mask8  = static_cast<uint8_t>(mask7 | ((1u ^ parity) << 7));
                for (size_t d = 0; d < kBlockDim; ++d) {
                    bool neg = ((mask8 >> d) & 1u) != 0;
                    float mag = (d == pos) ? 1.5f : 0.5f;
                    c[d] = neg ? -mag : mag;
                }
                ++idx;
            }
        }
        // assert idx == kNumRoots
    }

    alignas(64) std::array<float, kNumRoots * kBlockDim> cb_{};
    std::array<std::array<uint8_t, 4>, kDSubsetTabSize> d_subset_pos_{};
    alignas(64) std::array<int32_t, kDSubsetTabSize> quad_pos_table_{};
};

inline const E8Shell2Codebook& get_e8_shell2_codebook() {
    static const E8Shell2Codebook cb;
    return cb;
}

}  // namespace e8lib12
