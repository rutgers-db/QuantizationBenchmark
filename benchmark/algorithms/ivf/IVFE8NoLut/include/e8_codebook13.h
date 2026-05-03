/*
 * E8 lattice 13-bit (per 8-dim block) quantization, LUT-free format.
 *
 * Uses E8's third shell (norm^2 = 6), which has 6720 lattice points,
 * partitioned into three families:
 *   Type F: (+/-2, +/-1, +/-1, 0^5)              1344 codewords
 *           (8 pos_2 * 2 sign_2 * C(7,2) pairs * 2 * 2 signs = 1344)
 *   Type G: (+/-1^6, 0^2)                        1792 codewords
 *           (C(8,2) zero-pairs * 2^6 signs = 28 * 64 = 1792)
 *   Type H: (+/-3/2 at 2 pos, +/-1/2 elsewhere)  3584 codewords
 *           (C(8,2) pairs * 128 valid sign masks (even-parity) = 3584)
 *
 * 13-bit packed code layout:
 *   bit 12 : H flag (1 = H, 0 = F or G)
 *   if H (bit 12 = 1):
 *     bits 11..7 : pair_idx (0..27)
 *     bits  6..0 : signs s_0..s_6 (1 = negative); s_7 = parity(s_0..s_6)
 *                  so that the full 8-bit mask has even parity.
 *   if not H (bit 12 = 0):
 *     bit 11 : F flag (1 = F, 0 = G)
 *     if F (bit 11 = 1):
 *       bits 10..3 : tuple_idx (0..167) -> (pos_2, pos_a, pos_b)
 *       bits  2..0 : 3 sign bits (sign_2, sign_a, sign_b)
 *     if G (bit 11 = 0):
 *       bits 10..6 : zero_pair_idx (0..27)
 *       bits  5..0 : 6 sign bits (signs of nonzero positions in canonical
 *                     ascending order; bit k = sign of k-th nonzero coord)
 *
 * Codes are stored bit-packed in tiles of 16 lanes per (tile, block):
 *   16 bytes : low 8 bits of each lane (lane v at byte v)
 *    8 bytes : bits 8..11 of each lane (paired: lane 2k low nibble, 2k+1 high)
 *    2 bytes : bit 12 of each lane (8 lanes per byte, lane v at bit v of byte v/8)
 * Total: 26 bytes per (tile, block).
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace e8lib13 {

constexpr size_t kBlockDim       = 8;
constexpr size_t kNumTypeF       = 1344;
constexpr size_t kNumTypeG       = 1792;
constexpr size_t kNumTypeH       = 3584;
constexpr size_t kNumRoots       = 6720;

constexpr size_t kNumPairs       = 28;        // C(8,2) for H and G zero-pairs
constexpr size_t kPairsPad       = 32;
constexpr size_t kNumFTuples     = 168;       // 8 pos_2 * 21 (j,k) pairs
constexpr size_t kFTuplesPad     = 256;       // padded for SIMD lookup

constexpr size_t kCodeBitsPerBlock = 13;
constexpr size_t kPackedBytesPerTileBlock = 26;  // 16 + 8 + 2

constexpr uint16_t kFlagHMask    = 0x1000;    // bit 12
constexpr uint16_t kFlagFMask    = 0x0800;    // bit 11

// H fields
constexpr uint16_t kHPairShift   = 7;
constexpr uint16_t kHPairMask    = 0x1F;
constexpr uint16_t kHPayloadMask = 0x7F;

// F fields
constexpr uint16_t kFTupleShift  = 3;
constexpr uint16_t kFTupleMask   = 0xFF;
constexpr uint16_t kFSignsMask   = 0x07;

// G fields
constexpr uint16_t kGZeroPairShift = 6;
constexpr uint16_t kGZeroPairMask  = 0x1F;
constexpr uint16_t kGSignsMask     = 0x3F;

class E8Shell3Codebook {
public:
    E8Shell3Codebook() {
        fill_pair_tables();
        fill_f_tuples();
        fill_g_zero_pairs();
        fill_codebook();
    }

    const float* data() const { return cb_.data(); }
    const float* codeword(size_t k) const { return cb_.data() + k * kBlockDim; }

    /* Brute-force nearest in shell 3 by max IP. Returns raw index 0..6719. */
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

    /* Codebook order: [0..1343] = F, [1344..3135] = G, [3136..6719] = H. */
    uint16_t raw_to_packed(uint16_t k) const {
        if (k < kNumTypeF) {
            uint16_t tuple_idx = static_cast<uint16_t>(k >> 3);   // /8 sign combos
            uint16_t signs     = static_cast<uint16_t>(k & 0x7u);
            return static_cast<uint16_t>(kFlagFMask | (tuple_idx << kFTupleShift) | signs);
        }
        if (k < kNumTypeF + kNumTypeG) {
            uint16_t off = static_cast<uint16_t>(k - kNumTypeF);
            uint16_t zp  = static_cast<uint16_t>(off >> 6);       // /64
            uint16_t s   = static_cast<uint16_t>(off & 0x3Fu);
            return static_cast<uint16_t>((zp << kGZeroPairShift) | s);
        }
        uint16_t off  = static_cast<uint16_t>(k - kNumTypeF - kNumTypeG);
        uint16_t pair = static_cast<uint16_t>(off >> 7);          // /128
        uint16_t m7   = static_cast<uint16_t>(off & 0x7Fu);
        return static_cast<uint16_t>(kFlagHMask | (pair << kHPairShift) | m7);
    }

    void decode_packed(uint16_t packed, float* out) const {
        for (size_t d = 0; d < kBlockDim; ++d) out[d] = 0.0f;
        if (packed & kFlagHMask) {
            uint16_t pair  = (packed >> kHPairShift) & kHPairMask;
            uint16_t mask7 = packed & kHPayloadMask;
            uint8_t parity = static_cast<uint8_t>(__builtin_popcount(mask7) & 1u);
            uint8_t mask8  = static_cast<uint8_t>(mask7 | (parity << 7));
            uint8_t i = pair_to_i_[pair];
            uint8_t j = pair_to_j_[pair];
            for (size_t d = 0; d < kBlockDim; ++d) {
                bool neg = ((mask8 >> d) & 1u) != 0;
                float mag = (d == i || d == j) ? 1.5f : 0.5f;
                out[d] = neg ? -mag : mag;
            }
        } else if (packed & kFlagFMask) {
            uint16_t tuple_idx = (packed >> kFTupleShift) & kFTupleMask;
            uint16_t signs     = packed & kFSignsMask;
            uint8_t pos_2 = f_tuple_pos_2_[tuple_idx];
            uint8_t pa    = f_tuple_a_[tuple_idx];
            uint8_t pb    = f_tuple_b_[tuple_idx];
            out[pos_2] = (signs & 1u) ? -2.0f : 2.0f;
            out[pa]    = (signs & 2u) ? -1.0f : 1.0f;
            out[pb]    = (signs & 4u) ? -1.0f : 1.0f;
        } else {
            uint16_t zp    = (packed >> kGZeroPairShift) & kGZeroPairMask;
            uint16_t signs = packed & kGSignsMask;
            for (size_t k = 0; k < 6; ++k) {
                uint8_t pos = g_zp_pos_[zp][k];
                bool neg = ((signs >> k) & 1u) != 0;
                out[pos] = neg ? -1.0f : 1.0f;
            }
        }
    }

    /* SIMD lookup tables. */
    const int32_t* pair_to_i_arr() const { return pair_to_i32_.data(); }   // 32 ent
    const int32_t* pair_to_j_arr() const { return pair_to_j32_.data(); }   // 32 ent
    const int32_t* f_tuple_packed_arr() const { return f_tuple_packed_.data(); }
        // 256 ent: each int32 = pos_2 | pa<<8 | pb<<16
    const int32_t* g_zp_packed_lo_arr() const { return g_zp_packed_lo_.data(); }
        // 32 ent: pos[0] | pos[1]<<8 | pos[2]<<16
    const int32_t* g_zp_packed_hi_arr() const { return g_zp_packed_hi_.data(); }
        // 32 ent: pos[3] | pos[4]<<8 | pos[5]<<16

private:
    void fill_pair_tables() {
        size_t p = 0;
        for (uint8_t i = 0; i < kBlockDim; ++i) {
            for (uint8_t j = static_cast<uint8_t>(i + 1); j < kBlockDim; ++j) {
                pair_to_i_[p] = i;
                pair_to_j_[p] = j;
                pair_to_i32_[p] = static_cast<int32_t>(i);
                pair_to_j32_[p] = static_cast<int32_t>(j);
                ++p;
            }
        }
        for (; p < kPairsPad; ++p) {
            pair_to_i_[p] = pair_to_j_[p] = 0;
            pair_to_i32_[p] = pair_to_j32_[p] = 0;
        }
    }

    void fill_f_tuples() {
        size_t idx = 0;
        for (uint8_t pos_2 = 0; pos_2 < kBlockDim; ++pos_2) {
            for (uint8_t a = 0; a < kBlockDim; ++a) {
                if (a == pos_2) continue;
                for (uint8_t b = static_cast<uint8_t>(a + 1); b < kBlockDim; ++b) {
                    if (b == pos_2) continue;
                    f_tuple_pos_2_[idx] = pos_2;
                    f_tuple_a_[idx]     = a;
                    f_tuple_b_[idx]     = b;
                    f_tuple_packed_[idx] =
                        static_cast<int32_t>(pos_2)
                      | (static_cast<int32_t>(a) << 8)
                      | (static_cast<int32_t>(b) << 16);
                    ++idx;
                }
            }
        }
        for (; idx < kFTuplesPad; ++idx) {
            f_tuple_pos_2_[idx] = 0;
            f_tuple_a_[idx]     = 0;
            f_tuple_b_[idx]     = 0;
            f_tuple_packed_[idx] = 0;
        }
    }

    void fill_g_zero_pairs() {
        size_t p = 0;
        for (uint8_t i = 0; i < kBlockDim; ++i) {
            for (uint8_t j = static_cast<uint8_t>(i + 1); j < kBlockDim; ++j) {
                size_t kk = 0;
                for (uint8_t d = 0; d < kBlockDim; ++d) {
                    if (d == i || d == j) continue;
                    g_zp_pos_[p][kk++] = d;
                }
                g_zp_packed_lo_[p] =
                    static_cast<int32_t>(g_zp_pos_[p][0])
                  | (static_cast<int32_t>(g_zp_pos_[p][1]) << 8)
                  | (static_cast<int32_t>(g_zp_pos_[p][2]) << 16);
                g_zp_packed_hi_[p] =
                    static_cast<int32_t>(g_zp_pos_[p][3])
                  | (static_cast<int32_t>(g_zp_pos_[p][4]) << 8)
                  | (static_cast<int32_t>(g_zp_pos_[p][5]) << 16);
                ++p;
            }
        }
        for (; p < kPairsPad; ++p) {
            for (size_t kk = 0; kk < 6; ++kk) g_zp_pos_[p][kk] = 0;
            g_zp_packed_lo_[p] = 0;
            g_zp_packed_hi_[p] = 0;
        }
    }

    void fill_codebook() {
        size_t idx = 0;
        // F
        for (size_t t = 0; t < kNumFTuples; ++t) {
            uint8_t p2 = f_tuple_pos_2_[t];
            uint8_t pa = f_tuple_a_[t];
            uint8_t pb = f_tuple_b_[t];
            for (uint16_t s = 0; s < 8; ++s) {
                float* c = cb_.data() + idx * kBlockDim;
                for (size_t d = 0; d < kBlockDim; ++d) c[d] = 0.0f;
                c[p2] = (s & 1u) ? -2.0f : 2.0f;
                c[pa] = (s & 2u) ? -1.0f : 1.0f;
                c[pb] = (s & 4u) ? -1.0f : 1.0f;
                ++idx;
            }
        }
        // G
        for (size_t zp = 0; zp < kNumPairs; ++zp) {
            for (uint16_t s = 0; s < 64; ++s) {
                float* c = cb_.data() + idx * kBlockDim;
                for (size_t d = 0; d < kBlockDim; ++d) c[d] = 0.0f;
                for (size_t kk = 0; kk < 6; ++kk) {
                    bool neg = ((s >> kk) & 1u) != 0;
                    c[g_zp_pos_[zp][kk]] = neg ? -1.0f : 1.0f;
                }
                ++idx;
            }
        }
        // H
        for (size_t pair = 0; pair < kNumPairs; ++pair) {
            uint8_t i = pair_to_i_[pair];
            uint8_t j = pair_to_j_[pair];
            for (uint16_t mask7 = 0; mask7 < 128; ++mask7) {
                float* c = cb_.data() + idx * kBlockDim;
                uint8_t parity = static_cast<uint8_t>(__builtin_popcount(mask7) & 1u);
                uint8_t mask8  = static_cast<uint8_t>(mask7 | (parity << 7));
                for (size_t d = 0; d < kBlockDim; ++d) {
                    bool neg = ((mask8 >> d) & 1u) != 0;
                    float mag = (d == i || d == j) ? 1.5f : 0.5f;
                    c[d] = neg ? -mag : mag;
                }
                ++idx;
            }
        }
    }

    alignas(64) std::array<float, kNumRoots * kBlockDim> cb_{};

    std::array<uint8_t, kPairsPad> pair_to_i_{};
    std::array<uint8_t, kPairsPad> pair_to_j_{};
    alignas(64) std::array<int32_t, kPairsPad> pair_to_i32_{};
    alignas(64) std::array<int32_t, kPairsPad> pair_to_j32_{};

    std::array<uint8_t, kFTuplesPad> f_tuple_pos_2_{};
    std::array<uint8_t, kFTuplesPad> f_tuple_a_{};
    std::array<uint8_t, kFTuplesPad> f_tuple_b_{};
    alignas(64) std::array<int32_t, kFTuplesPad> f_tuple_packed_{};

    std::array<std::array<uint8_t, 6>, kPairsPad> g_zp_pos_{};
    alignas(64) std::array<int32_t, kPairsPad> g_zp_packed_lo_{};
    alignas(64) std::array<int32_t, kPairsPad> g_zp_packed_hi_{};
};

inline const E8Shell3Codebook& get_e8_shell3_codebook() {
    static const E8Shell3Codebook cb;
    return cb;
}

}  // namespace e8lib13
