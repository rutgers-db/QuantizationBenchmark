/*
 * Leech lattice (Λ24) minimum-norm codebook.
 *
 * Block dim 24. Codebook = 196,560 minimum vectors of Λ24 (squared norm 32 in
 * Conway-Sloane scaling). Three types:
 *   Type 2 (1,104):  (±4)² 0²²
 *   Type 3 (97,152): (±2)⁸ 0¹⁶, support is a Golay octad, even # of minus signs
 *   Type 4 (98,304): (∓3)(±1)²³, sign pattern of (1, -1) entries from Golay G24
 *
 * Total = 196,560. log2(196,560) ≈ 17.58, so 18 bits suffice; codes are packed
 * into 3 bytes per 24-dim block (1 bit per dimension, matching the IVFE8 1-bit
 * budget on an 8-dim block).
 *
 * Two physical layouts are kept:
 *   cb_int8_  : flat (kCodebookSize, 24) int8. Used for index -> codeword
 *               lookup at search and for scalar reference encoder.
 *   cb_simd_  : coord-broadcast (24, kNumSimdGroups, 16) int8. Used by the
 *               AVX-512 brute-force encoder during construct().
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace leechlib {

constexpr size_t kBlockDim = 24;
constexpr size_t kCodebookSize = 196560;
constexpr size_t kNumType2 = 1104;
constexpr size_t kNumType3 = 97152;
constexpr size_t kNumType4 = 98304;
constexpr size_t kSimdGroup = 16;
constexpr size_t kNumSimdGroups = kCodebookSize / kSimdGroup;  // 12,285 (exact)

constexpr size_t kCodeBytes = 3;  // packed 18-bit index, 1 bit/dim

/* Generator B in [I_12 | B] form for the extended binary Golay code G24.
 * Each entry encodes 12 bits: bit d is the (12+d)-th column of the generator.
 * Rows verified: B[0] has weight 11, B[1..11] have weight 7. Combined with the
 * I_12 part, all generator rows have weight 8 or 12 (≡ 0 mod 4). */
constexpr uint16_t kGolayB[12] = {
    0xFFE, 0x477, 0xA3B, 0xD1D, 0x68F, 0xB47,
    0xDA3, 0xED1, 0x769, 0x3B5, 0x1DB, 0x8ED
};

class LeechCodebook {
public:
    LeechCodebook() {
        build_golay();
        build_codebook();
        build_simd_encoder();
        build_hash_table();
    }

    const int8_t* data() const { return cb_int8_.data(); }
    const int8_t* codeword_int8(uint32_t idx) const {
        return cb_int8_.data() + idx * kBlockDim;
    }
    const int8_t* simd_data() const { return cb_simd_.data(); }

    void decode(uint32_t idx, float* out) const {
        const int8_t* c = codeword_int8(idx);
        for (size_t d = 0; d < kBlockDim; ++d) out[d] = static_cast<float>(c[d]);
    }

    /* Reference scalar encoder. Brute-force argmax<block, c_k>. Used as a
     * correctness oracle and as the fallback path. The SIMD encoder lives in
     * the binding (encode_block_simd). */
    uint32_t encode_block_scalar(const float* block) const {
        float best_ip = -1e30f;
        uint32_t best_k = 0;
        for (uint32_t k = 0; k < kCodebookSize; ++k) {
            const int8_t* c = cb_int8_.data() + k * kBlockDim;
            float ip = 0.0f;
            for (size_t d = 0; d < kBlockDim; ++d) {
                ip += block[d] * static_cast<float>(c[d]);
            }
            if (ip > best_ip) { best_ip = ip; best_k = k; }
        }
        return best_k;
    }

    /* If v is one of the kCodebookSize Shell-1 minimum vectors, return its
     * index in [0, kCodebookSize) and set *found = true. Otherwise *found
     * = false. Uses a content hash table built at static init. */
    uint32_t index_of_shell1(const int8_t* v, bool* found) const {
        uint64_t h = hash_codeword(v);
        uint32_t bucket = static_cast<uint32_t>(h) & (kHashSize - 1);
        while (hash_keys_[bucket] != kHashEmpty) {
            uint32_t idx = hash_keys_[bucket];
            const int8_t* c = cb_int8_.data() + idx * kBlockDim;
            bool eq = true;
            for (size_t d = 0; d < kBlockDim; ++d) {
                if (c[d] != v[d]) { eq = false; break; }
            }
            if (eq) { *found = true; return idx; }
            bucket = (bucket + 1) & (kHashSize - 1);
        }
        *found = false;
        return 0;
    }

private:
    /* Enumerate all 4,096 binary Golay codewords and identify the 759 octads
     * (weight-8 codewords). Each codeword is stored as a 24-bit integer where
     * bit d = coordinate d. */
    void build_golay() {
        golay_codewords_.assign(4096, 0);
        for (uint32_t info = 0; info < 4096; ++info) {
            uint32_t check = 0;
            for (int i = 0; i < 12; ++i) {
                if (info & (1u << i)) check ^= static_cast<uint32_t>(kGolayB[i]);
            }
            golay_codewords_[info] = info | (check << 12);
        }
        for (uint32_t cw : golay_codewords_) {
            if (__builtin_popcount(cw) == 8) octads_.push_back(cw);
        }
        if (octads_.size() != 759) {
            throw std::runtime_error("Golay G24 octad count mismatch (expected 759)");
        }
    }

    void build_codebook() {
        cb_int8_.assign(kCodebookSize * kBlockDim, 0);
        size_t idx = 0;

        // Type 2: (±4, ±4, 0^22). 24·23/2 pairs × 4 sign combos = 1104.
        for (size_t i = 0; i < kBlockDim; ++i) {
            for (size_t j = i + 1; j < kBlockDim; ++j) {
                for (int si = 0; si < 2; ++si) {
                    for (int sj = 0; sj < 2; ++sj) {
                        int8_t* c = cb_int8_.data() + idx * kBlockDim;
                        c[i] = si ? 4 : -4;
                        c[j] = sj ? 4 : -4;
                        ++idx;
                    }
                }
            }
        }
        if (idx != kNumType2) {
            throw std::runtime_error("Leech Type 2 count mismatch");
        }

        // Type 3: (±2)^8 0^16 with octad support and even # of minus signs.
        // 759 octads × 128 even-parity 8-bit sign masks = 97,152.
        for (uint32_t oct : octads_) {
            int positions[8];
            int p = 0;
            for (int d = 0; d < 24; ++d) {
                if (oct & (1u << d)) positions[p++] = d;
            }
            for (uint32_t mask = 0; mask < 256; ++mask) {
                if (__builtin_popcount(mask) & 1) continue;
                int8_t* c = cb_int8_.data() + idx * kBlockDim;
                for (int b = 0; b < 8; ++b) {
                    c[positions[b]] = (mask & (1u << b)) ? -2 : 2;
                }
                ++idx;
            }
        }
        if (idx != kNumType2 + kNumType3) {
            throw std::runtime_error("Leech Type 3 count mismatch");
        }

        // Type 4: (∓3)(±1)^23. For each k in 0..23 and each Golay codeword cw,
        // build v with v_d = (-1)^cw_d for d≠k and v_k = -3·(-1)^cw_k.
        // Squared norm = 23 + 9 = 32. The all-ones (mod 2) vector is in G24,
        // and Σ v ≡ 4 (mod 8) for all (k, cw), satisfying Λ24's congruence
        // condition for this orbit.
        for (size_t k = 0; k < kBlockDim; ++k) {
            for (uint32_t cw : golay_codewords_) {
                int8_t* c = cb_int8_.data() + idx * kBlockDim;
                for (size_t d = 0; d < kBlockDim; ++d) {
                    int sign_bit = (cw >> d) & 1;
                    if (d == k) {
                        c[d] = sign_bit ? 3 : -3;
                    } else {
                        c[d] = sign_bit ? -1 : 1;
                    }
                }
                ++idx;
            }
        }
        if (idx != kCodebookSize) {
            throw std::runtime_error("Leech codebook total count mismatch");
        }

        // Sanity: all codewords have squared norm 32.
        for (size_t k = 0; k < kCodebookSize; ++k) {
            const int8_t* c = cb_int8_.data() + k * kBlockDim;
            int n2 = 0;
            for (size_t d = 0; d < kBlockDim; ++d) n2 += int(c[d]) * int(c[d]);
            if (n2 != 32) {
                throw std::runtime_error("Leech codeword has wrong squared norm");
            }
        }
    }

    /* Open-addressing hash table mapping codeword content → Shell-1 index,
     * for fast lookup of decoder outputs. Sized to ~2× kCodebookSize so load
     * factor stays below 0.5 for fast probe. */
    static constexpr size_t kHashSize  = 524288;  // 2^19, > 2 * 196,560
    static constexpr uint32_t kHashEmpty = 0xFFFFFFFFu;

    static uint64_t hash_codeword(const int8_t* v) {
        // FNV-1a over 24 bytes.
        uint64_t h = 1469598103934665603ULL;
        for (size_t d = 0; d < kBlockDim; ++d) {
            h ^= static_cast<uint8_t>(v[d]);
            h *= 1099511628211ULL;
        }
        return h;
    }

    void build_hash_table() {
        hash_keys_.assign(kHashSize, kHashEmpty);
        for (uint32_t k = 0; k < kCodebookSize; ++k) {
            const int8_t* v = cb_int8_.data() + k * kBlockDim;
            uint64_t h = hash_codeword(v);
            uint32_t bucket = static_cast<uint32_t>(h) & (kHashSize - 1);
            while (hash_keys_[bucket] != kHashEmpty) {
                bucket = (bucket + 1) & (kHashSize - 1);
            }
            hash_keys_[bucket] = k;
        }
    }

    /* Coord-broadcast layout for SIMD encoder:
     *   cb_simd_[d * kNumSimdGroups * kSimdGroup + g * kSimdGroup + l]
     * holds coord d of codeword (g * 16 + l). With this layout the encoder
     * processes 16 codewords per inner iteration using 24 broadcast-FMAs. */
    void build_simd_encoder() {
        cb_simd_.assign(kBlockDim * kCodebookSize, 0);
        for (size_t k = 0; k < kCodebookSize; ++k) {
            size_t g = k / kSimdGroup;
            size_t l = k % kSimdGroup;
            for (size_t d = 0; d < kBlockDim; ++d) {
                cb_simd_[d * kCodebookSize + g * kSimdGroup + l] =
                    cb_int8_[k * kBlockDim + d];
            }
        }
    }

    std::vector<int8_t> cb_int8_;
    std::vector<int8_t> cb_simd_;
    std::vector<uint32_t> golay_codewords_;
    std::vector<uint32_t> octads_;
    std::vector<uint32_t> hash_keys_;
};

inline const LeechCodebook& get_leech_codebook() {
    static const LeechCodebook cb;
    return cb;
}

}  // namespace leechlib
