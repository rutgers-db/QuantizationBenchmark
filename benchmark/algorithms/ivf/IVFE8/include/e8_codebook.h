/*
 * E8 lattice codebook for 1-bit quantization.
 *
 * 240 minimum-norm vectors of the E8 lattice (all of norm sqrt(2)):
 *   Type A (112): (+/-1, +/-1, 0^6) over all C(8,2)=28 position pairs.
 *   Type B (128): (+/-1/2)^8 with even number of minus signs.
 *
 * 16 padding codewords (indices 240..255) duplicate Type A entries; the
 * encoder never selects them, so we get a clean 1-byte-per-block layout
 * matching the RaBitQ 1-bit storage budget (d/8 bytes per vector).
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace e8lib {

constexpr size_t kBlockDim = 8;
constexpr size_t kCodebookSize = 256;
constexpr size_t kNumRoots = 240;

class E8Codebook {
public:
    E8Codebook() { fill_codebook(); fill_transposed(); }

    const float* data() const { return cb_.data(); }
    const float* codeword(size_t k) const { return cb_.data() + k * kBlockDim; }

    /* Transposed layout: cb_T[d * 256 + k] = cb[k * 8 + d].
     * Lets a LUT builder do 16 output entries per SIMD slab: one broadcast
     * of q[d] times a loaded column of 16 codeword coords, accumulated into
     * the LUT slab. 32-byte aligned. */
    const float* data_transposed() const { return cb_T_.data(); }

    /* Nearest codeword to block b (8 floats). Scans only the first
     * kNumRoots entries; the padding slots are unreachable. */
    uint8_t encode_block(const float* __restrict__ block) const {
        float best_ip = -1e30f;
        uint8_t best_k = 0;
        for (size_t k = 0; k < kNumRoots; ++k) {
            const float* c = cb_.data() + k * kBlockDim;
            float ip = block[0] * c[0] + block[1] * c[1] +
                       block[2] * c[2] + block[3] * c[3] +
                       block[4] * c[4] + block[5] * c[5] +
                       block[6] * c[6] + block[7] * c[7];
            if (ip > best_ip) {
                best_ip = ip;
                best_k = static_cast<uint8_t>(k);
            }
        }
        return best_k;
    }

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

    void fill_transposed() {
        for (size_t d = 0; d < kBlockDim; ++d) {
            for (size_t k = 0; k < kCodebookSize; ++k) {
                cb_T_[d * kCodebookSize + k] = cb_[k * kBlockDim + d];
            }
        }
    }

    alignas(64) std::array<float, kCodebookSize * kBlockDim> cb_{};
    alignas(64) std::array<float, kBlockDim * kCodebookSize> cb_T_{};
};

inline const E8Codebook& get_e8_codebook() {
    static const E8Codebook cb;
    return cb;
}

}  // namespace e8lib
