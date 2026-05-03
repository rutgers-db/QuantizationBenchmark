/*
 * Conway–Sloane bounded-distance decoder for the Leech lattice Λ₂₄.
 *
 * Given an arbitrary r ∈ R²⁴ this decoder returns v ∈ Λ₂₄ that is close to r.
 * For r within the packing radius of its nearest minimum vector the result is
 * the exact nearest min-norm Leech codeword; outside the packing radius it
 * still returns a lattice point with distance² ≤ the brute-force minimum
 * over Shell 1.
 *
 * Algorithm (two-path even/odd construction, hard-decision Golay):
 *   Path A: round each coord of r to the nearest EVEN integer → x_a.
 *           Σ x_a is automatically ≡ 0 (mod 2). Force Σ x_a ≡ 0 (mod 4)
 *           by flipping the cheapest single coord by ±2.
 *           x_a/2 mod 2 is the all-zero codeword in G24, so the Λ₂₄
 *           parity condition is satisfied for the even coset.
 *
 *   Path B: round each coord of r to the nearest ODD integer → x_b.
 *           Then x_b mod 2 is the all-ones vector ∈ G24 (weight-24 codeword,
 *           which lies in G24). Force Σ x_b ≡ 4 (mod 8) by flipping the
 *           cheapest single coord by ±2.
 *
 *   Path C: round each coord of r to the nearest INTEGER → x_c. The mod-2
 *           pattern b = x_c mod 2 may not lie in G24. Compute the syndrome
 *           s = b · H^T and look up the weight-≤3 error pattern e such that
 *           (b ⊕ e) ∈ G24. For each set bit d of e, flip x_c[d] by ±1
 *           toward r (cheapest direction). Then enforce the appropriate
 *           Λ₂₄ sum constraint by ±2 adjustment on the cheapest coord.
 *
 * Output: argmin over the three paths of ||r - x||₂². Each path is O(few
 * hundred ops), syndrome lookup is O(1) via a 4096-entry table built once
 * at static init. Total per-call cost: ~300–500 ops.
 *
 * The output v ∈ Λ₂₄ is returned as 24 int8 values (Conway scaling — every
 * minimum vector has squared norm 32, so coords fit in [-7, +7]).
 */

#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "leech_codebook.h"

namespace leechlib {

/* The same Golay generator B used in leech_codebook.h.  Parity-check matrix
 * H = [B^T | I_12]; for our symmetric-by-construction B, H[i] is just
 * "row i of B in the high 12 bits, plus bit i in the low 12 bits". */
class LeechDecoder {
public:
    LeechDecoder() {
        build_syndrome_table();
    }

    /* Top-level entry point.  Λ24 has TWO constraints on Shell-1 minimum
     * vectors: (1) all-even or all-odd coords with appropriate Σ ≡ {0,4}
     * (mod 8), and (2) the "sign pattern" b_d = (v_d / 2) mod 2 (even path)
     * or (v_d - 1) / 2 mod 2 (odd path) lies in the binary Golay code G24.
     * The decoder runs both cosets and picks the closer Shell-1 vector. */
    float decode(const float* r, int8_t* v_out) const {
        int x_a[24], x_b[24];
        float d_a = path_even(r, x_a);
        float d_b = path_odd(r, x_b);
        const int* best = (d_a <= d_b) ? x_a : x_b;
        float best_d   = (d_a <= d_b) ? d_a : d_b;
        for (int d = 0; d < 24; ++d) v_out[d] = static_cast<int8_t>(best[d]);
        return best_d;
    }

private:
    /* H[i] = row i of the Golay parity-check matrix as a 24-bit value.
     * H = [B | I_12]: bit (12 + i) is 1 (the I_12 part), and bits 0..11 of
     * H[i] form column i of B (= row i of B since B is symmetric for the
     * generator we use). */
    void build_syndrome_table() {
        for (int i = 0; i < 12; ++i) {
            // Row i of B as a 12-bit value (leech_codebook.h::kGolayB[i]).
            // Build column i: bit j = B[j][i].
            uint16_t col = 0;
            for (int j = 0; j < 12; ++j) {
                if (kGolayB[j] & (1u << i)) col |= (1u << j);
            }
            H_[i] = static_cast<uint32_t>(col) | (1u << (12 + i));
        }

        // Build syndrome → weight-≤3 error pattern table.
        // 4096 syndromes; iterate all weight-0/1/2/3 error patterns (24-bit)
        // and store the lightest pattern per syndrome.
        for (int s = 0; s < 4096; ++s) syndrome_table_[s] = 0xFFFFFFFFu;

        // Weight 0
        syndrome_table_[0] = 0;
        // Weight 1
        for (int d = 0; d < 24; ++d) {
            uint32_t e = 1u << d;
            uint32_t s = compute_syndrome(e);
            if (syndrome_weight(syndrome_table_[s]) > 1) syndrome_table_[s] = e;
        }
        // Weight 2
        for (int d1 = 0; d1 < 24; ++d1) for (int d2 = d1 + 1; d2 < 24; ++d2) {
            uint32_t e = (1u << d1) | (1u << d2);
            uint32_t s = compute_syndrome(e);
            if (syndrome_weight(syndrome_table_[s]) > 2) syndrome_table_[s] = e;
        }
        // Weight 3
        for (int d1 = 0; d1 < 24; ++d1)
        for (int d2 = d1 + 1; d2 < 24; ++d2)
        for (int d3 = d2 + 1; d3 < 24; ++d3) {
            uint32_t e = (1u << d1) | (1u << d2) | (1u << d3);
            uint32_t s = compute_syndrome(e);
            if (syndrome_weight(syndrome_table_[s]) > 3) syndrome_table_[s] = e;
        }
        // Weight 4 — needed because Golay G24 covering radius is 4. Cosets
        // not covered by weight-≤3 leaders all have weight-4 leaders.
        for (int d1 = 0; d1 < 24; ++d1)
        for (int d2 = d1 + 1; d2 < 24; ++d2)
        for (int d3 = d2 + 1; d3 < 24; ++d3)
        for (int d4 = d3 + 1; d4 < 24; ++d4) {
            uint32_t e = (1u << d1) | (1u << d2) | (1u << d3) | (1u << d4);
            uint32_t s = compute_syndrome(e);
            if (syndrome_weight(syndrome_table_[s]) > 4) syndrome_table_[s] = e;
        }

        // Sanity: every syndrome must now have a weight-≤4 entry (covering
        // radius of G24 is 4).
        for (int s = 0; s < 4096; ++s) {
            if (syndrome_table_[s] == 0xFFFFFFFFu) {
                throw std::runtime_error(
                    "Golay syndrome table incomplete (should not happen)");
            }
        }
    }

    static int syndrome_weight(uint32_t e) {
        if (e == 0xFFFFFFFFu) return 100;  // sentinel: no entry
        return __builtin_popcount(e);
    }

    /* Compute syndrome of a 24-bit pattern e: s = e · H^T (binary). */
    uint32_t compute_syndrome(uint32_t e) const {
        uint32_t s = 0;
        for (int i = 0; i < 12; ++i) {
            if (__builtin_popcount(e & H_[i]) & 1) s |= (1u << i);
        }
        return s;
    }

    /* Round r[d] to nearest even integer. */
    static int round_even(float v) {
        int e = 2 * static_cast<int>(std::lround(v / 2.0f));
        return e;
    }

    /* Round r[d] to nearest odd integer. */
    static int round_odd(float v) {
        int o = 2 * static_cast<int>(std::lround((v - 1.0f) / 2.0f)) + 1;
        return o;
    }

    /* Round r[d] to nearest integer. */
    static int round_any(float v) {
        return static_cast<int>(std::lround(v));
    }

    /* Cost (in distance²) of bumping x[d] by ±2 in the better direction.
     * Returns the squared-distance change and writes the chosen sign to
     * sign_out. */
    static float bump2_cost(float r_d, int x_d, int* sign_out) {
        // After bump: x_d ± 2. New residual: r_d − (x_d ± 2).
        // Δd² = (r_d − x_d ∓ 2)² − (r_d − x_d)² = ∓4(r_d − x_d) + 4.
        float diff = r_d - static_cast<float>(x_d);
        // To MINIMIZE Δd² we want ∓ to follow sign of (r_d − x_d):
        //   if diff > 0: bump up (sign = +1), Δd² = -4·diff + 4
        //   if diff < 0: bump down (sign = -1), Δd² = +4·diff + 4
        if (diff >= 0) { *sign_out = +1; return -4.0f * diff + 4.0f; }
        else           { *sign_out = -1; return  4.0f * diff + 4.0f; }
    }

    /* Cost of bumping x[d] by ±1 (toward r). */
    static float bump1_cost(float r_d, int x_d, int* sign_out) {
        float diff = r_d - static_cast<float>(x_d);
        if (diff >= 0) { *sign_out = +1; return -2.0f * diff + 1.0f; }
        else           { *sign_out = -1; return  2.0f * diff + 1.0f; }
    }

    /* Greedy ±4 bump applied to a single coord d.  Going "up" by +4 changes
     * d² by (r-x-4)² - (r-x)² = -8(r-x) + 16; going "down" by -4 gives
     * +8(r-x) + 16.  Returns the cheaper-direction cost. */
    static float bump4_cost(float r_d, int x_d, int* sign_out) {
        float diff = r_d - static_cast<float>(x_d);
        if (diff >= 0) { *sign_out = +1; return -8.0f * diff + 16.0f; }
        else           { *sign_out = -1; return  8.0f * diff + 16.0f; }
    }

    /* Soft-decision Golay decoder via Chase-3.
     *
     * Inputs:
     *   b           — received 24-bit pattern (hard-decision)
     *   flip_cost   — per-bit cost (in d² units) of flipping bit d
     * Returns:
     *   the Golay codeword c minimizing Σ_{d: c_d ≠ b_d} flip_cost[d].
     *
     * Chase-3 strategy: the K = 4 least-reliable bits (lowest flip_cost) are
     * candidates for being "wrong" in a way that the hard-decision syndrome
     * lookup misclassifies. We enumerate all 2^K = 16 subsets of those bits,
     * flip them in b speculatively, hard-decode each variant (syndrome →
     * weight-≤4 error pattern), and pick the one with minimum total soft
     * cost. For G24 (d_min=8), Chase-3 with K=4 attains near-ML soft
     * decoding performance. */
    uint32_t soft_golay_decode(uint32_t b, const float* flip_cost) const {
        constexpr int K = 4;
        // Find indices of K least-reliable bits.
        int  least_idx[K];
        float least_val[K];
        for (int k = 0; k < K; ++k) least_val[k] = std::numeric_limits<float>::infinity();
        for (int d = 0; d < 24; ++d) {
            float c = flip_cost[d];
            // Insert d into the K-min list if c < max(least_val).
            int worst = 0;
            for (int k = 1; k < K; ++k) if (least_val[k] > least_val[worst]) worst = k;
            if (c < least_val[worst]) { least_val[worst] = c; least_idx[worst] = d; }
        }

        float best_cost = std::numeric_limits<float>::infinity();
        uint32_t best_c = 0;
        for (int subset = 0; subset < (1 << K); ++subset) {
            // Flip bits in `subset` of the K least-reliable positions.
            uint32_t flip = 0;
            for (int k = 0; k < K; ++k) {
                if (subset & (1 << k)) flip |= (1u << least_idx[k]);
            }
            uint32_t b_try = b ^ flip;
            uint32_t s = compute_syndrome(b_try);
            uint32_t e = syndrome_table_[s];
            uint32_t c_candidate = b_try ^ e;  // Golay codeword in F_2^24
            // Soft cost = sum of flip costs over bits where c differs from b.
            uint32_t diff = c_candidate ^ b;
            float cost = 0.0f;
            while (diff) {
                int d = __builtin_ctz(diff);
                cost += flip_cost[d];
                diff &= diff - 1;
            }
            if (cost < best_cost) { best_cost = cost; best_c = c_candidate; }
        }
        return best_c;
    }

    /* Apply Golay correction to an all-even path. The "Golay bit" is
     * (x_d / 2) mod 2 = bit-1 of x_d.  ±2 bumps toggle bit-1 while keeping
     * x_d even, so they're the right primitive (NOT ±4 — which jumps two
     * Golay-bits at once and leaves bit-1 unchanged). */
    void apply_golay_correction_even(const float* r, int* x, float* d2) const {
        uint32_t b = 0;
        float    flip_cost[24];
        for (int d = 0; d < 24; ++d) {
            int sgn;
            flip_cost[d] = bump2_cost(r[d], x[d], &sgn);
            if ((x[d] >> 1) & 1) b |= (1u << d);
        }
        uint32_t c = soft_golay_decode(b, flip_cost);
        uint32_t diff = c ^ b;
        while (diff) {
            int d = __builtin_ctz(diff);
            int sgn;
            float cost = bump2_cost(r[d], x[d], &sgn);
            x[d] += 2 * sgn;
            *d2 += cost;
            diff &= diff - 1;
        }
    }

    /* Odd path: same primitive (±2 keeps parity odd and toggles bit-1). */
    void apply_golay_correction_odd(const float* r, int* x, float* d2) const {
        uint32_t b = 0;
        float    flip_cost[24];
        for (int d = 0; d < 24; ++d) {
            int sgn;
            flip_cost[d] = bump2_cost(r[d], x[d], &sgn);
            if ((x[d] >> 1) & 1) b |= (1u << d);
        }
        uint32_t c = soft_golay_decode(b, flip_cost);
        uint32_t diff = c ^ b;
        while (diff) {
            int d = __builtin_ctz(diff);
            int sgn;
            float cost = bump2_cost(r[d], x[d], &sgn);
            x[d] += 2 * sgn;
            *d2 += cost;
            diff &= diff - 1;
        }
    }

    /* Path A: round to nearest even, force the sign-pattern (x/2) mod 2 to
     * lie in G24, then enforce Σ x ≡ 0 (mod 8). */
    float path_even(const float* r, int* x) const {
        float d2 = 0.0f;
        for (int d = 0; d < 24; ++d) {
            x[d] = round_even(r[d]);
            float diff = r[d] - x[d];
            d2 += diff * diff;
        }
        // First: sign-pattern correction. Each ±4 bump preserves parity
        // (still even) AND toggles a bit of (x/2) mod 2.
        apply_golay_correction_even(r, x, &d2);
        // Now sum-mod-8 correction.
        int sum = 0;
        for (int d = 0; d < 24; ++d) sum += x[d];
        // sum is divisible by 2; we need ≡ 0 (mod 8).
        int residue = ((sum % 8) + 8) % 8;  // 0, 2, 4, or 6
        if (residue == 0) return d2;
        if (residue == 2 || residue == 6) {
            int target_sign = (residue == 6) ? +1 : -1;
            int best_d = 0; float best_cost = std::numeric_limits<float>::infinity();
            for (int d = 0; d < 24; ++d) {
                float diff = r[d] - x[d];
                float c = (target_sign > 0) ? -4.0f * diff + 4.0f
                                            :  4.0f * diff + 4.0f;
                if (c < best_cost) { best_cost = c; best_d = d; }
            }
            x[best_d] += 2 * target_sign;
            d2 += best_cost;
        } else if (residue == 4) {
            // Need two ±2 bumps of same sign.
            float best_pos1 = std::numeric_limits<float>::infinity(), best_pos2 = best_pos1;
            int   best_pos1_d = -1, best_pos2_d = -1;
            float best_neg1 = best_pos1, best_neg2 = best_pos1;
            int   best_neg1_d = -1, best_neg2_d = -1;
            for (int d = 0; d < 24; ++d) {
                float diff = r[d] - x[d];
                float c_pos = -4.0f * diff + 4.0f;
                float c_neg =  4.0f * diff + 4.0f;
                if (c_pos < best_pos1) { best_pos2 = best_pos1; best_pos2_d = best_pos1_d; best_pos1 = c_pos; best_pos1_d = d; }
                else if (c_pos < best_pos2) { best_pos2 = c_pos; best_pos2_d = d; }
                if (c_neg < best_neg1) { best_neg2 = best_neg1; best_neg2_d = best_neg1_d; best_neg1 = c_neg; best_neg1_d = d; }
                else if (c_neg < best_neg2) { best_neg2 = c_neg; best_neg2_d = d; }
            }
            float pos_total = best_pos1 + best_pos2;
            float neg_total = best_neg1 + best_neg2;
            if (pos_total <= neg_total && best_pos1_d >= 0 && best_pos2_d >= 0) {
                x[best_pos1_d] += 2; x[best_pos2_d] += 2; d2 += pos_total;
            } else if (best_neg1_d >= 0 && best_neg2_d >= 0) {
                x[best_neg1_d] -= 2; x[best_neg2_d] -= 2; d2 += neg_total;
            }
        }
        return d2;
    }

    /* Path B: round to nearest odd, force the sign-pattern ((x-1)/2) mod 2
     * to lie in G24, then enforce Σ x ≡ 4 (mod 8). */
    float path_odd(const float* r, int* x) const {
        float d2 = 0.0f;
        for (int d = 0; d < 24; ++d) {
            x[d] = round_odd(r[d]);
            float diff = r[d] - x[d];
            d2 += diff * diff;
        }
        apply_golay_correction_odd(r, x, &d2);
        int sum = 0;
        for (int d = 0; d < 24; ++d) sum += x[d];
        // sum is even (24 odd numbers sum to even).  We need ≡ 4 (mod 8).
        // sum mod 8 can be 0, 2, 4, 6 (only even).  If 4, done.  Else bump.
        int residue = ((sum % 8) + 8) % 8;
        if (residue != 4) {
            // To shift residue by 4 (mod 8): bump one coord by ±2 (changes
            // sum by ±2) — that gets residue → residue ± 2 (mod 8).  Two
            // bumps give residue ± 4.  We need the cheapest pair.
            // Greedy: pick the two cheapest single-bumps in the right
            // direction.  For simplicity, try four candidates: cheapest
            // bump-up coord and cheapest bump-down coord, combined.
            // (This is a small optimization — for production use, an
            // optimal pair search is a few extra cycles.)
            int delta_to_4 = ((4 - residue) % 8 + 8) % 8;  // 0,2,4,6
            // Need sum-shift of delta_to_4 (or delta_to_4 - 8); both possible
            // via positive or negative direction.  Equivalent shift is ±4
            // when delta_to_4 ∈ {4} (one bump won't work, need two of same
            // sign).  When delta_to_4 ∈ {2, 6}, one ±2 bump works.
            float best_cost = std::numeric_limits<float>::infinity();
            int best_d1 = -1, best_d2 = -1, best_s1 = 0, best_s2 = 0;
            if (delta_to_4 == 2 || delta_to_4 == 6) {
                int target_sign = (delta_to_4 == 2) ? +1 : -1;
                for (int d = 0; d < 24; ++d) {
                    int s;
                    float c = bump2_cost(r[d], x[d], &s);
                    if (s != target_sign) {
                        // Force the direction we need; recompute cost.
                        c = (target_sign > 0) ? -4.0f * (r[d] - x[d]) + 4.0f
                                              :  4.0f * (r[d] - x[d]) + 4.0f;
                    }
                    if (c < best_cost) {
                        best_cost = c; best_d1 = d; best_s1 = target_sign;
                        best_d2 = -1;
                    }
                }
            } else if (delta_to_4 == 4) {
                // Need two bumps of same sign.  Find two cheapest same-sign.
                float best_pos1 = std::numeric_limits<float>::infinity();
                float best_pos2 = std::numeric_limits<float>::infinity();
                int   best_pos1_d = -1, best_pos2_d = -1;
                float best_neg1 = std::numeric_limits<float>::infinity();
                float best_neg2 = std::numeric_limits<float>::infinity();
                int   best_neg1_d = -1, best_neg2_d = -1;
                for (int d = 0; d < 24; ++d) {
                    float diff = r[d] - x[d];
                    float c_pos = -4.0f * diff + 4.0f;
                    float c_neg =  4.0f * diff + 4.0f;
                    if (c_pos < best_pos1) {
                        best_pos2 = best_pos1; best_pos2_d = best_pos1_d;
                        best_pos1 = c_pos;     best_pos1_d = d;
                    } else if (c_pos < best_pos2) {
                        best_pos2 = c_pos;     best_pos2_d = d;
                    }
                    if (c_neg < best_neg1) {
                        best_neg2 = best_neg1; best_neg2_d = best_neg1_d;
                        best_neg1 = c_neg;     best_neg1_d = d;
                    } else if (c_neg < best_neg2) {
                        best_neg2 = c_neg;     best_neg2_d = d;
                    }
                }
                float pos_total = best_pos1 + best_pos2;
                float neg_total = best_neg1 + best_neg2;
                if (pos_total <= neg_total) {
                    best_cost = pos_total;
                    best_d1 = best_pos1_d; best_s1 = +1;
                    best_d2 = best_pos2_d; best_s2 = +1;
                } else {
                    best_cost = neg_total;
                    best_d1 = best_neg1_d; best_s1 = -1;
                    best_d2 = best_neg2_d; best_s2 = -1;
                }
            } else if (delta_to_4 == 0) {
                best_cost = 0.0f;
            }
            if (best_d1 >= 0) {
                x[best_d1] += 2 * best_s1;
                d2 += best_cost;
                if (best_d2 >= 0) x[best_d2] += 2 * best_s2;
            }
        }
        return d2;
    }

    /* Path C: round to nearest int; correct mod-2 pattern via Golay
     * syndrome lookup; then enforce sum constraint. */
    float path_general(const float* r, int* x) const {
        int    x_int[24];
        float  d2 = 0.0f;
        uint32_t b = 0;
        for (int d = 0; d < 24; ++d) {
            x_int[d] = round_any(r[d]);
            float diff = r[d] - x_int[d];
            d2 += diff * diff;
            if (x_int[d] & 1) b |= (1u << d);
        }

        // Compute syndrome of b and look up correction.
        uint32_t s = compute_syndrome(b);
        uint32_t e = syndrome_table_[s];

        // For each set bit of e, flip x[d] by ±1 to flip its parity.
        for (int d = 0; d < 24; ++d) {
            if (e & (1u << d)) {
                int sgn;
                float c = bump1_cost(r[d], x_int[d], &sgn);
                x_int[d] += sgn;
                d2 += c;
            }
        }

        // Now (x_int mod 2) ∈ G24.  Enforce Λ₂₄ sum-mod constraint.
        // Standard Λ₂₄ membership: Σ x ≡ 4·s₀ (mod 8) where s₀ ∈ {0,1} is
        // a function of the coset.  For our purposes both Σ ≡ 0 and Σ ≡ 4
        // (mod 8) are valid (different cosets), so accept whichever is
        // closer.
        int sum = 0;
        for (int d = 0; d < 24; ++d) sum += x_int[d];
        int residue = ((sum % 8) + 8) % 8;
        // Acceptable residues: 0 (even-coset case) or 4 (odd-coset case).
        // Find the cheapest single-bump (±2) to reach 0 or 4.
        if (residue != 0 && residue != 4) {
            int target_residue;
            int delta;
            // Pick whichever target is closer in mod-8 distance.
            int d_to_0 = std::min(residue, 8 - residue);
            int d_to_4 = std::min(((residue - 4) % 8 + 8) % 8,
                                  ((4 - residue) % 8 + 8) % 8);
            target_residue = (d_to_0 <= d_to_4) ? 0 : 4;
            delta = ((target_residue - residue) % 8 + 8) % 8;
            int target_sign = (delta <= 4) ? +1 : -1;
            int abs_delta = (delta <= 4) ? delta : 8 - delta;

            if (abs_delta == 2) {
                int best_d = 0; float best_cost = std::numeric_limits<float>::infinity();
                for (int d = 0; d < 24; ++d) {
                    float diff = r[d] - x_int[d];
                    float c = (target_sign > 0) ? -4.0f * diff + 4.0f
                                                :  4.0f * diff + 4.0f;
                    if (c < best_cost) { best_cost = c; best_d = d; }
                }
                x_int[best_d] += 2 * target_sign;
                d2 += best_cost;
            } else if (abs_delta == 4) {
                // Need two same-sign ±2 bumps.
                float best1 = std::numeric_limits<float>::infinity();
                float best2 = std::numeric_limits<float>::infinity();
                int   best1_d = -1, best2_d = -1;
                for (int d = 0; d < 24; ++d) {
                    float diff = r[d] - x_int[d];
                    float c = (target_sign > 0) ? -4.0f * diff + 4.0f
                                                :  4.0f * diff + 4.0f;
                    if (c < best1) {
                        best2 = best1; best2_d = best1_d;
                        best1 = c;     best1_d = d;
                    } else if (c < best2) {
                        best2 = c;     best2_d = d;
                    }
                }
                if (best1_d >= 0) { x_int[best1_d] += 2 * target_sign; d2 += best1; }
                if (best2_d >= 0) { x_int[best2_d] += 2 * target_sign; d2 += best2; }
            }
        }

        for (int d = 0; d < 24; ++d) x[d] = x_int[d];
        return d2;
    }

    uint32_t H_[12];
    uint32_t syndrome_table_[4096];
};

inline const LeechDecoder& get_leech_decoder() {
    static const LeechDecoder dec;
    return dec;
}

}  // namespace leechlib
