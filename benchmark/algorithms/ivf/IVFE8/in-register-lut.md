**Role & Objective:**
You are an expert in high-performance C++ and AVX-512 SIMD optimizations. I need you to implement an ultra-fast vector distance computation pipeline known as "In-Register LUT" (or PQ Fast Scan) for an $E_8$ lattice-based vector search engine.

**Context & Data Structure:**
1. **The Database:** All database vectors have been quantized into a 1-Byte ID representing one of the 240 vertices of the $E_8$ lattice.
2. **The Query LUT:** For an incoming query, we have pre-computed its distance to all 240 $E_8$ vertices. To fit this Lookup Table (LUT) into AVX-512 registers, we have linearly quantized these 240 `float32` distances into 240 `uint8_t` values (0 to 255).
3. **The Goal:** For the inner scanning loop, scan 64 database IDs at a time, look up their corresponding 8-bit quantized distances from the LUT completely inside the CPU registers (without ANY memory/L1 cache access for the LUT), and accumulate the distances.

**Hardware Requirement:**
Assume the target CPU supports **AVX512BW** and **AVX512_VBMI** (specifically, we need the `_mm512_permutex2var_epi8` instruction).

**Implementation Details (The Inner Loop):**

**Step 1: LUT Initialization (Outside the loop)**
Load the 240-byte LUT into 4 `__m512i` (ZMM) registers.
- `zmm_lut0`: Holds distances for IDs 0 - 63.
- `zmm_lut1`: Holds distances for IDs 64 - 127.
- `zmm_lut2`: Holds distances for IDs 128 - 191.
- `zmm_lut3`: Holds distances for IDs 192 - 239 (padded with zeros up to 255).

**Step 2: The Core Pipeline (Inside the loop, processing 64 bytes/IDs per iteration)**
Please write the exact SIMD instructions following this logic:
1. **Load:** Load 64 `uint8_t` database IDs into a ZMM register (`zmm_idx`) using `_mm512_loadu_si512`.
2. **First Shuffle (IDs 0-127):** Look up distances using `_mm512_permutex2var_epi8(zmm_lut0, zmm_idx, zmm_lut1)`. Store as `zmm_dist_low`.
3. **Index Masking:** Because `_mm512_permutex2var_epi8` only looks at the lower 7 bits (0-127) for the index, we need to clear the 8th bit of our original IDs to safely look up the second half. Create `zmm_idx_masked` by bitwise ANDing `zmm_idx` with `0x7F`.
4. **Second Shuffle (IDs 128-239):** Look up distances using `_mm512_permutex2var_epi8(zmm_lut2, zmm_idx_masked, zmm_lut3)`. Store as `zmm_dist_high`.
5. **Mask Generation:** Compare the original `zmm_idx` to `127` using `_mm512_cmpgt_epi8_mask`. This generates a `__mmask64` where bits are 1 if the ID is >= 128.
6. **Blend:** Use `_mm512_mask_blend_epi8` with the generated mask to combine `zmm_dist_low` and `zmm_dist_high` into the final `zmm_final_dist`.

**Step 3: Accumulation**
Since the output distances are 8-bit, use `_mm512_sad_epu8` (Sum of Absolute Differences against zero) or a combination of unpack/add instructions to safely accumulate these 8-bit distances into 16-bit or 32-bit accumulators to prevent overflow.

**Deliverable:**
Provide the C++ function containing this inner loop. Ensure proper alignment pragmas or loading intrinsics are used. Comment the pipeline clearly. Do not use scalar fallbacks inside the block processing 64 elements. 