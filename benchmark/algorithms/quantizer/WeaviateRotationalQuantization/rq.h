// Rotational Quantization (RQ) — C++ implementation.
//
// Matches Weaviate's two code paths:
//   bits = 1        → BRQ: 1-bit sign data + 5-bit bit-sliced asymmetric query.
//   bits = 2, 4, 8  → uniform RQ: 1 byte per dim (data & query, symmetric),
//                     quantized to 2^bits levels.
//
// Interface is faiss-style: train / add / search.

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

namespace rq {

enum class Metric { L2, IP };

class RotationalQuantizer {
 public:
  RotationalQuantizer(int d, int bits = 8,
                      Metric metric = Metric::L2,
                      uint64_t seed = 0x517cc1b727220a95ULL);

  void train(size_t n, const float* x);           // no-op (RQ is data-independent)
  void add(size_t n, const float* x);
  void search(size_t nq, const float* queries, size_t k,
              float* distances, int64_t* labels) const;
  void reset();

  size_t ntotal() const { return ntotal_; }
  int d() const { return d_; }
  int out_dim() const { return out_d_; }
  int bits() const { return bits_; }
  size_t code_size() const { return code_bytes_; }
  Metric metric() const { return metric_; }

 private:
  int d_;
  int out_d_;
  int bits_;
  Metric metric_;
  int rounds_;
  static constexpr int kBlock = 64;
  static constexpr int kBrqMinDim = 256;   // Weaviate pads BRQ input to ≥256.

  std::vector<std::vector<int>>   swap_a_, swap_b_;
  std::vector<std::vector<float>> signs_;

  // Randomized rounding table used by BRQ query quantization (bits=1).
  std::vector<float> rounding_;

  size_t code_bytes_;      // per-data-vector code size
  size_t ntotal_ = 0;
  std::vector<uint8_t> codes_;
  std::vector<float>   meta_;  // 4 floats per vector

  void rotate(const float* x, float* out) const;

  // Uniform RQ path (bits=2, 4, 8): 1 byte per dim for both data and query.
  void encode_uniform(const float* rx, uint8_t* code, float* meta) const;
  float dist_uniform(const uint8_t* cx, const float* mx,
                     const uint8_t* cq, const float* mq) const;

  // BRQ path (bits=1):
  //   data  = 1-bit sign code, stored as uint64 words in `code`,
  //           meta = {step = ‖rx‖² / ‖rx‖₁,  ‖rx‖²}
  //   query = 5 bit-planes (bits0..bits4), stored as 5 * (out_d_/64) uint64 words,
  //           plus step = max|rx| / 31 and ‖rx‖².
  void encode_brq_data(const float* rx, uint8_t* code, float* meta) const;
  void encode_brq_query(const float* rx,
                        uint64_t* qbits_out,
                        float& step_out, float& norm2_out) const;
  float dist_brq(const uint8_t* cx, const float* mx,
                 const uint64_t* qbits, float q_step, float q_norm2) const;
};

}  // namespace rq
