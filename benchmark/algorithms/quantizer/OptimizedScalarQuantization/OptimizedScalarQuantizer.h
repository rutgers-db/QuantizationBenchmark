#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace osq {

enum class Similarity {
  EUCLIDEAN,
  COSINE,
  DOT_PRODUCT,
  MAX_INNER_PRODUCT,
};

enum class ScalarEncoding {
  UNSIGNED_BYTE,           // doc:8, query:8
  PACKED_NIBBLE,           // doc:4, query:4
  SEVEN_BIT,               // doc:7, query:7 (legacy compatibility)
  SINGLE_BIT_QUERY_NIBBLE, // doc:1, query:4
  DIBIT_QUERY_NIBBLE,      // doc:2, query:4
};

struct QuantizationResult {
  float lower_interval = 0.0f;
  float upper_interval = 0.0f;
  float additional_correction = 0.0f;
  int quantized_component_sum = 0;
};

struct EncodedVector {
  std::vector<uint8_t> packed;
  QuantizationResult correction;
};

class EncodingConfig {
 public:
  static uint8_t doc_bits(ScalarEncoding encoding);
  static uint8_t query_bits(ScalarEncoding encoding);
  static bool is_asymmetric(ScalarEncoding encoding);
  static size_t discrete_dims(ScalarEncoding encoding, size_t dims);
  static size_t doc_packed_len(ScalarEncoding encoding, size_t dims);
  static size_t query_packed_len(ScalarEncoding encoding, size_t dims);
};

class OptimizedScalarQuantizer {
 public:
  explicit OptimizedScalarQuantizer(Similarity similarity, float lambda = 0.1f, int iters = 5)
      : similarity_(similarity), lambda_(lambda), iters_(iters) {}

  QuantizationResult scalar_quantize(
      const float* vector,
      size_t dims,
      uint8_t bits,
      const float* centroid,
      uint8_t* destination) const;

  std::vector<QuantizationResult> multi_scalar_quantize(
      const float* vector,
      size_t dims,
      const std::vector<uint8_t>& bits,
      const float* centroid,
      const std::vector<uint8_t*>& destinations) const;

  static void dequantize(
      const uint8_t* quantized,
      size_t dims,
      uint8_t bits,
      float lower_interval,
      float upper_interval,
      const float* centroid,
      float* dequantized);

  static size_t discretize(size_t value, size_t bucket);

  static void transpose_half_byte(const uint8_t* in, size_t dims, uint8_t* out);
  static void pack_as_binary(const uint8_t* in, size_t dims, uint8_t* out);
  static void unpack_binary(const uint8_t* packed, size_t dims, uint8_t* out);
  static void transpose_dibit(const uint8_t* in, size_t dims, uint8_t* out);
  static void untranspose_dibit(const uint8_t* packed, size_t dims, uint8_t* out);

 private:
  double loss(const std::vector<float>& centered, float a, float b, int points, float norm2) const;

  void optimize_intervals(
      float& a,
      float& b,
      const std::vector<float>& centered,
      float norm2,
      int points) const;

  static float clampf(float x, float lo, float hi);

  Similarity similarity_;
  float lambda_;
  int iters_;
};

} // namespace osq
