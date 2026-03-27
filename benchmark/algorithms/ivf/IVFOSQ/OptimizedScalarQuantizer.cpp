#include "OptimizedScalarQuantizer.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace osq {

namespace {

constexpr float kMinimumMseGrid[8][2] = {
    {-0.798f, 0.798f},
    {-1.493f, 1.493f},
    {-2.051f, 2.051f},
    {-2.514f, 2.514f},
    {-2.916f, 2.916f},
    {-3.278f, 3.278f},
    {-3.611f, 3.611f},
    {-3.922f, 3.922f},
};

inline float l2_norm(const float* v, size_t d) {
  double s = 0.0;
  for (size_t i = 0; i < d; ++i) s += static_cast<double>(v[i]) * static_cast<double>(v[i]);
  return static_cast<float>(std::sqrt(s));
}

} // namespace

uint8_t EncodingConfig::doc_bits(ScalarEncoding encoding) {
  switch (encoding) {
    case ScalarEncoding::UNSIGNED_BYTE:
      return 8;
    case ScalarEncoding::PACKED_NIBBLE:
      return 4;
    case ScalarEncoding::SEVEN_BIT:
      return 7;
    case ScalarEncoding::SINGLE_BIT_QUERY_NIBBLE:
      return 1;
    case ScalarEncoding::DIBIT_QUERY_NIBBLE:
      return 2;
  }
  throw std::invalid_argument("unknown encoding");
}

uint8_t EncodingConfig::query_bits(ScalarEncoding encoding) {
  switch (encoding) {
    case ScalarEncoding::SINGLE_BIT_QUERY_NIBBLE:
    case ScalarEncoding::DIBIT_QUERY_NIBBLE:
      return 4;
    default:
      return doc_bits(encoding);
  }
}

bool EncodingConfig::is_asymmetric(ScalarEncoding encoding) {
  return doc_bits(encoding) != query_bits(encoding);
}

size_t EncodingConfig::discrete_dims(ScalarEncoding encoding, size_t dims) {
  const size_t d_bits = doc_bits(encoding);
  const size_t q_bits = query_bits(encoding);
  auto round_dims = [](size_t n, size_t bits_per_dim) {
    const size_t total_bits = n * bits_per_dim;
    const size_t rounded_bits = ((total_bits + 7) / 8) * 8;
    return rounded_bits / bits_per_dim;
  };

  if (encoding == ScalarEncoding::DIBIT_QUERY_NIBBLE) {
    // Match Lucene: force dibit storage to byte boundaries compatible with single-bit striping.
    const size_t q = round_dims(dims, 4);
    const size_t d = round_dims(dims, 1);
    return std::max(q, d);
  }

  if (d_bits == q_bits) return round_dims(dims, d_bits);
  return std::max(round_dims(dims, d_bits), round_dims(dims, q_bits));
}

size_t EncodingConfig::doc_packed_len(ScalarEncoding encoding, size_t dims) {
  const size_t dd = discrete_dims(encoding, dims);
  switch (encoding) {
    case ScalarEncoding::SEVEN_BIT:
      // Legacy OSQ stores 7-bit values one per byte instead of bit-packing.
      return dd;
    case ScalarEncoding::DIBIT_QUERY_NIBBLE:
      return 2 * ((dd + 7) / 8);
    default: {
      const size_t bits = dd * doc_bits(encoding);
      return (bits + 7) / 8;
    }
  }
}

size_t EncodingConfig::query_packed_len(ScalarEncoding encoding, size_t dims) {
  const size_t dd = discrete_dims(encoding, dims);
  const size_t bits = dd * query_bits(encoding);
  return (bits + 7) / 8;
}

float OptimizedScalarQuantizer::clampf(float x, float lo, float hi) {
  return std::max(lo, std::min(hi, x));
}

QuantizationResult OptimizedScalarQuantizer::scalar_quantize(
    const float* vector,
    size_t dims,
    uint8_t bits,
    const float* centroid,
    uint8_t* destination) const {
  if (bits == 0 || bits > 8) throw std::invalid_argument("bits must be in [1,8]");

  std::vector<float> centered(dims);
  double mean = 0.0;
  double var_acc = 0.0;
  float norm2 = 0.0f;
  float centroid_dot = 0.0f;
  float min_v = std::numeric_limits<float>::max();
  float max_v = -std::numeric_limits<float>::max();

  for (size_t i = 0; i < dims; ++i) {
    if (similarity_ != Similarity::EUCLIDEAN) centroid_dot += vector[i] * centroid[i];
    centered[i] = vector[i] - centroid[i];
    min_v = std::min(min_v, centered[i]);
    max_v = std::max(max_v, centered[i]);
    norm2 += centered[i] * centered[i];

    const double delta = centered[i] - mean;
    mean += delta / static_cast<double>(i + 1);
    var_acc += delta * (centered[i] - mean);
  }

  const double var = dims == 0 ? 0.0 : var_acc / static_cast<double>(dims);
  const double stddev = std::sqrt(std::max(0.0, var));

  float a = clampf(kMinimumMseGrid[bits - 1][0] * static_cast<float>(stddev) + static_cast<float>(mean), min_v, max_v);
  float b = clampf(kMinimumMseGrid[bits - 1][1] * static_cast<float>(stddev) + static_cast<float>(mean), min_v, max_v);

  const int points = 1 << bits;
  optimize_intervals(a, b, centered, norm2, points);

  const float steps = static_cast<float>((1u << bits) - 1u);
  const float step = (b - a) / steps;

  int sum = 0;
  for (size_t i = 0; i < dims; ++i) {
    const float xi = clampf(centered[i], a, b);
    const int q = static_cast<int>(std::lround((xi - a) / step));
    destination[i] = static_cast<uint8_t>(q);
    sum += q;
  }

  QuantizationResult res;
  res.lower_interval = a;
  res.upper_interval = b;
  res.additional_correction = (similarity_ == Similarity::EUCLIDEAN) ? norm2 : centroid_dot;
  res.quantized_component_sum = sum;
  return res;
}

std::vector<QuantizationResult> OptimizedScalarQuantizer::multi_scalar_quantize(
    const float* vector,
    size_t dims,
    const std::vector<uint8_t>& bits,
    const float* centroid,
    const std::vector<uint8_t*>& destinations) const {
  if (bits.size() != destinations.size()) {
    throw std::invalid_argument("bits and destinations must have same size");
  }

  std::vector<float> centered(dims);
  double mean = 0.0;
  double var_acc = 0.0;
  float norm2 = 0.0f;
  float centroid_dot = 0.0f;
  float min_v = std::numeric_limits<float>::max();
  float max_v = -std::numeric_limits<float>::max();

  for (size_t i = 0; i < dims; ++i) {
    if (similarity_ != Similarity::EUCLIDEAN) centroid_dot += vector[i] * centroid[i];
    centered[i] = vector[i] - centroid[i];
    min_v = std::min(min_v, centered[i]);
    max_v = std::max(max_v, centered[i]);
    norm2 += centered[i] * centered[i];

    const double delta = centered[i] - mean;
    mean += delta / static_cast<double>(i + 1);
    var_acc += delta * (centered[i] - mean);
  }

  const double var = dims == 0 ? 0.0 : var_acc / static_cast<double>(dims);
  const double stddev = std::sqrt(std::max(0.0, var));

  std::vector<QuantizationResult> out(bits.size());
  for (size_t bi = 0; bi < bits.size(); ++bi) {
    if (bits[bi] == 0 || bits[bi] > 8) throw std::invalid_argument("bits must be in [1,8]");

    float a = clampf(kMinimumMseGrid[bits[bi] - 1][0] * static_cast<float>(stddev) + static_cast<float>(mean), min_v, max_v);
    float b = clampf(kMinimumMseGrid[bits[bi] - 1][1] * static_cast<float>(stddev) + static_cast<float>(mean), min_v, max_v);
    const int points = 1 << bits[bi];
    optimize_intervals(a, b, centered, norm2, points);

    const float steps = static_cast<float>((1u << bits[bi]) - 1u);
    const float step = (b - a) / steps;

    int sum = 0;
    for (size_t i = 0; i < dims; ++i) {
      const float xi = clampf(centered[i], a, b);
      const int q = static_cast<int>(std::lround((xi - a) / step));
      destinations[bi][i] = static_cast<uint8_t>(q);
      sum += q;
    }

    out[bi] = QuantizationResult{a, b, (similarity_ == Similarity::EUCLIDEAN) ? norm2 : centroid_dot, sum};
  }

  return out;
}

void OptimizedScalarQuantizer::dequantize(
    const uint8_t* quantized,
    size_t dims,
    uint8_t bits,
    float lower_interval,
    float upper_interval,
    const float* centroid,
    float* dequantized) {
  const int nsteps = (1 << bits) - 1;
  const double step = (upper_interval - lower_interval) / static_cast<double>(nsteps);
  for (size_t i = 0; i < dims; ++i) {
    const double xi = static_cast<double>(quantized[i]) * step + lower_interval;
    dequantized[i] = static_cast<float>(xi + centroid[i]);
  }
}

double OptimizedScalarQuantizer::loss(
    const std::vector<float>& centered,
    float a,
    float b,
    int points,
    float norm2) const {
  const double step = (b - a) / (points - 1.0);
  const double step_inv = 1.0 / step;
  double xe = 0.0;
  double e = 0.0;

  for (float xif : centered) {
    const double xi = static_cast<double>(xif);
    const double xiq = a + step * std::round((clampf(xif, a, b) - a) * step_inv);
    xe += xi * (xi - xiq);
    e += (xi - xiq) * (xi - xiq);
  }
  return (1.0 - lambda_) * xe * xe / norm2 + lambda_ * e;
}

void OptimizedScalarQuantizer::optimize_intervals(
    float& a,
    float& b,
    const std::vector<float>& centered,
    float norm2,
    int points) const {
  if (!(norm2 > 0.0f)) return;

  double best_loss = loss(centered, a, b, points, norm2);
  const float scale = (1.0f - lambda_) / norm2;
  if (!std::isfinite(scale)) return;

  for (int it = 0; it < iters_; ++it) {
    const float step_inv = (points - 1.0f) / (b - a);

    double daa = 0.0;
    double dab = 0.0;
    double dbb = 0.0;
    double dax = 0.0;
    double dbx = 0.0;

    for (float xi : centered) {
      const float k = std::round((clampf(xi, a, b) - a) * step_inv);
      const float s = k / (points - 1.0f);
      daa += (1.0 - s) * (1.0 - s);
      dab += (1.0 - s) * s;
      dbb += s * s;
      dax += xi * (1.0 - s);
      dbx += xi * s;
    }

    const double m0 = scale * dax * dax + lambda_ * daa;
    const double m1 = scale * dax * dbx + lambda_ * dab;
    const double m2 = scale * dbx * dbx + lambda_ * dbb;
    const double det = m0 * m2 - m1 * m1;
    if (det == 0.0) return;

    const float a_opt = static_cast<float>((m2 * dax - m1 * dbx) / det);
    const float b_opt = static_cast<float>((m0 * dbx - m1 * dax) / det);

    if (std::fabs(a - a_opt) < 1e-8 && std::fabs(b - b_opt) < 1e-8) return;

    const double new_loss = loss(centered, a_opt, b_opt, points, norm2);
    if (new_loss > best_loss) return;

    a = a_opt;
    b = b_opt;
    best_loss = new_loss;
  }
}

size_t OptimizedScalarQuantizer::discretize(size_t value, size_t bucket) {
  return ((value + (bucket - 1)) / bucket) * bucket;
}

void OptimizedScalarQuantizer::transpose_half_byte(const uint8_t* in, size_t dims, uint8_t* out) {
  // out size must be dims / 2 bytes where dims is already discretized to multiples of 8.
  const size_t stripe = (dims + 7) / 8;
  for (size_t i = 0; i < dims; i += 8) {
    uint8_t b0 = 0, b1 = 0, b2 = 0, b3 = 0;
    for (size_t j = 0; j < 8; ++j) {
      const size_t idx = i + j;
      const uint8_t q = (idx < dims) ? in[idx] : 0;
      assert(q <= 15);
      b0 |= static_cast<uint8_t>((q & 1u) << (7 - j));
      b1 |= static_cast<uint8_t>(((q >> 1) & 1u) << (7 - j));
      b2 |= static_cast<uint8_t>(((q >> 2) & 1u) << (7 - j));
      b3 |= static_cast<uint8_t>(((q >> 3) & 1u) << (7 - j));
    }
    const size_t o = i / 8;
    out[o] = b0;
    out[o + stripe] = b1;
    out[o + 2 * stripe] = b2;
    out[o + 3 * stripe] = b3;
  }
}

void OptimizedScalarQuantizer::pack_as_binary(const uint8_t* in, size_t dims, uint8_t* out) {
  for (size_t i = 0; i < dims; i += 8) {
    uint8_t r = 0;
    for (size_t j = 0; j < 8; ++j) {
      const uint8_t v = (i + j < dims) ? in[i + j] : 0;
      assert(v <= 1);
      r |= static_cast<uint8_t>((v & 1u) << (7 - j));
    }
    out[i / 8] = r;
  }
}

void OptimizedScalarQuantizer::unpack_binary(const uint8_t* packed, size_t dims, uint8_t* out) {
  for (size_t i = 0; i < dims; ++i) {
    const uint8_t byte = packed[i / 8];
    out[i] = static_cast<uint8_t>((byte >> (7 - (i % 8))) & 1u);
  }
}

void OptimizedScalarQuantizer::transpose_dibit(const uint8_t* in, size_t dims, uint8_t* out) {
  const size_t stripe = (dims + 7) / 8;
  for (size_t i = 0; i < dims; i += 8) {
    uint8_t lo = 0, hi = 0;
    for (size_t j = 0; j < 8; ++j) {
      const size_t idx = i + j;
      const uint8_t v = (idx < dims) ? in[idx] : 0;
      assert(v <= 3);
      lo |= static_cast<uint8_t>((v & 1u) << (7 - j));
      hi |= static_cast<uint8_t>(((v >> 1) & 1u) << (7 - j));
    }
    const size_t o = i / 8;
    out[o] = lo;
    out[o + stripe] = hi;
  }
}

void OptimizedScalarQuantizer::untranspose_dibit(const uint8_t* packed, size_t dims, uint8_t* out) {
  const size_t stripe = (dims + 7) / 8;
  for (size_t i = 0; i < dims; ++i) {
    const uint8_t lo = packed[i / 8];
    const uint8_t hi = packed[i / 8 + stripe];
    const uint8_t b = static_cast<uint8_t>(7 - (i % 8));
    const uint8_t v0 = static_cast<uint8_t>((lo >> b) & 1u);
    const uint8_t v1 = static_cast<uint8_t>((hi >> b) & 1u);
    out[i] = static_cast<uint8_t>(v0 | (v1 << 1));
  }
}

} // namespace osq
