// nvfp4.cu - host-side NVFP4 codecs and tensor packing.
#include "nvfp4.h"

#include <algorithm>
#include <cmath>
#include <cstring>

// ---- E2M1 -----------------------------------------------------------------
// Positive E2M1 magnitudes indexed by (exp<<1 | mant), exp in [0,3], mant 0/1.
static const float kE2m1Mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
constexpr float kE2m1Max = 6.0f;

float e2m1_decode(uint8_t code) {
  float mag = kE2m1Mag[code & 0x7];
  return (code & 0x8) ? -mag : mag;
}

uint8_t e2m1_quantize(float x) {
  uint8_t sign = (x < 0.0f) ? 0x8 : 0x0;
  float a = std::fabs(x);
  // Nearest representable magnitude (round-half-to-even-ish via midpoints).
  int best = 0;
  float best_err = std::fabs(a - kE2m1Mag[0]);
  for (int i = 1; i < 8; ++i) {
    float e = std::fabs(a - kE2m1Mag[i]);
    if (e < best_err) { best_err = e; best = i; }
  }
  return sign | uint8_t(best);
}

// ---- E4M3 (positive scales) ----------------------------------------------
// Minimal, correctness-oriented (not bit-exact-IEEE) E4M3 codec for scales.
// E4M3: 1 sign, 4 exponent (bias 7), 3 mantissa. Max finite 448, min normal 2^-6.
float e4m3_decode(uint8_t code) {
  int sign = (code >> 7) & 0x1;
  int exp  = (code >> 3) & 0xF;
  int man  = code & 0x7;
  float val;
  if (exp == 0) {
    val = std::ldexp(float(man) / 8.0f, 1 - 7);          // subnormal
  } else {
    val = std::ldexp(1.0f + float(man) / 8.0f, exp - 7); // normal
  }
  return sign ? -val : val;
}

uint8_t e4m3_encode(float x) {
  if (x <= 0.0f) return 0;            // scales are positive; clamp <=0 to 0
  x = std::min(x, 448.0f);
  // Search the 0..126 positive code space for the nearest value (small table).
  uint8_t best = 0;
  float best_err = 1e30f;
  for (int c = 0; c < 128; ++c) {
    float v = e4m3_decode(uint8_t(c));
    float e = std::fabs(v - x);
    if (e < best_err) { best_err = e; best = uint8_t(c); }
  }
  return best;
}

// ---- Packing --------------------------------------------------------------
Nvfp4Tensor quantize_to_nvfp4(const std::vector<float>& src, int rows, int cols,
                              float global_scale) {
  Nvfp4Tensor t;
  t.rows = rows;
  t.cols = cols;
  t.global_scale = global_scale;
  t.packed.assign(size_t(rows) * (cols / 2), 0);
  t.scales.assign(size_t(rows) * (cols / kNvfp4BlockSize), 0);

  for (int r = 0; r < rows; ++r) {
    for (int b = 0; b < cols / kNvfp4BlockSize; ++b) {
      int c0 = b * kNvfp4BlockSize;
      // Per-block absmax -> scale so the largest element maps near E2M1 max.
      float amax = 0.0f;
      for (int j = 0; j < kNvfp4BlockSize; ++j)
        amax = std::max(amax, std::fabs(src[size_t(r) * cols + c0 + j]));
      float scale = (amax > 0.0f) ? (amax / kE2m1Max) / global_scale : 1.0f;
      uint8_t scale_code = e4m3_encode(scale);
      float dec_scale = e4m3_decode(scale_code) * global_scale;
      if (dec_scale <= 0.0f) dec_scale = 1.0f;
      t.scales[size_t(r) * (cols / kNvfp4BlockSize) + b] = scale_code;

      for (int j = 0; j < kNvfp4BlockSize; ++j) {
        int c = c0 + j;
        float q = src[size_t(r) * cols + c] / dec_scale;
        uint8_t code = e2m1_quantize(q);
        size_t byte_idx = (size_t(r) * cols + c) / 2;
        if ((c & 1) == 0)
          t.packed[byte_idx] = (t.packed[byte_idx] & 0xF0) | (code & 0x0F);
        else
          t.packed[byte_idx] = (t.packed[byte_idx] & 0x0F) | uint8_t((code & 0x0F) << 4);
      }
    }
  }
  return t;
}

std::vector<float> dequantize_nvfp4(const Nvfp4Tensor& t) {
  std::vector<float> out(size_t(t.rows) * t.cols);
  for (int r = 0; r < t.rows; ++r) {
    for (int c = 0; c < t.cols; ++c) {
      size_t byte_idx = (size_t(r) * t.cols + c) / 2;
      uint8_t byte = t.packed[byte_idx];
      uint8_t code = (c & 1) ? (byte >> 4) : (byte & 0x0F);
      int b = c / kNvfp4BlockSize;
      float scale = e4m3_decode(t.scales[size_t(r) * (t.cols / kNvfp4BlockSize) + b]) * t.global_scale;
      out[size_t(r) * t.cols + c] = e2m1_decode(code) * scale;
    }
  }
  return out;
}

// ---- Random matrix --------------------------------------------------------
std::vector<float> make_random_matrix(int rows, int cols, uint64_t seed) {
  std::vector<float> v(size_t(rows) * cols);
  // splitmix64 for deterministic, fast fill.
  uint64_t x = seed ? seed : 0x9E3779B97F4A7C15ull;
  for (size_t i = 0; i < v.size(); ++i) {
    x += 0x9E3779B97F4A7C15ull;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z = z ^ (z >> 31);
    // Map to [-1, 1].
    v[i] = (float(z >> 40) / float(1 << 24)) * 2.0f - 1.0f;
  }
  return v;
}
