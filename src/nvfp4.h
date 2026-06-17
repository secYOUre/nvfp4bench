// nvfp4.h - NVFP4 (E2M1 + per-16 e4m3 block scale + global FP32 scale) helpers.
//
// Canonical layout used by the harness, the FP32 reference, and the custom
// kernel (the CUTLASS path manages its own swizzled scale-factor layout):
//   * Values: E2M1, packed two per byte (low nibble = even index).
//   * Block scale: one e4m3 byte per 16 contiguous values along K.
//   * Global scale: one FP32 per tensor.
#pragma once

#include <cstdint>
#include <vector>

constexpr int kNvfp4BlockSize = 16;

// ---- Scalar codecs (host) ------------------------------------------------
// Decode a 4-bit E2M1 code (sign<<3 | exp<<1 | mant) to float.
float e2m1_decode(uint8_t code);
// Quantize a float to the nearest E2M1 code (returns 4-bit code).
uint8_t e2m1_quantize(float x);

// FP8 E4M3 (positive scales): encode/decode.
float   e4m3_decode(uint8_t code);
uint8_t e4m3_encode(float x);

// ---- Tensor packing ------------------------------------------------------
// A dense NVFP4 tensor in canonical layout. `rows` x `cols`, blocks of 16 run
// along `cols` (the K dimension for both A and B in this harness).
struct Nvfp4Tensor {
  int rows = 0;
  int cols = 0;                  // must be a multiple of kNvfp4BlockSize
  float global_scale = 1.0f;
  std::vector<uint8_t> packed;   // rows * (cols/2) bytes
  std::vector<uint8_t> scales;   // rows * (cols/16) e4m3 bytes

  int blocks_per_row() const { return cols / kNvfp4BlockSize; }
  size_t packed_bytes() const { return packed.size(); }
  size_t scale_bytes()  const { return scales.size(); }
};

// Quantize a row-major FP32 matrix [rows x cols] into an NVFP4 tensor.
Nvfp4Tensor quantize_to_nvfp4(const std::vector<float>& src, int rows, int cols,
                              float global_scale = 1.0f);

// Dequantize back to FP32 [rows x cols] (for the validation oracle).
std::vector<float> dequantize_nvfp4(const Nvfp4Tensor& t);

// Convenience: fill with deterministic pseudo-random values in [-1, 1].
std::vector<float> make_random_matrix(int rows, int cols, uint64_t seed);
