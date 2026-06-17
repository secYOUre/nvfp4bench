// gemm.h - unified launch interface for the NVFP4 GEMM backends.
#pragma once

#include <string>
#include <cuda_runtime.h>
#include "common.h"

// Device-resident NVFP4 operands in the canonical layout (see nvfp4.h).
//   A: M x K, packed FP4 (M*(K/2) bytes), scales M*(K/16) e4m3 bytes.
//   B: K x N, packed FP4 (K*(N/2) bytes), scales (K/16)*N e4m3 bytes,
//      i.e. blocks of 16 run along K (the contraction dim) for B too.
//   C: M x N row-major FP32 output.
struct Nvfp4DeviceOperands {
  const uint8_t* a_packed = nullptr;
  const uint8_t* a_scales = nullptr;
  float          a_global = 1.0f;
  const uint8_t* b_packed = nullptr;
  const uint8_t* b_scales = nullptr;
  float          b_global = 1.0f;
  float*         c = nullptr;
  int            M = 0, N = 0, K = 0;
  GemmMode       mode = GemmMode::Dense;
};

// Each backend returns true on success. On failure it sets *err and returns
// false (e.g. backend not compiled in, or unsupported on this device).
bool launch_gemm_stub(const Nvfp4DeviceOperands& op, cudaStream_t s, std::string* err);
bool launch_gemm_cutlass(const Nvfp4DeviceOperands& op, cudaStream_t s, std::string* err);
bool launch_gemm_custom(const Nvfp4DeviceOperands& op, cudaStream_t s, std::string* err);

inline bool launch_gemm(KernelKind kind, const Nvfp4DeviceOperands& op,
                        cudaStream_t s, std::string* err) {
  switch (kind) {
    case KernelKind::Stub:    return launch_gemm_stub(op, s, err);
    case KernelKind::Cutlass: return launch_gemm_cutlass(op, s, err);
    case KernelKind::Custom:  return launch_gemm_custom(op, s, err);
  }
  if (err) *err = "unknown kernel kind";
  return false;
}
