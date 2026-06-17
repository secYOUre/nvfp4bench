// common.h - shared types, error checking, small utilities.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

// ---- Error checking ------------------------------------------------------
#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      std::fprintf(stderr, "[CUDA error] %s:%d: %s (%s)\n", __FILE__, __LINE__,\
                   cudaGetErrorString(_e), #call);                            \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define CHECK_CUDA_NOEXIT(call)                                                \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      std::fprintf(stderr, "[CUDA warn] %s:%d: %s (%s)\n", __FILE__, __LINE__, \
                   cudaGetErrorString(_e), #call);                            \
    }                                                                          \
  } while (0)

// ---- GEMM problem description -------------------------------------------
enum class KernelKind { Stub, Cutlass, Custom };
enum class GemmMode { Dense, Sparse };  // Sparse == 2:4 structured sparsity on A

struct GemmProblem {
  int m = 8192;
  int n = 8192;
  int k = 8192;
  GemmMode mode = GemmMode::Dense;
};

inline const char* to_string(KernelKind k) {
  switch (k) {
    case KernelKind::Stub:    return "stub";
    case KernelKind::Cutlass: return "cutlass";
    case KernelKind::Custom:  return "custom";
  }
  return "?";
}
inline const char* to_string(GemmMode m) {
  return m == GemmMode::Dense ? "dense" : "2:4";
}

// FLOPs for an MxNxK GEMM. Dense and sparse perform the same logical FLOPs;
// sparse just does them faster by skipping structural zeros, so peak-% differs
// (compared against the sparse vs dense hardware ceiling), not the FLOP count.
inline double gemm_flops(const GemmProblem& p) {
  return 2.0 * double(p.m) * double(p.n) * double(p.k);
}

// Result of one measured GEMM configuration.
struct BenchResult {
  GemmProblem problem;
  KernelKind  kernel = KernelKind::Stub;
  double median_ms = 0.0;
  double min_ms = 0.0;
  double tflops = 0.0;       // sustained, from median time
  double peak_tflops = 0.0;  // best-case (burst), from min time -- least throttled
  double pct_dense_peak = 0.0;   // based on peak_tflops vs spec dense
  double pct_sparse_peak = 0.0;  // based on peak_tflops vs spec sparse
  bool   validated = false;
  bool   valid_pass = false;
  double max_rel_err = 0.0;
  bool   ok = false;         // kernel ran successfully
  std::string note;          // error / status message
};
