// device_probe.h - device info + theoretical NVFP4 peak (CUDA 13 safe).
#pragma once

#include <string>

struct DeviceInfo {
  int         device_id = 0;
  std::string name;
  int         cc_major = 0;
  int         cc_minor = 0;
  int         sm_count = 0;
  // Clocks in MHz. max_sm_clock_mhz comes from NVML (cudaDeviceProp::clockRate
  // was removed in CUDA 13). 0 means "unavailable".
  int         max_sm_clock_mhz = 0;
  int         cur_sm_clock_mhz = 0;
  size_t      total_mem_bytes = 0;
  int         l2_cache_bytes = 0;
  double      mem_bandwidth_gbs = 273.0;  // GB10 LPDDR5x spec; overridable.

  // Theoretical peaks (TFLOPS).
  // spec_*  : NVIDIA published figures (fixed).
  // est_*   : clock-scaled estimate = sm_count * flops_per_sm_cycle * clock.
  double spec_dense_tflops  = 500.0;
  double spec_sparse_tflops = 1000.0;
  double est_dense_tflops   = 0.0;
  double est_sparse_tflops  = 0.0;
};

// Dense FP4 FLOPs per SM per clock (counts FMA as 2 FLOPs). Chosen so that
// 48 SM x 2.42 GHz reproduces ~500 TFLOPS dense; documented as an estimate.
constexpr double kDenseFp4FlopsPerSmPerCycle = 4304.0;

DeviceInfo probe_device(int device_id);
void print_device_info(const DeviceInfo& d);
