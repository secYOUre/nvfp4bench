// device_probe.cu - query device, compute theoretical NVFP4 peak.
#include "device_probe.h"
#include "common.h"

#include <cuda_runtime.h>

#if NVFP4BENCH_HAVE_NVML
#include <nvml.h>
#endif

// Query SM clock via NVML. Returns false if unavailable. CUDA 13 removed
// cudaDeviceProp::clockRate, so NVML is the supported path.
static bool query_clocks_nvml(int device_id, int& max_mhz, int& cur_mhz) {
#if NVFP4BENCH_HAVE_NVML
  if (nvmlInit_v2() != NVML_SUCCESS) return false;
  bool ok = false;
  nvmlDevice_t dev;
  if (nvmlDeviceGetHandleByIndex_v2(static_cast<unsigned>(device_id), &dev) == NVML_SUCCESS) {
    unsigned int mx = 0, cur = 0;
    bool a = (nvmlDeviceGetMaxClockInfo(dev, NVML_CLOCK_SM, &mx) == NVML_SUCCESS);
    bool b = (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &cur) == NVML_SUCCESS);
    if (a) max_mhz = static_cast<int>(mx);
    if (b) cur_mhz = static_cast<int>(cur);
    ok = a || b;
  }
  nvmlShutdown();
  return ok;
#else
  (void)device_id; (void)max_mhz; (void)cur_mhz;
  return false;
#endif
}

DeviceInfo probe_device(int device_id) {
  DeviceInfo d;
  d.device_id = device_id;
  CHECK_CUDA(cudaSetDevice(device_id));

  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device_id));
  d.name           = prop.name;
  d.cc_major       = prop.major;
  d.cc_minor       = prop.minor;
  d.sm_count       = prop.multiProcessorCount;
  d.total_mem_bytes = prop.totalGlobalMem;
  d.l2_cache_bytes = prop.l2CacheSize;

  // Clocks via NVML (CUDA 13: no cudaDeviceProp::clockRate).
  int max_mhz = 0, cur_mhz = 0;
  if (query_clocks_nvml(device_id, max_mhz, cur_mhz)) {
    d.max_sm_clock_mhz = max_mhz;
    d.cur_sm_clock_mhz = cur_mhz;
  } else {
    // Fallback to GB10 nominal so the estimate is still meaningful.
    d.max_sm_clock_mhz = 2420;
    d.cur_sm_clock_mhz = 0;
  }

  // Clock-scaled estimate of dense/sparse FP4 peak.
  const double clk_hz = double(d.max_sm_clock_mhz) * 1.0e6;
  d.est_dense_tflops  = (double(d.sm_count) * kDenseFp4FlopsPerSmPerCycle * clk_hz) / 1.0e12;
  d.est_sparse_tflops = 2.0 * d.est_dense_tflops;

  return d;
}

void print_device_info(const DeviceInfo& d) {
  std::printf("Device %d: %s  (sm_%d%d)\n", d.device_id, d.name.c_str(), d.cc_major, d.cc_minor);
  std::printf("  SMs               : %d\n", d.sm_count);
  if (d.max_sm_clock_mhz > 0)
    std::printf("  Max SM clock      : %.2f GHz%s\n", d.max_sm_clock_mhz / 1000.0,
                d.cur_sm_clock_mhz == 0 ? " (NVML unavailable, nominal)" : "");
  if (d.cur_sm_clock_mhz > 0)
    std::printf("  Current SM clock  : %.2f GHz\n", d.cur_sm_clock_mhz / 1000.0);
  std::printf("  Total memory      : %.1f GB\n", d.total_mem_bytes / (1024.0 * 1024.0 * 1024.0));
  std::printf("  L2 cache          : %.1f MB\n", d.l2_cache_bytes / (1024.0 * 1024.0));
  std::printf("  Mem bandwidth(spec): %.0f GB/s\n", d.mem_bandwidth_gbs);
  std::printf("  NVFP4 peak (spec)  : %.0f TFLOPS dense | %.0f TFLOPS 2:4 sparse\n",
              d.spec_dense_tflops, d.spec_sparse_tflops);
  std::printf("  NVFP4 peak (est)   : %.1f TFLOPS dense | %.1f TFLOPS 2:4 sparse  (clock-scaled)\n",
              d.est_dense_tflops, d.est_sparse_tflops);
  if (d.cc_major != 12 || d.cc_minor != 1) {
    std::printf("  [warn] expected compute capability 12.1 (GB10); NVFP4 path may be unavailable.\n");
  }
  std::printf("\n");
}
