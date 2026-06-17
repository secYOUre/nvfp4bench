// bench.h - timing, operand setup, reporting, bandwidth probe.
#pragma once

#include <string>
#include <vector>

#include "common.h"
#include "device_probe.h"

struct BenchConfig {
  int    iters = 50;
  int    warmup = 10;
  bool   validate = false;
  double valid_tol = 5e-2;   // NVFP4 is lossy; tolerance-based pass/fail.
  uint64_t seed = 1234;
};

// Run one (kernel, problem) configuration end to end: build operands, warm up,
// time `iters` launches, optionally validate against the FP32 oracle.
BenchResult run_bench(KernelKind kind, const GemmProblem& prob,
                      const DeviceInfo& dev, const BenchConfig& cfg);

// STREAM-triad style achievable bandwidth (GB/s) to contextualize the roofline.
double measure_bandwidth_gbs(size_t bytes = size_t(2) << 30, int iters = 30);

// Reporting.
void print_result_header();
void print_result(const BenchResult& r);
void write_csv(const std::string& path, const std::vector<BenchResult>& rows);
void write_json(const std::string& path, const std::vector<BenchResult>& rows,
                const DeviceInfo& dev);
