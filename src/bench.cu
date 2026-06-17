// bench.cu - operand setup, timing loop, validation, reporting.
#include "bench.h"
#include "gemm.h"
#include "nvfp4.h"
#include "reference.h"

#include <algorithm>
#include <cstdio>
#include <fstream>

namespace {

struct DeviceBuffers {
  uint8_t *a_packed = nullptr, *a_scales = nullptr;
  uint8_t *b_packed = nullptr, *b_scales = nullptr;
  float   *c = nullptr;
  void free() {
    cudaFree(a_packed); cudaFree(a_scales);
    cudaFree(b_packed); cudaFree(b_scales);
    cudaFree(c);
  }
};

// Quantize random A (MxK) and B (KxN) and upload. Keep host copies for the oracle.
DeviceBuffers upload_operands(const GemmProblem& p, const BenchConfig& cfg,
                              Nvfp4Tensor& At, Nvfp4Tensor& Bt,
                              std::vector<float>& Adq, std::vector<float>& Bdq,
                              bool keep_host_dequant) {
  std::vector<float> Ah = make_random_matrix(p.m, p.k, cfg.seed);
  std::vector<float> Bh = make_random_matrix(p.k, p.n, cfg.seed ^ 0xABCDEF);
  At = quantize_to_nvfp4(Ah, p.m, p.k);
  Bt = quantize_to_nvfp4(Bh, p.k, p.n);
  if (keep_host_dequant) {
    Adq = dequantize_nvfp4(At);
    Bdq = dequantize_nvfp4(Bt);
  }

  DeviceBuffers d;
  CHECK_CUDA(cudaMalloc(&d.a_packed, At.packed_bytes()));
  CHECK_CUDA(cudaMalloc(&d.a_scales, At.scale_bytes()));
  CHECK_CUDA(cudaMalloc(&d.b_packed, Bt.packed_bytes()));
  CHECK_CUDA(cudaMalloc(&d.b_scales, Bt.scale_bytes()));
  CHECK_CUDA(cudaMalloc(&d.c, sizeof(float) * size_t(p.m) * p.n));
  CHECK_CUDA(cudaMemcpy(d.a_packed, At.packed.data(), At.packed_bytes(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d.a_scales, At.scales.data(), At.scale_bytes(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d.b_packed, Bt.packed.data(), Bt.packed_bytes(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d.b_scales, Bt.scales.data(), Bt.scale_bytes(), cudaMemcpyHostToDevice));
  return d;
}

}  // namespace

BenchResult run_bench(KernelKind kind, const GemmProblem& prob,
                      const DeviceInfo& dev, const BenchConfig& cfg) {
  BenchResult r;
  r.problem = prob;
  r.kernel = kind;

  Nvfp4Tensor At, Bt;
  std::vector<float> Adq, Bdq;
  DeviceBuffers d = upload_operands(prob, cfg, At, Bt, Adq, Bdq, cfg.validate);

  Nvfp4DeviceOperands op;
  op.a_packed = d.a_packed; op.a_scales = d.a_scales; op.a_global = At.global_scale;
  op.b_packed = d.b_packed; op.b_scales = d.b_scales; op.b_global = Bt.global_scale;
  op.c = d.c;
  op.M = prob.m; op.N = prob.n; op.K = prob.k; op.mode = prob.mode;

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  // Warmup (also surfaces "not built"/unsupported errors early).
  std::string err;
  for (int i = 0; i < std::max(1, cfg.warmup); ++i) {
    if (!launch_gemm(kind, op, stream, &err)) {
      r.ok = false; r.note = err;
      CHECK_CUDA_NOEXIT(cudaStreamDestroy(stream));
      d.free();
      return r;
    }
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // Timed iterations.
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  std::vector<double> times_ms;
  times_ms.reserve(cfg.iters);
  for (int i = 0; i < cfg.iters; ++i) {
    CHECK_CUDA(cudaEventRecord(start, stream));
    launch_gemm(kind, op, stream, &err);
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    times_ms.push_back(ms);
  }
  std::sort(times_ms.begin(), times_ms.end());
  r.median_ms = times_ms[times_ms.size() / 2];
  r.min_ms = times_ms.front();
  r.tflops = gemm_flops(prob) / (r.median_ms * 1e-3) / 1e12;       // sustained
  r.peak_tflops = gemm_flops(prob) / (r.min_ms * 1e-3) / 1e12;     // burst (least throttled)
  r.pct_dense_peak  = 100.0 * r.peak_tflops / dev.spec_dense_tflops;
  r.pct_sparse_peak = 100.0 * r.peak_tflops / dev.spec_sparse_tflops;
  r.ok = true;

  // Validation against the FP32 oracle. Only the stub consumes the harness
  // operands and writes op.c; the cutlass and custom (autotuner) backends
  // generate their own operands in CUTLASS's native layout, so they are trusted
  // for numerics and timed only. (The future from-scratch warp kernel, which
  // consumes op directly, will be validated here.)
  if (cfg.validate && kind == KernelKind::Stub) {
    std::vector<float> Cgot(size_t(prob.m) * prob.n);
    CHECK_CUDA(cudaMemcpy(Cgot.data(), d.c, sizeof(float) * Cgot.size(), cudaMemcpyDeviceToHost));
    std::vector<float> Cref = reference_gemm_fp32(Adq, Bdq, prob.m, prob.n, prob.k);
    r.max_rel_err = max_rel_error(Cgot, Cref, prob.m, prob.n);
    r.validated = true;
    r.valid_pass = (r.max_rel_err <= cfg.valid_tol);
  }

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaStreamDestroy(stream));
  d.free();
  return r;
}

// ---- Reporting ------------------------------------------------------------
void print_result_header() {
  std::printf("%-7s %-18s %-6s %9s %10s %10s %8s %8s  %s\n",
              "kernel", "MxNxK", "mode", "best_ms", "peakTFLOPS", "sustTFLOPS",
              "%dense", "%sparse", "validation");
  std::printf("%s\n", std::string(104, '-').c_str());
}

void print_result(const BenchResult& r) {
  char dims[64];
  std::snprintf(dims, sizeof(dims), "%dx%dx%d", r.problem.m, r.problem.n, r.problem.k);
  if (!r.ok) {
    std::printf("%-7s %-18s %-6s   --- not run: %s\n",
                to_string(r.kernel), dims, to_string(r.problem.mode), r.note.c_str());
    return;
  }
  char val[48] = "";
  if (r.validated)
    std::snprintf(val, sizeof(val), "%s (relerr %.2e)",
                  r.valid_pass ? "PASS" : "FAIL", r.max_rel_err);
  std::printf("%-7s %-18s %-6s %9.3f %10.1f %10.1f %7.1f%% %7.2f%%  %s\n",
              to_string(r.kernel), dims, to_string(r.problem.mode),
              r.min_ms, r.peak_tflops, r.tflops, r.pct_dense_peak, r.pct_sparse_peak, val);
}

void write_csv(const std::string& path, const std::vector<BenchResult>& rows) {
  std::ofstream f(path);
  f << "kernel,M,N,K,mode,median_ms,min_ms,sust_tflops,peak_tflops,pct_dense,pct_sparse,validated,pass,max_rel_err,note\n";
  for (const auto& r : rows) {
    f << to_string(r.kernel) << ',' << r.problem.m << ',' << r.problem.n << ',' << r.problem.k
      << ',' << to_string(r.problem.mode) << ',' << r.median_ms << ',' << r.min_ms << ','
      << r.tflops << ',' << r.peak_tflops << ',' << r.pct_dense_peak << ',' << r.pct_sparse_peak << ','
      << (r.validated ? 1 : 0) << ',' << (r.valid_pass ? 1 : 0) << ',' << r.max_rel_err
      << ',' << '"' << r.note << '"' << '\n';
  }
}

void write_json(const std::string& path, const std::vector<BenchResult>& rows,
                const DeviceInfo& dev) {
  std::ofstream f(path);
  f << "{\n  \"device\": {\"name\": \"" << dev.name << "\", \"sm_count\": " << dev.sm_count
    << ", \"max_clock_mhz\": " << dev.max_sm_clock_mhz
    << ", \"spec_dense_tflops\": " << dev.spec_dense_tflops
    << ", \"spec_sparse_tflops\": " << dev.spec_sparse_tflops << "},\n  \"results\": [\n";
  for (size_t i = 0; i < rows.size(); ++i) {
    const auto& r = rows[i];
    f << "    {\"kernel\": \"" << to_string(r.kernel) << "\", \"M\": " << r.problem.m
      << ", \"N\": " << r.problem.n << ", \"K\": " << r.problem.k
      << ", \"mode\": \"" << to_string(r.problem.mode) << "\", \"median_ms\": " << r.median_ms
      << ", \"min_ms\": " << r.min_ms << ", \"sust_tflops\": " << r.tflops
      << ", \"peak_tflops\": " << r.peak_tflops << ", \"pct_dense\": " << r.pct_dense_peak
      << ", \"pct_sparse\": " << r.pct_sparse_peak << ", \"ok\": " << (r.ok ? "true" : "false")
      << ", \"validated\": " << (r.validated ? "true" : "false")
      << ", \"pass\": " << (r.valid_pass ? "true" : "false")
      << ", \"max_rel_err\": " << r.max_rel_err << "}";
    f << (i + 1 < rows.size() ? ",\n" : "\n");
  }
  f << "  ]\n}\n";
}
