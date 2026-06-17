// main.cu - CLI for nvfp4bench.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "bench.h"
#include "common.h"
#include "device_probe.h"
#include "peak.h"

namespace {

void usage(const char* prog) {
  std::printf(
    "nvfp4bench - measure peak NVFP4 GEMM performance on NVIDIA DGX Spark (GB10)\n\n"
    "Usage: %s [options]\n\n"
    "  --kernel  stub|cutlass|custom|all   backends to run (default: all built)\n"
    "  --mode    dense|sparse|both         NVFP4 dense and/or 2:4 sparse (default: both)\n"
    "  --m N --n N --k N                   single GEMM size (default 4096x14336x4096,\n"
    "                                       the GB10 NVFP4 peak shape)\n"
    "  --sweep                            run a preset size sweep\n"
    "  --iters N                          timed iterations (default 50)\n"
    "  --warmup N                         warmup iterations (default 10)\n"
    "  --validate                         check vs FP32 oracle (use small sizes)\n"
    "  --tol F                            validation rel-error tolerance (default 0.05)\n"
    "  --peak                             measure pure tensor-core NVFP4 peak (the MMA\n"
    "                                       throughput ladder; the 1 PFLOP headline)\n"
    "  --bandwidth                        also run the memory-bandwidth roofline probe\n"
    "  --device N                         CUDA device index (default 0)\n"
    "  --csv FILE / --json FILE           write results\n"
    "  -h, --help                         this help\n", prog);
}

bool eq(const char* a, const char* b) { return std::strcmp(a, b) == 0; }

}  // namespace

int main(int argc, char** argv) {
  std::string kernel_sel = "all";
  std::string mode_sel = "both";
  // Default to the shape that peaks on GB10 (short kernel => least throttling).
  int m = 4096, n = 14336, k = 4096;
  bool sweep = false, do_bw = false, do_peak = false;
  int device = 0;
  BenchConfig cfg;
  std::string csv_path, json_path;

  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    auto next = [&](const char* name) -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", name); std::exit(2); }
      return argv[++i];
    };
    if (eq(a, "-h") || eq(a, "--help")) { usage(argv[0]); return 0; }
    else if (eq(a, "--kernel")) kernel_sel = next("--kernel");
    else if (eq(a, "--mode")) mode_sel = next("--mode");
    else if (eq(a, "--m")) m = std::atoi(next("--m"));
    else if (eq(a, "--n")) n = std::atoi(next("--n"));
    else if (eq(a, "--k")) k = std::atoi(next("--k"));
    else if (eq(a, "--sweep")) sweep = true;
    else if (eq(a, "--iters")) cfg.iters = std::atoi(next("--iters"));
    else if (eq(a, "--warmup")) cfg.warmup = std::atoi(next("--warmup"));
    else if (eq(a, "--validate")) cfg.validate = true;
    else if (eq(a, "--tol")) cfg.valid_tol = std::atof(next("--tol"));
    else if (eq(a, "--peak")) do_peak = true;
    else if (eq(a, "--bandwidth")) do_bw = true;
    else if (eq(a, "--device")) device = std::atoi(next("--device"));
    else if (eq(a, "--csv")) csv_path = next("--csv");
    else if (eq(a, "--json")) json_path = next("--json");
    else { std::fprintf(stderr, "unknown arg: %s\n", a); usage(argv[0]); return 2; }
  }

  DeviceInfo dev = probe_device(device);
  print_device_info(dev);

  if (do_bw) {
    double gbs = measure_bandwidth_gbs();
    std::printf("Memory bandwidth (triad): %.0f GB/s  (spec %.0f GB/s)\n\n",
                gbs, dev.mem_bandwidth_gbs);
  }

  // --peak is a dedicated mode: the pure tensor-core NVFP4 throughput ladder,
  // including the packed+2:4-sparse rung that reaches the 1 PFLOP headline. It is
  // memory-free by design, so it is reported on its own rather than per-GEMM-shape.
  if (do_peak) {
    run_peak_ladder(device);
    return 0;
  }

  // Build kernel list.
  std::vector<KernelKind> kernels;
  auto want = [&](const char* s) { return kernel_sel == "all" || kernel_sel == s; };
  if (want("stub"))    kernels.push_back(KernelKind::Stub);
  if (want("cutlass")) kernels.push_back(KernelKind::Cutlass);
  if (want("custom"))  kernels.push_back(KernelKind::Custom);
  if (kernels.empty()) { std::fprintf(stderr, "no kernel selected\n"); return 2; }

  // Build mode list.
  std::vector<GemmMode> modes;
  if (mode_sel == "dense" || mode_sel == "both") modes.push_back(GemmMode::Dense);
  if (mode_sel == "sparse" || mode_sel == "both") modes.push_back(GemmMode::Sparse);

  // Build problem sizes.
  std::vector<GemmProblem> sizes;
  if (sweep) {
    // Square shapes plus the non-square 4096x14336x4096 that is reported to peak
    // on GB10 (large N, shorter-running kernel => less thermal throttling).
    sizes.push_back(GemmProblem{4096, 14336, 4096, GemmMode::Dense});
    sizes.push_back(GemmProblem{4096, 28672, 4096, GemmMode::Dense});
    sizes.push_back(GemmProblem{6144, 6144, 6144, GemmMode::Dense});
    for (int s : {2048, 4096, 8192, 12288, 16384})
      sizes.push_back(GemmProblem{s, s, s, GemmMode::Dense});
  } else {
    sizes.push_back(GemmProblem{m, n, k, GemmMode::Dense});
  }

  print_result_header();
  std::vector<BenchResult> rows;
  for (auto sz : sizes) {
    for (auto mode : modes) {
      GemmProblem p = sz;
      p.mode = mode;
      for (auto kk : kernels) {
        BenchResult r = run_bench(kk, p, dev, cfg);
        print_result(r);
        rows.push_back(r);
      }
    }
  }

  if (!csv_path.empty())  { write_csv(csv_path, rows);  std::printf("\nWrote %s\n", csv_path.c_str()); }
  if (!json_path.empty()) { write_json(json_path, rows, dev); std::printf("Wrote %s\n", json_path.c_str()); }
  return 0;
}
