// gemm_autotune.cu - CUTLASS NVFP4 block-scaled GEMM: both the fixed reference
// (`cutlass` backend) and the tile autotuner (`custom` backend) live HERE, in a
// single translation unit, so each kernel type is defined exactly once. (Defining
// the same GemmUniversal __global__ in two TUs deduplicates the host stub and
// breaks cudaFuncSetAttribute -> kErrorInternal on initialize.)
//
// GB10/sm_121 has no tcgen05/TMEM/2-SM MMA; NVFP4 runs on the GeForce warp-level
// block-scaled MMA. The dominant tunable is the threadblock tile within GB10's
// ~101 KB shared-memory budget, so each candidate also reports its smem need.
#include "gemm.h"

#if !(NVFP4BENCH_ENABLE_CUTLASS)

bool launch_gemm_cutlass(const Nvfp4DeviceOperands&, cudaStream_t, std::string* err) {
  if (err) *err = "CUTLASS backend not built (configure with -DENABLE_CUTLASS=ON)";
  return false;
}
bool launch_gemm_autotune(const Nvfp4DeviceOperands&, cudaStream_t, std::string* err) {
  if (err) *err = "autotune backend needs ENABLE_CUTLASS=ON";
  return false;
}

#else

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"

#if defined(__CUDA_ARCH__)
#  if defined(CUTLASS_ARCH_MMA_SM121A_ENABLED) || defined(CUTLASS_ARCH_MMA_SM120A_ENABLED)
#    pragma message("nvfp4bench: NVFP4 accelerated MMA ENABLED (SM12xA) for this arch")
#  else
#    pragma message("nvfp4bench: WARNING - NVFP4 accelerated MMA NOT enabled; build is plain sm_121 (need sm_121a)")
#  endif
#endif

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

#include <cstdio>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using namespace cute;

using ElementA   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;
using ElementC   = cutlass::bfloat16_t;
using ElementD   = cutlass::bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
using ElementAccumulator = float;
using ArchTag       = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

template <int TM, int TN, int TK>
struct Cfg {
  using TileShape    = Shape<Int<TM>, Int<TN>, Int<TK>>;
  using ClusterShape = Shape<_1, _1, _1>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutCTag, AlignmentC,
      ElementD, LayoutDTag, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, LayoutATag, AlignmentA,
      ElementB, LayoutBTag, AlignmentB,
      ElementAccumulator, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

struct IConfig {
  virtual ~IConfig() = default;
  virtual const char* name() const = 0;
  virtual size_t smem_bytes() const = 0;
  virtual bool build(int M, int N, int K, std::string* err) = 0;
  virtual bool run(cudaStream_t s, std::string* err) = 0;
};

template <int TM, int TN, int TK>
struct ConfigImpl : IConfig {
  using G = Cfg<TM, TN, TK>;
  using Gemm = typename G::Gemm;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
  using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;

  int M = 0, N = 0, K = 0;
  cutlass::device_memory::allocation<typename ElementA::DataType>        a;
  cutlass::device_memory::allocation<typename ElementA::ScaleFactorType> a_sf;
  cutlass::device_memory::allocation<typename ElementB::DataType>        b;
  cutlass::device_memory::allocation<typename ElementB::ScaleFactorType> b_sf;
  cutlass::device_memory::allocation<ElementC> c;
  cutlass::device_memory::allocation<ElementD> d;
  cutlass::device_memory::allocation<uint8_t>  workspace;
  StrideA stride_a; StrideB stride_b; StrideC stride_c; StrideD stride_d;
  LayoutSFA layout_sfa; LayoutSFB layout_sfb;
  Gemm gemm;
  bool initialized = false;
  std::string nm;

  ConfigImpl() {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%dx%dx%d", TM, TN, TK);
    nm = buf;
  }
  const char* name() const override { return nm.c_str(); }
  size_t smem_bytes() const override { return size_t(Gemm::GemmKernel::SharedStorageSize); }

  typename Gemm::Arguments make_args() {
    return typename Gemm::Arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {a.get(), stride_a, b.get(), stride_b, a_sf.get(), layout_sfa, b_sf.get(), layout_sfb},
        {{1.0f, 0.0f}, c.get(), stride_c, d.get(), stride_d}};
  }

  bool build(int M_, int N_, int K_, std::string* err) override {
    M = M_; N = N_; K = K_;
    stride_a = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    stride_b = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    stride_c = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    stride_d = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
    layout_sfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    layout_sfb = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
    a.reset(size_t(M) * K);
    b.reset(size_t(N) * K);
    a_sf.reset(size_t(size(filter_zeros(layout_sfa))));
    b_sf.reset(size_t(size(filter_zeros(layout_sfb))));
    c.reset(size_t(M) * N);
    d.reset(size_t(M) * N);
    cutlass::Status s = gemm.can_implement(make_args());
    if (s != cutlass::Status::kSuccess) {
      if (err) *err = std::string("can_implement: ") + cutlassGetStatusString(s);
      return false;
    }
    workspace.reset(Gemm::get_workspace_size(make_args()));
    return true;
  }

  bool run(cudaStream_t s, std::string* err) override {
    if (!initialized) {
      auto args = make_args();
      cutlass::Status is = gemm.initialize(args, workspace.get(), s);
      if (is != cutlass::Status::kSuccess) {
        if (err) *err = std::string("initialize: ") + cutlassGetStatusString(is) +
                        " (smem=" + std::to_string(smem_bytes()) + "B)";
        return false;
      }
      initialized = true;
    }
    cutlass::Status rs = gemm.run(s);
    if (rs != cutlass::Status::kSuccess) {
      if (err) *err = std::string("run: ") + cutlassGetStatusString(rs);
      return false;
    }
    return true;
  }
};

int device_max_optin_smem() {
  static int v = -1;
  if (v < 0) {
    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  }
  return v;
}

double time_config_ms(IConfig* c, std::string* err) {
  cudaStream_t s; cudaStreamCreate(&s);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  double best = -1.0;
  if (c->run(s, err)) {  // warmup
    cudaStreamSynchronize(s);
    best = 1e30;
    for (int i = 0; i < 5; ++i) {
      cudaEventRecord(a, s);
      if (!c->run(s, err)) { best = -1.0; break; }
      cudaEventRecord(b, s);
      cudaEventSynchronize(b);
      float ms = 0.0f; cudaEventElapsedTime(&ms, a, b);
      best = ms < best ? ms : best;
    }
  }
  cudaEventDestroy(a); cudaEventDestroy(b); cudaStreamDestroy(s);
  return best;
}

uint64_t shape_key(int M, int N, int K) {
  return (uint64_t(M) << 42) ^ (uint64_t(N) << 21) ^ uint64_t(K);
}

// Candidate tile shapes. Constraints learned on GB10:
//   * M and N must be multiples of 128 (the SM120 block-scaled scale-factor TMA
//     cannot form a valid copy for 64-wide tiles).
//   * Compiling additional large-smem kernels into the module appears to break
//     kernel registration for ALL kernels (cudaFuncSetAttribute fails inside
//     initialize, even for the small 128^3 kernel that works on its own). So we
//     start with ONLY the known-good 128x128x128, then re-introduce tiles one at
//     a time to find which the module tolerates.
std::vector<std::unique_ptr<IConfig>> make_candidates() {
  std::vector<std::unique_ptr<IConfig>> v;
  v.emplace_back(new ConfigImpl<128, 128, 128>());
  return v;
}

std::unordered_map<uint64_t, std::unique_ptr<IConfig>> g_winner;

IConfig* autotune(int M, int N, int K, std::string* err) {
  uint64_t key = shape_key(M, N, K);
  auto it = g_winner.find(key);
  if (it != g_winner.end()) return it->second.get();

  double flops = 2.0 * double(M) * double(N) * double(K);
  int optin = device_max_optin_smem();
  std::printf("[autotune %dx%dx%d] device max optin smem = %d B; trying tile configs...\n",
              M, N, K, optin);
  std::string best_name; double best_ms = 1e30;
  auto cands = make_candidates();
  for (auto& cand : cands) {
    size_t smem = cand->smem_bytes();
    if (int(smem) > optin) {
      std::printf("    %-14s smem=%6zuB  skipped (exceeds optin)\n", cand->name(), smem);
      continue;
    }
    std::string e;
    if (!cand->build(M, N, K, &e)) {
      std::printf("    %-14s smem=%6zuB  skipped (%s)\n", cand->name(), smem, e.c_str());
      continue;
    }
    double ms = time_config_ms(cand.get(), &e);
    if (ms <= 0) { std::printf("    %-14s smem=%6zuB  failed (%s)\n", cand->name(), smem, e.c_str()); continue; }
    double tflops = flops / (ms * 1e-3) / 1e12;
    bool better = ms < best_ms;
    std::printf("    %-14s smem=%6zuB  %8.3f ms  %7.1f TFLOPS%s\n",
                cand->name(), smem, ms, tflops, better ? "  <= best" : "");
    if (better) { best_ms = ms; best_name = cand->name(); }
    cand.reset();  // free this config's device memory before the next
  }
  if (best_name.empty()) { if (err) *err = "no viable tile config for this shape"; return nullptr; }

  auto winners = make_candidates();
  std::unique_ptr<IConfig> w;
  for (auto& cptr : winners) if (cptr->name() == best_name) { w = std::move(cptr); break; }
  std::string e;
  if (!w || !w->build(M, N, K, &e)) { if (err) *err = "winner rebuild failed: " + e; return nullptr; }
  std::printf("[autotune %dx%dx%d] winner: %s (%.1f TFLOPS)\n", M, N, K, best_name.c_str(),
              flops / (best_ms * 1e-3) / 1e12);
  IConfig* raw = w.get();
  g_winner[key] = std::move(w);
  return raw;
}

// Fixed 128x128x128 reference (the `cutlass` backend). Same TU, same kernel type
// as the autotuner candidate -> defined exactly once.
std::unordered_map<uint64_t, std::unique_ptr<ConfigImpl<128, 128, 128>>> g_fixed;

}  // namespace

bool launch_gemm_cutlass(const Nvfp4DeviceOperands& op, cudaStream_t s, std::string* err) {
  if (op.mode == GemmMode::Sparse) {
    if (err) *err = "CUTLASS sparse NVFP4 path not yet wired (dense only)";
    return false;
  }
  uint64_t key = shape_key(op.M, op.N, op.K);
  auto it = g_fixed.find(key);
  ConfigImpl<128, 128, 128>* c;
  if (it == g_fixed.end()) {
    auto p = std::make_unique<ConfigImpl<128, 128, 128>>();
    if (!p->build(op.M, op.N, op.K, err)) return false;
    c = p.get();
    g_fixed[key] = std::move(p);
  } else {
    c = it->second.get();
  }
  return c->run(s, err);
}

bool launch_gemm_autotune(const Nvfp4DeviceOperands& op, cudaStream_t s, std::string* err) {
  if (op.mode == GemmMode::Sparse) {
    if (err) *err = "autotune sparse NVFP4 path not yet wired (dense only)";
    return false;
  }
  IConfig* w = autotune(op.M, op.N, op.K, err);
  if (!w) return false;
  return w->run(s, err);
}

#else  // arch not supported

bool launch_gemm_cutlass(const Nvfp4DeviceOperands&, cudaStream_t, std::string* err) {
  if (err) *err = "CUTLASS NVFP4 not supported for this compile arch (need sm_121a, CUDA>=12.9)";
  return false;
}
bool launch_gemm_autotune(const Nvfp4DeviceOperands&, cudaStream_t, std::string* err) {
  if (err) *err = "autotune: NVFP4 not supported for this compile arch (need sm_121a, CUDA>=12.9)";
  return false;
}

#endif  // CUTLASS_ARCH_MMA_SM120/121_SUPPORTED
#endif  // ENABLE_CUTLASS
