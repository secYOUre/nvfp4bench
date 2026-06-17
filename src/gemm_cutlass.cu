// gemm_cutlass.cu - CUTLASS NVFP4 block-scaled GEMM (the proven concrete path).
//
// This is the configuration that reaches ~386 TFLOPS (77% of the 500 dense spec)
// on GB10. It deliberately uses CONCRETE, namespace-scope kernel types (NOT a
// class template) and a single 128x128x128 tile: on this CUTLASS/GB10 combo the
// templated multi-tile autotuner (gemm_autotune.cu) failed to register its
// kernels (cudaFuncSetAttribute -> kErrorInternal inside initialize()).
//
// Both the `cutlass` reference and the `custom` backend route here for now; the
// higher-value "custom" work is the from-scratch warp-MMA kernel (task #8).
//
// Mirrors CUTLASS example 79a. ArchTag is Sm120 even on sm_121 (sm_121a is
// selected by the compile flag; CUTLASS dispatches via the SM120/121 macros).
#include "gemm.h"

#if !NVFP4BENCH_ENABLE_CUTLASS

bool launch_gemm_cutlass(const Nvfp4DeviceOperands&, cudaStream_t, std::string* err) {
  if (err) *err = "CUTLASS backend not built (configure with -DENABLE_CUTLASS=ON)";
  return false;
}
bool launch_gemm_autotune(const Nvfp4DeviceOperands&, cudaStream_t, std::string* err) {
  if (err) *err = "CUTLASS backend not built (configure with -DENABLE_CUTLASS=ON)";
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
#define NVFP4BENCH_CUTLASS_OK 1
#else
#define NVFP4BENCH_CUTLASS_OK 0
#endif

#if NVFP4BENCH_CUTLASS_OK

#include <memory>
#include <unordered_map>

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

using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape     = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator, ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;

struct CutlassState {
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
};

uint64_t key(int M, int N, int K) {
  return (uint64_t(M) << 42) ^ (uint64_t(N) << 21) ^ uint64_t(K);
}

typename Gemm::Arguments make_args(CutlassState* st) {
  return typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {st->M, st->N, st->K, 1},
      {st->a.get(), st->stride_a, st->b.get(), st->stride_b,
       st->a_sf.get(), st->layout_sfa, st->b_sf.get(), st->layout_sfb},
      {{1.0f, 0.0f}, st->c.get(), st->stride_c, st->d.get(), st->stride_d}};
}

CutlassState* get_or_build(std::unordered_map<uint64_t, std::unique_ptr<CutlassState>>& cache,
                           const Nvfp4DeviceOperands& op, std::string* err) {
  uint64_t kk = key(op.M, op.N, op.K);
  auto it = cache.find(kk);
  if (it != cache.end()) return it->second.get();

  auto st = std::make_unique<CutlassState>();
  st->M = op.M; st->N = op.N; st->K = op.K;
  const int M = op.M, N = op.N, K = op.K;
  st->stride_a = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
  st->stride_b = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
  st->stride_c = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
  st->stride_d = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
  st->layout_sfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  st->layout_sfb = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
  st->a.reset(size_t(M) * K);
  st->b.reset(size_t(N) * K);
  st->a_sf.reset(size_t(size(filter_zeros(st->layout_sfa))));
  st->b_sf.reset(size_t(size(filter_zeros(st->layout_sfb))));
  st->c.reset(size_t(M) * N);
  st->d.reset(size_t(M) * N);

  cutlass::Status s = st->gemm.can_implement(make_args(st.get()));
  if (s != cutlass::Status::kSuccess) {
    if (err) *err = std::string("can_implement: ") + cutlassGetStatusString(s);
    return nullptr;
  }
  st->workspace.reset(Gemm::get_workspace_size(make_args(st.get())));
  CutlassState* raw = st.get();
  cache[kk] = std::move(st);
  return raw;
}

bool run_state(CutlassState* st, cudaStream_t s, std::string* err) {
  if (!st->initialized) {
    auto args = make_args(st);
    cutlass::Status is = st->gemm.initialize(args, st->workspace.get(), s);
    if (is != cutlass::Status::kSuccess) {
      if (err) *err = std::string("initialize: ") + cutlassGetStatusString(is);
      return false;
    }
    st->initialized = true;
  }
  cutlass::Status rs = st->gemm.run(s);
  if (rs != cutlass::Status::kSuccess) {
    if (err) *err = std::string("run: ") + cutlassGetStatusString(rs);
    return false;
  }
  return true;
}

// Separate caches so `cutlass` and `custom` keep independent state instances.
std::unordered_map<uint64_t, std::unique_ptr<CutlassState>> g_cache_cutlass;
std::unordered_map<uint64_t, std::unique_ptr<CutlassState>> g_cache_custom;

bool run_shared(std::unordered_map<uint64_t, std::unique_ptr<CutlassState>>& cache,
                const Nvfp4DeviceOperands& op, cudaStream_t s, std::string* err) {
  if (op.mode == GemmMode::Sparse) {
    if (err) *err = "CUTLASS sparse NVFP4 path not yet wired (dense only)";
    return false;
  }
  CutlassState* st = get_or_build(cache, op, err);
  if (!st) return false;
  return run_state(st, s, err);
}

}  // namespace

bool launch_gemm_cutlass(const Nvfp4DeviceOperands& op, cudaStream_t s, std::string* err) {
  return run_shared(g_cache_cutlass, op, s, err);
}
// The `custom` backend currently uses the same proven config (see header note).
bool launch_gemm_autotune(const Nvfp4DeviceOperands& op, cudaStream_t s, std::string* err) {
  return run_shared(g_cache_custom, op, s, err);
}

#else  // arch not supported

bool launch_gemm_cutlass(const Nvfp4DeviceOperands&, cudaStream_t, std::string* err) {
  if (err) *err = "CUTLASS NVFP4 not supported for this compile arch (need sm_121a, CUDA>=12.9)";
  return false;
}
bool launch_gemm_autotune(const Nvfp4DeviceOperands&, cudaStream_t, std::string* err) {
  if (err) *err = "CUTLASS NVFP4 not supported for this compile arch (need sm_121a, CUDA>=12.9)";
  return false;
}

#endif  // NVFP4BENCH_CUTLASS_OK
#endif  // NVFP4BENCH_ENABLE_CUTLASS
