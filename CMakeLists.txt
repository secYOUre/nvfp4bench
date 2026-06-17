cmake_minimum_required(VERSION 3.24)

# DGX Spark / GB10 is compute capability 12.1. The arch-specific "121a" target
# is REQUIRED to unlock the NVFP4 (tcgen05 ... kind::mxf4nvf4) MMA instructions.
# Plain sm_121 will compile but will NOT emit the NVFP4 tensor-core path.
project(nvfp4bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# GB10 needs the ACCELERATED arch (the trailing "a"): sm_121a. The NVFP4
# block-scaled MMA path is gated on CUTLASS_ARCH_MMA_SM121A_ENABLED, which in
# turn requires nvcc to define __CUDA_ARCH_FEAT_SM121_ALL -- only emitted when
# the code is genuinely compiled as sm_121a. Relying on CMAKE_CUDA_ARCHITECTURES
# can silently drop the "a" (producing plain sm_121), which compiles fine but
# aborts at runtime with "Arch conditional MMA instruction used without
# targeting appropriate compute capability". So we emit the gencode explicitly.
set(NVFP4BENCH_CUDA_ARCH "121a" CACHE STRING
    "Target GPU arch suffix, e.g. 121a (GB10/DGX Spark) or 120a (GeForce RTX 50).")
# Disable CMake's own -arch handling; we add the accelerated gencode ourselves.
set(CMAKE_CUDA_ARCHITECTURES OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---- Options -------------------------------------------------------------
# Build the harness first with both OFF to confirm the foundation compiles,
# then turn ENABLE_CUTLASS on, then ENABLE_CUSTOM on, iterating as needed.
option(ENABLE_CUTLASS "Build the CUTLASS NVFP4 reference GEMM path" ON)
option(ENABLE_CUSTOM  "Build the custom tcgen05 NVFP4 GEMM kernel"  ON)
set(CUTLASS_DIR "${CMAKE_SOURCE_DIR}/third_party/cutlass" CACHE PATH
    "Path to a CUTLASS >= 4.2.1 checkout (header-only include/ used).")

# ---- Dependencies --------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

# NVML provides the GPU clock in CUDA 13, where cudaDeviceProp::clockRate was removed.
# CUDAToolkit exposes it as CUDA::nvml.
if(NOT TARGET CUDA::nvml)
  message(WARNING "CUDA::nvml target not found; clock query will fall back to a default.")
endif()

# ---- Sources -------------------------------------------------------------
# Both GEMM backend files always compile; their real bodies are guarded by the
# ENABLE_* macros below, falling back to a "not built" stub. This keeps the
# launch_gemm_* symbols defined regardless of which backends are enabled.
set(NVFP4BENCH_SOURCES
  src/main.cu
  src/device_probe.cu
  src/nvfp4.cu
  src/reference.cu
  src/bench.cu
  src/bandwidth.cu
  src/peak.cu
  src/gemm_stub.cu
  src/gemm_custom.cu
  src/gemm_cutlass.cu
)
# NOTE: gemm_autotune.cu (templated multi-tile autotuner) is intentionally NOT
# compiled. On this CUTLASS/GB10 combo the templated kernel path failed to
# register (cudaFuncSetAttribute -> kErrorInternal). The proven concrete
# single-config path in gemm_cutlass.cu serves both `cutlass` and `custom`.

add_executable(nvfp4bench ${NVFP4BENCH_SOURCES})

target_include_directories(nvfp4bench PRIVATE ${CMAKE_SOURCE_DIR}/src)

# CUDA compile flags. --expt-relaxed-constexpr & extended lambdas are needed by CUTLASS.
# The explicit accelerated gencode guarantees the "a" suffix (sm_121a) reaches nvcc.
target_compile_options(nvfp4bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_${NVFP4BENCH_CUDA_ARCH},code=sm_${NVFP4BENCH_CUDA_ARCH}>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
message(STATUS "nvfp4bench: compiling for sm_${NVFP4BENCH_CUDA_ARCH} "
               "(arch=compute_${NVFP4BENCH_CUDA_ARCH},code=sm_${NVFP4BENCH_CUDA_ARCH})")

target_link_libraries(nvfp4bench PRIVATE CUDA::cudart)
if(TARGET CUDA::nvml)
  target_link_libraries(nvfp4bench PRIVATE CUDA::nvml)
  target_compile_definitions(nvfp4bench PRIVATE NVFP4BENCH_HAVE_NVML=1)
else()
  target_compile_definitions(nvfp4bench PRIVATE NVFP4BENCH_HAVE_NVML=0)
endif()

# Optional cuBLAS for an independent FP-reference cross-check (not required).
if(TARGET CUDA::cublas)
  target_link_libraries(nvfp4bench PRIVATE CUDA::cublas)
  target_compile_definitions(nvfp4bench PRIVATE NVFP4BENCH_HAVE_CUBLAS=1)
endif()

if(ENABLE_CUTLASS)
  if(NOT EXISTS "${CUTLASS_DIR}/include/cutlass/cutlass.h")
    message(FATAL_ERROR
      "ENABLE_CUTLASS=ON but CUTLASS headers not found at ${CUTLASS_DIR}/include.\n"
      "Clone CUTLASS >= 4.2.1 there, e.g.:\n"
      "  git clone --depth 1 --branch v4.2.1 https://github.com/NVIDIA/cutlass third_party/cutlass\n"
      "or pass -DCUTLASS_DIR=/path/to/cutlass")
  endif()
  target_include_directories(nvfp4bench PRIVATE
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include)
  target_compile_definitions(nvfp4bench PRIVATE NVFP4BENCH_ENABLE_CUTLASS=1)
endif()
if(ENABLE_CUSTOM)
  target_compile_definitions(nvfp4bench PRIVATE NVFP4BENCH_ENABLE_CUSTOM=1)
endif()

install(TARGETS nvfp4bench RUNTIME DESTINATION bin)
