// peak.cu - pure tensor-core peak NVFP4 microbenchmark (the `--peak` mode).
//
// Issues many INDEPENDENT back-to-back FP4 MMAs from registers, with no global/
// shared memory traffic, so it measures the raw warp-MMA issue rate -> the true
// compute peak, free of the L2/bandwidth ceiling that caps the GEMM backends.
//
// Four rungs establish where NVIDIA's headline figures actually come from on GB10:
//   1. mxf8f6f4 dense  m16n8k32   - byte-padded FP4 (1 code/byte)         ~256 TFLOPS
//   2. mxf8f6f4 sparse m16n8k64   - + 2:4 structured sparsity             ~511 TFLOPS
//   3. mxf4nvf4 dense  m16n8k64   - native PACKED FP4 (2 codes/byte)      ~511 TFLOPS
//   4. mxf4nvf4 sparse m16n8k128  - packed + 2:4 sparse  = the 1 PFLOP   ~1022 TFLOPS
// Numeric model (decode/scale) is reverse-engineered in mma_probe.cu; throughput is
// data-independent so the calibration constants are irrelevant to these numbers.
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include "peak.h"

namespace {

#define NACC 16   // independent accumulators per warp -> hide MMA latency

__global__ void k_mxf8_dense(int iters, float* sink) {
  uint32_t a0=0x02020202,a1=a0,a2=a0,a3=a0,b0=a0,b1=a0;
  const uint32_t sf=0x81u; const uint16_t z=0;
  float d[NACC][4];
  #pragma unroll
  for (int j=0;j<NACC;++j) d[j][0]=d[j][1]=d[j][2]=d[j][3]=0.f;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  for (int i=0;i<iters;++i)
    #pragma unroll
    for (int j=0;j<NACC;++j)
      asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col."
        "f32.e2m1.e2m1.f32.ue8m0 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
        "{%14},{%15,%16},{%17},{%18,%19};\n"
        : "=f"(d[j][0]),"=f"(d[j][1]),"=f"(d[j][2]),"=f"(d[j][3])
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
          "f"(d[j][0]),"f"(d[j][1]),"f"(d[j][2]),"f"(d[j][3]),
          "r"(sf),"h"(z),"h"(z),"r"(sf),"h"(z),"h"(z));
#endif
  float s=0; for (int j=0;j<NACC;++j) s+=d[j][0]+d[j][1]+d[j][2]+d[j][3];
  if (s<0) sink[threadIdx.x]=s;
}

__global__ void k_mxf8_sparse(int iters, float* sink) {
  uint32_t a0=0x02020202,a1=a0,a2=a0,a3=a0,b0=a0,b1=a0,b2=a0,b3=a0;
  uint32_t meta=0xEEEEEEEEu; const uint32_t sf=0x81u; const uint16_t z=0;
  float d[NACC][4];
  #pragma unroll
  for (int j=0;j<NACC;++j) d[j][0]=d[j][1]=d[j][2]=d[j][3]=0.f;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  for (int i=0;i<iters;++i)
    #pragma unroll
    for (int j=0;j<NACC;++j)
      asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
        "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9,%10,%11},{%12,%13,%14,%15},"
        "%16,0x0,{%17},{%18,%19},{%20},{%21,%22};\n"
        : "=f"(d[j][0]),"=f"(d[j][1]),"=f"(d[j][2]),"=f"(d[j][3])
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(b2),"r"(b3),
          "f"(d[j][0]),"f"(d[j][1]),"f"(d[j][2]),"f"(d[j][3]),"r"(meta),
          "r"(sf),"h"(z),"h"(z),"r"(sf),"h"(z),"h"(z));
#endif
  float s=0; for (int j=0;j<NACC;++j) s+=d[j][0]+d[j][1]+d[j][2]+d[j][3];
  if (s<0) sink[threadIdx.x]=s;
}

__global__ void k_mxf4_dense(int iters, float* sink) {
  uint32_t a0=0x02020202,a1=a0,a2=a0,a3=a0,b0=a0,b1=a0;
  uint32_t sf=0x00008181u; const uint16_t z=0;
  float d[NACC][4];
  #pragma unroll
  for (int j=0;j<NACC;++j) d[j][0]=d[j][1]=d[j][2]=d[j][3]=0.f;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  for (int i=0;i<iters;++i)
    #pragma unroll
    for (int j=0;j<NACC;++j)
      asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col."
        "f32.e2m1.e2m1.f32.ue8m0 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
        "{%14},{%15,%16},{%17},{%18,%19};\n"
        : "=f"(d[j][0]),"=f"(d[j][1]),"=f"(d[j][2]),"=f"(d[j][3])
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
          "f"(d[j][0]),"f"(d[j][1]),"f"(d[j][2]),"f"(d[j][3]),
          "r"(sf),"h"(z),"h"(z),"r"(sf),"h"(z),"h"(z));
#endif
  float s=0; for (int j=0;j<NACC;++j) s+=d[j][0]+d[j][1]+d[j][2]+d[j][3];
  if (s<0) sink[threadIdx.x]=s;
}

__global__ void k_mxf4_sparse(int iters, float* sink) {
  uint32_t a0=0x02020202,a1=a0,a2=a0,a3=a0,b0=a0,b1=a0,b2=a0,b3=a0;
  uint32_t meta=0xEEEEEEEEu, sf=0x00008181u; const uint16_t z=0;
  float d[NACC][4];
  #pragma unroll
  for (int j=0;j<NACC;++j) d[j][0]=d[j][1]=d[j][2]=d[j][3]=0.f;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  for (int i=0;i<iters;++i)
    #pragma unroll
    for (int j=0;j<NACC;++j)
      asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X."
        "m16n8k128.row.col.f32.e2m1.e2m1.f32.ue8m0 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9,%10,%11},{%12,%13,%14,%15},"
        "%16,0x0,{%17},{%18,%19},{%20},{%21,%22};\n"
        : "=f"(d[j][0]),"=f"(d[j][1]),"=f"(d[j][2]),"=f"(d[j][3])
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(b2),"r"(b3),
          "f"(d[j][0]),"f"(d[j][1]),"f"(d[j][2]),"f"(d[j][3]),"r"(meta),
          "r"(sf),"h"(z),"h"(z),"r"(sf),"h"(z),"h"(z));
#endif
  float s=0; for (int j=0;j<NACC;++j) s+=d[j][0]+d[j][1]+d[j][2]+d[j][3];
  if (s<0) sink[threadIdx.x]=s;
}

}  // namespace

double run_peak_ladder(int device) {
  cudaSetDevice(device);
  cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, device);
  const int blocks = prop.multiProcessorCount * 8;   // saturate the GPU
  const int block  = 128;                            // 4 warps/block
  const int iters  = 700;                            // short bursts -> un-throttled peak
  const long warps = long(blocks) * (block / 32);
  float* sink; cudaMalloc(&sink, sizeof(float) * block);

  cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
  auto bench = [&](const char* name, void(*kern)(int,float*), double fpm, double spec) {
    kern<<<blocks,block>>>(iters, sink); cudaDeviceSynchronize();   // warmup
    double best = 1e30;
    for (int r=0;r<10;++r) {
      cudaEventRecord(s); kern<<<blocks,block>>>(iters, sink);
      cudaEventRecord(e); cudaEventSynchronize(e);
      float ms=0; cudaEventElapsedTime(&ms,s,e); if (ms<best) best=ms;
    }
    double tf = double(warps)*iters*NACC*fpm / (best*1e-3) / 1e12;
    std::printf("  %-16s %7.1f TFLOPS  (%5.1f%% of %.0f spec)\n", name, tf, 100.0*tf/spec, spec);
    return tf;
  };

  std::printf("\nPeak NVFP4 tensor-core throughput (register-resident, no memory traffic)\n");
  std::printf("  device: %s, %d SMs, %ld warps x %d acc x %d iters\n",
              prop.name, prop.multiProcessorCount, warps, NACC, iters);
  double d0 = bench("mxf8f6f4 dense",  k_mxf8_dense,   2.0*16*8*32,  500.0);
  double d1 = bench("mxf8f6f4 sparse", k_mxf8_sparse,  2.0*16*8*64,  1000.0);
  double d2 = bench("mxf4nvf4 dense",  k_mxf4_dense,   2.0*16*8*64,  500.0);
  double d3 = bench("mxf4nvf4 sparse", k_mxf4_sparse,  2.0*16*8*128, 1000.0);
  std::printf("  ladder: %.0f -> packed %.0f -> packed+2:4 %.0f TFLOPS"
              "  (packed %.2fx, sparse %.2fx)\n", d0, d2, d3, d2/d0, d3/d2);
  std::printf("  => GB10 reaches %.0f%% of the 500 TFLOPS dense and %.0f%% of the "
              "1 PFLOP sparse NVFP4 spec.\n", 100.0*d2/500.0, 100.0*d3/1000.0);

  cudaEventDestroy(s); cudaEventDestroy(e); cudaFree(sink);
  return d3;
}
