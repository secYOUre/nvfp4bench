// mxf4_sparse_probe.cu - the top of the ladder. Native packed NVFP4 (2 codes/byte)
// + 2:4 sparsity = m16n8k128. If GB10's warp path assembles this, it is the literal
// 1 PFLOP NVFP4 instruction (256 mxf8f6f4-dense x2 packed x2 sparse = 1024 TFLOPS).
// Operand sizes scaled from the working mxf8f6f4 sparse k64 (A 4xu32 compressed,
// B doubles to 4xu32 for k128, metadata, 2X->4X scale for the doubled K).
//
//   nvcc -gencode=arch=compute_121a,code=sm_121a -o mxf4_sparse_probe src/mxf4_sparse_probe.cu && ./mxf4_sparse_probe
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

__global__ void probe(float* out) {
  uint32_t a0=0x02020202,a1=a0,a2=a0,a3=a0;          // compressed A (k64 packed)
  uint32_t b0=0x02020202,b1=b0,b2=b0,b3=b0;          // full B (k128 packed)
  float d0=0,d1=0,d2=0,d3=0,c0=0,c1=0,c2=0,c3=0;
  uint32_t meta=0xEEEEEEEEu;                          // 2:4 metadata
  uint32_t sf=0x00008181u; const uint16_t z=0;        // 2 ue8m0 unit bytes (2X, block-32)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  asm volatile(
    "mma.sp::ordered_metadata.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X."
    "m16n8k128.row.col.f32.e2m1.e2m1.f32.ue8m0 "
    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9,%10,%11},{%12,%13,%14,%15},"
    "%16,0x0,{%17},{%18,%19},{%20},{%21,%22};\n"
    : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),"r"(b2),"r"(b3),
      "f"(c0),"f"(c1),"f"(c2),"f"(c3),"r"(meta),
      "r"(sf),"h"(z),"h"(z),"r"(sf),"h"(z),"h"(z));
#endif
  if (threadIdx.x == 0) { out[0]=d0; out[1]=d1; out[2]=d2; out[3]=d3; }
}

int main() {
  float* d; cudaMalloc(&d, 16); cudaMemset(d, 0, 16);
  probe<<<1,32>>>(d);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) { std::printf("launch: %s\n", cudaGetErrorString(e)); return 1; }
  e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { std::printf("run: %s\n", cudaGetErrorString(e)); return 1; }
  float h[4]; cudaMemcpy(h, d, 16, cudaMemcpyDeviceToHost);
  std::printf("mxf4nvf4 SPARSE k128 assembled+ran. d0..3 = %.2f %.2f %.2f %.2f\n",
              h[0],h[1],h[2],h[3]);
  std::printf("=> if this assembled, GB10 exposes the full packed+sparse NVFP4 (1 PFLOP) MMA.\n");
  cudaFree(d);
  return 0;
}
