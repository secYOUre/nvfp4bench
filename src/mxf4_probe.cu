// mxf4_probe.cu - the missing 2x. We've been measuring kind::mxf8f6f4 (byte-padded
// FP4, 1 code/byte, m16n8k32 -> 256 TFLOPS ceiling). The NATIVE NVFP4 path is
// kind::mxf4nvf4: 2 codes/byte, packed, m16n8k64 (dense) -> 2x the K per instruction
// at the same issue rate. That's the 2x that separates our 256 from CUTLASS's 375 and
// underpins the 1 PFLOP figure (256 x2 packed x2 sparse = 1024). This probe assembles
// the dense mxf4nvf4 k64 MMA and checks the all-ones result == 64.00 (vs 32 for k32).
//
//   nvcc -gencode=arch=compute_121a,code=sm_121a -o mxf4_probe src/mxf4_probe.cu && ./mxf4_probe
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

// Variant A: scale_vec::2X with ue8m0 (block 32 -> 2 scales across k64).
__global__ void probe2x(float* out) {
  uint32_t a0=0x02020202,a1=a0,a2=a0,a3=a0;   // A: 4xu32, 2 codes/byte -> k64
  uint32_t b0=0x02020202,b1=b0;               // B: 2xu32
  float d0=0,d1=0,d2=0,d3=0,c0=0,c1=0,c2=0,c3=0;
  uint32_t sf = 0x00008181u;                  // two ue8m0 unit bytes (0x81) for 2X
  const uint16_t z = 0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  asm volatile(
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col."
    "f32.e2m1.e2m1.f32.ue8m0 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
    "{%14},{%15,%16},{%17},{%18,%19};\n"
    : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
      "f"(c0),"f"(c1),"f"(c2),"f"(c3),
      "r"(sf),"h"(z),"h"(z),"r"(sf),"h"(z),"h"(z));
#endif
  if (threadIdx.x == 0) { out[0]=d0; out[1]=d1; out[2]=d2; out[3]=d3; }
}

int main() {
  float* d; cudaMalloc(&d, 16); cudaMemset(d, 0, 16);
  probe2x<<<1,32>>>(d);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) { std::printf("2X launch: %s\n", cudaGetErrorString(e)); return 1; }
  e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { std::printf("2X run: %s\n", cudaGetErrorString(e)); return 1; }
  float h[4]; cudaMemcpy(h, d, 16, cudaMemcpyDeviceToHost);
  std::printf("mxf4nvf4 dense k64 (2X/ue8m0) assembled+ran. d0..3 = %.2f %.2f %.2f %.2f\n",
              h[0],h[1],h[2],h[3]);
  std::printf("expect 64.00 (k64 all-ones) -> packed FP4 is the 2x path to 1 PFLOP.\n");
  cudaFree(d);
  return 0;
}
