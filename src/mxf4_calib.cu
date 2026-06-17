// mxf4_calib.cu - calibrate the PACKED mxf4nvf4 numeric model (step 1 of the
// validated packed-GEMM path). The dense mxf8f6f4 model is solved (code*0.5 decode,
// ue8m0 0x81 = unit). mxf4nvf4 packs 2 codes/byte, so we must learn (a) the nibble
// order (which nibble is which K), (b) confirm the decode carries over, and (c) the
// ue8m0 scale base in this instruction (all-ones gave 512, not 64 -> something is
// scaled or both nibbles contribute differently than assumed).
//
// One run, many controlled experiments. A is 4xu32, B is 2xu32, each filled with a
// repeated byte. Output D[0] (one accumulator lane) is printed per experiment; the
// ratios between experiments isolate decode, nibble contribution, and scale.
//
//   nvcc -gencode=arch=compute_121a,code=sm_121a -o mxf4_calib src/mxf4_calib.cu && ./mxf4_calib
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

__global__ void run(uint32_t ab, uint32_t bb, uint32_t sfa, uint32_t sfb, float* out) {
  uint32_t a0=ab,a1=ab,a2=ab,a3=ab, b0=bb,b1=bb;
  float d0=0,d1=0,d2=0,d3=0,c0=0,c1=0,c2=0,c3=0;
  const uint16_t z=0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  asm volatile(
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col."
    "f32.e2m1.e2m1.f32.ue8m0 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
    "{%14},{%15,%16},{%17},{%18,%19};\n"
    : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
      "f"(c0),"f"(c1),"f"(c2),"f"(c3),
      "r"(sfa),"h"(z),"h"(z),"r"(sfb),"h"(z),"h"(z));
#endif
  if (threadIdx.x==0) { out[0]=d0; out[1]=d1; out[2]=d2; out[3]=d3; }
}

static float once(uint32_t ab, uint32_t bb, uint32_t sfa, uint32_t sfb) {
  float* d; cudaMalloc(&d,16); cudaMemset(d,0,16);
  run<<<1,32>>>(ab,bb,sfa,sfb,d);
  if (cudaDeviceSynchronize()!=cudaSuccess) { cudaFree(d); return -999; }
  float h[4]; cudaMemcpy(h,d,16,cudaMemcpyDeviceToHost); cudaFree(d);
  return h[0];
}

int main() {
  const uint32_t S = 0x00008181u;  // the "unit-ish" 2X scale that gave 512 for 0x02
  std::printf("packed mxf4nvf4 calibration (D[0], A=4xu32 B=2xu32 of a repeated byte)\n\n");

  std::printf("-- nibble contribution & decode (scale fixed = 0x%08X) --\n", S);
  std::printf("  A=B=0x00 : %.2f   (sanity: expect 0)\n",            once(0x00000000,0x00000000,S,S));
  std::printf("  A=B=0x02 : %.2f   (lo nibble=code2, hi=0)\n",       once(0x02020202,0x02020202,S,S));
  std::printf("  A=B=0x20 : %.2f   (hi nibble=code2, lo=0)\n",       once(0x20202020,0x20202020,S,S));
  std::printf("  A=B=0x22 : %.2f   (both nibbles=code2=1.0)\n",      once(0x22222222,0x22222222,S,S));
  std::printf("  A=B=0x11 : %.2f   (both nibbles=code1=0.5)\n",      once(0x11111111,0x11111111,S,S));
  std::printf("  A=B=0x33 : %.2f   (both nibbles=code3=1.5)\n",      once(0x33333333,0x33333333,S,S));

  std::printf("\n-- scale base (A=B=0x22 fixed; vary one ue8m0 byte) --\n");
  std::printf("  sfA=0x..8181 : %.2f   (reference)\n",  once(0x22222222,0x22222222,0x00008181u,S));
  std::printf("  sfA=0x..8182 : %.2f   (low byte +1)\n",once(0x22222222,0x22222222,0x00008182u,S));
  std::printf("  sfA=0x..8281 : %.2f   (high byte+1)\n",once(0x22222222,0x22222222,0x00008281u,S));
  std::printf("  sfA=0x..0081 : %.2f   (high byte=0)\n",once(0x22222222,0x22222222,0x00000081u,S));
  std::printf("  sfA=0x..8100 : %.2f   (low byte=0)\n", once(0x22222222,0x22222222,0x00008100u,S));
  return 0;
}
