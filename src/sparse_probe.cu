// sparse_probe.cu - does GB10/sm_121a expose a WARP-LEVEL sparse block-scaled FP4
// MMA? (task #9, step 1). 2:4 structured sparsity is where NVIDIA's 1 PFLOP NVFP4
// figure comes from. Evidence (CUTLASS docs) says sparse block-scaled FP4 is a
// datacenter tcgen05 (sm_100) feature and GB10 has no tcgen05 -- this probe tests
// it directly: if ptxas assembles the instruction, GB10 supports it (and we
// calibrate/reverse-engineer like dense); if ptxas rejects the feature, the 1 PFLOP
// is not reachable via the programmable warp MMA path on this chip.
//
//   nvcc -gencode=arch=compute_121a,code=sm_121a -o sparse_probe src/sparse_probe.cu && ./sparse_probe
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

// Sparse m16n8k64 FP4 (2:4 compresses 2x along K, so k64 vs the dense k32), with
// block scale + metadata. Operand shapes are a best guess; the point is whether
// ptxas accepts the .sp + .block_scale + mxf8f6f4 FEATURE combination at all.
__global__ void probe(float* out) {
  uint32_t a0=0x02020202,a1=a0,a2=a0,a3=a0;          // compressed A (2:4)
  uint32_t b0=0x02020202,b1=b0,b2=b0,b3=b0;          // full B (k64)
  float d0=0,d1=0,d2=0,d3=0,c0=0,c1=0,c2=0,c3=0;
  uint32_t meta = 0xEEEEEEEEu;                       // 2:4 metadata (positions 1,3 of each 4)
  const uint32_t sf = 0x81u; const uint16_t z = 0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  asm volatile(
    "mma.sp::ordered_metadata.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
    "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15},"
    "%16, 0x0, {%17},{%18,%19}, {%20},{%21,%22};\n"
    : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
    : "r"(a0),"r"(a1),"r"(a2),"r"(a3), "r"(b0),"r"(b1),"r"(b2),"r"(b3),
      "f"(c0),"f"(c1),"f"(c2),"f"(c3), "r"(meta),
      "r"(sf),"h"(z),"h"(z), "r"(sf),"h"(z),"h"(z));
#endif
  if (threadIdx.x == 0) { out[0]=d0; out[1]=d1; out[2]=d2; out[3]=d3; }
}

int main() {
  float* d; if (cudaMalloc(&d, 16) != cudaSuccess) { std::printf("malloc fail\n"); return 1; }
  cudaMemset(d, 0, 16);
  probe<<<1,32>>>(d);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) { std::printf("launch: %s\n", cudaGetErrorString(e)); return 1; }
  e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { std::printf("run: %s\n", cudaGetErrorString(e)); return 1; }
  float h[4]; cudaMemcpy(h, d, 16, cudaMemcpyDeviceToHost);
  std::printf("sparse block-scaled FP4 MMA assembled and ran. d0..3 = %.2f %.2f %.2f %.2f\n",
              h[0],h[1],h[2],h[3]);
  std::printf("=> GB10 DOES expose a warp-level sparse FP4 MMA; proceed to calibrate.\n");
  cudaFree(d);
  return 0;
}
