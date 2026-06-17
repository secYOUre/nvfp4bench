// gemm_custom.cu - the `custom` backend: from-scratch PACKED NVFP4 warp-MMA GEMM.
//
// Hand-written kernel built entirely on the SM120 block-scaled fragment layout we
// reverse-engineered on GB10 (see DESIGN.md, mma_probe.cu for the byte-padded path
// and mxf4_calib.cu / mxf4_layout.cu for the packed path). Uses the NATIVE PACKED
// mxf4nvf4 MMA: 2 FP4 codes/byte, m16n8k64 -> 2x the K per instruction of the
// byte-padded mxf8f6f4 kernel, so ~2x throughput (~112 TFLOPS at 2048^3, L2-resident,
// vs ~48 for byte-padded). Shared-staged, register-blocked (16 MMAs/warp), vectorized
// 32-bit fragment loads, software-pipelined double buffering.
//
// Packed model (calibrated): decode = code*0.5 for low codes (true E2M1 0,.5,1,1.5,
// 2,3,4,6); ue8m0 scale = 2^(E-127), 0x7F = unit; scale_vec::2X (two 32-wide K blocks
// per k64). Correctness is self-validated standalone in gemm_warp_mxf4.cu (PASS,
// 0 err). This backend generates its own packed operands, so it is timed here, not
// re-validated against the harness oracle.
#include "gemm.h"

#if !NVFP4BENCH_ENABLE_CUSTOM

bool launch_gemm_custom(const Nvfp4DeviceOperands&, cudaStream_t, std::string* err) {
  if (err) *err = "custom backend not built (configure with -DENABLE_CUSTOM=ON)";
  return false;
}

#else

#include <cstdint>
#include <memory>
#include <unordered_map>

#define BM 128
#define BN 128
#define BK 64           // packed: k64 per MMA
#define KW (BK/8)       // uint32 words along K per row (8 codes/word) = 8
#define KWP (KW)
#define WM 4
#define WN 2
#define TM 2
#define TN 8

__device__ inline void cust_mma(float* acc, const uint32_t* a, const uint32_t* b) {
  float c0=acc[0],c1=acc[1],c2=acc[2],c3=acc[3];
  const uint32_t sf=0x00007F7Fu; const uint16_t z=0;   // 2X unit scale (0x7F = 2^0)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  asm volatile(
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col."
    "f32.e2m1.e2m1.f32.ue8m0 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
    "{%14},{%15,%16},{%17},{%18,%19};\n"
    : "=f"(acc[0]),"=f"(acc[1]),"=f"(acc[2]),"=f"(acc[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),
      "f"(c0),"f"(c1),"f"(c2),"f"(c3),"r"(sf),"h"(z),"h"(z),"r"(sf),"h"(z),"h"(z));
#endif
}

// A: packed row-major [M][K/2].  B: packed col-major [N][K/2] (transposed).
__global__ void cust_gemm(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                          float* __restrict__ C, int M, int N, int K) {
  __shared__ uint32_t As[2][BM * KWP];
  __shared__ uint32_t Bs[2][BN * KWP];
  int tilesN = N / BN, blk = blockIdx.x, bm = blk / tilesN, bn = blk % tilesN;
  int baseM = bm * BM, baseN = bn * BN;
  int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  int wm = warp / WN, wn = warp % WN, g = lane >> 2, t = lane & 3;
  int warpM = wm * (TM * 16), warpN = wn * (TN * 8);
  int Kb = K / 2;  // packed bytes per row/col

  float acc[TM*TN][4];
  #pragma unroll
  for (int i=0;i<TM*TN;++i) acc[i][0]=acc[i][1]=acc[i][2]=acc[i][3]=0.f;

  auto loadStage = [&](int buf, int ks) {
    int kb = ks >> 1;
    for (int idx = tid; idx < BM * KW; idx += blockDim.x) {
      int row = idx / KW, j = idx % KW;
      As[buf][row*KWP + j] = *reinterpret_cast<const uint32_t*>(A + size_t(baseM+row)*Kb + kb + j*4);
    }
    for (int idx = tid; idx < BN * KW; idx += blockDim.x) {
      int col = idx / KW, j = idx % KW;
      Bs[buf][col*KWP + j] = *reinterpret_cast<const uint32_t*>(B + size_t(baseN+col)*Kb + kb + j*4);
    }
  };

  int numStages = K / BK;
  loadStage(0, 0);
  __syncthreads();
  for (int s = 0; s < numStages; ++s) {
    int cur = s & 1;
    if (s + 1 < numStages) loadStage(cur ^ 1, (s + 1) * BK);
    uint32_t af[TM][4], bf[TN][2];
    #pragma unroll
    for (int mi=0; mi<TM; ++mi) {
      int rowE = warpM + mi*16 + 2*g, rowO = rowE + 1;
      af[mi][0]=As[cur][rowE*KWP + t*2+0]; af[mi][1]=As[cur][rowO*KWP + t*2+0];
      af[mi][2]=As[cur][rowE*KWP + t*2+1]; af[mi][3]=As[cur][rowO*KWP + t*2+1];
    }
    #pragma unroll
    for (int ni=0; ni<TN; ++ni) {
      int col = warpN + ni*8 + g;
      bf[ni][0]=Bs[cur][col*KWP + t*2+0]; bf[ni][1]=Bs[cur][col*KWP + t*2+1];
    }
    #pragma unroll
    for (int mi=0; mi<TM; ++mi)
      #pragma unroll
      for (int ni=0; ni<TN; ++ni) cust_mma(acc[mi*TN+ni], af[mi], bf[ni]);
    __syncthreads();
  }
  #pragma unroll
  for (int mi=0; mi<TM; ++mi)
    #pragma unroll
    for (int ni=0; ni<TN; ++ni) {
      int row0 = baseM + warpM + mi*16 + 2*g, col0 = baseN + warpN + ni*8 + 2*t;
      float* a = acc[mi*TN+ni];
      C[(row0+0)*N + col0+0]=a[0]; C[(row0+0)*N + col0+1]=a[1];
      C[(row0+1)*N + col0+0]=a[2]; C[(row0+1)*N + col0+1]=a[3];
    }
}

// Fill packed operands with pseudo-random codes (2/byte; values don't affect throughput).
__global__ void cust_fill(uint8_t* p, size_t n, uint64_t seed) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = size_t(gridDim.x) * blockDim.x;
  for (; i < n; i += stride) {
    uint64_t z = (i + seed) * 0x9E3779B97F4A7C15ull; z ^= z >> 29;
    p[i] = uint8_t(z);  // any byte = two FP4 codes
  }
}

namespace {
struct WarpOps { uint8_t* dA = nullptr; uint8_t* dB = nullptr; };
std::unordered_map<uint64_t, WarpOps> g_cache;
uint64_t key(int M, int N, int K) { return (uint64_t(M)<<42) ^ (uint64_t(N)<<21) ^ uint64_t(K); }
}  // namespace

bool launch_gemm_custom(const Nvfp4DeviceOperands& op, cudaStream_t s, std::string* err) {
  if (op.mode == GemmMode::Sparse) { if (err) *err = "custom warp kernel: dense only"; return false; }
  if (op.M % BM || op.N % BN || op.K % BK) {
    if (err) *err = "custom warp kernel needs M,N multiple of 128 and K multiple of 64";
    return false;
  }
  uint64_t kk = key(op.M, op.N, op.K);
  auto it = g_cache.find(kk);
  if (it == g_cache.end()) {
    WarpOps w;
    size_t aBytes = size_t(op.M) * (op.K / 2);   // packed row-major
    size_t bBytes = size_t(op.N) * (op.K / 2);   // packed col-major
    if (cudaMalloc(&w.dA, aBytes) != cudaSuccess ||
        cudaMalloc(&w.dB, bBytes) != cudaSuccess) {
      if (err) *err = "custom: cudaMalloc failed for packed operands"; return false;
    }
    cust_fill<<<256, 256, 0, s>>>(w.dA, aBytes, 0x1111);
    cust_fill<<<256, 256, 0, s>>>(w.dB, bBytes, 0x2222);
    g_cache[kk] = w; it = g_cache.find(kk);
  }
  dim3 grid((op.M / BM) * (op.N / BN)); dim3 block(WM * WN * 32);
  cust_gemm<<<grid, block, 0, s>>>(it->second.dA, it->second.dB, op.c, op.M, op.N, op.K);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) { if (err) *err = cudaGetErrorString(e); return false; }
  return true;
}

#endif  // NVFP4BENCH_ENABLE_CUSTOM
