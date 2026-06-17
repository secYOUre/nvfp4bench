// gemm_warp.cu - from-scratch FP4 warp-MMA GEMM (task #8), on the reverse-
// engineered SM120 block-scaled fragment layout (see DESIGN.md).
//
// v3: shared-memory staged + register-blocked + vectorized fragment loads.
// Each MMA fragment register is 4 consecutive-K bytes, so storing the A-tile
// row-major and the B-tile transposed ([col][k]) makes every fragment a single
// aligned 32-bit shared load instead of 4 byte loads (v2 was byte-gather bound).
//
//   nvcc -gencode=arch=compute_121a,code=sm_121a -O3 -o gemm_warp src/gemm_warp.cu
//   ./gemm_warp [M N K]
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  std::printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); std::exit(1);} }while(0)

__host__ __device__ inline float fp4_decode(uint8_t c) { return float(c & 7) * 0.5f; }

#define BM 128
#define BN 128
#define BK 32
#define KW (BK/4)     // uint32 words along K per row (=8)
#define KWP (KW)      // row stride (padding to KW+1 regressed -> no padding)
#define WM 4
#define WN 2
#define TM 2
#define TN 8   // 128x128 block, 256-thread block, 16 MMAs/warp

__device__ inline void do_mma(float* acc, const uint32_t* a, const uint32_t* b) {
  float c0=acc[0],c1=acc[1],c2=acc[2],c3=acc[3]; const uint32_t sf=0x81u; const uint16_t z=0;
  asm volatile(
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col."
    "f32.e2m1.e2m1.f32.ue8m0 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
    "{%14},{%15,%16},{%17},{%18,%19};\n"
    : "=f"(acc[0]),"=f"(acc[1]),"=f"(acc[2]),"=f"(acc[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),
      "f"(c0),"f"(c1),"f"(c2),"f"(c3),"r"(sf),"h"(z),"h"(z),"r"(sf),"h"(z),"h"(z));
}

__global__ void gemm_smem(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                          float* __restrict__ C, int M, int N, int K) {
  __shared__ uint32_t As[2][BM * KWP];  // double-buffered, [row][kword], padded stride
  __shared__ uint32_t Bs[2][BN * KWP];  // double-buffered, [col][kword] (transposed), padded
  int tilesN = N / BN;
  int blk = blockIdx.x, bm = blk / tilesN, bn = blk % tilesN;
  int baseM = bm * BM, baseN = bn * BN;
  int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  int wm = warp / WN, wn = warp % WN;
  int g = lane >> 2, t = lane & 3;
  int warpM = wm * (TM * 16), warpN = wn * (TN * 8);

  float acc[TM*TN][4];
  #pragma unroll
  for (int i=0;i<TM*TN;++i) acc[i][0]=acc[i][1]=acc[i][2]=acc[i][3]=0.f;

  auto loadStage = [&](int buf, int ks) {
    for (int idx = tid; idx < BM * KW; idx += blockDim.x) {
      int row = idx / KW, j = idx % KW;
      As[buf][row * KWP + j] = *reinterpret_cast<const uint32_t*>(A + size_t(baseM + row) * K + ks + j * 4);
    }
    for (int idx = tid; idx < BN * KW; idx += blockDim.x) {
      int col = idx / KW, j = idx % KW; uint32_t v = 0;
      #pragma unroll
      for (int p = 0; p < 4; ++p) v |= uint32_t(B[size_t(ks + j*4 + p) * N + baseN + col]) << (p*8);
      Bs[buf][col * KWP + j] = v;
    }
  };

  int numStages = K / BK;
  loadStage(0, 0);
  __syncthreads();

  for (int s = 0; s < numStages; ++s) {
    int cur = s & 1;
    // Prefetch next stage into the other buffer (overlaps the compute below).
    if (s + 1 < numStages) loadStage(cur ^ 1, (s + 1) * BK);

    uint32_t af[TM][4], bf[TN][2];
    #pragma unroll
    for (int mi=0; mi<TM; ++mi) {
      int rowE = warpM + mi*16 + 2*g, rowO = rowE + 1;
      af[mi][0] = As[cur][rowE*KWP + t*2 + 0];
      af[mi][1] = As[cur][rowO*KWP + t*2 + 0];
      af[mi][2] = As[cur][rowE*KWP + t*2 + 1];
      af[mi][3] = As[cur][rowO*KWP + t*2 + 1];
    }
    #pragma unroll
    for (int ni=0; ni<TN; ++ni) {
      int col = warpN + ni*8 + g;
      bf[ni][0] = Bs[cur][col*KWP + t*2 + 0];
      bf[ni][1] = Bs[cur][col*KWP + t*2 + 1];
    }
    #pragma unroll
    for (int mi=0; mi<TM; ++mi)
      #pragma unroll
      for (int ni=0; ni<TN; ++ni)
        do_mma(acc[mi*TN+ni], af[mi], bf[ni]);
    __syncthreads();
  }
  #pragma unroll
  for (int mi=0; mi<TM; ++mi)
    #pragma unroll
    for (int ni=0; ni<TN; ++ni) {
      int row0 = baseM + warpM + mi*16 + 2*g;
      int col0 = baseN + warpN + ni*8 + 2*t;
      float* a = acc[mi*TN+ni];
      C[(row0+0)*N + col0+0]=a[0]; C[(row0+0)*N + col0+1]=a[1];
      C[(row0+1)*N + col0+0]=a[2]; C[(row0+1)*N + col0+1]=a[3];
    }
}

static void fill_codes(std::vector<uint8_t>& v, uint64_t seed){ uint64_t s=seed;
  for (auto& x:v){ s=s*6364136223846793005ull+1; x=uint8_t((s>>33)%8);} }

static void run(int M, int N, int K, bool validate) {
  std::vector<uint8_t> hA(size_t(M)*K), hB(size_t(K)*N);
  fill_codes(hA,0x111); fill_codes(hB,0x222);
  uint8_t *dA,*dB; float* dC;
  CK(cudaMalloc(&dA,hA.size())); CK(cudaMalloc(&dB,hB.size())); CK(cudaMalloc(&dC,sizeof(float)*size_t(M)*N));
  CK(cudaMemcpy(dA,hA.data(),hA.size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB,hB.data(),hB.size(),cudaMemcpyHostToDevice));
  dim3 grid((M/BM)*(N/BN)); dim3 block(WM*WN*32);
  gemm_smem<<<grid,block>>>(dA,dB,dC,M,N,K); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  if (validate) {
    std::vector<float> hC(size_t(M)*N);
    CK(cudaMemcpy(hC.data(),dC,sizeof(float)*size_t(M)*N,cudaMemcpyDeviceToHost));
    double maxerr=0; int nbad=0;
    for (int m=0;m<M;++m) for(int n=0;n<N;++n){ double acc=0;
      for(int k=0;k<K;++k) acc+=fp4_decode(hA[size_t(m)*K+k])*fp4_decode(hB[size_t(k)*N+n]);
      double e=std::fabs(acc-hC[size_t(m)*N+n]); if(e>maxerr)maxerr=e; if(e>1e-2)++nbad; }
    std::printf("validate %dx%dx%d: max abs err=%.4f, mismatched=%d/%d  %s\n",
                M,N,K,maxerr,nbad,M*N,nbad==0?"PASS":"FAIL");
  } else {
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); double best=1e30;
    for(int i=0;i<20;++i){ CK(cudaEventRecord(a)); gemm_smem<<<grid,block>>>(dA,dB,dC,M,N,K);
      CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b)); float ms=0; CK(cudaEventElapsedTime(&ms,a,b)); if(ms<best)best=ms; }
    std::printf("time %dx%dx%d: best %.3f ms, %.1f TFLOPS (custom warp-MMA, v3 vec-smem)\n",
                M,N,K,best,2.0*M*N*K/(best*1e-3)/1e12);
  }
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

int main(int argc, char** argv){
  run(128,128,128,true);
  int M=2048,N=2048,K=2048; if(argc==4){M=atoi(argv[1]);N=atoi(argv[2]);K=atoi(argv[3]);}
  run(M,N,K,false);
  return 0;
}
