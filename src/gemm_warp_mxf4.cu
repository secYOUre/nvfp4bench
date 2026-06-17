// gemm_warp_mxf4.cu - from-scratch PACKED NVFP4 warp-MMA GEMM (task #12).
//
// The native packed mxf4nvf4 path: 2 FP4 codes/byte, m16n8k64 -> 2x the K per MMA of
// the byte-padded mxf8f6f4 kernel (gemm_warp.cu), so ~2x the throughput. Built on the
// reverse-engineered packed fragment layout (src/mxf4_layout.cu, self-validated):
//   g=lane>>2, t=lane&3, nibble p:  A row=2g+((p>>3)&1), k=16t+8*((p>>3)>>1)+(p&7)
//                                   B col=g,             k=16t+p
//                                   D m=2g+(dreg>>1),    n=2t+(dreg&1)
// With the A-tile stored packed (8 codes / 32-bit word, lo=even-k) and the B-tile
// packed-transposed, the per-warp fragment gather and the C store are byte-for-byte
// identical to the dense kernel; only BK (32->64) and the MMA opcode change.
//
//   nvcc -gencode=arch=compute_121a,code=sm_121a -O3 -o gemm_warp_mxf4 src/gemm_warp_mxf4.cu
//   ./gemm_warp_mxf4 [M N K]
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  std::printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); std::exit(1);} }while(0)

// True hardware E2M1 decode (measured): codes 0..7 -> 0,.5,1,1.5,2,3,4,6.
__host__ __device__ inline float fp4_decode(uint8_t c) {
  const float t[8] = {0.f,0.5f,1.f,1.5f,2.f,3.f,4.f,6.f}; return t[c & 7];
}

#define BM 128
#define BN 128
#define BK 64           // packed: k64 per MMA
#define KW (BK/8)       // uint32 words along K per row (8 codes/word) = 8
#define KWP (KW)
#define WM 4
#define WN 2
#define TM 2
#define TN 8            // 128x128 block, 256-thread block, 16 MMAs/warp

__device__ inline void do_mma(float* acc, const uint32_t* a, const uint32_t* b) {
  float c0=acc[0],c1=acc[1],c2=acc[2],c3=acc[3];
  const uint32_t sf=0x00007F7Fu; const uint16_t z=0;   // 2X unit scale (0x7F=2^0)
  asm volatile(
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col."
    "f32.e2m1.e2m1.f32.ue8m0 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
    "{%14},{%15,%16},{%17},{%18,%19};\n"
    : "=f"(acc[0]),"=f"(acc[1]),"=f"(acc[2]),"=f"(acc[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),
      "f"(c0),"f"(c1),"f"(c2),"f"(c3),"r"(sf),"h"(z),"h"(z),"r"(sf),"h"(z),"h"(z));
}

// A: packed row-major, M*(K/2) bytes.  B: packed col-major (transposed), N*(K/2) bytes.
__global__ void gemm_smem(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                          float* __restrict__ C, int M, int N, int K) {
  __shared__ uint32_t As[2][BM * KWP];
  __shared__ uint32_t Bs[2][BN * KWP];
  int tilesN = N / BN;
  int blk = blockIdx.x, bm = blk / tilesN, bn = blk % tilesN;
  int baseM = bm * BM, baseN = bn * BN;
  int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  int wm = warp / WN, wn = warp % WN;
  int g = lane >> 2, t = lane & 3;
  int warpM = wm * (TM * 16), warpN = wn * (TN * 8);
  int Kb = K / 2;  // packed bytes per row/col

  float acc[TM*TN][4];
  #pragma unroll
  for (int i=0;i<TM*TN;++i) acc[i][0]=acc[i][1]=acc[i][2]=acc[i][3]=0.f;

  auto loadStage = [&](int buf, int ks) {
    int kb = ks >> 1;  // packed byte offset for this stage
    for (int idx = tid; idx < BM * KW; idx += blockDim.x) {
      int row = idx / KW, j = idx % KW;
      As[buf][row * KWP + j] = *reinterpret_cast<const uint32_t*>(A + size_t(baseM + row) * Kb + kb + j * 4);
    }
    for (int idx = tid; idx < BN * KW; idx += blockDim.x) {
      int col = idx / KW, j = idx % KW;
      Bs[buf][col * KWP + j] = *reinterpret_cast<const uint32_t*>(B + size_t(baseN + col) * Kb + kb + j * 4);
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

// pack 2 codes/byte: lo nibble = even k, hi = odd k.
static std::vector<uint8_t> pack_rowmajor(const std::vector<uint8_t>& c, int rows, int K) {
  std::vector<uint8_t> p(size_t(rows)*(K/2));
  for (int r=0;r<rows;++r) for (int b=0;b<K/2;++b)
    p[size_t(r)*(K/2)+b] = (c[size_t(r)*K+2*b]&0xF) | ((c[size_t(r)*K+2*b+1]&0xF)<<4);
  return p;
}

static void run(int M, int N, int K, bool validate) {
  std::vector<uint8_t> hA(size_t(M)*K), hB(size_t(K)*N);
  fill_codes(hA,0x111); fill_codes(hB,0x222);
  // A packed row-major [M][K/2]; B packed col-major [N][K/2] (transpose hB[k][n]->[n][k]).
  std::vector<uint8_t> hBt(size_t(N)*K);
  for (int k=0;k<K;++k) for (int n=0;n<N;++n) hBt[size_t(n)*K+k]=hB[size_t(k)*N+n];
  auto pA = pack_rowmajor(hA, M, K);
  auto pB = pack_rowmajor(hBt, N, K);
  uint8_t *dA,*dB; float* dC;
  CK(cudaMalloc(&dA,pA.size())); CK(cudaMalloc(&dB,pB.size())); CK(cudaMalloc(&dC,sizeof(float)*size_t(M)*N));
  CK(cudaMemcpy(dA,pA.data(),pA.size(),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB,pB.data(),pB.size(),cudaMemcpyHostToDevice));
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
    std::printf("time %dx%dx%d: best %.3f ms, %.1f TFLOPS (packed mxf4nvf4 warp-MMA)\n",
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
