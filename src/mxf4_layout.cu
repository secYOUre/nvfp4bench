// mxf4_layout.cu - packed mxf4nvf4 (k64) fragment-layout extractor. Adapts the proven
// dense mxf8f6f4 extractor (mma_probe.cu) to the PACKED format where each byte holds
// TWO FP4 codes (lo/hi nibble), so the physical slot is a NIBBLE: A has 32 lanes x 32
// nibbles = 1024 slots, B has 32 x 16 = 512 slots, K=64. Same self-calibrating method:
//   1) one-hot A nibble (B all ones) -> lit D set reveals its ROW.
//   2) one-hot B nibble (A all ones) -> lit D set reveals its COL.
//   3) one-hot A x one-hot B          -> pairs A/B nibbles sharing a K.
//   4) random tile through the maps, validated against the hardware's own decode.
// Calibrated model (src/mxf4_calib.cu): decode = code*0.5; ue8m0 scale = 2^(E-127),
// so 0x7F = unit; scale_vec::2X uses two unit bytes -> 0x00007F7F.
//
//   nvcc -gencode=arch=compute_121a,code=sm_121a -o mxf4_layout src/mxf4_layout.cu && ./mxf4_layout
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <utility>
#include <cuda_runtime.h>

// A: 32 lanes x 16 bytes (=512), each byte = 2 nibbles -> 1024 A nibble-slots.
// B: 32 lanes x  8 bytes (=256), each byte = 2 nibbles ->  512 B nibble-slots.
// Nibble slot n (per lane): byte = n>>1, half = n&1 (0=lo,1=hi).
__global__ void calib(const uint8_t* Aop, const uint8_t* Bop, float* Dout) {
  int lane = threadIdx.x;
  const uint8_t* A = Aop + lane*16; const uint8_t* B = Bop + lane*8;
  auto P=[&](const uint8_t* x){ return uint32_t(x[0])|uint32_t(x[1])<<8|uint32_t(x[2])<<16|uint32_t(x[3])<<24; };
  uint32_t a0=P(A+0),a1=P(A+4),a2=P(A+8),a3=P(A+12),b0=P(B+0),b1=P(B+4);
  float d0=0,d1=0,d2=0,d3=0,c0=0,c1=0,c2=0,c3=0;
  const uint32_t sf=0x00007F7Fu; const uint16_t z=0;   // 2X unit scale (0x7F = 2^0)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  asm volatile(
   "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col."
   "f32.e2m1.e2m1.f32.ue8m0 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
   "{%14},{%15,%16},{%17},{%18,%19};\n"
   : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
   : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
     "f"(c0),"f"(c1),"f"(c2),"f"(c3),"r"(sf),"h"(z),"h"(z),"r"(sf),"h"(z),"h"(z));
#endif
  Dout[lane*4+0]=d0; Dout[lane*4+1]=d1; Dout[lane*4+2]=d2; Dout[lane*4+3]=d3;
}

static uint8_t *dA,*dB; static float* dD;
// Pack per-lane nibble arrays (Anib: 1024, Bnib: 512) into the byte buffers and run.
static void run(const std::vector<uint8_t>& Anib, const std::vector<uint8_t>& Bnib, float* Dout) {
  uint8_t Ab[512], Bb[256];
  for (int lane=0; lane<32; ++lane) {
    for (int by=0; by<16; ++by) { int n=lane*32+by*2; Ab[lane*16+by]=(Anib[n]&0xF)|((Anib[n+1]&0xF)<<4); }
    for (int by=0; by<8;  ++by) { int n=lane*16+by*2; Bb[lane*8 +by]=(Bnib[n]&0xF)|((Bnib[n+1]&0xF)<<4); }
  }
  cudaMemcpy(dA,Ab,512,cudaMemcpyHostToDevice);
  cudaMemcpy(dB,Bb,256,cudaMemcpyHostToDevice);
  calib<<<1,32>>>(dA,dB,dD);
  cudaMemcpy(Dout,dD,128*sizeof(float),cudaMemcpyDeviceToHost);
}

int main() {
  cudaMalloc(&dA,512); cudaMalloc(&dB,256); cudaMalloc(&dD,128*sizeof(float));
  const int NA=1024, NB=512; const uint8_t ONE=0x2;   // code 2 = 1.0
  std::vector<uint8_t> An(NA), Bn(NB); std::vector<float> Dout(128);

  // --- Step 1: rows. one-hot A nibble, B all ones. ---
  std::vector<uint64_t> litA(NA), litA_hi(NA);
  for (int ai=0; ai<NA; ++ai) {
    std::fill(An.begin(),An.end(),0); An[ai]=ONE; std::fill(Bn.begin(),Bn.end(),ONE);
    run(An,Bn,Dout.data());
    uint64_t lo=0,hi=0; for (int d=0; d<128; ++d) if (Dout[d]>0.5f) (d<64?lo:hi)|=(1ull<<(d&63));
    litA[ai]=lo; litA_hi[ai]=hi;
  }
  std::vector<int> rowOf(NA,-1); std::vector<std::pair<uint64_t,uint64_t>> rowMask;
  for (int ai=0; ai<NA; ++ai) {
    if (litA[ai]==0 && litA_hi[ai]==0) continue;
    int found=-1; for (size_t r=0;r<rowMask.size();++r) if (rowMask[r].first==litA[ai]&&rowMask[r].second==litA_hi[ai]){found=(int)r;break;}
    if (found<0){ found=(int)rowMask.size(); rowMask.push_back({litA[ai],litA_hi[ai]}); }
    rowOf[ai]=found;
  }
  std::vector<int> dRow(128,-1);
  for (int r=0;r<(int)rowMask.size();++r) for (int d=0; d<128; ++d)
    if ((d<64?(rowMask[r].first>>(d&63)):(rowMask[r].second>>(d&63)))&1) dRow[d]=r;
  std::printf("rows discovered: %zu (expect 16)\n", rowMask.size());

  // --- Step 2: cols. one-hot B nibble, A all ones. ---
  std::vector<uint64_t> litB(NB), litB_hi(NB);
  for (int bj=0; bj<NB; ++bj) {
    std::fill(Bn.begin(),Bn.end(),0); Bn[bj]=ONE; std::fill(An.begin(),An.end(),ONE);
    run(An,Bn,Dout.data());
    uint64_t lo=0,hi=0; for (int d=0; d<128; ++d) if (Dout[d]>0.5f) (d<64?lo:hi)|=(1ull<<(d&63));
    litB[bj]=lo; litB_hi[bj]=hi;
  }
  std::vector<int> colOf(NB,-1); std::vector<std::pair<uint64_t,uint64_t>> colMask;
  for (int bj=0; bj<NB; ++bj) {
    int found=-1; for (size_t c=0;c<colMask.size();++c) if (colMask[c].first==litB[bj]&&colMask[c].second==litB_hi[bj]){found=(int)c;break;}
    if (found<0){ found=(int)colMask.size(); colMask.push_back({litB[bj],litB_hi[bj]}); }
    colOf[bj]=found;
  }
  std::vector<int> dCol(128,-1);
  for (int c=0;c<(int)colMask.size();++c) for (int d=0; d<128; ++d)
    if ((d<64?(colMask[c].first>>(d&63)):(colMask[c].second>>(d&63)))&1) dCol[d]=c;
  std::printf("cols discovered: %zu (expect 8)\n", colMask.size());

  std::vector<int> dM(128), dN(128), dSlotOf(16*8,-1);
  for (int d=0; d<128; ++d){ dM[d]=dRow[d]; dN[d]=dCol[d]; if(dM[d]>=0&&dN[d]>=0) dSlotOf[dM[d]*8+dN[d]]=d; }
  int dt = dSlotOf[0];

  // --- Step 3: K pairing. ---
  std::vector<int> a0slots,b0slots;
  for (int ai=0; ai<NA; ++ai) if (rowOf[ai]==0) a0slots.push_back(ai);
  for (int bj=0; bj<NB; ++bj) if (colOf[bj]==0) b0slots.push_back(bj);
  std::printf("row0 A-slots: %zu, col0 B-slots: %zu, target D=%d\n", a0slots.size(), b0slots.size(), dt);
  std::vector<int> kA(NA,-1), kB(NB,-1);
  auto pair_nonzero=[&](int ai,int bj,int ds)->bool{
    std::fill(An.begin(),An.end(),0); An[ai]=ONE; std::fill(Bn.begin(),Bn.end(),0); Bn[bj]=ONE;
    run(An,Bn,Dout.data()); return ds>=0 && Dout[ds]>0.5f;
  };
  int kc=0;
  for (int ai : a0slots) for (int bj : b0slots)
    if (kB[bj]<0 && pair_nonzero(ai,bj,dt)) { kA[ai]=kc; kB[bj]=kc; ++kc; break; }
  std::printf("seed K labels (row0/col0): %d (expect 64)\n", kc);
  for (int ai=0; ai<NA; ++ai) {
    if (kA[ai]>=0 || rowOf[ai]<0) continue; int ds=dSlotOf[rowOf[ai]*8+0];
    for (int bj : b0slots) if (kB[bj]>=0 && pair_nonzero(ai,bj,ds)) { kA[ai]=kB[bj]; break; }
  }
  for (int bj=0; bj<NB; ++bj) {
    if (kB[bj]>=0 || colOf[bj]<0) continue; int ds=dSlotOf[0*8+colOf[bj]];
    for (int ai : a0slots) if (kA[ai]>=0 && pair_nonzero(ai,bj,ds)) { kB[bj]=kA[ai]; break; }
  }
  int nKA=0,nKB=0; for(int x:kA)nKA+=(x>=0); for(int x:kB)nKB+=(x>=0);
  std::printf("K assigned: A %d/%d, B %d/%d\n", nKA,NA,nKB,NB);

  // --- hardware decode by code (A=all code c, B=all 1.0, unit scale -> D=64*dec). ---
  float hwdec[8];
  for (int c=0;c<8;++c){ std::fill(An.begin(),An.end(),(uint8_t)c); std::fill(Bn.begin(),Bn.end(),(uint8_t)0x2);
    run(An,Bn,Dout.data()); hwdec[c]=Dout[0]/64.0f; }
  std::printf("hardware decode code 0..7: "); for(int c=0;c<8;++c)std::printf("%.3f ",hwdec[c]); std::printf("\n");

  // --- Step 4: random 16x64 * 64x8 tile through the maps. ---
  uint8_t Al[16*64], Bl[64*8]; uint64_t s=0xABCDEF;
  auto rnd=[&]{ s=s*6364136223846793005ull+1; return uint8_t((s>>33)%8); };
  for (auto&x:Al)x=rnd(); for (auto&x:Bl)x=rnd();
  float ref[128]; for(int m=0;m<16;++m)for(int n=0;n<8;++n){float a=0;for(int k=0;k<64;++k)a+=hwdec[Al[m*64+k]]*hwdec[Bl[k*8+n]];ref[m*8+n]=a;}
  for (int ai=0; ai<NA; ++ai) An[ai]=(rowOf[ai]<0||kA[ai]<0)?0:Al[rowOf[ai]*64+kA[ai]];
  for (int bj=0; bj<NB; ++bj) Bn[bj]=(colOf[bj]<0||kB[bj]<0)?0:Bl[kB[bj]*8+colOf[bj]];
  run(An,Bn,Dout.data());
  float got[128]; for (int d=0; d<128; ++d) if(dM[d]>=0&&dN[d]>=0) got[dM[d]*8+dN[d]]=Dout[d];
  float maxerr=0; int nbad=0; for(int i=0;i<128;++i){float e=fabsf(got[i]-ref[i]); maxerr=fmaxf(maxerr,e); if(e>1e-3f)++nbad;}
  std::printf("VALIDATION: max abs err=%.4f, mismatched=%d/128\n", maxerr,nbad);
  { std::vector<float> a(got,got+128),b(ref,ref+128); std::sort(a.begin(),a.end()); std::sort(b.begin(),b.end());
    float md=0; for(int i=0;i<128;++i)md=fmaxf(md,fabsf(a[i]-b[i]));
    std::printf("  multiset diff=%.4f %s\n", md, md<1e-3f?"(PERMUTATION: D-map)":"(values: K/decode)"); }

  if (nbad==0) {
    std::printf("SUCCESS: packed mxf4nvf4 layout extracted + self-validated.\n");
    std::printf("static const short A_row[1024]={"); for(int i=0;i<NA;++i)std::printf("%d,",rowOf[i]); std::printf("};\n");
    std::printf("static const short A_k[1024]={");   for(int i=0;i<NA;++i)std::printf("%d,",kA[i]);    std::printf("};\n");
    std::printf("static const short B_col[512]={");  for(int i=0;i<NB;++i)std::printf("%d,",colOf[i]); std::printf("};\n");
    std::printf("static const short B_k[512]={");    for(int i=0;i<NB;++i)std::printf("%d,",kB[i]);    std::printf("};\n");
    std::printf("static const char  D_m[128]={");    for(int i=0;i<128;++i)std::printf("%d,",dM[i]);   std::printf("};\n");
    std::printf("static const char  D_n[128]={");    for(int i=0;i<128;++i)std::printf("%d,",dN[i]);   std::printf("};\n");
  } else {
    std::printf("Extraction incomplete (rows=%zu cols=%zu kc=%d A %d/%d B %d/%d) -- send output.\n",
                rowMask.size(),colMask.size(),kc,nKA,NA,nKB,NB);
  }
  return 0;
}
