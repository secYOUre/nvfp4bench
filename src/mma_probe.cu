// mma_probe.cu - FP4 warp-MMA self-calibrating layout extractor (task #8, step 4).
//
// Brute-forcing layout combinations failed, so we MEASURE the hardware contraction
// directly via one-hot inputs, then self-validate:
//   1) one-hot A (B=all ones)  -> each physical A byte's lit output set reveals its ROW.
//   2) one-hot B (A=all ones)  -> each physical B byte's lit output set reveals its COL.
//   3) one-hot A x one-hot B   -> pairs A-bytes with B-bytes that share the same K.
//   4) build per-physical-slot maps, run a random tile through them, compare to ref.
// Exploits the relabel freedom: any consistent (row,col,K) labeling that matches the
// hardware's bilinear structure yields a correct GEMM.
//
//   nvcc -gencode=arch=compute_121a,code=sm_121a -o mma_probe src/mma_probe.cu && ./mma_probe
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <utility>
#include <cuda_runtime.h>

__host__ __device__ inline float e2m1_decode(uint8_t c) {
  const float m[8] = {0,0.5f,1,1.5f,2,3,4,6}; return m[c & 7];
}

// Physical operand layout: Aop[lane*16 + areg*4 + byte], Bop[lane*8 + breg*4 + byte].
// Output: Dout[lane*4 + dreg]. Runs the FP4 block-scaled MMA at unit scale (0x81).
__global__ void calib(const uint8_t* Aop, const uint8_t* Bop, float* Dout) {
  int lane = threadIdx.x;
  uint32_t a0,a1,a2,a3,b0,b1;
  const uint8_t* A = Aop + lane*16; const uint8_t* B = Bop + lane*8;
  auto P=[&](const uint8_t* x){ return uint32_t(x[0])|uint32_t(x[1])<<8|uint32_t(x[2])<<16|uint32_t(x[3])<<24; };
  a0=P(A+0); a1=P(A+4); a2=P(A+8); a3=P(A+12); b0=P(B+0); b1=P(B+4);
  float d0=0,d1=0,d2=0,d3=0,c0=0,c1=0,c2=0,c3=0; const uint32_t sf=0x81u; const uint16_t z=0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  asm volatile(
   "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col."
   "f32.e2m1.e2m1.f32.ue8m0 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
   "{%14},{%15,%16},{%17},{%18,%19};\n"
   : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
   : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
     "f"(c0),"f"(c1),"f"(c2),"f"(c3),"r"(sf),"h"(z),"h"(z),"r"(sf),"h"(z),"h"(z));
#endif
  Dout[lane*4+0]=d0; Dout[lane*4+1]=d1; Dout[lane*4+2]=d2; Dout[lane*4+3]=d3;
}

static uint8_t *dA,*dB; static float* dD;
static void run(const uint8_t* Aop, const uint8_t* Bop, float* Dout) {
  cudaMemcpy(dA,Aop,512,cudaMemcpyHostToDevice);
  cudaMemcpy(dB,Bop,256,cudaMemcpyHostToDevice);
  calib<<<1,32>>>(dA,dB,dD);
  cudaMemcpy(Dout,dD,128*sizeof(float),cudaMemcpyDeviceToHost);
}

int main() {
  cudaMalloc(&dA,512); cudaMalloc(&dB,256); cudaMalloc(&dD,128*sizeof(float));
  std::vector<uint8_t> Aop(512), Bop(256); std::vector<float> Dout(128);
  const uint8_t ONE = 0x2;  // E2M1 code 1.0

  // --- Step 1: rows. one-hot A, B all ones. lit D set (8 slots) identifies row. ---
  std::vector<uint64_t> litA(512);  // bitmask over 128 D slots (use 2x64 -> pack into hi/lo via vector<pair>)
  std::vector<uint64_t> litA_hi(512);
  for (int ai=0; ai<512; ++ai) {
    std::fill(Aop.begin(),Aop.end(),0); Aop[ai]=ONE; std::fill(Bop.begin(),Bop.end(),ONE);
    run(Aop.data(),Bop.data(),Dout.data());
    uint64_t lo=0,hi=0; for (int d=0; d<128; ++d) if (Dout[d]>0.5f) (d<64?lo:hi) |= (1ull<<(d&63));
    litA[ai]=lo; litA_hi[ai]=hi;
  }
  // cluster identical (lo,hi) -> row ids
  std::vector<int> rowOf(512,-1); std::vector<std::pair<uint64_t,uint64_t>> rowMask;
  for (int ai=0; ai<512; ++ai) {
    if (litA[ai]==0 && litA_hi[ai]==0) continue; // zero element (shouldn't happen for ONE)
    int found=-1; for (size_t r=0;r<rowMask.size();++r) if (rowMask[r].first==litA[ai]&&rowMask[r].second==litA_hi[ai]){found=(int)r;break;}
    if (found<0){ found=(int)rowMask.size(); rowMask.push_back({litA[ai],litA_hi[ai]}); }
    rowOf[ai]=found;
  }
  // D slot -> row: for each D slot, which row's mask contains it
  std::vector<int> dRow(128,-1);
  for (int r=0;r<(int)rowMask.size();++r) for (int d=0; d<128; ++d)
    if ((d<64? (rowMask[r].first>>(d&63)) : (rowMask[r].second>>(d&63)))&1) dRow[d]=r;
  std::printf("rows discovered: %zu (expect 16)\n", rowMask.size());

  // --- Step 2: cols. one-hot B, A all ones. lit D set (16 slots) identifies col. ---
  std::vector<uint64_t> litB(256), litB_hi(256);
  for (int bj=0; bj<256; ++bj) {
    std::fill(Bop.begin(),Bop.end(),0); Bop[bj]=ONE; std::fill(Aop.begin(),Aop.end(),ONE);
    run(Aop.data(),Bop.data(),Dout.data());
    uint64_t lo=0,hi=0; for (int d=0; d<128; ++d) if (Dout[d]>0.5f) (d<64?lo:hi)|=(1ull<<(d&63));
    litB[bj]=lo; litB_hi[bj]=hi;
  }
  std::vector<int> colOf(256,-1); std::vector<std::pair<uint64_t,uint64_t>> colMask;
  for (int bj=0; bj<256; ++bj) {
    int found=-1; for (size_t c=0;c<colMask.size();++c) if (colMask[c].first==litB[bj]&&colMask[c].second==litB_hi[bj]){found=(int)c;break;}
    if (found<0){ found=(int)colMask.size(); colMask.push_back({litB[bj],litB_hi[bj]}); }
    colOf[bj]=found;
  }
  std::vector<int> dCol(128,-1);
  for (int c=0;c<(int)colMask.size();++c) for (int d=0; d<128; ++d)
    if ((d<64?(colMask[c].first>>(d&63)):(colMask[c].second>>(d&63)))&1) dCol[d]=c;
  std::printf("cols discovered: %zu (expect 8)\n", colMask.size());

  // D slot -> (m,n); and inverse (m,n) -> physical D slot.
  std::vector<int> dM(128), dN(128); std::vector<int> dSlotOf(16*8,-1);
  for (int d=0; d<128; ++d){ dM[d]=dRow[d]; dN[d]=dCol[d]; if(dM[d]>=0&&dN[d]>=0) dSlotOf[dM[d]*8+dN[d]]=d; }
  int dt = dSlotOf[0*8+0];  // target for (row0,col0)

  // --- Step 3: K pairing, measured per-slot (no structural assumption). ---
  std::vector<int> a0slots, b0slots;
  for (int ai=0; ai<512; ++ai) if (rowOf[ai]==0) a0slots.push_back(ai);
  for (int bj=0; bj<256; ++bj) if (colOf[bj]==0) b0slots.push_back(bj);
  std::printf("row0 A-slots: %zu, col0 B-slots: %zu, target D=%d\n", a0slots.size(), b0slots.size(), dt);
  std::vector<int> kA(512,-1), kB(256,-1);

  auto pair_nonzero = [&](int ai, int bj, int dslot)->bool {
    std::fill(Aop.begin(),Aop.end(),0); Aop[ai]=ONE;
    std::fill(Bop.begin(),Bop.end(),0); Bop[bj]=ONE;
    run(Aop.data(),Bop.data(),Dout.data());
    return dslot>=0 && Dout[dslot]>0.5f;
  };

  // (a) seed a consistent K labeling on row0 A-slots and col0 B-slots.
  int kc=0;
  for (int ai : a0slots) for (int bj : b0slots)
    if (kB[bj]<0 && pair_nonzero(ai,bj,dt)) { kA[ai]=kc; kB[bj]=kc; ++kc; break; }
  std::printf("seed K labels (row0/col0): %d (expect 32)\n", kc);

  // (b) every A-slot: pair with col0 B-slots (kB known) at D(rowOf,0) to get its K.
  for (int ai=0; ai<512; ++ai) {
    if (kA[ai]>=0 || rowOf[ai]<0) continue;
    int dslot = dSlotOf[rowOf[ai]*8 + 0];
    for (int bj : b0slots) if (kB[bj]>=0 && pair_nonzero(ai,bj,dslot)) { kA[ai]=kB[bj]; break; }
  }
  // (c) every B-slot: pair with row0 A-slots (kA known) at D(0,colOf) to get its K.
  for (int bj=0; bj<256; ++bj) {
    if (kB[bj]>=0 || colOf[bj]<0) continue;
    int dslot = dSlotOf[0*8 + colOf[bj]];
    for (int ai : a0slots) if (kA[ai]>=0 && pair_nonzero(ai,bj,dslot)) { kB[bj]=kA[ai]; break; }
  }
  int nKA=0,nKB=0; for(int x:kA)nKA+=(x>=0); for(int x:kB)nKB+=(x>=0);
  std::printf("K assigned: A %d/512, B %d/256\n", nKA, nKB);

  // --- Measure the true hardware E2M1 decode for each code 0..7. ---
  // A = all code c, B = all 1.0 (code 2), unit scale => D = 32 * hwdec[c] * 1.0.
  float hwdec[8];
  for (int c=0;c<8;++c){
    std::fill(Aop.begin(),Aop.end(),(uint8_t)c); std::fill(Bop.begin(),Bop.end(),(uint8_t)0x2);
    run(Aop.data(),Bop.data(),Dout.data());
    hwdec[c]=Dout[0]/32.0f;
  }
  std::printf("hardware E2M1 decode by code 0..7: ");
  for (int c=0;c<8;++c) std::printf("%.4f ", hwdec[c]); std::printf("\n");

  // --- Step 4: random tile through the extracted maps, validated with hwdec. ---
  uint8_t Al[16*32], Bl[32*8]; uint64_t s=0xABCDEF;
  auto rnd=[&]{ s=s*6364136223846793005ull+1; return uint8_t((s>>33)%8); };
  for (auto&x:Al)x=rnd(); for (auto&x:Bl)x=rnd();
  float ref[128]; for (int m=0;m<16;++m)for(int n=0;n<8;++n){float acc=0;for(int k=0;k<32;++k)acc+=hwdec[Al[m*32+k]]*hwdec[Bl[k*8+n]];ref[m*8+n]=acc;}
  for (int ai=0; ai<512; ++ai) Aop[ai] = (rowOf[ai]<0||kA[ai]<0)?0:Al[rowOf[ai]*32 + kA[ai]];
  for (int bj=0; bj<256; ++bj) Bop[bj] = (colOf[bj]<0||kB[bj]<0)?0:Bl[kB[bj]*8 + colOf[bj]];
  run(Aop.data(),Bop.data(),Dout.data());
  float got[128]; for (int d=0; d<128; ++d) if (dM[d]>=0&&dN[d]>=0) got[dM[d]*8+dN[d]]=Dout[d];
  float maxerr=0; int nbad=0; for (int i=0;i<128;++i){float e=fabsf(got[i]-ref[i]); maxerr=fmaxf(maxerr,e); if(e>1e-3f)++nbad;}
  std::printf("VALIDATION: max abs err=%.4f, mismatched=%d/128\n", maxerr, nbad);

  // --- Diagnostics to localize the discrepancy ---
  std::printf("  m=0 got vs ref: "); for(int n=0;n<8;++n) std::printf("(%.1f|%.1f) ", got[n], ref[n]); std::printf("\n");
  std::printf("  m=8 got vs ref: "); for(int n=0;n<8;++n) std::printf("(%.1f|%.1f) ", got[8*8+n], ref[8*8+n]); std::printf("\n");
  // Is got a permutation of ref? (would indicate a D index mixup, not value error)
  { std::vector<float> a(got,got+128), b(ref,ref+128); std::sort(a.begin(),a.end()); std::sort(b.begin(),b.end());
    float md=0; for(int i=0;i<128;++i) md=fmaxf(md,fabsf(a[i]-b[i]));
    std::printf("  multiset(got) vs multiset(ref) max diff = %.4f %s\n", md, md<1e-3f?"(PERMUTATION: D-map issue)":"(values differ: K/decode)"); }
  // Average ratio got/ref (partial-sum hint).
  { double sg=0,sr=0; for(int i=0;i<128;++i){sg+=got[i];sr+=ref[i];} std::printf("  sum(got)/sum(ref) = %.4f\n", sg/sr); }
  // Control: route ALL-ONES (code 2) through the tables -> must be 32 everywhere (tests plumbing).
  { for(int i=0;i<512;++i)Aop[i]=(rowOf[i]<0||kA[i]<0)?0:0x2; for(int i=0;i<256;++i)Bop[i]=(colOf[i]<0||kB[i]<0)?0:0x2;
    run(Aop.data(),Bop.data(),Dout.data()); int b32=0; for(int d=0;d<128;++d) if(fabsf(Dout[d]-32.0f)>1e-3f)++b32;
    std::printf("  control all-ones-via-tables: %d/128 != 32\n", b32); }
  if (nbad==0) {
    std::printf("SUCCESS: layout extracted and self-validated. Baking tables:\n");
    // Compact per-lane view: for lane L, A slot s=areg*4+byte -> (row,k); B slot -> (k,col); D dreg -> (m,n).
    std::printf("static const int A_row[512]={"); for(int i=0;i<512;++i)std::printf("%d,",rowOf[i]); std::printf("};\n");
    std::printf("static const int A_k[512]={");   for(int i=0;i<512;++i)std::printf("%d,",kA[i]);    std::printf("};\n");
    std::printf("static const int B_col[256]={"); for(int i=0;i<256;++i)std::printf("%d,",colOf[i]); std::printf("};\n");
    std::printf("static const int B_k[256]={");   for(int i=0;i<256;++i)std::printf("%d,",kB[i]);    std::printf("};\n");
    std::printf("static const int D_m[128]={");   for(int i=0;i<128;++i)std::printf("%d,",dM[i]);    std::printf("};\n");
    std::printf("static const int D_n[128]={");   for(int i=0;i<128;++i)std::printf("%d,",dN[i]);    std::printf("};\n");
  } else {
    std::printf("Extraction incomplete. (rows=%zu cols=%zu kc=%d, A %d/512, B %d/256) -- send this output.\n",
                rowMask.size(), colMask.size(), kc, nKA, nKB);
  }
  return 0;
}
