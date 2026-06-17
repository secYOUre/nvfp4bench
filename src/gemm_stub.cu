// gemm_stub.cu - correctness-oriented, on-the-fly dequant GEMM.
//
// This is NOT a performance kernel. It reads the canonical packed NVFP4 layout
// directly and dequantizes on the fly, so it validates the packing/layout and
// the full harness before the tensor-core kernels exist. Use small sizes.
#include "gemm.h"

namespace {

__device__ inline float decode_e2m1(uint8_t code) {
  const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  float m = mag[code & 0x7];
  return (code & 0x8) ? -m : m;
}

__device__ inline float decode_e4m3(uint8_t code) {
  int sign = (code >> 7) & 0x1;
  int exp  = (code >> 3) & 0xF;
  int man  = code & 0x7;
  float v = (exp == 0) ? ldexpf(man / 8.0f, 1 - 7)
                       : ldexpf(1.0f + man / 8.0f, exp - 7);
  return sign ? -v : v;
}

constexpr int kBlk = 16;

__global__ void dequant_gemm(Nvfp4DeviceOperands op) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= op.M || col >= op.N) return;
  const int K = op.K, N = op.N;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    // A[row,k]
    size_t ai = size_t(row) * K + k;
    uint8_t abyte = op.a_packed[ai / 2];
    uint8_t acode = (k & 1) ? (abyte >> 4) : (abyte & 0x0F);
    float as = decode_e4m3(op.a_scales[size_t(row) * (K / kBlk) + k / kBlk]) * op.a_global;
    float a = decode_e2m1(acode) * as;
    // B[k,col]
    size_t bi = size_t(k) * N + col;
    uint8_t bbyte = op.b_packed[bi / 2];
    uint8_t bcode = (col & 1) ? (bbyte >> 4) : (bbyte & 0x0F);
    float bs = decode_e4m3(op.b_scales[size_t(k / kBlk) * N + col]) * op.b_global;
    float b = decode_e2m1(bcode) * bs;
    acc += a * b;
  }
  op.c[size_t(row) * N + col] = acc;
}

}  // namespace

bool launch_gemm_stub(const Nvfp4DeviceOperands& op, cudaStream_t s, std::string* err) {
  dim3 block(16, 16);
  dim3 grid((op.N + block.x - 1) / block.x, (op.M + block.y - 1) / block.y);
  dequant_gemm<<<grid, block, 0, s>>>(op);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) { if (err) *err = cudaGetErrorString(e); return false; }
  return true;
}
