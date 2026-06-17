// reference.cu - naive FP32 GEMM oracle on the GPU.
#include "reference.h"
#include "common.h"

#include <cmath>

namespace {

__global__ void naive_gemm(const float* __restrict__ A, const float* __restrict__ B,
                           float* __restrict__ C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  float acc = 0.0f;
  for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
  C[row * N + col] = acc;
}

}  // namespace

std::vector<float> reference_gemm_fp32(const std::vector<float>& A,
                                       const std::vector<float>& B,
                                       int M, int N, int K) {
  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CHECK_CUDA(cudaMalloc(&dA, sizeof(float) * size_t(M) * K));
  CHECK_CUDA(cudaMalloc(&dB, sizeof(float) * size_t(K) * N));
  CHECK_CUDA(cudaMalloc(&dC, sizeof(float) * size_t(M) * N));
  CHECK_CUDA(cudaMemcpy(dA, A.data(), sizeof(float) * size_t(M) * K, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, B.data(), sizeof(float) * size_t(K) * N, cudaMemcpyHostToDevice));

  dim3 block(16, 16);
  dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
  naive_gemm<<<grid, block>>>(dA, dB, dC, M, N, K);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  std::vector<float> C(size_t(M) * N);
  CHECK_CUDA(cudaMemcpy(C.data(), dC, sizeof(float) * size_t(M) * N, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(dA));
  CHECK_CUDA(cudaFree(dB));
  CHECK_CUDA(cudaFree(dC));
  return C;
}

double max_rel_error(const std::vector<float>& got, const std::vector<float>& ref,
                     int M, int N) {
  double worst = 0.0;
  for (size_t i = 0; i < size_t(M) * N; ++i) {
    double r = ref[i];
    double g = got[i];
    double denom = std::max(1e-6, std::fabs(r));
    double e = std::fabs(g - r) / denom;
    if (e > worst) worst = e;
  }
  return worst;
}
