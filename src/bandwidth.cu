// bandwidth.cu - STREAM-triad style achievable bandwidth probe.
#include "bench.h"
#include "common.h"

namespace {
__global__ void triad(const float* __restrict__ a, const float* __restrict__ b,
                       float* __restrict__ c, float s, size_t n) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  size_t stride = size_t(gridDim.x) * blockDim.x;
  for (; i < n; i += stride) c[i] = a[i] + s * b[i];
}
}  // namespace

double measure_bandwidth_gbs(size_t bytes, int iters) {
  size_t n = bytes / sizeof(float);
  float *a = nullptr, *b = nullptr, *c = nullptr;
  CHECK_CUDA(cudaMalloc(&a, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&b, n * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&c, n * sizeof(float)));
  CHECK_CUDA(cudaMemset(a, 1, n * sizeof(float)));
  CHECK_CUDA(cudaMemset(b, 2, n * sizeof(float)));

  int block = 256, grid = 0;
  CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&grid, triad, block, 0));
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
  grid *= prop.multiProcessorCount;

  for (int i = 0; i < 5; ++i) triad<<<grid, block>>>(a, b, c, 1.5f, n);  // warmup
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t s, e;
  CHECK_CUDA(cudaEventCreate(&s));
  CHECK_CUDA(cudaEventCreate(&e));
  CHECK_CUDA(cudaEventRecord(s));
  for (int i = 0; i < iters; ++i) triad<<<grid, block>>>(a, b, c, 1.5f, n);
  CHECK_CUDA(cudaEventRecord(e));
  CHECK_CUDA(cudaEventSynchronize(e));
  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));

  // triad reads 2 arrays, writes 1 => 3 * bytes per iter.
  double moved = 3.0 * double(n) * sizeof(float) * iters;
  double gbs = moved / (ms * 1e-3) / 1e9;

  CHECK_CUDA(cudaEventDestroy(s));
  CHECK_CUDA(cudaEventDestroy(e));
  CHECK_CUDA(cudaFree(a));
  CHECK_CUDA(cudaFree(b));
  CHECK_CUDA(cudaFree(c));
  return gbs;
}
