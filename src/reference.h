// reference.h - FP32 reference GEMM used as the validation oracle.
#pragma once

#include <vector>

// Computes C[M x N] = A[M x K] * B[K x N], all row-major FP32, on the GPU with a
// simple (un-optimized) kernel. Used only to validate the NVFP4 kernels, so it
// favours clarity over speed. Use modest sizes with --validate.
std::vector<float> reference_gemm_fp32(const std::vector<float>& A,
                                       const std::vector<float>& B,
                                       int M, int N, int K);

// Max relative error between two row-major MxN matrices.
double max_rel_error(const std::vector<float>& got,
                     const std::vector<float>& ref,
                     int M, int N);
