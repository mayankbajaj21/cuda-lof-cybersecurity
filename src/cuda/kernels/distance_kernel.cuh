/**
 * distance_kernel.cuh - CUDA Pairwise Euclidean Distance Matrix Kernel Declarations
 *
 * Project: CUDA LOF Cybersecurity
 * Milestone: M3 - CUDA LOF Baseline (Step 2)
 *
 * Computes:
 *   dist(i, j) = sqrt( sum_{d=0}^{D-1} (X[i, d] - X[j, d])^2 )
 *
 * Designed for:
 *   - N samples (arbitrary N, tested on 10K, 25K)
 *   - D = 68 standardized numerical features
 *   - float32 row-major contiguous input X (N, D)
 *   - float32 row-major contiguous output dist_matrix (N, N)
 *   - Target: sm_86 (NVIDIA GeForce RTX 3050 Laptop GPU)
 */

#ifndef DISTANCE_KERNEL_CUH
#define DISTANCE_KERNEL_CUH

#include <cuda_runtime.h>

#ifdef _WIN32
  #define DISTANCE_API extern "C" __declspec(dllexport)
#else
  #define DISTANCE_API extern "C"
#endif

// ---------------------------------------------------------------------------
// Kernel declaration (naive 2D grid/block)
// ---------------------------------------------------------------------------
__global__ void pairwise_distance_naive_kernel(
    const float* __restrict__ X,
    float* __restrict__ dist_matrix,
    int N,
    int D
);

// ---------------------------------------------------------------------------
// Host launcher declaration (operates on device pointers)
// ---------------------------------------------------------------------------
cudaError_t launch_pairwise_distance_kernel(
    const float* d_X,
    float* d_dist_matrix,
    int N,
    int D,
    cudaStream_t stream = 0
);

// ---------------------------------------------------------------------------
// C-compatible API for Python ctypes / external callers (host memory in/out)
// ---------------------------------------------------------------------------
DISTANCE_API int compute_pairwise_distances_cuda(
    const float* h_X,
    float* h_dist_matrix,
    int N,
    int D
);

#endif  // DISTANCE_KERNEL_CUH
