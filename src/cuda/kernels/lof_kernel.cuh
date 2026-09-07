/**
 * lof_kernel.cuh - CUDA Local Outlier Factor (LOF) Scoring Kernel Declarations
 *
 * Project: CUDA LOF Cybersecurity
 * Milestone: M3 - CUDA LOF Baseline (Step 4)
 *
 * Formula (M2 Reproduction):
 *   LOF_k(p) = ( sum_{o in N_k(p)} (lrd(o) / lrd(p)) ) / |N_k(p)|
 *            = mean_{o in N_k(p)} (lrd(o)) / (lrd(p) + eps)
 *
 * Input:
 *   - lrd: (N,) float32 array of local reachability densities
 *   - knn_indices: (N, k) int32 row-major array of neighbor indices
 *   - k: number of neighbors
 *   - eps: numerical stability epsilon (default 1e-10f)
 *
 * Output:
 *   - lof_scores: (N,) float32 array
 */

#ifndef LOF_KERNEL_CUH
#define LOF_KERNEL_CUH

#include <cuda_runtime.h>
#include "lrd_kernel.cuh"

#ifdef _WIN32
  #define LOF_API extern "C" __declspec(dllexport)
#else
  #define LOF_API extern "C"
#endif

// ---------------------------------------------------------------------------
// CUDA Kernel Declaration
// ---------------------------------------------------------------------------
__global__ void lof_scoring_naive_kernel(
    const float* __restrict__ lrd,
    const int* __restrict__ knn_indices,
    float* __restrict__ lof_scores,
    int N,
    int k,
    float eps = LOF_DEFAULT_EPS
);

// ---------------------------------------------------------------------------
// Host Kernel Launcher (Operates directly on device memory)
// ---------------------------------------------------------------------------
cudaError_t launch_lof_scoring_kernel(
    const float* d_lrd,
    const int* d_knn_indices,
    float* d_lof_scores,
    int N,
    int k,
    float eps = LOF_DEFAULT_EPS,
    cudaStream_t stream = 0
);

// ---------------------------------------------------------------------------
// C API Export for Python ctypes / External Invocations (Host Memory)
// ---------------------------------------------------------------------------
LOF_API int compute_lof_scores_cuda(
    const float* h_lrd,
    const int* h_knn_indices,
    float* h_lof_scores,
    int N,
    int k,
    float eps = LOF_DEFAULT_EPS
);

#endif  // LOF_KERNEL_CUH
