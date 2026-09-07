/**
 * lrd_kernel.cuh - CUDA Local Reachability Density (LRD) Kernel Declarations
 *
 * Project: CUDA LOF Cybersecurity
 * Milestone: M3 - CUDA LOF Baseline (Step 4)
 *
 * Formula (M2 Reproduction):
 *   lrd_k(p) = 1.0 / ( mean_{o in N_k(p)} reach-dist_k(p, o) + eps )
 *            = 1.0 / ( ( (1/k) * sum_{m=0}^{k-1} reach_dists[p, m] ) + eps )
 *
 * Input:
 *   - reach_dists: (N, k) float32 row-major array
 *   - k: number of neighbors
 *   - eps: numerical stability epsilon (default 1e-10f matching M2 CPU)
 *
 * Output:
 *   - lrd: (N,) float32 array
 */

#ifndef LRD_KERNEL_CUH
#define LRD_KERNEL_CUH

#include <cuda_runtime.h>

#ifdef _WIN32
  #define LRD_API extern "C" __declspec(dllexport)
#else
  #define LRD_API extern "C"
#endif

#define LOF_DEFAULT_EPS 1e-10f

// ---------------------------------------------------------------------------
// CUDA Kernel Declaration
// ---------------------------------------------------------------------------
__global__ void lrd_naive_kernel(
    const float* __restrict__ reach_dists,
    float* __restrict__ lrd,
    int N,
    int k,
    float eps = LOF_DEFAULT_EPS
);

// ---------------------------------------------------------------------------
// Host Kernel Launcher (Operates directly on device memory)
// ---------------------------------------------------------------------------
cudaError_t launch_lrd_kernel(
    const float* d_reach_dists,
    float* d_lrd,
    int N,
    int k,
    float eps = LOF_DEFAULT_EPS,
    cudaStream_t stream = 0
);

// ---------------------------------------------------------------------------
// C API Export for Python ctypes / External Invocations (Host Memory)
// ---------------------------------------------------------------------------
LRD_API int compute_lrd_cuda(
    const float* h_reach_dists,
    float* h_lrd,
    int N,
    int k,
    float eps = LOF_DEFAULT_EPS
);

#endif  // LRD_KERNEL_CUH
