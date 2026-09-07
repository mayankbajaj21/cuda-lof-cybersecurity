/**
 * reachability_kernel.cuh - CUDA Reachability Distance Kernel Declarations
 *
 * Project: CUDA LOF Cybersecurity
 * Milestone: M3 - CUDA LOF Baseline (Step 4)
 *
 * Formula (M2 Reproduction):
 *   reach-dist_k(p, o) = max(k-distance(o), d(p, o))
 *
 * where:
 *   - p is query sample i
 *   - o is the m-th neighbor of p (knn_indices[i, m])
 *   - d(p, o) is the direct distance knn_distances[i, m]
 *   - k-distance(o) is the distance from neighbor o to its k-th nearest neighbor (knn_distances[o, k-1])
 *
 * Input:
 *   - knn_indices: (N, k) int32 row-major array
 *   - knn_distances: (N, k) float32 row-major array
 *
 * Output:
 *   - reach_dists: (N, k) float32 row-major array
 */

#ifndef REACHABILITY_KERNEL_CUH
#define REACHABILITY_KERNEL_CUH

#include <cuda_runtime.h>

#ifdef _WIN32
  #define REACH_API extern "C" __declspec(dllexport)
#else
  #define REACH_API extern "C"
#endif

// ---------------------------------------------------------------------------
// CUDA Kernel Declaration
// ---------------------------------------------------------------------------
__global__ void reachability_distance_naive_kernel(
    const int* __restrict__ knn_indices,
    const float* __restrict__ knn_distances,
    float* __restrict__ reach_dists,
    int N,
    int k
);

// ---------------------------------------------------------------------------
// Host Kernel Launcher (Operates directly on device memory)
// ---------------------------------------------------------------------------
cudaError_t launch_reachability_distance_kernel(
    const int* d_knn_indices,
    const float* d_knn_distances,
    float* d_reach_dists,
    int N,
    int k,
    cudaStream_t stream = 0
);

// ---------------------------------------------------------------------------
// C API Export for Python ctypes / External Invocations (Host Memory)
// ---------------------------------------------------------------------------
REACH_API int compute_reachability_distances_cuda(
    const int* h_knn_indices,
    const float* h_knn_distances,
    float* h_reach_dists,
    int N,
    int k
);

#endif  // REACHABILITY_KERNEL_CUH
