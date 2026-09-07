/**
 * knn_kernel.cuh - CUDA k-Nearest Neighbors (k-NN) Kernel Declarations
 *
 * Project: CUDA LOF Cybersecurity
 * Milestone: M3 - CUDA LOF Baseline (Step 3)
 *
 * Computes:
 *   For each sample i, identifies its k nearest distinct neighbors (excluding i == j),
 *   sorted in ascending order of distance:
 *     neighbor_distances[i, 0] <= neighbor_distances[i, 1] <= ... <= neighbor_distances[i, k-1]
 *   and records the corresponding neighbor sample indices.
 *
 * Input:
 *   - dist_matrix: (N, N) float32 row-major pairwise Euclidean distance matrix
 *   - k: number of neighbors (default k=20)
 *
 * Output:
 *   - neighbor_indices: (N, k) int32 row-major array
 *   - neighbor_distances: (N, k) float32 row-major array
 */

#ifndef KNN_KERNEL_CUH
#define KNN_KERNEL_CUH

#include <cuda_runtime.h>

#ifdef _WIN32
  #define KNN_API extern "C" __declspec(dllexport)
#else
  #define KNN_API extern "C"
#endif

// Maximum supported k in thread registers / local memory
#define KNN_MAX_K 256

// ---------------------------------------------------------------------------
// CUDA Kernel Declaration
// ---------------------------------------------------------------------------
__global__ void knn_naive_kernel(
    const float* __restrict__ dist_matrix,
    int* __restrict__ neighbor_indices,
    float* __restrict__ neighbor_distances,
    int N,
    int k
);

// ---------------------------------------------------------------------------
// Host Kernel Launcher (Operates directly on device memory)
// ---------------------------------------------------------------------------
cudaError_t launch_knn_kernel(
    const float* d_dist_matrix,
    int* d_neighbor_indices,
    float* d_neighbor_distances,
    int N,
    int k,
    cudaStream_t stream = 0
);

// ---------------------------------------------------------------------------
// C API Export for Python ctypes / External Invocations (Host Memory)
// ---------------------------------------------------------------------------
KNN_API int compute_knn_cuda(
    const float* h_dist_matrix,
    int* h_neighbor_indices,
    float* h_neighbor_distances,
    int N,
    int k
);

#endif  // KNN_KERNEL_CUH
