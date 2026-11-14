/**
 * distance_kernel_optimized.cuh - Shared-Memory Tiled Pairwise Euclidean Distance Kernel
 *
 * Project: CUDA LOF Cybersecurity
 * Milestone: M5 - Profiling-Driven CUDA Optimization (Step 1: Pairwise Distance)
 *
 * M4 profiling evidence that justifies this optimization:
 *   On this machine (RTX 4050 Laptop, sm_89), the naive pairwise-distance
 *   kernel (src/cuda/kernels/distance_kernel.cu) consumed 78.9%/80.0% of
 *   total kernel time at N=10K/25K. Its global-memory access pattern reads
 *   each feature row D times per distance computed with NO reuse across
 *   threads in a warp/block, even though every row of X is read by many
 *   (i, j) pairs. This optimized kernel tiles X into shared memory so each
 *   feature value is loaded from global memory once per TILE output rows
 *   (or columns) instead of once per distance computed, trading redundant
 *   global loads for cheap shared-memory broadcasts.
 *
 * Math is IDENTICAL to the naive kernel (no approximation):
 *   d(i, j) = sqrt( sum_{d=0}^{D-1} (X[i, d] - X[j, d])^2 ), d(i, i) = 0.
 *
 * Tile size: 16x16 (256 threads/block - deliberately NOT 32x32/1024
 * threads). A first attempt at DIST_TILE=32 measured ~5x SLOWER than the
 * naive kernel: 1024 threads/block is the hardware maximum, so only one
 * block is resident per SM at a time, leaving no second block to hide
 * __syncthreads()/memory latency behind - occupancy collapsed. 16x16
 * (256 threads/block, matching the naive kernel's own block size) lets
 * several blocks reside per SM simultaneously while still reusing every
 * loaded feature value across 16 output distances instead of 1. Shared
 * memory cost per block = 2 * 16 * 16 * 4 bytes = 2048 bytes, far under
 * this GPU's 49152 bytes/block budget. Arbitrary D (not required to be a
 * multiple of 16) is supported via a bounds-checked tail chunk.
 */

#ifndef DISTANCE_KERNEL_OPTIMIZED_CUH
#define DISTANCE_KERNEL_OPTIMIZED_CUH

#include <cuda_runtime.h>

#ifdef _WIN32
  #define DISTANCE_OPT_API extern "C" __declspec(dllexport)
#else
  #define DISTANCE_OPT_API extern "C"
#endif

#define DIST_TILE 16

// ---------------------------------------------------------------------------
// Kernel declaration (shared-memory tiled 2D grid/block)
// ---------------------------------------------------------------------------
__global__ void pairwise_distance_tiled_kernel(
    const float* __restrict__ X,
    float* __restrict__ dist_matrix,
    int N,
    int D
);

// ---------------------------------------------------------------------------
// Host launcher declaration (operates on device pointers)
// ---------------------------------------------------------------------------
cudaError_t launch_pairwise_distance_kernel_optimized(
    const float* d_X,
    float* d_dist_matrix,
    int N,
    int D,
    cudaStream_t stream = 0
);

// ---------------------------------------------------------------------------
// C-compatible API for Python ctypes / external callers (host memory in/out)
// ---------------------------------------------------------------------------
DISTANCE_OPT_API int compute_pairwise_distances_cuda_optimized(
    const float* h_X,
    float* h_dist_matrix,
    int N,
    int D
);

#endif  // DISTANCE_KERNEL_OPTIMIZED_CUH
