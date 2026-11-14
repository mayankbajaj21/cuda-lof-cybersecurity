/**
 * knn_kernel_optimized.cuh - Register-Right-Sized k-NN Kernel Declarations
 *
 * Project: CUDA LOF Cybersecurity
 * Milestone: M5 - Profiling-Driven CUDA Optimization (Step 2: k-NN)
 *
 * M4 profiling evidence that justifies this optimization:
 *   The naive k-NN kernel (src/cuda/kernels/knn_kernel.cu) declares its
 *   per-thread top-k buffers as `float best_dist[KNN_MAX_K]` with
 *   KNN_MAX_K = 256, even though the project always runs with k = 20.
 *   256 floats + 256 ints = 2048 bytes of per-thread local state is far
 *   beyond what the register file can hold per thread, so the compiler
 *   spills most of this array to local memory (physically global memory,
 *   cached but far slower than registers), on every one of the N threads,
 *   for the entire O(N) scan loop. This optimized kernel keeps the exact
 *   same insertion-sort top-k algorithm (no approximation) but right-sizes
 *   the buffer to OPT_KNN_MAX_K = 32 - enough headroom above the project's
 *   k = 20 default to stay safe, small enough to fit in registers without
 *   spilling for most launch configurations.
 *
 * Constraint: this optimized kernel only supports k <= OPT_KNN_MAX_K (32).
 * The naive kernel's k <= KNN_MAX_K (256) ceiling remains available via
 * the unmodified naive path for callers that need larger k.
 */

#ifndef KNN_KERNEL_OPTIMIZED_CUH
#define KNN_KERNEL_OPTIMIZED_CUH

#include <cuda_runtime.h>

#ifdef _WIN32
  #define KNN_OPT_API extern "C" __declspec(dllexport)
#else
  #define KNN_OPT_API extern "C"
#endif

// Right-sized maximum k - large enough for this project's k=20 default
// with headroom, small enough to avoid register spill to local memory.
#define OPT_KNN_MAX_K 32

// ---------------------------------------------------------------------------
// CUDA Kernel Declaration
// ---------------------------------------------------------------------------
__global__ void knn_optimized_kernel(
    const float* __restrict__ dist_matrix,
    int* __restrict__ neighbor_indices,
    float* __restrict__ neighbor_distances,
    int N,
    int k
);

// ---------------------------------------------------------------------------
// Host Kernel Launcher (Operates directly on device memory)
// ---------------------------------------------------------------------------
cudaError_t launch_knn_kernel_optimized(
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
KNN_OPT_API int compute_knn_cuda_optimized(
    const float* h_dist_matrix,
    int* h_neighbor_indices,
    float* h_neighbor_distances,
    int N,
    int k
);

#endif  // KNN_KERNEL_OPTIMIZED_CUH
