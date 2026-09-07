/**
 * cuda_lof_pipeline.cuh - Complete End-to-End CUDA LOF Baseline Pipeline
 *
 * Project: CUDA LOF Cybersecurity
 * Milestone: M3 - CUDA LOF Baseline (Step 4)
 *
 * Orchestrates the full 5-stage native CUDA LOF algorithm:
 *   1. Pairwise Euclidean Distance Matrix (sm_86)
 *   2. k-Nearest Neighbors (k-NN) with self-distance exclusion
 *   3. Reachability Distance calculation
 *   4. Local Reachability Density (LRD)
 *   5. Local Outlier Factor (LOF) scoring
 */

#ifndef CUDA_LOF_PIPELINE_CUH
#define CUDA_LOF_PIPELINE_CUH

#include <cuda_runtime.h>
#include "../kernels/distance_kernel.cuh"
#include "../kernels/knn_kernel.cuh"
#include "../kernels/reachability_kernel.cuh"
#include "../kernels/lrd_kernel.cuh"
#include "../kernels/lof_kernel.cuh"

#ifdef _WIN32
  #define PIPELINE_API extern "C" __declspec(dllexport)
#else
  #define PIPELINE_API extern "C"
#endif

// ---------------------------------------------------------------------------
// Timing Structure for CUDA Hardware Events
// ---------------------------------------------------------------------------
struct CudaLofTimings {
    float h2d_ms;
    float distance_kernel_ms;
    float knn_kernel_ms;
    float reachability_kernel_ms;
    float lrd_kernel_ms;
    float lof_kernel_ms;
    float d2h_ms;
    float total_kernel_ms;
    float end_to_end_ms;
};

// ---------------------------------------------------------------------------
// End-to-End Native CUDA LOF Pipeline
// ---------------------------------------------------------------------------
PIPELINE_API int compute_lof_pipeline_cuda(
    const float* h_X,
    float* h_lof_scores,
    int N,
    int D,
    int k,
    float eps = LOF_DEFAULT_EPS
);

PIPELINE_API int compute_lof_pipeline_cuda_timed(
    const float* h_X,
    float* h_lof_scores,
    int N,
    int D,
    int k,
    float eps,
    CudaLofTimings* out_timings
);

#endif  // CUDA_LOF_PIPELINE_CUH

