/**
 * cuda_lof_pipeline_optimized.cuh - Optimized End-to-End CUDA LOF Pipeline
 *
 * Project: CUDA LOF Cybersecurity
 * Milestone: M5 - Profiling-Driven CUDA Optimization
 *
 * Same 5-stage pipeline and VRAM-freeing strategy as the naive baseline
 * (src/cuda/lof/cuda_lof_pipeline.cu), but stages 1-2 (the two kernels M4
 * measured as the bottleneck, 78.9-80.0% and 19.9-20.9% of kernel time)
 * use the optimized kernels in src/cuda/kernels_optimized/. Stages 3-5
 * (reachability, LRD, LOF scoring) are measured by M4 as sub-millisecond
 * and together under 0.2% of total kernel time at both 10K and 25K - no
 * optimization was attempted there; this pipeline reuses the ORIGINAL,
 * unmodified reachability/LRD/LOF-scoring kernels for completeness.
 */

#ifndef CUDA_LOF_PIPELINE_OPTIMIZED_CUH
#define CUDA_LOF_PIPELINE_OPTIMIZED_CUH

#include <cuda_runtime.h>
#include "../kernels_optimized/distance_kernel_optimized.cuh"
#include "../kernels_optimized/knn_kernel_optimized.cuh"
#include "../kernels/reachability_kernel.cuh"
#include "../kernels/lrd_kernel.cuh"
#include "../kernels/lof_kernel.cuh"

#ifdef _WIN32
  #define PIPELINE_OPT_API extern "C" __declspec(dllexport)
#else
  #define PIPELINE_OPT_API extern "C"
#endif

// ---------------------------------------------------------------------------
// Timing Structure for CUDA Hardware Events (identical layout to the naive
// pipeline's CudaLofTimings so the Python ctypes.Structure binding is reusable)
// ---------------------------------------------------------------------------
struct CudaLofTimingsOptimized {
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
// End-to-End Optimized CUDA LOF Pipeline
// ---------------------------------------------------------------------------
PIPELINE_OPT_API int compute_lof_pipeline_cuda_optimized(
    const float* h_X,
    float* h_lof_scores,
    int N,
    int D,
    int k,
    float eps = LOF_DEFAULT_EPS
);

PIPELINE_OPT_API int compute_lof_pipeline_cuda_optimized_timed(
    const float* h_X,
    float* h_lof_scores,
    int N,
    int D,
    int k,
    float eps,
    CudaLofTimingsOptimized* out_timings
);

#endif  // CUDA_LOF_PIPELINE_OPTIMIZED_CUH
