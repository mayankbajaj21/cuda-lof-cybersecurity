/**
 * cuda_lof_pipeline_variants.cuh - M8 single-entry LOF pipeline for V0..V5
 *
 *   V0 naive                       (src/cuda/kernels/)
 *   V1 syllabus-optimized (M5)     (src/cuda/kernels_optimized/)
 *   V2 V1 + register blocking                      exact
 *   V3 V1 + cp.async double buffering              exact
 *   V4 V1 + FP16 storage                           APPROXIMATE
 *   V5 register blocking + cp.async + FP16         APPROXIMATE
 *
 * Reachability / LRD / LOF-scoring always use the original naive kernels.
 * Allocation order mirrors the original pipelines (each buffer is freed as
 * soon as the next stage no longer needs it). GPU memory is reported two
 * ways: a running tally of the pipeline's own cudaMalloc bytes, and the
 * device-wide used-memory delta from cudaMemGetInfo sampled after every
 * allocation.
 */

#ifndef CUDA_LOF_PIPELINE_VARIANTS_CUH
#define CUDA_LOF_PIPELINE_VARIANTS_CUH

#include <cuda_runtime.h>

#ifdef _WIN32
  #define VARIANTS_API extern "C" __declspec(dllexport)
#else
  #define VARIANTS_API extern "C"
#endif

#define LOF_VARIANT_COUNT 6

struct VariantTimings {
    float h2d_ms;
    float fp16_convert_ms;
    float distance_kernel_ms;
    float knn_kernel_ms;
    float reachability_kernel_ms;
    float lrd_kernel_ms;
    float lof_kernel_ms;
    float d2h_ms;
    float total_kernel_ms;
    float end_to_end_ms;
};

struct VariantMemStats {
    unsigned long long tracked_peak_bytes;
    unsigned long long memgetinfo_peak_used_delta_bytes;
    unsigned long long memgetinfo_free_before_bytes;
    unsigned long long memgetinfo_total_bytes;
    unsigned long long dist_matrix_bytes;
    unsigned int fp16_overflow_count;
    int dist_element_bytes;
};

struct VariantKernelAttributes {
    int num_regs;
    int static_shared_bytes;
    int local_bytes;
    int max_threads_per_block;
    int block_threads;
    int max_active_blocks_per_sm;
    float theoretical_occupancy_pct;
};

// h_knn_indices_out may be NULL; when given (N*k ints) it receives the
// final neighbor indices, copied after the end-to-end timer stops.
VARIANTS_API int compute_lof_pipeline_variant(
    const float* h_X, float* h_lof_scores, int N, int D, int k, float eps,
    int variant, VariantTimings* out_timings, VariantMemStats* out_mem,
    int* h_knn_indices_out);

// which: 0 = distance kernel, 1 = k-NN kernel. Source: cudaFuncGetAttributes.
VARIANTS_API int get_variant_kernel_attributes(int variant, int which, VariantKernelAttributes* out);

VARIANTS_API int variant_max_k(int variant);

#endif  // CUDA_LOF_PIPELINE_VARIANTS_CUH
