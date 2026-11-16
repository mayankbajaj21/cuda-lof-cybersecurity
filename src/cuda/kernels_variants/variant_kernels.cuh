/**
 * variant_kernels.cuh - M8 advanced (outside-syllabus) optimization kernels
 *
 * Project: CUDA LOF Cybersecurity
 * Milestone: M8 - Advanced Optimizations
 *
 * Every variant starts from the M5 syllabus-optimized kernels (V1:
 * src/cuda/kernels_optimized/) and adds ONE technique, except V5 which
 * combines all three:
 *
 *   V2  register blocking / thread coarsening (fp32, exact)
 *       16x16 threads, each computing a 4x4 micro-tile -> 64x64 output
 *       tile per block. Partial sums live in registers, feature chunks in
 *       padded shared memory ([64][17]) so the per-column reads are
 *       bank-conflict free.
 *   V3  asynchronous global->shared copies + double buffering (fp32, exact)
 *       V1's 16x16 tiling, but tile c+1 is fetched with cp.async while
 *       tile c is being consumed. Falls back to synchronous loads below
 *       sm_80 so the default sm_86 build still runs.
 *   V4  FP16 storage (APPROXIMATE)
 *       Features and the N x N distance matrix are stored as __half
 *       (halves the dominant N^2 allocation); arithmetic and accumulation
 *       stay fp32. Feature rows are re-laid out with a stride padded to a
 *       multiple of 8 halves (zero-filled) so 16-byte copies stay aligned.
 *   V5  V2 + V3 + V4 combined (APPROXIMATE)
 *
 * All exact variants sum squared differences in the same sequential
 * feature order as V0/V1, so their distances are bit-identical to V1.
 */

#ifndef VARIANT_KERNELS_CUH
#define VARIANT_KERNELS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define RB_THREADS 16
#define RB_MICRO 4
#define RB_TILE (RB_THREADS * RB_MICRO)
#define RB_DK 16

#define FP16_PAD_MULTIPLE 8
#define FP16_MAX_FINITE 65504.0f

#define V3_TILE 16
#define V5_SMEM_STRIDE 24

int fp16_padded_dim(int D);

__global__ void convert_features_to_half_padded_kernel(
    const float* __restrict__ X, __half* __restrict__ Xh, int N, int D, int Dp);

__global__ void pairwise_distance_regblock_kernel(
    const float* __restrict__ X, float* __restrict__ dist, int N, int D);

__global__ void pairwise_distance_tiled_async_kernel(
    const float* __restrict__ X, float* __restrict__ dist, int N, int D);

__global__ void pairwise_distance_tiled_fp16_kernel(
    const __half* __restrict__ Xh, __half* __restrict__ dist, int N, int Dp,
    unsigned int* __restrict__ overflow_count);

__global__ void pairwise_distance_combined_fp16_kernel(
    const __half* __restrict__ Xh, __half* __restrict__ dist, int N, int Dp,
    unsigned int* __restrict__ overflow_count);

__global__ void knn_fp16_kernel(
    const __half* __restrict__ dist, int* __restrict__ neighbor_indices,
    float* __restrict__ neighbor_distances, int N, int k);

cudaError_t launch_convert_features_to_half(const float* d_X, __half* d_Xh, int N, int D, int Dp);
cudaError_t launch_distance_regblock(const float* d_X, float* d_dist, int N, int D);
cudaError_t launch_distance_tiled_async(const float* d_X, float* d_dist, int N, int D);
cudaError_t launch_distance_tiled_fp16(const __half* d_Xh, __half* d_dist, int N, int Dp, unsigned int* d_overflow);
cudaError_t launch_distance_combined_fp16(const __half* d_Xh, __half* d_dist, int N, int Dp, unsigned int* d_overflow);
cudaError_t launch_knn_fp16(const __half* d_dist, int* d_idx, float* d_knn_dist, int N, int k);

#endif  // VARIANT_KERNELS_CUH
