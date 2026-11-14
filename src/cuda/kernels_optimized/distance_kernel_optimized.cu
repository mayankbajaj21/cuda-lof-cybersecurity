/**
 * distance_kernel_optimized.cu - Shared-Memory Tiled Pairwise Distance Implementation
 *
 * Project: CUDA LOF Cybersecurity
 * Milestone: M5 - Profiling-Driven CUDA Optimization (Step 1: Pairwise Distance)
 *
 * See distance_kernel_optimized.cuh for the M4 profiling evidence and
 * rationale. Same math as the naive baseline; only the memory-access
 * strategy changes.
 */

#include "distance_kernel_optimized.cuh"
#include "../common/cuda_utils.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Shared-Memory Tiled Pairwise Euclidean Distance Kernel
//
// Each block computes a DIST_TILE x DIST_TILE sub-block of the output
// distance matrix. The feature dimension D is processed in chunks of
// DIST_TILE. For each chunk, the block cooperatively loads:
//   row_tile[ty][c] = X[row0 + ty, d0 + c]   (coalesced: c == threadIdx.x)
//   col_tile[ty][c] = X[col0 + ty, d0 + c]   (coalesced: c == threadIdx.x)
// into shared memory once, then every thread in the block reuses both
// tiles via shared-memory broadcast reads instead of re-fetching from
// global memory per (i, j) pair.
// ---------------------------------------------------------------------------
__global__ void pairwise_distance_tiled_kernel(
    const float* __restrict__ X,
    float* __restrict__ dist_matrix,
    int N,
    int D
) {
    __shared__ float row_tile[DIST_TILE][DIST_TILE];
    __shared__ float col_tile[DIST_TILE][DIST_TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row0 = blockIdx.y * DIST_TILE;
    const int col0 = blockIdx.x * DIST_TILE;

    const int i = row0 + ty; // output row this thread is responsible for
    const int j = col0 + tx; // output column this thread is responsible for

    float sum_sq = 0.0f;

    const int n_chunks = (D + DIST_TILE - 1) / DIST_TILE;
    for (int c = 0; c < n_chunks; ++c) {
        const int d0 = c * DIST_TILE;
        const int d_row = d0 + tx; // feature index loaded by this thread for row_tile
        const int d_col = d0 + tx; // feature index loaded by this thread for col_tile

        // Coalesced load: threads with consecutive tx read consecutive
        // feature addresses for the SAME sample row (row0 + ty) / (col0 + ty).
        row_tile[ty][tx] = (i < N && d_row < D)
            ? X[(size_t)(row0 + ty) * D + d_row]
            : 0.0f;
        col_tile[ty][tx] = (col0 + ty < N && d_col < D)
            ? X[(size_t)(col0 + ty) * D + d_col]
            : 0.0f;

        __syncthreads();

        // Accumulate squared differences for this chunk using the
        // broadcast shared-memory tiles (no further global memory traffic).
        // No bounds check needed here: out-of-range feature slots were
        // already zero-padded above, so padded (row-col) terms are 0-0=0
        // and contribute nothing - adding a branch here would only cost
        // warp divergence for zero benefit.
        #pragma unroll
        for (int t = 0; t < DIST_TILE; ++t) {
            float diff = row_tile[ty][t] - col_tile[tx][t];
            sum_sq += diff * diff;
        }

        __syncthreads();
    }

    if (i < N && j < N) {
        dist_matrix[(size_t)i * N + j] = (i == j) ? 0.0f : sqrtf(sum_sq);
    }
}

// ---------------------------------------------------------------------------
// Host Kernel Launcher (Operates directly on device memory)
// ---------------------------------------------------------------------------
cudaError_t launch_pairwise_distance_kernel_optimized(
    const float* d_X,
    float* d_dist_matrix,
    int N,
    int D,
    cudaStream_t stream
) {
    if (N <= 0 || D <= 0 || !d_X || !d_dist_matrix) {
        return cudaErrorInvalidValue;
    }

    dim3 block(DIST_TILE, DIST_TILE);
    dim3 grid(
        (N + DIST_TILE - 1) / DIST_TILE,
        (N + DIST_TILE - 1) / DIST_TILE
    );

    pairwise_distance_tiled_kernel<<<grid, block, 0, stream>>>(
        d_X, d_dist_matrix, N, D
    );

    return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// C API Export for Python ctypes / External Invocations
// ---------------------------------------------------------------------------
DISTANCE_OPT_API int compute_pairwise_distances_cuda_optimized(
    const float* h_X,
    float* h_dist_matrix,
    int N,
    int D
) {
    if (!h_X || !h_dist_matrix) {
        fprintf(stderr, "[CUDA ERROR] Null pointer passed to compute_pairwise_distances_cuda_optimized\n");
        return -1;
    }
    if (N <= 0 || D <= 0) {
        fprintf(stderr, "[CUDA ERROR] Invalid dimensions (N=%d, D=%d)\n", N, D);
        return -2;
    }

    size_t x_bytes = (size_t)N * D * sizeof(float);
    size_t dist_bytes = (size_t)N * N * sizeof(float);

    float* d_X = nullptr;
    float* d_dist_matrix = nullptr;

    cudaError_t err = cudaMalloc(&d_X, x_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMalloc failed for X (%zu bytes): %s\n", x_bytes, cudaGetErrorString(err));
        return -3;
    }

    err = cudaMalloc(&d_dist_matrix, dist_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMalloc failed for dist_matrix (%zu bytes): %s\n", dist_bytes, cudaGetErrorString(err));
        cudaFree(d_X);
        return -4;
    }

    err = cudaMemcpy(d_X, h_X, x_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMemcpy H2D failed for X: %s\n", cudaGetErrorString(err));
        cudaFree(d_X);
        cudaFree(d_dist_matrix);
        return -5;
    }

    err = launch_pairwise_distance_kernel_optimized(d_X, d_dist_matrix, N, D);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_X);
        cudaFree(d_dist_matrix);
        return -6;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_X);
        cudaFree(d_dist_matrix);
        return -7;
    }

    err = cudaMemcpy(h_dist_matrix, d_dist_matrix, dist_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMemcpy D2H failed for dist_matrix: %s\n", cudaGetErrorString(err));
        cudaFree(d_X);
        cudaFree(d_dist_matrix);
        return -8;
    }

    cudaFree(d_X);
    cudaFree(d_dist_matrix);

    return 0;
}
