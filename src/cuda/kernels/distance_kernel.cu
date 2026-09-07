/**
 * distance_kernel.cu - CUDA Pairwise Euclidean Distance Matrix Kernel Implementation
 *
 * Project: CUDA LOF Cybersecurity
 * Milestone: M3 - CUDA LOF Baseline (Step 2)
 *
 * Mathematical Formulation:
 *   d(i, j) = sqrt( sum_{d=0}^{D-1} (X[i, d] - X[j, d])^2 )
 *   d(i, i) = 0.0f
 *
 * Designed for:
 *   - N samples (arbitrary N, validated on 10K and 25K)
 *   - D = 68 standardized numerical features
 *   - float32 row-major contiguous feature matrix X
 *   - float32 row-major contiguous distance matrix (N x N)
 *   - Straightforward 2D grid/block naive implementation
 *   - sm_86 architecture (RTX 3050 Laptop GPU)
 */

#include "distance_kernel.cuh"
#include "../common/cuda_utils.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Naive Pairwise Euclidean Distance Kernel
// ---------------------------------------------------------------------------
__global__ void pairwise_distance_naive_kernel(
    const float* __restrict__ X,
    float* __restrict__ dist_matrix,
    int N,
    int D
) {
    // 2D grid decomposition: thread (x, y) computes distance between sample i and sample j
    int j = blockIdx.x * blockDim.x + threadIdx.x; // column index (sample j)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row index (sample i)

    // Boundary check for arbitrary N
    if (i >= N || j >= N) {
        return;
    }

    // Explicitly set self-distance on the diagonal to exactly zero
    if (i == j) {
        dist_matrix[(size_t)i * N + j] = 0.0f;
        return;
    }

    // Accumulate sum of squared differences across all D features
    float sum_sq = 0.0f;
    const float* row_i = X + (size_t)i * D;
    const float* row_j = X + (size_t)j * D;

    for (int d = 0; d < D; ++d) {
        float diff = row_i[d] - row_j[d];
        sum_sq += diff * diff;
    }

    // Euclidean distance
    dist_matrix[(size_t)i * N + j] = sqrtf(sum_sq);
}

// ---------------------------------------------------------------------------
// Host Kernel Launcher (Operates directly on device memory)
// ---------------------------------------------------------------------------
cudaError_t launch_pairwise_distance_kernel(
    const float* d_X,
    float* d_dist_matrix,
    int N,
    int D,
    cudaStream_t stream
) {
    if (N <= 0 || D <= 0 || !d_X || !d_dist_matrix) {
        return cudaErrorInvalidValue;
    }

    // 16x16 threads per block = 256 threads per block
    dim3 block(16, 16);
    dim3 grid(
        (N + block.x - 1) / block.x,
        (N + block.y - 1) / block.y
    );

    pairwise_distance_naive_kernel<<<grid, block, 0, stream>>>(
        d_X, d_dist_matrix, N, D
    );

    return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// C API Export for Python ctypes / External Invocations
// ---------------------------------------------------------------------------
DISTANCE_API int compute_pairwise_distances_cuda(
    const float* h_X,
    float* h_dist_matrix,
    int N,
    int D
) {
    if (!h_X || !h_dist_matrix) {
        fprintf(stderr, "[CUDA ERROR] Null pointer passed to compute_pairwise_distances_cuda\n");
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

    // Allocate device memory for feature matrix X
    cudaError_t err = cudaMalloc(&d_X, x_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMalloc failed for feature matrix X (%zu bytes): %s\n",
                x_bytes, cudaGetErrorString(err));
        return -3;
    }

    // Allocate device memory for dense N x N distance matrix
    err = cudaMalloc(&d_dist_matrix, dist_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMalloc failed for dist_matrix (%zu bytes = %.2f MB): %s\n",
                dist_bytes, (double)dist_bytes / (1024.0 * 1024.0), cudaGetErrorString(err));
        cudaFree(d_X);
        return -4;
    }

    // Copy feature matrix from host to device
    err = cudaMemcpy(d_X, h_X, x_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMemcpy H2D failed for X: %s\n", cudaGetErrorString(err));
        cudaFree(d_X);
        cudaFree(d_dist_matrix);
        return -5;
    }

    // Launch naive pairwise distance kernel
    err = launch_pairwise_distance_kernel(d_X, d_dist_matrix, N, D);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_X);
        cudaFree(d_dist_matrix);
        return -6;
    }

    // Synchronize to ensure kernel completes
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_X);
        cudaFree(d_dist_matrix);
        return -7;
    }

    // Copy distance matrix from device back to host
    err = cudaMemcpy(h_dist_matrix, d_dist_matrix, dist_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMemcpy D2H failed for dist_matrix: %s\n",
                cudaGetErrorString(err));
        cudaFree(d_X);
        cudaFree(d_dist_matrix);
        return -8;
    }

    // Clean up device memory
    cudaFree(d_X);
    cudaFree(d_dist_matrix);

    return 0; // Success
}
