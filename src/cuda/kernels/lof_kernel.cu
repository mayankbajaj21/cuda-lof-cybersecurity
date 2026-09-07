/**
 * lof_kernel.cu - CUDA Local Outlier Factor (LOF) Scoring Naive Baseline Implementation
 *
 * Project: CUDA LOF Cybersecurity
 * Milestone: M3 - CUDA LOF Baseline (Step 4)
 *
 * Formula (M2 Reproduction):
 *   LOF_k(p) = mean_{o in N_k(p)} (lrd(o)) / (lrd(p) + eps)
 */

#include "lof_kernel.cuh"
#include "../common/cuda_utils.cuh"
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Naive Row-Parallel LOF Scoring Kernel
// ---------------------------------------------------------------------------
__global__ void lof_scoring_naive_kernel(
    const float* __restrict__ lrd,
    const int* __restrict__ knn_indices,
    float* __restrict__ lof_scores,
    int N,
    int k,
    float eps
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    if (k <= 0) {
        return;
    }

    float p_lrd = lrd[i];
    const int* neighbors = knn_indices + (size_t)i * k;
    float sum_neighbor_lrd = 0.0f;

    for (int m = 0; m < k; ++m) {
        int o = neighbors[m];
        if (o >= 0 && o < N) {
            sum_neighbor_lrd += lrd[o];
        }
    }

    float avg_neighbor_lrd = sum_neighbor_lrd / (float)k;
    lof_scores[i] = avg_neighbor_lrd / (p_lrd + eps);
}

// ---------------------------------------------------------------------------
// Host Kernel Launcher (Operates directly on device memory)
// ---------------------------------------------------------------------------
cudaError_t launch_lof_scoring_kernel(
    const float* d_lrd,
    const int* d_knn_indices,
    float* d_lof_scores,
    int N,
    int k,
    float eps,
    cudaStream_t stream
) {
    if (N <= 0 || k <= 0 || !d_lrd || !d_knn_indices || !d_lof_scores) {
        return cudaErrorInvalidValue;
    }

    const int threads_per_block = 256;
    const int blocks = (N + threads_per_block - 1) / threads_per_block;

    lof_scoring_naive_kernel<<<blocks, threads_per_block, 0, stream>>>(
        d_lrd, d_knn_indices, d_lof_scores, N, k, eps
    );

    return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// C API Export for Python ctypes / External Invocations (Host Memory)
// ---------------------------------------------------------------------------
LOF_API int compute_lof_scores_cuda(
    const float* h_lrd,
    const int* h_knn_indices,
    float* h_lof_scores,
    int N,
    int k,
    float eps
) {
    if (!h_lrd || !h_knn_indices || !h_lof_scores) {
        fprintf(stderr, "[CUDA ERROR] Null pointer passed to compute_lof_scores_cuda\n");
        return -1;
    }
    if (N <= 0 || k <= 0) {
        fprintf(stderr, "[CUDA ERROR] Invalid parameters for compute_lof_scores_cuda (N=%d, k=%d)\n", N, k);
        return -2;
    }

    size_t lrd_bytes = (size_t)N * sizeof(float);
    size_t idx_bytes = (size_t)N * k * sizeof(int);

    float* d_lrd = nullptr;
    int* d_knn_indices = nullptr;
    float* d_lof_scores = nullptr;

    cudaError_t err = cudaMalloc(&d_lrd, lrd_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMalloc failed for lrd (%zu bytes): %s\n",
                lrd_bytes, cudaGetErrorString(err));
        return -3;
    }

    err = cudaMalloc(&d_knn_indices, idx_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMalloc failed for knn_indices (%zu bytes): %s\n",
                idx_bytes, cudaGetErrorString(err));
        cudaFree(d_lrd);
        return -4;
    }

    err = cudaMalloc(&d_lof_scores, lrd_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMalloc failed for lof_scores (%zu bytes): %s\n",
                lrd_bytes, cudaGetErrorString(err));
        cudaFree(d_lrd);
        cudaFree(d_knn_indices);
        return -5;
    }

    err = cudaMemcpy(d_lrd, h_lrd, lrd_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMemcpy H2D failed for lrd: %s\n", cudaGetErrorString(err));
        cudaFree(d_lrd);
        cudaFree(d_knn_indices);
        cudaFree(d_lof_scores);
        return -6;
    }

    err = cudaMemcpy(d_knn_indices, h_knn_indices, idx_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMemcpy H2D failed for knn_indices: %s\n", cudaGetErrorString(err));
        cudaFree(d_lrd);
        cudaFree(d_knn_indices);
        cudaFree(d_lof_scores);
        return -7;
    }

    err = launch_lof_scoring_kernel(d_lrd, d_knn_indices, d_lof_scores, N, k, eps);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_lrd);
        cudaFree(d_knn_indices);
        cudaFree(d_lof_scores);
        return -8;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_lrd);
        cudaFree(d_knn_indices);
        cudaFree(d_lof_scores);
        return -9;
    }

    err = cudaMemcpy(h_lof_scores, d_lof_scores, lrd_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMemcpy D2H failed for lof_scores: %s\n", cudaGetErrorString(err));
        cudaFree(d_lrd);
        cudaFree(d_knn_indices);
        cudaFree(d_lof_scores);
        return -10;
    }

    cudaFree(d_lrd);
    cudaFree(d_knn_indices);
    cudaFree(d_lof_scores);

    return 0; // Success
}
