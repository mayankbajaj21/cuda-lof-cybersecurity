/**
 * lrd_kernel.cu - CUDA Local Reachability Density (LRD) Naive Baseline Implementation
 *
 * Project: CUDA LOF Cybersecurity
 * Milestone: M3 - CUDA LOF Baseline (Step 4)
 *
 * Formula (M2 Reproduction):
 *   lrd_k(p) = 1.0 / ( ( (1/k) * sum_{m=0}^{k-1} reach_dists[p, m] ) + eps )
 */

#include "lrd_kernel.cuh"
#include "../common/cuda_utils.cuh"
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Naive Row-Parallel LRD Kernel
// ---------------------------------------------------------------------------
__global__ void lrd_naive_kernel(
    const float* __restrict__ reach_dists,
    float* __restrict__ lrd,
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

    const float* row = reach_dists + (size_t)i * k;
    float sum_reach = 0.0f;

    for (int m = 0; m < k; ++m) {
        sum_reach += row[m];
    }

    float avg_reach = sum_reach / (float)k;
    lrd[i] = 1.0f / (avg_reach + eps);
}

// ---------------------------------------------------------------------------
// Host Kernel Launcher (Operates directly on device memory)
// ---------------------------------------------------------------------------
cudaError_t launch_lrd_kernel(
    const float* d_reach_dists,
    float* d_lrd,
    int N,
    int k,
    float eps,
    cudaStream_t stream
) {
    if (N <= 0 || k <= 0 || !d_reach_dists || !d_lrd) {
        return cudaErrorInvalidValue;
    }

    const int threads_per_block = 256;
    const int blocks = (N + threads_per_block - 1) / threads_per_block;

    lrd_naive_kernel<<<blocks, threads_per_block, 0, stream>>>(
        d_reach_dists, d_lrd, N, k, eps
    );

    return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// C API Export for Python ctypes / External Invocations (Host Memory)
// ---------------------------------------------------------------------------
LRD_API int compute_lrd_cuda(
    const float* h_reach_dists,
    float* h_lrd,
    int N,
    int k,
    float eps
) {
    if (!h_reach_dists || !h_lrd) {
        fprintf(stderr, "[CUDA ERROR] Null pointer passed to compute_lrd_cuda\n");
        return -1;
    }
    if (N <= 0 || k <= 0) {
        fprintf(stderr, "[CUDA ERROR] Invalid parameters for compute_lrd_cuda (N=%d, k=%d)\n", N, k);
        return -2;
    }

    size_t reach_bytes = (size_t)N * k * sizeof(float);
    size_t lrd_bytes = (size_t)N * sizeof(float);

    float* d_reach_dists = nullptr;
    float* d_lrd = nullptr;

    cudaError_t err = cudaMalloc(&d_reach_dists, reach_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMalloc failed for reach_dists (%zu bytes): %s\n",
                reach_bytes, cudaGetErrorString(err));
        return -3;
    }

    err = cudaMalloc(&d_lrd, lrd_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMalloc failed for lrd (%zu bytes): %s\n",
                lrd_bytes, cudaGetErrorString(err));
        cudaFree(d_reach_dists);
        return -4;
    }

    err = cudaMemcpy(d_reach_dists, h_reach_dists, reach_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMemcpy H2D failed for reach_dists: %s\n", cudaGetErrorString(err));
        cudaFree(d_reach_dists);
        cudaFree(d_lrd);
        return -5;
    }

    err = launch_lrd_kernel(d_reach_dists, d_lrd, N, k, eps);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_reach_dists);
        cudaFree(d_lrd);
        return -6;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_reach_dists);
        cudaFree(d_lrd);
        return -7;
    }

    err = cudaMemcpy(h_lrd, d_lrd, lrd_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMemcpy D2H failed for lrd: %s\n", cudaGetErrorString(err));
        cudaFree(d_reach_dists);
        cudaFree(d_lrd);
        return -8;
    }

    cudaFree(d_reach_dists);
    cudaFree(d_lrd);

    return 0; // Success
}
