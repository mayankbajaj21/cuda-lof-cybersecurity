/**
 * reachability_kernel.cu - CUDA Reachability Distance Naive Baseline Implementation
 *
 * Project: CUDA LOF Cybersecurity
 * Milestone: M3 - CUDA LOF Baseline (Step 4)
 *
 * Formula (M2 Reproduction):
 *   reach-dist_k(p, o) = max(k-distance(o), d(p, o))
 *
 * Designed for:
 *   - N samples (validated on 10K, 25K)
 *   - k neighbors (default k=20)
 *   - sm_86 architecture (RTX 3050 Laptop GPU)
 */

#include "reachability_kernel.cuh"
#include "../common/cuda_utils.cuh"
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Naive Row-Parallel Reachability Distance Kernel
// ---------------------------------------------------------------------------
__global__ void reachability_distance_naive_kernel(
    const int* __restrict__ knn_indices,
    const float* __restrict__ knn_distances,
    float* __restrict__ reach_dists,
    int N,
    int k
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    if (k <= 0) {
        return;
    }

    const int* neighbors = knn_indices + (size_t)i * k;
    const float* direct_dists = knn_distances + (size_t)i * k;
    float* out_reach = reach_dists + (size_t)i * k;

    for (int m = 0; m < k; ++m) {
        int o = neighbors[m];
        float direct_dist = direct_dists[m];

        // Retrieve k-distance of neighbor o: distance to its k-th neighbor (column k-1)
        float o_k_dist = 0.0f;
        if (o >= 0 && o < N) {
            o_k_dist = knn_distances[(size_t)o * k + (k - 1)];
        }

        // reach-dist_k(p, o) = max(k-distance(o), d(p, o))
        out_reach[m] = fmaxf(direct_dist, o_k_dist);
    }
}

// ---------------------------------------------------------------------------
// Host Kernel Launcher (Operates directly on device memory)
// ---------------------------------------------------------------------------
cudaError_t launch_reachability_distance_kernel(
    const int* d_knn_indices,
    const float* d_knn_distances,
    float* d_reach_dists,
    int N,
    int k,
    cudaStream_t stream
) {
    if (N <= 0 || k <= 0 || !d_knn_indices || !d_knn_distances || !d_reach_dists) {
        return cudaErrorInvalidValue;
    }

    const int threads_per_block = 256;
    const int blocks = (N + threads_per_block - 1) / threads_per_block;

    reachability_distance_naive_kernel<<<blocks, threads_per_block, 0, stream>>>(
        d_knn_indices, d_knn_distances, d_reach_dists, N, k
    );

    return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// C API Export for Python ctypes / External Invocations (Host Memory)
// ---------------------------------------------------------------------------
REACH_API int compute_reachability_distances_cuda(
    const int* h_knn_indices,
    const float* h_knn_distances,
    float* h_reach_dists,
    int N,
    int k
) {
    if (!h_knn_indices || !h_knn_distances || !h_reach_dists) {
        fprintf(stderr, "[CUDA ERROR] Null pointer passed to compute_reachability_distances_cuda\n");
        return -1;
    }
    if (N <= 0 || k <= 0) {
        fprintf(stderr, "[CUDA ERROR] Invalid parameters for compute_reachability_distances_cuda (N=%d, k=%d)\n",
                N, k);
        return -2;
    }

    size_t idx_bytes = (size_t)N * k * sizeof(int);
    size_t dist_bytes = (size_t)N * k * sizeof(float);

    int* d_knn_indices = nullptr;
    float* d_knn_distances = nullptr;
    float* d_reach_dists = nullptr;

    cudaError_t err = cudaMalloc(&d_knn_indices, idx_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMalloc failed for knn_indices (%zu bytes): %s\n",
                idx_bytes, cudaGetErrorString(err));
        return -3;
    }

    err = cudaMalloc(&d_knn_distances, dist_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMalloc failed for knn_distances (%zu bytes): %s\n",
                dist_bytes, cudaGetErrorString(err));
        cudaFree(d_knn_indices);
        return -4;
    }

    err = cudaMalloc(&d_reach_dists, dist_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMalloc failed for reach_dists (%zu bytes): %s\n",
                dist_bytes, cudaGetErrorString(err));
        cudaFree(d_knn_indices);
        cudaFree(d_knn_distances);
        return -5;
    }

    err = cudaMemcpy(d_knn_indices, h_knn_indices, idx_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMemcpy H2D failed for knn_indices: %s\n", cudaGetErrorString(err));
        cudaFree(d_knn_indices);
        cudaFree(d_knn_distances);
        cudaFree(d_reach_dists);
        return -6;
    }

    err = cudaMemcpy(d_knn_distances, h_knn_distances, dist_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMemcpy H2D failed for knn_distances: %s\n", cudaGetErrorString(err));
        cudaFree(d_knn_indices);
        cudaFree(d_knn_distances);
        cudaFree(d_reach_dists);
        return -7;
    }

    err = launch_reachability_distance_kernel(d_knn_indices, d_knn_distances, d_reach_dists, N, k);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_knn_indices);
        cudaFree(d_knn_distances);
        cudaFree(d_reach_dists);
        return -8;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_knn_indices);
        cudaFree(d_knn_distances);
        cudaFree(d_reach_dists);
        return -9;
    }

    err = cudaMemcpy(h_reach_dists, d_reach_dists, dist_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMemcpy D2H failed for reach_dists: %s\n", cudaGetErrorString(err));
        cudaFree(d_knn_indices);
        cudaFree(d_knn_distances);
        cudaFree(d_reach_dists);
        return -10;
    }

    cudaFree(d_knn_indices);
    cudaFree(d_knn_distances);
    cudaFree(d_reach_dists);

    return 0; // Success
}
