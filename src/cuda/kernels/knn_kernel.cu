/**
 * knn_kernel.cu - CUDA k-Nearest Neighbors (k-NN) Naive Baseline Implementation
 *
 * Project: CUDA LOF Cybersecurity
 * Milestone: M3 - CUDA LOF Baseline (Step 3)
 *
 * Algorithm (Naive Baseline):
 *   - 1D grid decomposition across query samples i in [0, N-1].
 *   - Each thread scans the i-th row of the pairwise distance matrix.
 *   - The diagonal self-distance (j == i) is strictly excluded.
 *   - Maintains a local sorted list of the k smallest distances via insertion.
 *   - Stores ordered neighbor indices and distances into global memory.
 *
 * Constraints:
 *   - Target: sm_86 (RTX 3050 Laptop GPU)
 *   - Baseline only: no shared memory tiling, no warp shuffles, no streams.
 */

#include "knn_kernel.cuh"
#include "../common/cuda_utils.cuh"
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Naive Row-Parallel k-NN Kernel
// ---------------------------------------------------------------------------
__global__ void knn_naive_kernel(
    const float* __restrict__ dist_matrix,
    int* __restrict__ neighbor_indices,
    float* __restrict__ neighbor_distances,
    int N,
    int k
) {
    // 1D thread mapping: each thread processes one query sample row i
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    if (k <= 0 || k > KNN_MAX_K || k >= N) {
        return;
    }

    // Local top-k buffers kept sorted in ascending order:
    // best_dist[0] <= best_dist[1] <= ... <= best_dist[k-1]
    float best_dist[KNN_MAX_K];
    int best_idx[KNN_MAX_K];

    for (int m = 0; m < k; ++m) {
        best_dist[m] = 1e38f;
        best_idx[m] = -1;
    }

    const float* row = dist_matrix + (size_t)i * N;

    for (int j = 0; j < N; ++j) {
        // Exclude self-distance
        if (i == j) {
            continue;
        }

        float d = row[j];

        // Check against current k-th smallest distance
        if (d < best_dist[k - 1]) {
            // Find insertion point from right to left
            int pos = k - 1;
            while (pos > 0 && d < best_dist[pos - 1]) {
                best_dist[pos] = best_dist[pos - 1];
                best_idx[pos] = best_idx[pos - 1];
                --pos;
            }
            best_dist[pos] = d;
            best_idx[pos] = j;
        }
    }

    // Write top-k neighbors to global memory for sample i
    int* out_indices = neighbor_indices + (size_t)i * k;
    float* out_distances = neighbor_distances + (size_t)i * k;

    for (int m = 0; m < k; ++m) {
        out_indices[m] = best_idx[m];
        out_distances[m] = best_dist[m];
    }
}

// ---------------------------------------------------------------------------
// Host Kernel Launcher (Operates directly on device memory)
// ---------------------------------------------------------------------------
cudaError_t launch_knn_kernel(
    const float* d_dist_matrix,
    int* d_neighbor_indices,
    float* d_neighbor_distances,
    int N,
    int k,
    cudaStream_t stream
) {
    if (N <= 0 || k <= 0 || k >= N || k > KNN_MAX_K ||
        !d_dist_matrix || !d_neighbor_indices || !d_neighbor_distances) {
        return cudaErrorInvalidValue;
    }

    const int threads_per_block = 256;
    const int blocks = (N + threads_per_block - 1) / threads_per_block;

    knn_naive_kernel<<<blocks, threads_per_block, 0, stream>>>(
        d_dist_matrix, d_neighbor_indices, d_neighbor_distances, N, k
    );

    return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// C API Export for Python ctypes / External Invocations (Host Memory)
// ---------------------------------------------------------------------------
KNN_API int compute_knn_cuda(
    const float* h_dist_matrix,
    int* h_neighbor_indices,
    float* h_neighbor_distances,
    int N,
    int k
) {
    if (!h_dist_matrix || !h_neighbor_indices || !h_neighbor_distances) {
        fprintf(stderr, "[CUDA ERROR] Null pointer passed to compute_knn_cuda\n");
        return -1;
    }
    if (N <= 0 || k <= 0 || k >= N || k > KNN_MAX_K) {
        fprintf(stderr, "[CUDA ERROR] Invalid parameters for compute_knn_cuda (N=%d, k=%d, max_k=%d)\n",
                N, k, KNN_MAX_K);
        return -2;
    }

    size_t dist_bytes = (size_t)N * N * sizeof(float);
    size_t out_idx_bytes = (size_t)N * k * sizeof(int);
    size_t out_dist_bytes = (size_t)N * k * sizeof(float);

    float* d_dist_matrix = nullptr;
    int* d_neighbor_indices = nullptr;
    float* d_neighbor_distances = nullptr;

    // Allocate device memory for input distance matrix
    cudaError_t err = cudaMalloc(&d_dist_matrix, dist_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMalloc failed for dist_matrix (%zu bytes): %s\n",
                dist_bytes, cudaGetErrorString(err));
        return -3;
    }

    // Allocate device memory for neighbor indices
    err = cudaMalloc(&d_neighbor_indices, out_idx_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMalloc failed for neighbor_indices (%zu bytes): %s\n",
                out_idx_bytes, cudaGetErrorString(err));
        cudaFree(d_dist_matrix);
        return -4;
    }

    // Allocate device memory for neighbor distances
    err = cudaMalloc(&d_neighbor_distances, out_dist_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMalloc failed for neighbor_distances (%zu bytes): %s\n",
                out_dist_bytes, cudaGetErrorString(err));
        cudaFree(d_dist_matrix);
        cudaFree(d_neighbor_indices);
        return -5;
    }

    // Copy distance matrix from host to device
    err = cudaMemcpy(d_dist_matrix, h_dist_matrix, dist_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMemcpy H2D failed for dist_matrix: %s\n", cudaGetErrorString(err));
        cudaFree(d_dist_matrix);
        cudaFree(d_neighbor_indices);
        cudaFree(d_neighbor_distances);
        return -6;
    }

    // Launch naive k-NN kernel
    err = launch_knn_kernel(d_dist_matrix, d_neighbor_indices, d_neighbor_distances, N, k);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_dist_matrix);
        cudaFree(d_neighbor_indices);
        cudaFree(d_neighbor_distances);
        return -7;
    }

    // Synchronize
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_dist_matrix);
        cudaFree(d_neighbor_indices);
        cudaFree(d_neighbor_distances);
        return -8;
    }

    // Copy results from device back to host
    err = cudaMemcpy(h_neighbor_indices, d_neighbor_indices, out_idx_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMemcpy D2H failed for neighbor_indices: %s\n", cudaGetErrorString(err));
        cudaFree(d_dist_matrix);
        cudaFree(d_neighbor_indices);
        cudaFree(d_neighbor_distances);
        return -9;
    }

    err = cudaMemcpy(h_neighbor_distances, d_neighbor_distances, out_dist_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMemcpy D2H failed for neighbor_distances: %s\n", cudaGetErrorString(err));
        cudaFree(d_dist_matrix);
        cudaFree(d_neighbor_indices);
        cudaFree(d_neighbor_distances);
        return -10;
    }

    // Free device memory
    cudaFree(d_dist_matrix);
    cudaFree(d_neighbor_indices);
    cudaFree(d_neighbor_distances);

    return 0; // Success
}
