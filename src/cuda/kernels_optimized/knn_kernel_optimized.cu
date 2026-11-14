/**
 * knn_kernel_optimized.cu - Register-Right-Sized k-NN Kernel Implementation
 *
 * Project: CUDA LOF Cybersecurity
 * Milestone: M5 - Profiling-Driven CUDA Optimization (Step 2: k-NN)
 *
 * See knn_kernel_optimized.cuh for the M4 profiling evidence and rationale.
 * Algorithm is UNCHANGED from the naive baseline (sorted insertion top-k,
 * self excluded, same ordering guarantees) - only the buffer size changed,
 * plus float4-vectorized global loads of the distance row to halve the
 * instruction count for the dominant memory-read loop.
 */

#include "knn_kernel_optimized.cuh"
#include "../common/cuda_utils.cuh"
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Register-Right-Sized Row-Parallel k-NN Kernel
// ---------------------------------------------------------------------------
__global__ void knn_optimized_kernel(
    const float* __restrict__ dist_matrix,
    int* __restrict__ neighbor_indices,
    float* __restrict__ neighbor_distances,
    int N,
    int k
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) {
        return;
    }

    if (k <= 0 || k > OPT_KNN_MAX_K || k >= N) {
        return;
    }

    // Right-sized local top-k buffers (OPT_KNN_MAX_K = 32, vs. the naive
    // kernel's 256) - small enough to stay in registers for typical
    // occupancy instead of spilling to local memory.
    float best_dist[OPT_KNN_MAX_K];
    int best_idx[OPT_KNN_MAX_K];

    for (int m = 0; m < k; ++m) {
        best_dist[m] = 1e38f;
        best_idx[m] = -1;
    }

    const float* row = dist_matrix + (size_t)i * N;

    // Vectorized scan: read 4 floats at a time where alignment allows,
    // reducing the number of global memory load instructions issued by
    // each thread over its O(N) row scan. Only safe when N is a multiple
    // of 4 (so every row start is 16-byte aligned); otherwise fall back
    // to the scalar loop below for the whole row.
    int j = 0;
    const int n_vec4 = (N % 4 == 0) ? (N / 4) : 0;
    const float4* row4 = reinterpret_cast<const float4*>(row);

    for (int v = 0; v < n_vec4; ++v) {
        float4 chunk = row4[v];
        float vals[4] = {chunk.x, chunk.y, chunk.z, chunk.w};
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            int jj = v * 4 + t;
            if (jj == i) continue;
            float d = vals[t];
            if (d < best_dist[k - 1]) {
                int pos = k - 1;
                while (pos > 0 && d < best_dist[pos - 1]) {
                    best_dist[pos] = best_dist[pos - 1];
                    best_idx[pos] = best_idx[pos - 1];
                    --pos;
                }
                best_dist[pos] = d;
                best_idx[pos] = jj;
            }
        }
    }

    // Tail elements not covered by the float4 scan (N not a multiple of 4).
    for (j = n_vec4 * 4; j < N; ++j) {
        if (j == i) continue;
        float d = row[j];
        if (d < best_dist[k - 1]) {
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
cudaError_t launch_knn_kernel_optimized(
    const float* d_dist_matrix,
    int* d_neighbor_indices,
    float* d_neighbor_distances,
    int N,
    int k,
    cudaStream_t stream
) {
    if (N <= 0 || k <= 0 || k >= N || k > OPT_KNN_MAX_K ||
        !d_dist_matrix || !d_neighbor_indices || !d_neighbor_distances) {
        return cudaErrorInvalidValue;
    }

    const int threads_per_block = 256;
    const int blocks = (N + threads_per_block - 1) / threads_per_block;

    knn_optimized_kernel<<<blocks, threads_per_block, 0, stream>>>(
        d_dist_matrix, d_neighbor_indices, d_neighbor_distances, N, k
    );

    return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// C API Export for Python ctypes / External Invocations (Host Memory)
// ---------------------------------------------------------------------------
KNN_OPT_API int compute_knn_cuda_optimized(
    const float* h_dist_matrix,
    int* h_neighbor_indices,
    float* h_neighbor_distances,
    int N,
    int k
) {
    if (!h_dist_matrix || !h_neighbor_indices || !h_neighbor_distances) {
        fprintf(stderr, "[CUDA ERROR] Null pointer passed to compute_knn_cuda_optimized\n");
        return -1;
    }
    if (N <= 0 || k <= 0 || k >= N || k > OPT_KNN_MAX_K) {
        fprintf(stderr, "[CUDA ERROR] Invalid parameters for compute_knn_cuda_optimized (N=%d, k=%d, max_k=%d)\n",
                N, k, OPT_KNN_MAX_K);
        return -2;
    }

    size_t dist_bytes = (size_t)N * N * sizeof(float);
    size_t out_idx_bytes = (size_t)N * k * sizeof(int);
    size_t out_dist_bytes = (size_t)N * k * sizeof(float);

    float* d_dist_matrix = nullptr;
    int* d_neighbor_indices = nullptr;
    float* d_neighbor_distances = nullptr;

    cudaError_t err = cudaMalloc(&d_dist_matrix, dist_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMalloc failed for dist_matrix (%zu bytes): %s\n", dist_bytes, cudaGetErrorString(err));
        return -3;
    }

    err = cudaMalloc(&d_neighbor_indices, out_idx_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMalloc failed for neighbor_indices: %s\n", cudaGetErrorString(err));
        cudaFree(d_dist_matrix);
        return -4;
    }

    err = cudaMalloc(&d_neighbor_distances, out_dist_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMalloc failed for neighbor_distances: %s\n", cudaGetErrorString(err));
        cudaFree(d_dist_matrix);
        cudaFree(d_neighbor_indices);
        return -5;
    }

    err = cudaMemcpy(d_dist_matrix, h_dist_matrix, dist_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMemcpy H2D failed for dist_matrix: %s\n", cudaGetErrorString(err));
        cudaFree(d_dist_matrix);
        cudaFree(d_neighbor_indices);
        cudaFree(d_neighbor_distances);
        return -6;
    }

    err = launch_knn_kernel_optimized(d_dist_matrix, d_neighbor_indices, d_neighbor_distances, N, k);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_dist_matrix);
        cudaFree(d_neighbor_indices);
        cudaFree(d_neighbor_distances);
        return -7;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_dist_matrix);
        cudaFree(d_neighbor_indices);
        cudaFree(d_neighbor_distances);
        return -8;
    }

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

    cudaFree(d_dist_matrix);
    cudaFree(d_neighbor_indices);
    cudaFree(d_neighbor_distances);

    return 0;
}
