/**
 * cuda_lof_pipeline.cu - Complete End-to-End CUDA LOF Baseline Pipeline Implementation
 *
 * Project: CUDA LOF Cybersecurity
 * Milestone: M3 - CUDA LOF Baseline (Step 4)
 *
 * Pipeline Flow:
 *   X (Host) -> d_X (Device)
 *     -> pairwise_distance_kernel -> d_dist (Device)
 *     -> knn_kernel -> d_knn_indices, d_knn_distances (Device) [d_dist freed]
 *     -> reachability_kernel -> d_reach_dists (Device)
 *     -> lrd_kernel -> d_lrd (Device) [d_reach_dists freed]
 *     -> lof_kernel -> d_lof_scores (Device)
 *     -> h_lof_scores (Host)
 */

#include "cuda_lof_pipeline.cuh"
#include "../common/cuda_utils.cuh"
#include <cstdio>
#include <cstdlib>

PIPELINE_API int compute_lof_pipeline_cuda_timed(
    const float* h_X,
    float* h_lof_scores,
    int N,
    int D,
    int k,
    float eps,
    CudaLofTimings* out_timings
) {
    if (!h_X || !h_lof_scores) {
        fprintf(stderr, "[CUDA ERROR] Null pointer in compute_lof_pipeline_cuda_timed\n");
        return -1;
    }
    if (N <= 0 || D <= 0 || k <= 0 || k >= N) {
        fprintf(stderr, "[CUDA ERROR] Invalid parameters (N=%d, D=%d, k=%d)\n", N, D, k);
        return -2;
    }

    // CUDA Timing Events setup if requested
    cudaEvent_t ev_e2e_start = nullptr, ev_e2e_stop = nullptr;
    cudaEvent_t ev_h2d_start = nullptr, ev_h2d_stop = nullptr;
    cudaEvent_t ev_dist_start = nullptr, ev_dist_stop = nullptr;
    cudaEvent_t ev_knn_start = nullptr, ev_knn_stop = nullptr;
    cudaEvent_t ev_reach_start = nullptr, ev_reach_stop = nullptr;
    cudaEvent_t ev_lrd_start = nullptr, ev_lrd_stop = nullptr;
    cudaEvent_t ev_lof_start = nullptr, ev_lof_stop = nullptr;
    cudaEvent_t ev_d2h_start = nullptr, ev_d2h_stop = nullptr;

    if (out_timings) {
        cudaEventCreate(&ev_e2e_start);
        cudaEventCreate(&ev_e2e_stop);
        cudaEventCreate(&ev_h2d_start);
        cudaEventCreate(&ev_h2d_stop);
        cudaEventCreate(&ev_dist_start);
        cudaEventCreate(&ev_dist_stop);
        cudaEventCreate(&ev_knn_start);
        cudaEventCreate(&ev_knn_stop);
        cudaEventCreate(&ev_reach_start);
        cudaEventCreate(&ev_reach_stop);
        cudaEventCreate(&ev_lrd_start);
        cudaEventCreate(&ev_lrd_stop);
        cudaEventCreate(&ev_lof_start);
        cudaEventCreate(&ev_lof_stop);
        cudaEventCreate(&ev_d2h_start);
        cudaEventCreate(&ev_d2h_stop);

        cudaEventRecord(ev_e2e_start, 0);
    }

    auto cleanup_events = [&]() {
        if (out_timings) {
            if (ev_e2e_start) cudaEventDestroy(ev_e2e_start);
            if (ev_e2e_stop) cudaEventDestroy(ev_e2e_stop);
            if (ev_h2d_start) cudaEventDestroy(ev_h2d_start);
            if (ev_h2d_stop) cudaEventDestroy(ev_h2d_stop);
            if (ev_dist_start) cudaEventDestroy(ev_dist_start);
            if (ev_dist_stop) cudaEventDestroy(ev_dist_stop);
            if (ev_knn_start) cudaEventDestroy(ev_knn_start);
            if (ev_knn_stop) cudaEventDestroy(ev_knn_stop);
            if (ev_reach_start) cudaEventDestroy(ev_reach_start);
            if (ev_reach_stop) cudaEventDestroy(ev_reach_stop);
            if (ev_lrd_start) cudaEventDestroy(ev_lrd_start);
            if (ev_lrd_stop) cudaEventDestroy(ev_lrd_stop);
            if (ev_lof_start) cudaEventDestroy(ev_lof_start);
            if (ev_lof_stop) cudaEventDestroy(ev_lof_stop);
            if (ev_d2h_start) cudaEventDestroy(ev_d2h_start);
            if (ev_d2h_stop) cudaEventDestroy(ev_d2h_stop);
        }
    };

    // Byte sizes
    size_t x_bytes = (size_t)N * D * sizeof(float);
    size_t dist_bytes = (size_t)N * N * sizeof(float);
    size_t knn_idx_bytes = (size_t)N * k * sizeof(int);
    size_t knn_dist_bytes = (size_t)N * k * sizeof(float);
    size_t lrd_bytes = (size_t)N * sizeof(float);

    // Device pointers
    float* d_X = nullptr;
    float* d_dist = nullptr;
    int* d_knn_indices = nullptr;
    float* d_knn_distances = nullptr;
    float* d_reach_dists = nullptr;
    float* d_lrd = nullptr;
    float* d_lof_scores = nullptr;

    // -----------------------------------------------------------------------
    // Stage 1: Allocate X and copy from host
    // -----------------------------------------------------------------------
    cudaError_t err = cudaMalloc(&d_X, x_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMalloc failed for d_X: %s\n", cudaGetErrorString(err));
        cleanup_events();
        return -3;
    }

    if (out_timings) cudaEventRecord(ev_h2d_start, 0);
    err = cudaMemcpy(d_X, h_X, x_bytes, cudaMemcpyHostToDevice);
    if (out_timings) cudaEventRecord(ev_h2d_stop, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMemcpy H2D failed for d_X: %s\n", cudaGetErrorString(err));
        cudaFree(d_X);
        cleanup_events();
        return -4;
    }

    // -----------------------------------------------------------------------
    // Stage 2: Pairwise Distance Matrix
    // -----------------------------------------------------------------------
    err = cudaMalloc(&d_dist, dist_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMalloc failed for d_dist (%zu bytes): %s\n",
                dist_bytes, cudaGetErrorString(err));
        cudaFree(d_X);
        cleanup_events();
        return -5;
    }

    if (out_timings) cudaEventRecord(ev_dist_start, 0);
    err = launch_pairwise_distance_kernel(d_X, d_dist, N, D);
    if (out_timings) cudaEventRecord(ev_dist_stop, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] launch_pairwise_distance_kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_X);
        cudaFree(d_dist);
        cleanup_events();
        return -6;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] sync failed after distance kernel: %s\n", cudaGetErrorString(err));
        cudaFree(d_X);
        cudaFree(d_dist);
        cleanup_events();
        return -7;
    }

    // Free feature matrix d_X immediately
    cudaFree(d_X);
    d_X = nullptr;

    // -----------------------------------------------------------------------
    // Stage 3: k-Nearest Neighbors
    // -----------------------------------------------------------------------
    err = cudaMalloc(&d_knn_indices, knn_idx_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMalloc failed for d_knn_indices: %s\n", cudaGetErrorString(err));
        cudaFree(d_dist);
        cleanup_events();
        return -8;
    }

    err = cudaMalloc(&d_knn_distances, knn_dist_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMalloc failed for d_knn_distances: %s\n", cudaGetErrorString(err));
        cudaFree(d_dist);
        cudaFree(d_knn_indices);
        cleanup_events();
        return -9;
    }

    if (out_timings) cudaEventRecord(ev_knn_start, 0);
    err = launch_knn_kernel(d_dist, d_knn_indices, d_knn_distances, N, k);
    if (out_timings) cudaEventRecord(ev_knn_stop, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] launch_knn_kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_dist);
        cudaFree(d_knn_indices);
        cudaFree(d_knn_distances);
        cleanup_events();
        return -10;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] sync failed after k-NN kernel: %s\n", cudaGetErrorString(err));
        cudaFree(d_dist);
        cudaFree(d_knn_indices);
        cudaFree(d_knn_distances);
        cleanup_events();
        return -11;
    }

    // Free distance matrix d_dist immediately to reclaim massive VRAM!
    cudaFree(d_dist);
    d_dist = nullptr;

    // -----------------------------------------------------------------------
    // Stage 4: Reachability Distance
    // -----------------------------------------------------------------------
    err = cudaMalloc(&d_reach_dists, knn_dist_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMalloc failed for d_reach_dists: %s\n", cudaGetErrorString(err));
        cudaFree(d_knn_indices);
        cudaFree(d_knn_distances);
        cleanup_events();
        return -12;
    }

    if (out_timings) cudaEventRecord(ev_reach_start, 0);
    err = launch_reachability_distance_kernel(d_knn_indices, d_knn_distances, d_reach_dists, N, k);
    if (out_timings) cudaEventRecord(ev_reach_stop, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] launch_reachability_distance_kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_knn_indices);
        cudaFree(d_knn_distances);
        cudaFree(d_reach_dists);
        cleanup_events();
        return -13;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] sync failed after reachability kernel: %s\n", cudaGetErrorString(err));
        cudaFree(d_knn_indices);
        cudaFree(d_knn_distances);
        cudaFree(d_reach_dists);
        cleanup_events();
        return -14;
    }

    // Free d_knn_distances as it is no longer required
    cudaFree(d_knn_distances);
    d_knn_distances = nullptr;

    // -----------------------------------------------------------------------
    // Stage 5: Local Reachability Density (LRD)
    // -----------------------------------------------------------------------
    err = cudaMalloc(&d_lrd, lrd_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMalloc failed for d_lrd: %s\n", cudaGetErrorString(err));
        cudaFree(d_knn_indices);
        cudaFree(d_reach_dists);
        cleanup_events();
        return -15;
    }

    if (out_timings) cudaEventRecord(ev_lrd_start, 0);
    err = launch_lrd_kernel(d_reach_dists, d_lrd, N, k, eps);
    if (out_timings) cudaEventRecord(ev_lrd_stop, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] launch_lrd_kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_knn_indices);
        cudaFree(d_reach_dists);
        cudaFree(d_lrd);
        cleanup_events();
        return -16;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] sync failed after LRD kernel: %s\n", cudaGetErrorString(err));
        cudaFree(d_knn_indices);
        cudaFree(d_reach_dists);
        cudaFree(d_lrd);
        cleanup_events();
        return -17;
    }

    // Free d_reach_dists
    cudaFree(d_reach_dists);
    d_reach_dists = nullptr;

    // -----------------------------------------------------------------------
    // Stage 6: LOF Scoring
    // -----------------------------------------------------------------------
    err = cudaMalloc(&d_lof_scores, lrd_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMalloc failed for d_lof_scores: %s\n", cudaGetErrorString(err));
        cudaFree(d_knn_indices);
        cudaFree(d_lrd);
        cleanup_events();
        return -18;
    }

    if (out_timings) cudaEventRecord(ev_lof_start, 0);
    err = launch_lof_scoring_kernel(d_lrd, d_knn_indices, d_lof_scores, N, k, eps);
    if (out_timings) cudaEventRecord(ev_lof_stop, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] launch_lof_scoring_kernel failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_knn_indices);
        cudaFree(d_lrd);
        cudaFree(d_lof_scores);
        cleanup_events();
        return -19;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] sync failed after LOF scoring kernel: %s\n", cudaGetErrorString(err));
        cudaFree(d_knn_indices);
        cudaFree(d_lrd);
        cudaFree(d_lof_scores);
        cleanup_events();
        return -20;
    }

    // -----------------------------------------------------------------------
    // Stage 7: Copy LOF Scores back to Host
    // -----------------------------------------------------------------------
    if (out_timings) cudaEventRecord(ev_d2h_start, 0);
    err = cudaMemcpy(h_lof_scores, d_lof_scores, lrd_bytes, cudaMemcpyDeviceToHost);
    if (out_timings) {
        cudaEventRecord(ev_d2h_stop, 0);
        cudaEventRecord(ev_e2e_stop, 0);
    }
    if (err != cudaSuccess) {
        fprintf(stderr, "[CUDA ERROR] cudaMemcpy D2H failed for lof_scores: %s\n", cudaGetErrorString(err));
        cudaFree(d_knn_indices);
        cudaFree(d_lrd);
        cudaFree(d_lof_scores);
        cleanup_events();
        return -21;
    }

    // Clean up remaining device buffers
    cudaFree(d_knn_indices);
    cudaFree(d_lrd);
    cudaFree(d_lof_scores);

    // Extract timings if requested
    if (out_timings) {
        cudaEventSynchronize(ev_e2e_stop);

        cudaEventElapsedTime(&out_timings->h2d_ms, ev_h2d_start, ev_h2d_stop);
        cudaEventElapsedTime(&out_timings->distance_kernel_ms, ev_dist_start, ev_dist_stop);
        cudaEventElapsedTime(&out_timings->knn_kernel_ms, ev_knn_start, ev_knn_stop);
        cudaEventElapsedTime(&out_timings->reachability_kernel_ms, ev_reach_start, ev_reach_stop);
        cudaEventElapsedTime(&out_timings->lrd_kernel_ms, ev_lrd_start, ev_lrd_stop);
        cudaEventElapsedTime(&out_timings->lof_kernel_ms, ev_lof_start, ev_lof_stop);
        cudaEventElapsedTime(&out_timings->d2h_ms, ev_d2h_start, ev_d2h_stop);
        cudaEventElapsedTime(&out_timings->end_to_end_ms, ev_e2e_start, ev_e2e_stop);

        out_timings->total_kernel_ms = out_timings->distance_kernel_ms +
                                       out_timings->knn_kernel_ms +
                                       out_timings->reachability_kernel_ms +
                                       out_timings->lrd_kernel_ms +
                                       out_timings->lof_kernel_ms;

        cleanup_events();
    }

    return 0; // Success
}

PIPELINE_API int compute_lof_pipeline_cuda(
    const float* h_X,
    float* h_lof_scores,
    int N,
    int D,
    int k,
    float eps
) {
    return compute_lof_pipeline_cuda_timed(h_X, h_lof_scores, N, D, k, eps, nullptr);
}
