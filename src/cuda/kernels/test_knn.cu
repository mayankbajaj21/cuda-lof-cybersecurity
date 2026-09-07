/**
 * test_knn.cu - Standalone C++ verification for CUDA k-NN Baseline Kernel
 *
 * Project: CUDA LOF Cybersecurity
 * Milestone: M3 - CUDA LOF Baseline (Step 3)
 */

#include "knn_kernel.cuh"
#include "distance_kernel.cuh"
#include "../common/cuda_utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

static bool verify_knn(
    const float* dist_matrix,
    const int* neighbor_indices,
    const float* neighbor_distances,
    int N,
    int k,
    float tolerance = 1e-5f
) {
    int errors = 0;

    for (int i = 0; i < N; ++i) {
        const float* row = dist_matrix + (size_t)i * N;
        const int* row_idx = neighbor_indices + (size_t)i * k;
        const float* row_dist = neighbor_distances + (size_t)i * k;

        // 1. Check no self-neighbor
        for (int m = 0; m < k; ++m) {
            if (row_idx[m] == i) {
                fprintf(stderr, "[FAIL] Sample %d has self as neighbor at rank %d\n", i, m);
                if (++errors >= 5) return false;
            }
            if (row_idx[m] < 0 || row_idx[m] >= N) {
                fprintf(stderr, "[FAIL] Sample %d has out-of-bounds neighbor %d at rank %d\n", i, row_idx[m], m);
                if (++errors >= 5) return false;
            }
        }

        // 2. Check sorted order: d[m] <= d[m+1]
        for (int m = 0; m < k - 1; ++m) {
            if (row_dist[m] > row_dist[m + 1] + tolerance) {
                fprintf(stderr, "[FAIL] Distances not sorted for sample %d: d[%d]=%f > d[%d]=%f\n",
                        i, m, row_dist[m], m + 1, row_dist[m + 1]);
                if (++errors >= 5) return false;
            }
        }

        // 3. Check distance value corresponds to dist_matrix[i, idx]
        for (int m = 0; m < k; ++m) {
            float actual_d = row[row_idx[m]];
            if (fabsf(row_dist[m] - actual_d) > tolerance) {
                fprintf(stderr, "[FAIL] Distance mismatch for sample %d rank %d: recorded=%f, matrix=%f\n",
                        i, m, row_dist[m], actual_d);
                if (++errors >= 5) return false;
            }
        }

        // 4. Check against independent CPU reference for sample i
        std::vector<std::pair<float, int>> cpu_pairs;
        cpu_pairs.reserve(N - 1);
        for (int j = 0; j < N; ++j) {
            if (i != j) {
                cpu_pairs.emplace_back(row[j], j);
            }
        }
        std::partial_sort(
            cpu_pairs.begin(),
            cpu_pairs.begin() + k,
            cpu_pairs.end(),
            [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                return (a.first < b.first) || (a.first == b.first && a.second < b.second);
            }
        );

        for (int m = 0; m < k; ++m) {
            float expected_dist = cpu_pairs[m].first;
            if (fabsf(row_dist[m] - expected_dist) > tolerance) {
                fprintf(stderr, "[FAIL] Sample %d rank %d mismatch: GPU dist=%f, CPU dist=%f\n",
                        i, m, row_dist[m], expected_dist);
                if (++errors >= 5) return false;
            }
        }
    }

    return (errors == 0);
}

int main() {
    printf("============================================================\n");
    printf("  CUDA k-NN Kernel Standalone Verification\n");
    printf("  Milestone: M3 Step 3\n");
    printf("============================================================\n\n");

    print_device_info(0);

    const int D = 68; // CICIDS2017 feature count
    const int k = 20; // Default neighborhood size for LOF
    const int test_sizes[] = {64, 256, 1024};
    const int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);

    for (int t = 0; t < num_tests; ++t) {
        int N = test_sizes[t];
        printf("[INFO] Testing k-NN on synthetic data: N=%d, D=%d, k=%d...\n", N, D, k);

        // 1. Generate synthetic feature data
        std::vector<float> h_X((size_t)N * D);
        for (int i = 0; i < N; ++i) {
            for (int d = 0; d < D; ++d) {
                h_X[(size_t)i * D + d] = sinf((float)(i * D + d) * 0.05f) + 0.01f * (float)i;
            }
        }

        // 2. Compute pairwise distance matrix
        std::vector<float> h_dist((size_t)N * N);
        int dist_status = compute_pairwise_distances_cuda(h_X.data(), h_dist.data(), N, D);
        if (dist_status != 0) {
            fprintf(stderr, "[FAIL] compute_pairwise_distances_cuda failed with status %d\n", dist_status);
            return EXIT_FAILURE;
        }

        // 3. Compute k-NN
        std::vector<int> h_indices((size_t)N * k);
        std::vector<float> h_distances((size_t)N * k);
        int knn_status = compute_knn_cuda(h_dist.data(), h_indices.data(), h_distances.data(), N, k);
        if (knn_status != 0) {
            fprintf(stderr, "[FAIL] compute_knn_cuda failed with status %d\n", knn_status);
            return EXIT_FAILURE;
        }

        // 4. Verify results
        bool ok = verify_knn(h_dist.data(), h_indices.data(), h_distances.data(), N, k);
        if (!ok) {
            fprintf(stderr, "[FAIL] Verification failed for N=%d, k=%d\n", N, k);
            return EXIT_FAILURE;
        }
        printf("[PASS] N=%d, k=%d: valid indices, no self-neighbor, sorted distances, exact CPU match\n\n", N, k);
    }

    printf("============================================================\n");
    printf("  ALL C++ k-NN KERNEL VERIFICATIONS PASSED\n");
    printf("============================================================\n");

    return EXIT_SUCCESS;
}
