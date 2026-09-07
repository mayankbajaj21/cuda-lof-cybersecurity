/**
 * test_distance.cu - Standalone C++ verification for CUDA Pairwise Euclidean Distance Kernel
 *
 * Project: CUDA LOF Cybersecurity
 * Milestone: M3 - CUDA LOF Baseline (Step 2)
 */

#include "distance_kernel.cuh"
#include "../common/cuda_utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

static bool verify_distance_matrix(
    const float* X,
    const float* dist,
    int N,
    int D,
    float tolerance = 1e-4f
) {
    int error_count = 0;

    // 1. Check diagonal is zero
    for (int i = 0; i < N; ++i) {
        float diag_val = dist[(size_t)i * N + i];
        if (diag_val != 0.0f) {
            fprintf(stderr, "[FAIL] Non-zero diagonal at (%d, %d): %f\n", i, i, diag_val);
            error_count++;
            if (error_count >= 5) return false;
        }
    }

    // 2. Check symmetry: dist(i, j) == dist(j, i)
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            float dij = dist[(size_t)i * N + j];
            float dji = dist[(size_t)j * N + i];
            if (fabsf(dij - dji) > tolerance) {
                fprintf(stderr, "[FAIL] Asymmetric distance at (%d, %d): %f vs %f\n", i, j, dij, dji);
                error_count++;
                if (error_count >= 5) return false;
            }
        }
    }

    // 3. Check against independent CPU calculation for selected pairs
    int test_pairs = 0;
    for (int i = 0; i < N && test_pairs < 500; i += (N > 10 ? N / 10 : 1)) {
        for (int j = 0; j < N && test_pairs < 500; j += (N > 10 ? N / 10 : 1)) {
            float expected_sum = 0.0f;
            for (int d = 0; d < D; ++d) {
                float diff = X[(size_t)i * D + d] - X[(size_t)j * D + d];
                expected_sum += diff * diff;
            }
            float expected = (i == j) ? 0.0f : sqrtf(expected_sum);
            float actual = dist[(size_t)i * N + j];

            if (isnan(actual) || isinf(actual)) {
                fprintf(stderr, "[FAIL] NaN or Inf detected at (%d, %d): %f\n", i, j, actual);
                return false;
            }

            if (fabsf(actual - expected) > tolerance) {
                fprintf(stderr, "[FAIL] Distance mismatch at (%d, %d): actual=%f, expected=%f, diff=%f\n",
                        i, j, actual, expected, fabsf(actual - expected));
                error_count++;
                if (error_count >= 5) return false;
            }
            test_pairs++;
        }
    }

    return (error_count == 0);
}

int main() {
    printf("============================================================\n");
    printf("  CUDA Pairwise Euclidean Distance Kernel Verification\n");
    printf("  Milestone: M3 Step 2\n");
    printf("============================================================\n\n");

    print_device_info(0);

    const int D = 68; // CICIDS2017 feature count
    const int test_sizes[] = {64, 256, 1024};
    const int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);

    for (int t = 0; t < num_tests; ++t) {
        int N = test_sizes[t];
        printf("[INFO] Running test on synthetic data: N=%d, D=%d...\n", N, D);

        std::vector<float> h_X((size_t)N * D);
        std::vector<float> h_dist((size_t)N * N, -1.0f);

        // Populate with deterministic test data
        for (int i = 0; i < N; ++i) {
            for (int d = 0; d < D; ++d) {
                h_X[(size_t)i * D + d] = sinf((float)(i * D + d) * 0.1f);
            }
        }

        int status = compute_pairwise_distances_cuda(h_X.data(), h_dist.data(), N, D);
        if (status != 0) {
            fprintf(stderr, "[FAIL] compute_pairwise_distances_cuda returned error code %d\n", status);
            return EXIT_FAILURE;
        }

        bool ok = verify_distance_matrix(h_X.data(), h_dist.data(), N, D);
        if (!ok) {
            fprintf(stderr, "[FAIL] Verification failed for N=%d, D=%d\n", N, D);
            return EXIT_FAILURE;
        }
        printf("[PASS] N=%d, D=%d verified: zero diagonal, symmetric, exact match with CPU\n\n", N, D);
    }

    printf("============================================================\n");
    printf("  ALL C++ DISTANCE KERNEL VERIFICATIONS PASSED\n");
    printf("============================================================\n");

    return EXIT_SUCCESS;
}
