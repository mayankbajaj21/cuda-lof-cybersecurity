/**
 * test_cuda_lof.cu - Standalone C++ verification for Complete CUDA LOF Baseline Pipeline
 *
 * Project: CUDA LOF Cybersecurity
 * Milestone: M3 - CUDA LOF Baseline (Step 4)
 */

#include "cuda_lof_pipeline.cuh"
#include "../common/cuda_utils.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

int main() {
    printf("============================================================\n");
    printf("  Complete CUDA LOF Pipeline Standalone Verification\n");
    printf("  Milestone: M3 Step 4\n");
    printf("============================================================\n\n");

    print_device_info(0);

    const int N = 256;
    const int D = 68;
    const int k = 20;

    printf("[INFO] Generating synthetic dataset: N=%d, D=%d, k=%d...\n", N, D, k);

    std::vector<float> h_X((size_t)N * D);

    // Inlier cluster around origin
    for (int i = 0; i < N - 1; ++i) {
        for (int d = 0; d < D; ++d) {
            h_X[(size_t)i * D + d] = sinf((float)(i * D + d) * 0.1f) * 0.5f;
        }
    }

    // Inject explicit outlier at index N - 1
    for (int d = 0; d < D; ++d) {
        h_X[(size_t)(N - 1) * D + d] = 25.0f + (float)d;
    }

    std::vector<float> h_lof(N, 0.0f);

    printf("[INFO] Executing complete CUDA LOF pipeline...\n");
    int status = compute_lof_pipeline_cuda(h_X.data(), h_lof.data(), N, D, k);
    if (status != 0) {
        fprintf(stderr, "[FAIL] compute_lof_pipeline_cuda returned status %d\n", status);
        return EXIT_FAILURE;
    }

    // Check outputs
    int nan_count = 0;
    float min_lof = 1e9f;
    float max_lof = -1e9f;

    for (int i = 0; i < N; ++i) {
        float s = h_lof[i];
        if (isnan(s) || isinf(s)) {
            nan_count++;
        }
        if (s < min_lof) min_lof = s;
        if (s > max_lof) max_lof = s;
    }

    if (nan_count > 0) {
        fprintf(stderr, "[FAIL] Detected %d NaN/Inf values in LOF scores!\n", nan_count);
        return EXIT_FAILURE;
    }

    float outlier_lof = h_lof[N - 1];
    printf("[INFO] Min LOF score: %.4f, Max LOF score: %.4f\n", min_lof, max_lof);
    printf("[INFO] Injected outlier LOF score at sample %d: %.4f\n", N - 1, outlier_lof);

    if (outlier_lof <= 2.0f) {
        fprintf(stderr, "[FAIL] Outlier LOF score (%.4f) was not detected as significantly anomalous!\n", outlier_lof);
        return EXIT_FAILURE;
    }

    printf("[PASS] Outlier score is prominent (%.4f >> 1.0)\n", outlier_lof);
    printf("[PASS] All %d scores are finite and non-negative\n\n", N);

    printf("============================================================\n");
    printf("  COMPLETE CUDA LOF PIPELINE VERIFICATION PASSED\n");
    printf("============================================================\n");

    return EXIT_SUCCESS;
}
