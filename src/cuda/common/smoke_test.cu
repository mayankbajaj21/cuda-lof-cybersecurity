/**
 * smoke_test.cu - CUDA Smoke Test Program
 *
 * Project: CUDA LOF Cybersecurity
 * Milestone: M3 - CUDA LOF Baseline (Step 1: Build Infrastructure Verification)
 *
 * Purpose:
 *   Prove that the CUDA build toolchain works end-to-end on this system:
 *   1. nvcc can compile a .cu source file.
 *   2. The executable can launch a CUDA kernel on the GPU.
 *   3. The NVIDIA RTX 3050 is visible to the CUDA runtime.
 *   4. A simple GPU vector addition produces the expected result.
 *   5. Device properties are reported.
 *
 * This file does NOT contain any LOF implementation.
 *
 * Build:
 *   nvcc -arch=sm_86 -o smoke_test.exe src/cuda/common/smoke_test.cu
 *
 * Run:
 *   smoke_test.exe
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Error-checking macros (self-contained for standalone smoke test)
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "[FAIL] CUDA Error at %s:%d - %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Kernel: Vector Addition  (C[i] = A[i] + B[i])
// ---------------------------------------------------------------------------
__global__ void vector_add_kernel(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// ---------------------------------------------------------------------------
// Device info printer
// ---------------------------------------------------------------------------
static void print_device_info(int device_id) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    int runtime_version = 0, driver_version = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));

    printf("============================================================\n");
    printf("  CUDA Device Information\n");
    printf("============================================================\n");
    printf("  Device ID             : %d\n", device_id);
    printf("  Device Name           : %s\n", prop.name);
    printf("  Compute Capability    : %d.%d\n", prop.major, prop.minor);
    printf("  Global Memory         : %.2f GB (%zu bytes)\n",
           (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
           prop.totalGlobalMem);
    printf("  Shared Memory / Block : %zu bytes\n", prop.sharedMemPerBlock);
    printf("  Warp Size             : %d\n", prop.warpSize);
    printf("  Max Threads / Block   : %d\n", prop.maxThreadsPerBlock);
    printf("  Max Block Dimensions  : [%d, %d, %d]\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Max Grid Dimensions   : [%d, %d, %d]\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Multiprocessors (SMs) : %d\n", prop.multiProcessorCount);
    int clock_khz = 0, mem_clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device_id);
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, device_id);
    printf("  Clock Rate            : %.2f GHz\n", clock_khz / 1.0e6);
    printf("  Memory Clock Rate     : %.2f GHz\n", mem_clock_khz / 1.0e6);
    printf("  Memory Bus Width      : %d-bit\n", prop.memoryBusWidth);
    printf("  L2 Cache Size         : %d KB\n", prop.l2CacheSize / 1024);
    printf("  CUDA Runtime Version  : %d.%d\n",
           runtime_version / 1000, (runtime_version % 100) / 10);
    printf("  CUDA Driver Version   : %d.%d\n",
           driver_version / 1000, (driver_version % 100) / 10);
    printf("============================================================\n\n");
}

// ---------------------------------------------------------------------------
// Main: smoke test entry point
// ---------------------------------------------------------------------------
int main() {
    printf("============================================================\n");
    printf("  CUDA Smoke Test — M3 Build Infrastructure Verification\n");
    printf("============================================================\n\n");

    // --- Step 1: Check CUDA device count ---
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    printf("[INFO] CUDA devices found: %d\n", device_count);
    if (device_count == 0) {
        fprintf(stderr, "[FAIL] No CUDA-capable devices found!\n");
        return EXIT_FAILURE;
    }

    // --- Step 2: Select device 0 and print info ---
    int device_id = 0;
    CUDA_CHECK(cudaSetDevice(device_id));
    printf("[INFO] Selected device: %d\n\n", device_id);
    print_device_info(device_id);

    // --- Step 3: Vector addition smoke test ---
    const int N = 1 << 16;   // 65,536 elements
    const size_t bytes = N * sizeof(float);

    printf("[INFO] Vector addition smoke test\n");
    printf("       N = %d elements (%zu bytes per vector)\n\n", N, bytes);

    // Allocate host memory
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "[FAIL] Host memory allocation failed!\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors: A[i] = i,  B[i] = 2*i
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(2 * i);
    }

    // Allocate device memory
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    printf("[PASS] Device memory allocated (3 x %zu bytes = %.2f KB)\n", bytes, 3.0 * bytes / 1024.0);

    // Copy host -> device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    printf("[PASS] Host-to-device memcpy completed\n");

    // Launch kernel
    const int threads_per_block = 256;
    const int blocks = (N + threads_per_block - 1) / threads_per_block;
    printf("[INFO] Launching kernel: <<<%d, %d>>> (%d total threads)\n",
           blocks, threads_per_block, blocks * threads_per_block);

    vector_add_kernel<<<blocks, threads_per_block>>>(d_A, d_B, d_C, N);

    // Check for launch errors
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
        fprintf(stderr, "[FAIL] Kernel launch error: %s\n", cudaGetErrorString(launch_err));
        return EXIT_FAILURE;
    }
    printf("[PASS] Kernel launched successfully\n");

    // Synchronize and check for execution errors
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("[PASS] Kernel execution completed (cudaDeviceSynchronize OK)\n");

    // Copy device -> host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    printf("[PASS] Device-to-host memcpy completed\n\n");

    // --- Step 4: Verify correctness ---
    printf("[INFO] Verifying results...\n");
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = (float)(3 * i);  // A[i] + B[i] = i + 2i = 3i
        if (fabsf(h_C[i] - expected) > 1e-5f) {
            if (errors < 5) {
                fprintf(stderr, "       Mismatch at index %d: got %.6f, expected %.6f\n",
                        i, h_C[i], expected);
            }
            errors++;
        }
    }

    if (errors == 0) {
        printf("[PASS] All %d elements correct! (C[i] = A[i] + B[i] = 3*i)\n\n", N);
    } else {
        fprintf(stderr, "[FAIL] %d / %d elements incorrect!\n\n", errors, N);
    }

    // --- Step 5: Cleanup ---
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    printf("[PASS] Memory freed successfully\n");

    // --- Final result ---
    printf("\n============================================================\n");
    if (errors == 0) {
        printf("  CUDA SMOKE TEST PASSED\n");
        printf("  All checks succeeded. CUDA build infrastructure is ready.\n");
    } else {
        printf("  CUDA SMOKE TEST FAILED\n");
        printf("  %d errors detected during verification.\n", errors);
    }
    printf("============================================================\n");

    return (errors == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
