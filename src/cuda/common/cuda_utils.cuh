/**
 * cuda_utils.cuh - Common CUDA utility macros and helpers
 * 
 * Project: CUDA LOF Cybersecurity
 * Milestone: M3 - CUDA LOF Baseline
 * 
 * Provides:
 *   - CUDA_CHECK: Macro for runtime API error checking
 *   - CUDA_CHECK_LAST: Macro for checking the last kernel launch error
 *   - print_device_info: Function to query and print GPU device properties
 */

#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// Error-checking macros
// ---------------------------------------------------------------------------

/**
 * CUDA_CHECK(call)
 * Wraps any cudaError_t-returning CUDA runtime API call.
 * Prints file, line, and error string on failure, then exits.
 */
#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

/**
 * CUDA_CHECK_LAST()
 * Checks for errors after a kernel launch (which doesn't return cudaError_t).
 * Must be called after the kernel launch and before cudaDeviceSynchronize().
 */
#define CUDA_CHECK_LAST()                                                      \
    do {                                                                        \
        cudaError_t err = cudaGetLastError();                                   \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA Kernel Error at %s:%d - %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Device info utility
// ---------------------------------------------------------------------------

/**
 * Print key properties of the specified CUDA device.
 */
inline void print_device_info(int device_id = 0) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    int runtime_version = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    int driver_version = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));

    printf("============================================\n");
    printf("  CUDA Device Information\n");
    printf("============================================\n");
    printf("  Device ID            : %d\n", device_id);
    printf("  Device Name          : %s\n", prop.name);
    printf("  Compute Capability   : %d.%d\n", prop.major, prop.minor);
    printf("  Global Memory        : %.2f GB (%zu bytes)\n",
           (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
           prop.totalGlobalMem);
    printf("  Shared Memory / Block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("  Warp Size            : %d\n", prop.warpSize);
    printf("  Max Threads / Block  : %d\n", prop.maxThreadsPerBlock);
    printf("  Max Block Dimensions : [%d, %d, %d]\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Max Grid Dimensions  : [%d, %d, %d]\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Multiprocessors (SMs): %d\n", prop.multiProcessorCount);
    int clock_khz = 0, mem_clock_khz = 0;
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device_id);
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, device_id);
    printf("  Clock Rate           : %.2f GHz\n", clock_khz / 1.0e6);
    printf("  Memory Clock Rate    : %.2f GHz\n", mem_clock_khz / 1.0e6);
    printf("  Memory Bus Width     : %d-bit\n", prop.memoryBusWidth);
    printf("  L2 Cache Size        : %d bytes\n", prop.l2CacheSize);
    printf("  CUDA Runtime Version : %d.%d\n",
           runtime_version / 1000, (runtime_version % 100) / 10);
    printf("  CUDA Driver Version  : %d.%d\n",
           driver_version / 1000, (driver_version % 100) / 10);
    printf("============================================\n\n");
}

#endif  // CUDA_UTILS_CUH
