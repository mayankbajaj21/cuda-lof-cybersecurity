/**
 * cuda_lof_pipeline_variants.cu - M8 LOF pipeline for V0..V5 (see .cuh)
 */

#include "cuda_lof_pipeline_variants.cuh"
#include "../kernels/distance_kernel.cuh"
#include "../kernels/knn_kernel.cuh"
#include "../kernels/reachability_kernel.cuh"
#include "../kernels/lrd_kernel.cuh"
#include "../kernels/lof_kernel.cuh"
#include "../kernels_optimized/distance_kernel_optimized.cuh"
#include "../kernels_optimized/knn_kernel_optimized.cuh"
#include "../kernels_variants/variant_kernels.cuh"
#include <cuda_fp16.h>
#include <cstdio>

namespace {

struct MemTracker {
    size_t current = 0;
    size_t peak = 0;
    size_t free_before = 0;
    size_t total = 0;
    size_t used_before = 0;
    size_t peak_used_delta = 0;

    void begin() {
        cudaMemGetInfo(&free_before, &total);
        used_before = total - free_before;
    }

    void sample() {
        size_t f = 0, t = 0;
        if (cudaMemGetInfo(&f, &t) == cudaSuccess) {
            size_t used = t - f;
            if (used > used_before && used - used_before > peak_used_delta) {
                peak_used_delta = used - used_before;
            }
        }
    }

    cudaError_t alloc(void** p, size_t bytes) {
        cudaError_t e = cudaMalloc(p, bytes);
        if (e == cudaSuccess) {
            current += bytes;
            if (current > peak) peak = current;
            sample();
        }
        return e;
    }

    void release(void** p, size_t bytes) {
        if (*p) {
            cudaFree(*p);
            *p = nullptr;
            current -= bytes;
        }
    }
};

struct EventPair {
    cudaEvent_t start = nullptr, stop = nullptr;
    bool used = false;
    void create() { cudaEventCreate(&start); cudaEventCreate(&stop); }
    void destroy() {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
    }
    void begin() { cudaEventRecord(start, 0); used = true; }
    void end() { cudaEventRecord(stop, 0); }
    float ms() const {
        float v = 0.0f;
        if (used) cudaEventElapsedTime(&v, start, stop);
        return v;
    }
};

bool is_fp16_variant(int v) { return v == 4 || v == 5; }

}  // namespace

VARIANTS_API int variant_max_k(int variant) {
    return variant == 0 ? KNN_MAX_K : OPT_KNN_MAX_K;
}

VARIANTS_API int compute_lof_pipeline_variant(
    const float* h_X, float* h_lof_scores, int N, int D, int k, float eps,
    int variant, VariantTimings* out_timings, VariantMemStats* out_mem,
    int* h_knn_indices_out
) {
    if (!h_X || !h_lof_scores) return -1;
    if (variant < 0 || variant >= LOF_VARIANT_COUNT) return -2;
    if (N <= 0 || D <= 0 || k <= 0 || k >= N || k > variant_max_k(variant)) return -2;

    const bool fp16 = is_fp16_variant(variant);
    const int Dp = fp16 ? fp16_padded_dim(D) : D;
    const size_t elem = fp16 ? sizeof(__half) : sizeof(float);
    const size_t x_bytes = (size_t)N * D * sizeof(float);
    const size_t xh_bytes = (size_t)N * Dp * sizeof(__half);
    const size_t dist_bytes = (size_t)N * N * elem;
    const size_t knn_idx_bytes = (size_t)N * k * sizeof(int);
    const size_t knn_dist_bytes = (size_t)N * k * sizeof(float);
    const size_t vec_bytes = (size_t)N * sizeof(float);

    MemTracker mem;
    mem.begin();

    EventPair e2e, h2d, conv, dist_ev, knn_ev, reach_ev, lrd_ev, lof_ev, d2h;
    EventPair* all_events[] = {&e2e, &h2d, &conv, &dist_ev, &knn_ev, &reach_ev, &lrd_ev, &lof_ev, &d2h};
    for (EventPair* e : all_events) e->create();

    float* d_X = nullptr;
    __half* d_Xh = nullptr;
    void* d_dist = nullptr;
    unsigned int* d_overflow = nullptr;
    int* d_knn_idx = nullptr;
    float* d_knn_dist = nullptr;
    float* d_reach = nullptr;
    float* d_lrd = nullptr;
    float* d_lof = nullptr;
    unsigned int overflow_host = 0;
    int rc = 0;
    cudaError_t err = cudaSuccess;

#define VSTEP(call, code)                                                              \
    do {                                                                               \
        err = (call);                                                                  \
        if (err != cudaSuccess) {                                                      \
            fprintf(stderr, "[CUDA ERROR] variant %d: %s -> %s\n", variant, #call,    \
                    cudaGetErrorString(err));                                          \
            rc = (code);                                                               \
            goto cleanup;                                                              \
        }                                                                              \
    } while (0)

    e2e.begin();

    VSTEP(mem.alloc((void**)&d_X, x_bytes), -3);
    h2d.begin();
    VSTEP(cudaMemcpy(d_X, h_X, x_bytes, cudaMemcpyHostToDevice), -4);
    h2d.end();

    if (fp16) {
        VSTEP(mem.alloc((void**)&d_overflow, sizeof(unsigned int)), -5);
        VSTEP(cudaMemset(d_overflow, 0, sizeof(unsigned int)), -5);
        VSTEP(mem.alloc((void**)&d_Xh, xh_bytes), -5);
        conv.begin();
        VSTEP(launch_convert_features_to_half(d_X, d_Xh, N, D, Dp), -6);
        conv.end();
        VSTEP(cudaDeviceSynchronize(), -6);
        mem.release((void**)&d_X, x_bytes);
    }

    VSTEP(mem.alloc(&d_dist, dist_bytes), -7);
    dist_ev.begin();
    switch (variant) {
        case 0: err = launch_pairwise_distance_kernel(d_X, (float*)d_dist, N, D); break;
        case 1: err = launch_pairwise_distance_kernel_optimized(d_X, (float*)d_dist, N, D); break;
        case 2: err = launch_distance_regblock(d_X, (float*)d_dist, N, D); break;
        case 3: err = launch_distance_tiled_async(d_X, (float*)d_dist, N, D); break;
        case 4: err = launch_distance_tiled_fp16(d_Xh, (__half*)d_dist, N, Dp, d_overflow); break;
        default: err = launch_distance_combined_fp16(d_Xh, (__half*)d_dist, N, Dp, d_overflow); break;
    }
    dist_ev.end();
    VSTEP(err, -8);
    VSTEP(cudaDeviceSynchronize(), -8);
    mem.release((void**)&d_X, x_bytes);
    mem.release((void**)&d_Xh, xh_bytes);

    VSTEP(mem.alloc((void**)&d_knn_idx, knn_idx_bytes), -9);
    VSTEP(mem.alloc((void**)&d_knn_dist, knn_dist_bytes), -9);
    knn_ev.begin();
    if (variant == 0) err = launch_knn_kernel((const float*)d_dist, d_knn_idx, d_knn_dist, N, k);
    else if (fp16) err = launch_knn_fp16((const __half*)d_dist, d_knn_idx, d_knn_dist, N, k);
    else err = launch_knn_kernel_optimized((const float*)d_dist, d_knn_idx, d_knn_dist, N, k);
    knn_ev.end();
    VSTEP(err, -10);
    VSTEP(cudaDeviceSynchronize(), -10);
    mem.release(&d_dist, dist_bytes);

    VSTEP(mem.alloc((void**)&d_reach, knn_dist_bytes), -11);
    reach_ev.begin();
    VSTEP(launch_reachability_distance_kernel(d_knn_idx, d_knn_dist, d_reach, N, k), -12);
    reach_ev.end();
    VSTEP(cudaDeviceSynchronize(), -12);
    mem.release((void**)&d_knn_dist, knn_dist_bytes);

    VSTEP(mem.alloc((void**)&d_lrd, vec_bytes), -13);
    lrd_ev.begin();
    VSTEP(launch_lrd_kernel(d_reach, d_lrd, N, k, eps), -14);
    lrd_ev.end();
    VSTEP(cudaDeviceSynchronize(), -14);
    mem.release((void**)&d_reach, knn_dist_bytes);

    VSTEP(mem.alloc((void**)&d_lof, vec_bytes), -15);
    lof_ev.begin();
    VSTEP(launch_lof_scoring_kernel(d_lrd, d_knn_idx, d_lof, N, k, eps), -16);
    lof_ev.end();
    VSTEP(cudaDeviceSynchronize(), -16);

    d2h.begin();
    VSTEP(cudaMemcpy(h_lof_scores, d_lof, vec_bytes, cudaMemcpyDeviceToHost), -17);
    d2h.end();
    e2e.end();
    VSTEP(cudaEventSynchronize(e2e.stop), -17);

    if (h_knn_indices_out) {
        VSTEP(cudaMemcpy(h_knn_indices_out, d_knn_idx, knn_idx_bytes, cudaMemcpyDeviceToHost), -18);
    }
    if (d_overflow) {
        VSTEP(cudaMemcpy(&overflow_host, d_overflow, sizeof(unsigned int), cudaMemcpyDeviceToHost), -18);
    }

    if (out_timings) {
        out_timings->h2d_ms = h2d.ms();
        out_timings->fp16_convert_ms = conv.ms();
        out_timings->distance_kernel_ms = dist_ev.ms();
        out_timings->knn_kernel_ms = knn_ev.ms();
        out_timings->reachability_kernel_ms = reach_ev.ms();
        out_timings->lrd_kernel_ms = lrd_ev.ms();
        out_timings->lof_kernel_ms = lof_ev.ms();
        out_timings->d2h_ms = d2h.ms();
        out_timings->end_to_end_ms = e2e.ms();
        out_timings->total_kernel_ms = out_timings->fp16_convert_ms + out_timings->distance_kernel_ms +
                                       out_timings->knn_kernel_ms + out_timings->reachability_kernel_ms +
                                       out_timings->lrd_kernel_ms + out_timings->lof_kernel_ms;
    }

cleanup:
#undef VSTEP
    mem.release((void**)&d_X, x_bytes);
    mem.release((void**)&d_Xh, xh_bytes);
    mem.release(&d_dist, dist_bytes);
    mem.release((void**)&d_knn_idx, knn_idx_bytes);
    mem.release((void**)&d_knn_dist, knn_dist_bytes);
    mem.release((void**)&d_reach, knn_dist_bytes);
    mem.release((void**)&d_lrd, vec_bytes);
    mem.release((void**)&d_lof, vec_bytes);
    mem.release((void**)&d_overflow, sizeof(unsigned int));

    if (out_mem) {
        out_mem->tracked_peak_bytes = mem.peak;
        out_mem->memgetinfo_peak_used_delta_bytes = mem.peak_used_delta;
        out_mem->memgetinfo_free_before_bytes = mem.free_before;
        out_mem->memgetinfo_total_bytes = mem.total;
        out_mem->dist_matrix_bytes = dist_bytes;
        out_mem->fp16_overflow_count = overflow_host;
        out_mem->dist_element_bytes = (int)elem;
    }
    for (EventPair* e : all_events) e->destroy();
    return rc;
}

VARIANTS_API int get_variant_kernel_attributes(int variant, int which, VariantKernelAttributes* out) {
    if (!out || variant < 0 || variant >= LOF_VARIANT_COUNT || which < 0 || which > 1) return -1;
    const void* fn = nullptr;
    int block_threads = 256;
    if (which == 0) {
        switch (variant) {
            case 0: fn = (const void*)pairwise_distance_naive_kernel; break;
            case 1: fn = (const void*)pairwise_distance_tiled_kernel; break;
            case 2: fn = (const void*)pairwise_distance_regblock_kernel; break;
            case 3: fn = (const void*)pairwise_distance_tiled_async_kernel; break;
            case 4: fn = (const void*)pairwise_distance_tiled_fp16_kernel; break;
            default: fn = (const void*)pairwise_distance_combined_fp16_kernel; break;
        }
    } else {
        if (variant == 0) fn = (const void*)knn_naive_kernel;
        else if (is_fp16_variant(variant)) fn = (const void*)knn_fp16_kernel;
        else fn = (const void*)knn_optimized_kernel;
    }
    cudaFuncAttributes attr;
    cudaError_t e = cudaFuncGetAttributes(&attr, fn);
    if (e != cudaSuccess) return -2;
    out->num_regs = attr.numRegs;
    out->static_shared_bytes = (int)attr.sharedSizeBytes;
    out->local_bytes = (int)attr.localSizeBytes;
    out->max_threads_per_block = attr.maxThreadsPerBlock;
    out->block_threads = block_threads;

    int blocks = 0, device = 0;
    cudaDeviceProp prop;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fn, block_threads, 0) != cudaSuccess) return -3;
    out->max_active_blocks_per_sm = blocks;
    out->theoretical_occupancy_pct = 100.0f * (float)(blocks * block_threads) / (float)prop.maxThreadsPerMultiProcessor;
    return 0;
}
