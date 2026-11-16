/**
 * variant_kernels.cu - M8 advanced optimization kernels (see variant_kernels.cuh)
 */

#include "variant_kernels.cuh"
#include "../kernels_optimized/knn_kernel_optimized.cuh"
#include <cuda_pipeline.h>
#include <cstdint>

// cp.async is an sm_80+ instruction; older targets fall back to plain loads
// so the same source builds for the project's default sm_86 and lower.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define VAR_HAS_CP_ASYNC 1
#else
#define VAR_HAS_CP_ASYNC 0
#endif

__device__ __forceinline__ void async_copy_4(void* dst, const void* src) {
#if VAR_HAS_CP_ASYNC
    __pipeline_memcpy_async(dst, src, 4);
#else
    *reinterpret_cast<float*>(dst) = *reinterpret_cast<const float*>(src);
#endif
}

__device__ __forceinline__ void async_copy_16(void* dst, const void* src) {
#if VAR_HAS_CP_ASYNC
    __pipeline_memcpy_async(dst, src, 16);
#else
    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
#endif
}

__device__ __forceinline__ void async_commit() {
#if VAR_HAS_CP_ASYNC
    __pipeline_commit();
#endif
}

__device__ __forceinline__ void async_wait_all_but_one() {
#if VAR_HAS_CP_ASYNC
    __pipeline_wait_prior(1);
#endif
}

int fp16_padded_dim(int D) {
    return ((D + FP16_PAD_MULTIPLE - 1) / FP16_PAD_MULTIPLE) * FP16_PAD_MULTIPLE;
}

__device__ __forceinline__ __half store_distance_fp16(float v, unsigned int* overflow_count) {
    if (v > FP16_MAX_FINITE) {
        atomicAdd(overflow_count, 1u);
    }
    return __float2half_rn(v);
}

// ---------------------------------------------------------------------------
// FP16 conversion with padded row stride (pad columns are zero)
// ---------------------------------------------------------------------------
__global__ void convert_features_to_half_padded_kernel(
    const float* __restrict__ X, __half* __restrict__ Xh, int N, int D, int Dp
) {
    size_t total = (size_t)N * Dp;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (size_t)gridDim.x * blockDim.x) {
        int i = (int)(idx / Dp);
        int d = (int)(idx % Dp);
        Xh[idx] = (d < D) ? __float2half_rn(X[(size_t)i * D + d]) : __float2half_rn(0.0f);
    }
}

// ---------------------------------------------------------------------------
// V2: register-blocked distance (fp32, exact)
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(RB_THREADS * RB_THREADS)
pairwise_distance_regblock_kernel(
    const float* __restrict__ X, float* __restrict__ dist, int N, int D
) {
    __shared__ float As[RB_TILE][RB_DK + 1];
    __shared__ float Bs[RB_TILE][RB_DK + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * RB_THREADS + tx;
    const int row0 = blockIdx.y * RB_TILE;
    const int col0 = blockIdx.x * RB_TILE;

    float acc[RB_MICRO][RB_MICRO];
    #pragma unroll
    for (int a = 0; a < RB_MICRO; ++a)
        #pragma unroll
        for (int b = 0; b < RB_MICRO; ++b)
            acc[a][b] = 0.0f;

    constexpr int LOADS_PER_THREAD = (RB_TILE * RB_DK) / (RB_THREADS * RB_THREADS);

    for (int d0 = 0; d0 < D; d0 += RB_DK) {
        #pragma unroll
        for (int p = 0; p < LOADS_PER_THREAD; ++p) {
            int idx = tid + p * RB_THREADS * RB_THREADS;
            int r = idx / RB_DK;
            int c = idx % RB_DK;
            int gd = d0 + c;
            As[r][c] = (row0 + r < N && gd < D) ? X[(size_t)(row0 + r) * D + gd] : 0.0f;
            Bs[r][c] = (col0 + r < N && gd < D) ? X[(size_t)(col0 + r) * D + gd] : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int t = 0; t < RB_DK; ++t) {
            float av[RB_MICRO], bv[RB_MICRO];
            #pragma unroll
            for (int a = 0; a < RB_MICRO; ++a) av[a] = As[ty + a * RB_THREADS][t];
            #pragma unroll
            for (int b = 0; b < RB_MICRO; ++b) bv[b] = Bs[tx + b * RB_THREADS][t];
            #pragma unroll
            for (int a = 0; a < RB_MICRO; ++a)
                #pragma unroll
                for (int b = 0; b < RB_MICRO; ++b) {
                    float diff = av[a] - bv[b];
                    acc[a][b] += diff * diff;
                }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int a = 0; a < RB_MICRO; ++a) {
        int i = row0 + ty + a * RB_THREADS;
        if (i >= N) continue;
        #pragma unroll
        for (int b = 0; b < RB_MICRO; ++b) {
            int j = col0 + tx + b * RB_THREADS;
            if (j < N) dist[(size_t)i * N + j] = (i == j) ? 0.0f : sqrtf(acc[a][b]);
        }
    }
}

// ---------------------------------------------------------------------------
// V3: V1 tiling + cp.async double buffering (fp32, exact)
// ---------------------------------------------------------------------------
__global__ void pairwise_distance_tiled_async_kernel(
    const float* __restrict__ X, float* __restrict__ dist, int N, int D
) {
    __shared__ __align__(16) float row_tile[2][V3_TILE][V3_TILE];
    __shared__ __align__(16) float col_tile[2][V3_TILE][V3_TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row0 = blockIdx.y * V3_TILE;
    const int col0 = blockIdx.x * V3_TILE;
    const int i = row0 + ty;
    const int j = col0 + tx;
    const int n_chunks = (D + V3_TILE - 1) / V3_TILE;

    auto issue_tile = [&](int buf, int chunk) {
        int d = chunk * V3_TILE + tx;
        float* rdst = &row_tile[buf][ty][tx];
        float* cdst = &col_tile[buf][ty][tx];
        if (row0 + ty < N && d < D) async_copy_4(rdst, &X[(size_t)(row0 + ty) * D + d]);
        else *rdst = 0.0f;
        if (col0 + ty < N && d < D) async_copy_4(cdst, &X[(size_t)(col0 + ty) * D + d]);
        else *cdst = 0.0f;
    };

    float sum_sq = 0.0f;
    issue_tile(0, 0);
    async_commit();

    for (int c = 0; c < n_chunks; ++c) {
        if (c + 1 < n_chunks) issue_tile((c + 1) & 1, c + 1);
        async_commit();
        async_wait_all_but_one();
        __syncthreads();

        const int buf = c & 1;
        #pragma unroll
        for (int t = 0; t < V3_TILE; ++t) {
            float diff = row_tile[buf][ty][t] - col_tile[buf][tx][t];
            sum_sq += diff * diff;
        }
        __syncthreads();
    }

    if (i < N && j < N) {
        dist[(size_t)i * N + j] = (i == j) ? 0.0f : sqrtf(sum_sq);
    }
}

// ---------------------------------------------------------------------------
// V4: V1 tiling with fp16 storage (APPROXIMATE), fp32 accumulation
// ---------------------------------------------------------------------------
__global__ void pairwise_distance_tiled_fp16_kernel(
    const __half* __restrict__ Xh, __half* __restrict__ dist, int N, int Dp,
    unsigned int* __restrict__ overflow_count
) {
    __shared__ __half row_tile[V3_TILE][V3_TILE];
    __shared__ __half col_tile[V3_TILE][V3_TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row0 = blockIdx.y * V3_TILE;
    const int col0 = blockIdx.x * V3_TILE;
    const int i = row0 + ty;
    const int j = col0 + tx;
    const __half zero = __float2half_rn(0.0f);

    float sum_sq = 0.0f;
    for (int d0 = 0; d0 < Dp; d0 += V3_TILE) {
        int d = d0 + tx;
        row_tile[ty][tx] = (row0 + ty < N && d < Dp) ? Xh[(size_t)(row0 + ty) * Dp + d] : zero;
        col_tile[ty][tx] = (col0 + ty < N && d < Dp) ? Xh[(size_t)(col0 + ty) * Dp + d] : zero;
        __syncthreads();

        #pragma unroll
        for (int t = 0; t < V3_TILE; ++t) {
            float diff = __half2float(row_tile[ty][t]) - __half2float(col_tile[tx][t]);
            sum_sq += diff * diff;
        }
        __syncthreads();
    }

    if (i < N && j < N) {
        dist[(size_t)i * N + j] = (i == j) ? zero : store_distance_fp16(sqrtf(sum_sq), overflow_count);
    }
}

// ---------------------------------------------------------------------------
// V5: register blocking + cp.async double buffering + fp16 storage
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(RB_THREADS * RB_THREADS)
pairwise_distance_combined_fp16_kernel(
    const __half* __restrict__ Xh, __half* __restrict__ dist, int N, int Dp,
    unsigned int* __restrict__ overflow_count
) {
    // Row stride of 24 halves (48 bytes) keeps every 16-byte cp.async
    // destination aligned; it costs a 2-way bank conflict on column reads.
    __shared__ __align__(16) __half As[2][RB_TILE][V5_SMEM_STRIDE];
    __shared__ __align__(16) __half Bs[2][RB_TILE][V5_SMEM_STRIDE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * RB_THREADS + tx;
    const int row0 = blockIdx.y * RB_TILE;
    const int col0 = blockIdx.x * RB_TILE;
    const int n_chunks = (Dp + RB_DK - 1) / RB_DK;

    // 64 rows x 2 segments of 8 halves per matrix = 128 copies; threads
    // 0..127 load A, 128..255 load B (one 16-byte copy each).
    const bool loads_a = tid < 128;
    const int seg_id = loads_a ? tid : tid - 128;
    const int seg_row = seg_id >> 1;
    const int seg_col = (seg_id & 1) * 8;
    const int g_row = (loads_a ? row0 : col0) + seg_row;

    auto issue_tile = [&](int buf, int chunk) {
        int gd = chunk * RB_DK + seg_col;
        __half* dst = loads_a ? &As[buf][seg_row][seg_col] : &Bs[buf][seg_row][seg_col];
        if (g_row < N && gd < Dp) async_copy_16(dst, &Xh[(size_t)g_row * Dp + gd]);
        else *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
    };

    float acc[RB_MICRO][RB_MICRO];
    #pragma unroll
    for (int a = 0; a < RB_MICRO; ++a)
        #pragma unroll
        for (int b = 0; b < RB_MICRO; ++b)
            acc[a][b] = 0.0f;

    issue_tile(0, 0);
    async_commit();

    for (int c = 0; c < n_chunks; ++c) {
        if (c + 1 < n_chunks) issue_tile((c + 1) & 1, c + 1);
        async_commit();
        async_wait_all_but_one();
        __syncthreads();

        const int buf = c & 1;
        #pragma unroll
        for (int t = 0; t < RB_DK; t += 2) {
            float2 av[RB_MICRO], bv[RB_MICRO];
            #pragma unroll
            for (int a = 0; a < RB_MICRO; ++a)
                av[a] = __half22float2(*reinterpret_cast<const __half2*>(&As[buf][ty + a * RB_THREADS][t]));
            #pragma unroll
            for (int b = 0; b < RB_MICRO; ++b)
                bv[b] = __half22float2(*reinterpret_cast<const __half2*>(&Bs[buf][tx + b * RB_THREADS][t]));
            #pragma unroll
            for (int a = 0; a < RB_MICRO; ++a)
                #pragma unroll
                for (int b = 0; b < RB_MICRO; ++b) {
                    float dx = av[a].x - bv[b].x;
                    acc[a][b] += dx * dx;
                    float dy = av[a].y - bv[b].y;
                    acc[a][b] += dy * dy;
                }
        }
        __syncthreads();
    }

    const __half zero = __float2half_rn(0.0f);
    #pragma unroll
    for (int a = 0; a < RB_MICRO; ++a) {
        int i = row0 + ty + a * RB_THREADS;
        if (i >= N) continue;
        #pragma unroll
        for (int b = 0; b < RB_MICRO; ++b) {
            int j = col0 + tx + b * RB_THREADS;
            if (j < N) {
                dist[(size_t)i * N + j] = (i == j) ? zero : store_distance_fp16(sqrtf(acc[a][b]), overflow_count);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// k-NN over an fp16 distance matrix (same insertion top-k as the M5 kernel)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void knn_insert(float d, int jj, float* best_dist, int* best_idx, int k) {
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

__global__ void knn_fp16_kernel(
    const __half* __restrict__ dist, int* __restrict__ neighbor_indices,
    float* __restrict__ neighbor_distances, int N, int k
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || k <= 0 || k > OPT_KNN_MAX_K || k >= N) return;

    float best_dist[OPT_KNN_MAX_K];
    int best_idx[OPT_KNN_MAX_K];
    for (int m = 0; m < k; ++m) {
        best_dist[m] = 1e38f;
        best_idx[m] = -1;
    }

    const __half* row = dist + (size_t)i * N;

    // 8 halves per 16-byte load when every row start is 16-byte aligned.
    const int n_vec8 = (N % 8 == 0) ? (N / 8) : 0;
    const uint4* row8 = reinterpret_cast<const uint4*>(row);
    for (int v = 0; v < n_vec8; ++v) {
        uint4 chunk = row8[v];
        const __half2* h2 = reinterpret_cast<const __half2*>(&chunk);
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            float2 f = __half22float2(h2[q]);
            int jj = v * 8 + q * 2;
            if (jj != i) knn_insert(f.x, jj, best_dist, best_idx, k);
            if (jj + 1 != i) knn_insert(f.y, jj + 1, best_dist, best_idx, k);
        }
    }
    for (int j = n_vec8 * 8; j < N; ++j) {
        if (j != i) knn_insert(__half2float(row[j]), j, best_dist, best_idx, k);
    }

    int* out_idx = neighbor_indices + (size_t)i * k;
    float* out_dist = neighbor_distances + (size_t)i * k;
    for (int m = 0; m < k; ++m) {
        out_idx[m] = best_idx[m];
        out_dist[m] = best_dist[m];
    }
}

// ---------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------
cudaError_t launch_convert_features_to_half(const float* d_X, __half* d_Xh, int N, int D, int Dp) {
    size_t total = (size_t)N * Dp;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535 * 16) blocks = 65535 * 16;
    convert_features_to_half_padded_kernel<<<blocks, threads>>>(d_X, d_Xh, N, D, Dp);
    return cudaGetLastError();
}

cudaError_t launch_distance_regblock(const float* d_X, float* d_dist, int N, int D) {
    dim3 block(RB_THREADS, RB_THREADS);
    dim3 grid((N + RB_TILE - 1) / RB_TILE, (N + RB_TILE - 1) / RB_TILE);
    pairwise_distance_regblock_kernel<<<grid, block>>>(d_X, d_dist, N, D);
    return cudaGetLastError();
}

cudaError_t launch_distance_tiled_async(const float* d_X, float* d_dist, int N, int D) {
    dim3 block(V3_TILE, V3_TILE);
    dim3 grid((N + V3_TILE - 1) / V3_TILE, (N + V3_TILE - 1) / V3_TILE);
    pairwise_distance_tiled_async_kernel<<<grid, block>>>(d_X, d_dist, N, D);
    return cudaGetLastError();
}

cudaError_t launch_distance_tiled_fp16(const __half* d_Xh, __half* d_dist, int N, int Dp, unsigned int* d_overflow) {
    dim3 block(V3_TILE, V3_TILE);
    dim3 grid((N + V3_TILE - 1) / V3_TILE, (N + V3_TILE - 1) / V3_TILE);
    pairwise_distance_tiled_fp16_kernel<<<grid, block>>>(d_Xh, d_dist, N, Dp, d_overflow);
    return cudaGetLastError();
}

cudaError_t launch_distance_combined_fp16(const __half* d_Xh, __half* d_dist, int N, int Dp, unsigned int* d_overflow) {
    dim3 block(RB_THREADS, RB_THREADS);
    dim3 grid((N + RB_TILE - 1) / RB_TILE, (N + RB_TILE - 1) / RB_TILE);
    pairwise_distance_combined_fp16_kernel<<<grid, block>>>(d_Xh, d_dist, N, Dp, d_overflow);
    return cudaGetLastError();
}

cudaError_t launch_knn_fp16(const __half* d_dist, int* d_idx, float* d_knn_dist, int N, int k) {
    if (N <= 0 || k <= 0 || k >= N || k > OPT_KNN_MAX_K) return cudaErrorInvalidValue;
    const int threads = 256;
    knn_fp16_kernel<<<(N + threads - 1) / threads, threads>>>(d_dist, d_idx, d_knn_dist, N, k);
    return cudaGetLastError();
}
