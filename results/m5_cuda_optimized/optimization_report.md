# M5 — Profiling-Driven CUDA Optimization (MEASURED)

Hardware: NVIDIA GeForce RTX 4050 Laptop GPU (sm_89, 20 SMs, 6GB VRAM, 24MB L2 cache),
CUDA Toolkit 13.5, MSVC (VS2022 Community). Same machine as the M4 profiling pass —
**not** the RTX 3050 (sm_86) that M3's README numbers were measured on; do not
conflate the two.

Methodology: 2 discarded warm-up runs + 10 timed runs per subset, CUDA-event timing
(`cudaEventElapsedTime`) for every pipeline stage, k=20, contamination=0.10 — identical
to M4's methodology, applied to both the naive and optimized pipelines for a fair
before/after comparison.

## What was optimized, and why (M4 evidence)

| Stage | Naive share of kernel time (M4) | Action |
| :--- | :---: | :--- |
| Pairwise distance | 78.9% (10K) / 80.0% (25K) | **Optimized** — shared-memory tiled kernel |
| k-NN | 19.9% (10K) / 20.9% (25K) | **Optimized** — right-sized register buffer + vectorized scan |
| Reachability / LRD / LOF scoring | <0.2% combined | **Not optimized** — M4 showed negligible, unmodified |

## Optimization 1: Pairwise distance — shared-memory tiling

`src/cuda/kernels_optimized/distance_kernel_optimized.cu` tiles both the row- and
column-blocks of `X` into shared memory so each feature value is loaded from global
memory once per 16 output distances instead of once per distance computed.

**Important negative result en route, reported honestly:** the first implementation
used a 32×32 tile (1024 threads/block, the hardware maximum). That version measured
**~5x SLOWER** than the naive kernel (140ms vs 28ms at 10K) — with 1024 threads/block
only one block can be resident per SM at a time, so there is no second block available
to hide `__syncthreads()` and memory latency behind, and occupancy collapsed. Dropping
to a 16×16 tile (256 threads/block, matching the naive kernel's own block size) let
several blocks reside per SM concurrently and fixed this.

A second, smaller fix: the first 16×16 version also carried a bounds-check branch
(`if (t < valid_width)`) in the innermost accumulation loop that was **functionally
redundant** — the shared-memory tiles are already zero-padded for out-of-range
features, so the "extra" terms already evaluate to `0 - 0 = 0`. Removing the branch
(pure divergence overhead, zero semantic benefit) further cut the gap.

**Net result:** the distance kernel is **1.14–1.18x faster** than naive. This is a
real but modest win, not the order-of-magnitude speedup tiling sometimes produces —
and there is a concrete hardware reason why: `X` is only `N × 68 × 4` bytes
(2.7MB at N=10K, 6.8MB at N=25K), which fits entirely inside this GPU's 24MB L2 cache.
The naive kernel's "redundant" reads of `X` were therefore already mostly L2 cache
hits, not full global-memory round-trips — so shared-memory tiling has much less
headroom to exploit here than it would for a feature matrix too large for L2.

## Optimization 2: k-NN — register right-sizing + vectorized scan

`src/cuda/kernels_optimized/knn_kernel_optimized.cu` keeps the exact same sorted-
insertion top-k algorithm as the naive kernel, with two changes:
1. The per-thread top-k buffer shrinks from `KNN_MAX_K=256` to `OPT_KNN_MAX_K=32`.
   The naive kernel allocates registers/local memory for 256 candidate slots on
   every thread even though the project always runs k=20 — almost certainly causing
   local-memory spill. (Constraint: the optimized kernel supports k ≤ 32; the naive
   kernel's k ≤ 256 path remains available unmodified for larger k.)
2. `float4`-vectorized reads of the distance row (only when N is a multiple of 4,
   so every row stays 16-byte aligned; falls back to scalar reads otherwise).

**Net result:** k-NN kernel is **1.02–1.07x faster**. Modest — k-NN was already the
smaller of the two bottlenecks, and insertion-sort for k=20 is cheap relative to the
O(N) scan it's embedded in either way.

## End-to-end results

| Subset | Naive mean (ms) | Optimized mean (ms) | Speedup |
| :--- | :---: | :---: | :---: |
| 10K | 52.26 (σ=9.33) | 43.63 (σ=1.50) | **1.20x** |
| 25K | 278.26 (σ=2.07) | 264.99 (σ=15.55) | **1.05x** |

Run-to-run end-to-end variance (σ) is noticeably larger than per-kernel event-timed
variance — expected, since end-to-end also includes `cudaMalloc`/`cudaFree` and driver
scheduling overhead not wrapped in CUDA events. The direction of the result (optimized
faster) was reproduced consistently across repeated full runs of this script; the exact
magnitude varies run-to-run in the 1.05x–1.6x range.

## Correctness validation

All three implementations agree to within float32 precision:

| Comparison | 10K Pearson r | 25K Pearson r | Anomaly agreement |
| :--- | :---: | :---: | :---: |
| CPU (fresh baseline) vs. naive CUDA | 0.99999997 | 0.99999999 | 100% |
| CPU (fresh baseline) vs. optimized CUDA | 0.99999997 | 0.99999999 | 100% |
| Naive CUDA vs. optimized CUDA | **1.00000000** | **1.00000000** | 100% |

Naive-vs-optimized distance matrices and k-NN indices/distances are **bit-identical**
(`max_abs_error = 0.0` for both). The optimized kernels introduce zero new numerical
discrepancy relative to naive — the small (~1e-7 mean, ~1e-3 max) CPU-vs-CUDA gap is
the same pre-existing float32 cancellation effect already documented in `README.md`
§7, unrelated to this milestone's changes.

The fresh CPU baseline used here (`results/m2_cpu_baseline/benchmarks_current/`) was
regenerated against the CURRENT `data/processed/subsets/*.npz` specifically to fix the
stale-baseline mismatch flagged during M4 (the original M2 artifacts were computed
against an older, since-regenerated version of those subsets). The original M2
artifacts are untouched.

## What was deliberately not optimized, and why

Reachability distance, LRD, and LOF-scoring kernels are copied unmodified into the
optimized pipeline. M4 measured all three combined at under 0.2% of total kernel time
at both 10K and 25K — any speedup there is numerically incapable of moving end-to-end
performance by a measurable amount, so no optimization effort was spent on them.

## Architecture

Three implementations remain cleanly separated and independently buildable:
- CPU: `src/python/cpu/cpu_lof.py` (unmodified)
- Naive CUDA: `src/cuda/kernels/*.cu`, `src/cuda/lof/cuda_lof_pipeline.cu` → `bin/cuda_lof.dll` (unmodified)
- Optimized CUDA: `src/cuda/kernels_optimized/*.cu`, `src/cuda/lof/cuda_lof_pipeline_optimized.cu` → `bin/cuda_lof_optimized.dll` (new)

Python selection: `src/python/cuda/lof.py` (naive) vs. `src/python/cuda/lof_optimized.py`
(optimized) — two clearly separate modules, both available to callers.
