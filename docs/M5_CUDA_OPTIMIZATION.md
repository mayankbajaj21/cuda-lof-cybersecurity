# Milestone M5: CUDA Optimization — MEASURED

**Status: complete.** All numbers below are real CUDA-event measurements on this
machine (NVIDIA GeForce RTX 4050 Laptop GPU, sm_89, CUDA Toolkit 13.5), not
projections. Full detail and raw artifacts: `results/m5_cuda_optimized/optimization_report.md`.

## 1. What M4 justified optimizing

M4's real per-kernel profiling (also on this machine) measured:

| Stage | Share of kernel time |
| :--- | :---: |
| Pairwise distance | 78.9% (N=10K) / 80.0% (N=25K) |
| k-NN | 19.9% (N=10K) / 20.9% (N=25K) |
| Reachability + LRD + LOF scoring | <0.2% combined |

Only the first two were worth optimizing; the last three were left unmodified.

## 2. What was implemented

- **Pairwise distance**: shared-memory tiled kernel, 16×16 tiles (not 32×32 —
  see the negative result below), `src/cuda/kernels_optimized/distance_kernel_optimized.cu`.
- **k-NN**: right-sized top-k register buffer (32 slots vs. the naive kernel's 256,
  avoiding local-memory spill) + `float4`-vectorized row scan,
  `src/cuda/kernels_optimized/knn_kernel_optimized.cu`.
- **Reachability / LRD / LOF scoring**: copied unmodified — M4 evidence showed
  optimizing them could not move end-to-end time measurably.

Built as a fully separate binary, `bin/cuda_lof_optimized.dll`, from a separate
source tree (`src/cuda/kernels_optimized/`, `src/cuda/lof/cuda_lof_pipeline_optimized.cu`).
The naive baseline (`bin/cuda_lof.dll`, `src/cuda/kernels/`) is untouched and remains
available for direct before/after comparison. Build with:
```
set ARCH=sm_89 && scripts\build_cuda_lof_optimized.bat
```

## 3. A real negative result, reported honestly

The first tiled-distance attempt used 32×32 tiles (1024 threads/block — the hardware
maximum). It measured **~5x slower** than the naive kernel: only one block of 1024
threads can be resident per SM at a time, so there's no second block to hide memory/
sync latency behind, and occupancy collapsed. Dropping to 16×16 tiles (256
threads/block) fixed this. A second fix (removing a bounds-check branch made
redundant by zero-padding) closed the remaining gap. This history is kept here
deliberately — per the project's experimental-integrity principle, a false start that
got fixed is more informative than silently only showing the final number.

## 4. Final measured results

| Subset | Naive end-to-end (ms) | Optimized end-to-end (ms) | Speedup |
| :--- | :---: | :---: | :---: |
| 10K | 52.26 | 43.63 | **1.20x** |
| 25K | 278.26 | 264.99 | **1.05x** |

Modest, not dramatic — and there's a concrete hardware reason: the feature matrix
(2.7–6.8MB at these sizes) fits entirely inside this GPU's 24MB L2 cache, so the
naive kernel's "redundant" reads were already mostly L2 hits, leaving tiling less
headroom to exploit than it would have on a larger, L2-exceeding dataset.

## 5. Correctness

Naive vs. optimized: **bit-identical** LOF scores, distance matrices, and k-NN
results (max abs error = 0.0, Pearson r = 1.00000000) at both 10K and 25K. Both
agree with a freshly-regenerated CPU baseline at r ≈ 0.9999999971, 100% anomaly-flag
agreement. Full numbers: `results/m5_cuda_optimized/correctness_report.json`.

## 6. Fixed: stale CPU baseline from M4

M4 flagged that `results/m2_cpu_baseline/benchmarks/*.npy` was computed against an
older version of `data/processed/subsets/*.npz`. For M5's correctness validation, a
fresh, consistent CPU baseline was regenerated into
`results/m2_cpu_baseline/benchmarks_current/` against the CURRENT subset files. The
original M2 artifacts (different hardware, historical record) are untouched.

## 7. Artifacts

`results/m5_cuda_optimized/`: `optimization_summary.json`, `optimization_benchmarks.csv`,
`kernel_comparison.csv`, `correctness_report.json`, `optimization_metadata.json`,
`optimization_report.md`, `charts/` (5 PNGs).
