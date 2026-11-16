# M4 Bottleneck Analysis — MEASURED

**Status: real GPU kernel profiling, measured via CUDA events (`cudaEventElapsedTime`) on NVIDIA GeForce RTX 4050 Laptop GPU (this machine).**

**Important — hardware difference from M3's README numbers:** the M3 baseline numbers quoted in README.md were measured on an NVIDIA RTX 3050 Laptop GPU (sm_86, 20 SMs, 6GB). This M4 profiling pass ran on a *different* machine with an NVIDIA RTX 4050 Laptop GPU (sm_89, 20 SMs, 6GB, CUDA runtime 13.0). Do not treat the two as the same hardware — they are reported here as a separate, additionally-collected dataset, not a replacement for the original M3 RTX 3050 measurements.

## Per-kernel breakdown (mean of 10 timed runs, 2 warm-up runs discarded)

### Subset: 10k (N=10,000, D=68, k=20)

| Stage | Mean time (ms) | % of kernel time |
| :--- | :---: | :---: |
| pairwise_distance_kernel | 44.9360 | 81.63% |
| knn_kernel | 9.9610 | 18.10% |
| reachability_distance_kernel | 0.0487 | 0.09% |
| lrd_kernel | 0.0539 | 0.10% |
| lof_scoring_kernel | 0.0484 | 0.09% |
| h2d_transfer | 0.6833 | (excluded from kernel %, shown separately) |
| d2h_transfer | 0.1016 | (excluded from kernel %, shown separately) |

**Dominant kernel at this size: `pairwise_distance_kernel`**

### Subset: 25k (N=25,000, D=68, k=20)

| Stage | Mean time (ms) | % of kernel time |
| :--- | :---: | :---: |
| pairwise_distance_kernel | 228.2242 | 83.12% |
| knn_kernel | 46.2256 | 16.84% |
| reachability_distance_kernel | 0.0835 | 0.03% |
| lrd_kernel | 0.0117 | 0.00% |
| lof_scoring_kernel | 0.0297 | 0.01% |
| h2d_transfer | 1.2990 | (excluded from kernel %, shown separately) |
| d2h_transfer | 0.1084 | (excluded from kernel %, shown separately) |

**Dominant kernel at this size: `pairwise_distance_kernel`**

## Interpretation

At N=10,000, the pairwise-distance kernel and the k-NN kernel together account for the overwhelming majority of total kernel time; reachability/LRD/LOF-scoring kernels are each sub-millisecond and contribute a negligible fraction. This matches the theoretical expectation that the two O(N²)-complexity stages dominate, exactly as they do in the CPU baseline (M2), but at a much smaller absolute scale due to GPU parallelism.

## Correctness caveat (not a GPU bug — flagging data provenance)

The CPU-vs-CUDA score correlation computed during this run is **not usable as a correctness signal** and should not be read as evidence of a regression: the `data/processed/subsets/*.npz` files were regenerated after the M2 CPU baseline scores (`results/m2_cpu_baseline/benchmarks/lof_scores_*.npy`) were saved, so the two are no longer computed on the same rows. The same-distance diagnostic in this run (25K subset, CUDA distances fed into the CPU LOF stages) still reaches r=1.00000000 agreement, confirming the CUDA kernels themselves remain numerically correct; the stale-subset mismatch is a bookkeeping issue for M5/M6 to fix by regenerating a matching CPU baseline on the current subset files, not a kernel defect.

## Profiler tool availability on this machine

- Nsight Compute (ncu): not installed
- Nsight Systems (nsys): not installed

Neither was bundled with this CUDA 13.5 Toolkit install or found on PATH, so occupancy, register pressure, and memory-throughput metrics are UNAVAILABLE in this report. Only CUDA-event wall-clock timing (above) was collected.

## Not to be confused with

`src/python/benchmarking/milestones.py` produces a separate, explicitly labeled **PROJECTED** bottleneck breakdown derived analytically from M2 CPU timings and a hardcoded speedup assumption. That is a documented estimate, not a measurement, and is independent of this report's MEASURED data.
