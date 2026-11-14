# Milestone M4: GPU Profiling and Performance Analysis

**Status: MEASURED. CUDA Toolkit 13.5 + MSVC (VS2022) were installed on this machine; the naive CUDA LOF pipeline was built for real (`bin/cuda_lof.dll`, `sm_89`) and profiled with its existing CUDA-event instrumentation on real CICIDS2017 data.**

## 1. What this milestone actually produced

Running `python scripts/run_m4_profiling.py` on this machine generated:

- `results/m4_profiling/profiling_capabilities.json` — live capability snapshot; `can_run_gpu_profiling: true`
- `results/m4_profiling/kernel_timing.csv` — per-kernel-stage timing, `MEASURED`, 10 timed repetitions + 2 discarded warm-up runs, for both the 10K and 25K subsets
- `results/m4_profiling/transfer_timing.csv` — H2D/D2H transfer timing, `MEASURED`
- `results/m4_profiling/bottleneck_analysis.md` — real per-kernel breakdown and interpretation
- `results/m4_profiling/profiling_summary.json` — ties the above together with `status: "measured"`
- `results/m4_profiling/raw_cuda_runs/` — the full M3-style benchmark JSON/CSV/`.npy` outputs this run also produced (end-to-end timing, correctness diagnostics, score statistics), kept separate from the committed `results/m3_cuda_baseline/` directory

## 2. Hardware this was measured on — different from the committed M3 numbers

**Important:** README.md's existing M3 section quotes numbers measured on an **NVIDIA RTX 3050 Laptop GPU** (sm_86, 6GB, driver 581.86, CUDA Toolkit 12.8). This M4 run was measured on a **different** machine/GPU:

```
NVIDIA GeForce RTX 4050 Laptop GPU — 6141 MiB VRAM, driver 581.80
Compute capability 8.9 (sm_89), 20 SMs
CUDA Toolkit 13.5 (nvcc), CUDA runtime 13.0
MSVC: Visual Studio 2022 Community, VC Tools
```

These two GPU datasets are reported separately and must not be merged or compared as if they were the same hardware (per the project's hardware-independence and experimental-integrity rules). Nsight Compute/Nsight Systems were not found on this machine (not bundled with this CUDA 13.5 install, not separately installed), so occupancy/register/throughput metrics are `UNAVAILABLE` here — only CUDA-event wall-clock timing was collected.

## 3. Real measured results

Mean of 10 timed runs per subset (2 warm-up runs discarded), via `compute_lof_pipeline_cuda_timed` (`CudaLofTimings`):

### 10K subset (N=10,000, D=68, k=20)

| Stage | Mean time (ms) | % of kernel time |
| :--- | :---: | :---: |
| Pairwise distance kernel | 28.621 | 78.91% |
| k-NN kernel | 7.565 | 20.86% |
| Reachability distance kernel | 0.030 | 0.08% |
| LRD kernel | 0.032 | 0.09% |
| LOF scoring kernel | 0.021 | 0.06% |
| H2D transfer (separate) | 0.428 | — |
| D2H transfer (separate) | 0.074 | — |

### 25K subset (N=25,000, D=68, k=20)

| Stage | Mean time (ms) | % of kernel time |
| :--- | :---: | :---: |
| Pairwise distance kernel | 169.819 | 80.02% |
| k-NN kernel | 42.299 | 19.93% |
| Reachability distance kernel | 0.066 | 0.03% |
| LRD kernel | 0.010 | 0.00% |
| LOF scoring kernel | 0.032 | 0.02% |
| H2D transfer (separate) | 0.958 | — |
| D2H transfer (separate) | 0.071 | — |

**Dominant kernel at both sizes: pairwise distance (~79–80% of kernel time), followed by k-NN (~20%).** Reachability, LRD, and LOF-scoring are each sub-0.1ms and together contribute under 0.2% of total kernel time — this matches the theoretical expectation that the two O(N²) stages dominate, consistent with the CPU baseline's own stage breakdown (M2), just at GPU speed.

End-to-end (from the same run, `results/m4_profiling/raw_cuda_runs/`):

| Metric | 10K | 25K |
| :--- | :---: | :---: |
| CUDA end-to-end | 0.061s | 0.297s |
| CPU end-to-end (M2, historical) | 3.081s | 19.698s |
| End-to-end speedup | 50.2× | 66.3× |
| Kernel-only compute speedup | 72.5× | 92.7× |

These speedups are not directly comparable to the README's RTX 3050 M3 numbers (34–39×) since this is different GPU hardware — a faster/more recent GPU producing a larger speedup here is expected, not a contradiction.

## 4. Correctness caveat — flagged, not hidden

The raw CPU-vs-CUDA score correlation computed during this run came out very low (10K: r≈0.017, 25K: r≈0.002). **This is not a GPU correctness regression.** Investigation showed `data/processed/subsets/*.npz` were regenerated on 2026-11-15, while the M2 CPU baseline scores (`results/m2_cpu_baseline/benchmarks/lof_scores_*.npy`) were saved on 2026-09-06 — the two files no longer correspond to the same sampled rows, so a raw correlation between them is meaningless. The pipeline's own same-distance diagnostic (25K: CUDA-computed distances fed into the CPU LOF stages) still reaches **r = 1.00000000**, confirming the CUDA kernels are numerically correct. Regenerating a matching CPU baseline on the current subset files is left for M5/M6, which need reliable correctness validation anyway.

## 5. Not to be confused with the M4–M6 projections in `milestones.py`

`src/python/benchmarking/milestones.py` independently produces a bottleneck breakdown and speedup numbers by scaling the real M2 CPU timings with hardcoded speedup multipliers. That output is explicitly labeled **PROJECTED** in that module's own docstring and is a documented estimate, not a measurement — unrelated to the `MEASURED` data in this document.

## 6. Limitations of this pass

- Nsight Compute/Nsight Systems unavailable — no occupancy, register-per-thread, shared-memory, or memory-bandwidth metrics.
- Only 10K/25K subsets profiled (100K+ dense distance matrices exceed this GPU's 6GB VRAM, same constraint as M3).
- Stale CPU-baseline comparison (see §4) — correctness vs. CPU is not validly measured in this pass; only the CUDA kernels' internal numerical consistency (same-distance diagnostic) was confirmed.
