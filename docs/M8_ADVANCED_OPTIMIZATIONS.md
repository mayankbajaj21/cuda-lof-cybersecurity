# M8 — Advanced Optimizations Beyond the Syllabus

**Status:** Done — measured on real hardware.
**Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB, compute capability 8.9), AMD 8-core / 16-thread CPU, 15.3 GB RAM, Windows 11, CUDA Toolkit 13.5, driver 581.80.
**Run date:** 2026-11-16 (single session; `results/m8_advanced_optimizations/benchmark_metadata.json`).
**Dataset:** synthetic CICIDS2017-shaped placeholder (see README §3). Timing and memory depend only on rows × features, so they are representative; anomaly content is not a real-world finding.

---

## 1. Goal

M5 applied optimization techniques from the course syllabus (Modules 2–4). M8 asks how much each technique from **outside** the syllabus adds on its own, and what all of them give together. The comparison is meant to be read as a ladder:

CPU → V0 naive GPU → V1 syllabus-optimized GPU → V2 / V3 / V4 (one outside-syllabus technique each, built on V1) → V5 (all three combined)

## 2. Variants

| ID | Name | Distance kernel | k-NN kernel | Exact? |
| :--- | :--- | :--- | :--- | :--- |
| V0 | Naive | One thread per distance, direct global reads (M3 code) | Naive per-thread insertion sort (M3) | Exact |
| V1 | Syllabus-optimized | M5: coalesced loads, 16×16 shared-memory tiles, occupancy-tuned, `#pragma unroll`, divergence removed | M5: register-local top-k | Exact |
| V2 | Register blocking | 16×16 threads compute a 64×64 output tile; each thread keeps a 4×4 block of partial sums in registers | M5 | Exact |
| V3 | Async copy + double buffering | V1 tiling, but the next tile is fetched with `cp.async` into a second shared-memory buffer while the current tile is computed | M5 | Exact |
| V4 | FP16 storage | V1 tiling over FP16 features; distance matrix stored as FP16; all arithmetic in FP32 | Reads the FP16 matrix, outputs FP32 distances | **Approximate** |
| V5 | Combined | Register blocking + `cp.async` double buffering + FP16 storage | Same as V4 | **Approximate** |

Reachability, LRD and LOF-scoring kernels are the unchanged naive kernels for every variant (M4 measured them at under 0.2% of runtime).

All six are built into one library, `bin/cuda_lof_variants.dll`, from `src/cuda/kernels_variants/variant_kernels.cu` and `src/cuda/lof/cuda_lof_pipeline_variants.cu`, with a `variant` argument selecting which one runs. V0 and V1 link the original M3/M5 kernel source unchanged; the M3 and M5 libraries themselves were not modified. The orchestrator frees buffers in the same order as the original pipelines, so memory behaviour matches. `cp.async` is compiled only for compute capability 8.0+; older targets fall back to ordinary loads.

### Why each technique was expected to help

- **Register blocking (V2).** In V1 most instructions are shared-memory loads: every distance needs two loads per feature. Computing a 4×4 block per thread reuses each loaded value four times, cutting loads per distance by about 4×. It costs registers (54 per thread vs 40), which lowers theoretical occupancy to 66.7%, but each thread has 16 independent sums to work on, so the GPU still has enough work to hide latency.
- **Async copies + double buffering (V3).** Overlaps loading the next tile with computing the current one, hiding global-memory latency. It only helps if the kernel is actually waiting on memory.
- **FP16 storage (V4).** Halves the bytes moved and the VRAM used by the N×N distance matrix, which is what limits dataset size. Arithmetic stays in FP32 to limit accuracy loss.

## 3. Methodology

- Script: `scripts/run_m8_advanced_optimizations.py` (one session; no other GPU work running).
- k = 20, contamination = 0.10, ε = 1e-10.
- GPU: 3 discarded warm-ups, then 10 timed runs, **taken in rotation across variants** so slow drift affects all of them equally. Per-stage CUDA-event timings plus end-to-end time. Medians are reported because 25k timings vary a lot from run to run.
- CPU: `src/python/cpu/cpu_lof.py`, 1 warm-up + 5 timed runs.
- Sizes: 10k and 25k subsets; plus a 45k slice (first 45,000 rows of the 100k subset, nothing written to `data/`). A live free-VRAM check runs before every 45k attempt.
- Peak GPU memory: tracked inside the pipeline (running total of bytes allocated by the pipeline, maximum recorded) and cross-checked against `cudaMemGetInfo`; the two agree within about 0.5%. This replaces M6's after-the-fact `nvidia-smi` poll, which reported 0 MB.
- Power: on AC, Windows "Balanced" plan (recorded in metadata).

## 4. Results

Medians from `results/m8_advanced_optimizations/variant_benchmarks.csv` and `kernel_breakdown.csv`. The README §16 table is generated from the same CSV by `scripts/update_readme_tables.py`.

### End-to-end

| Variant | 10k | × vs V1 | 25k | × vs V1 | 45k |
| :--- | ---: | ---: | ---: | ---: | ---: |
| CPU | 2,989.6 ms | – | 21,504.3 ms | – | not run (≈70 s, RAM) |
| V0 Naive | 48.55 ms | 0.92× | 293.70 ms | 0.91× | doesn't fit |
| V1 Syllabus-optimized | 44.47 ms | 1.00× | 268.18 ms | 1.00× | doesn't fit |
| V2 Register blocking | 25.46 ms | 1.75× | 149.66 ms | 1.79× | doesn't fit |
| V3 Async + double buffering | 49.37 ms | **0.90×** | 299.52 ms | **0.90×** | doesn't fit |
| V4 FP16 storage | 25.68 ms | 1.73× | 148.87 ms | 1.80× | 466.16 ms |
| V5 Combined | 16.27 ms | 2.73× | 88.15 ms | 3.04× | 270.45 ms |

Speedup over CPU: V1 67× (10k) / 80× (25k); V5 184× / 244×.

### Per kernel (median ms)

| Variant | Distance 10k | k-NN 10k | Distance 25k | k-NN 25k |
| :--- | ---: | ---: | ---: | ---: |
| V0 | 28.17 | 7.61 | 175.96 | 43.43 |
| V1 | 24.65 | 7.20 | 154.17 | 42.69 |
| V2 | 5.82 | 7.22 | 35.83 | 42.71 |
| V3 | 29.63 | 7.23 | 185.56 | 42.65 |
| V4 | 13.90 | 4.37 | 88.93 | 22.32 |
| V5 | 4.23 | 4.36 | 26.40 | 22.36 |

Observations:
- **V2's distance kernel is 4.2–4.3× faster than V1's.** End-to-end gains are smaller (about 1.8×) because the rest of each run — allocating the large distance matrix, the k-NN kernel and transfers — is unchanged.
- **V3 is about 10% slower than V1.** Its copies are 4 bytes each, so there is little to overlap; the second shared-memory buffer lowers theoretical occupancy to 83.3%; and the feature matrix (2.7–6.8 MB) already fits in the 24 MB L2 cache, so V1 was hardly waiting on memory. This matches M5's finding that this workload is not memory-latency-bound.
- **V4 speeds up both kernels**, including k-NN (about 1.7–1.9×), because the k-NN kernel reads a matrix half the size.
- **V5 combines V2's compute savings with V4's memory savings**, giving the fastest distance and k-NN kernels.

### Memory

| Variant | Peak GPU memory 10k | 25k | 45k | Largest N that fits (this GPU) |
| :--- | ---: | ---: | ---: | ---: |
| V0–V3 (FP32 matrix) | 384 MiB | 2,391 MiB | needs ~7.56 GiB | 37,974 |
| V4, V5 (FP16 matrix) | 192 MiB | 1,196 MiB | 3,869 MiB | 53,712 |

"Largest N" comes from `memory_analysis.csv`, computed from live free VRAM at benchmark time. At 45k, 5.67 GiB was free, so V0–V3 were refused before running (`feasibility_checks.csv`).

## 5. Correctness

From `correctness_summary.csv` / `correctness_report.json`:

- **V1, V2 and V3 are bit-identical to V0** at 10k and 25k: zero score difference, identical neighbour indices, identical anomaly flags.
- **V4 and V5 are approximate** (and identical to each other):

| Compared with exact V0 | 10k | 25k |
| :--- | ---: | ---: |
| Pearson r of LOF scores | 0.99988581 | 0.99988543 |
| k-NN recall@20 | 0.9933 | 0.9936 |
| Anomaly-flag agreement | 0.9990 | 0.9990 |
| Max absolute score error | 0.0068 | 0.0102 |

  No FP16 overflow occurred. At 45k no FP32 GPU reference fits in VRAM, so only V4 vs V5 is compared there (identical).
- Every GPU variant differs slightly from the CPU reference (max error about 0.0009 at 10k for the exact variants) because the CPU computes distances with the matrix-expansion formula; this is the known float32 cancellation issue documented since M3.

Tests: `tests/cuda/test_cuda_lof_variants.py` (36 tests: exact variants match V0, FP16 variants within tolerance, odd shapes D = 67 / D = 5 / N = 1000, self-exclusion, injected outlier ranked highest).

## 6. Kernel resources and Nsight status

Nsight Compute 2026.4 is installed but every profiling attempt failed with `ERR_NVGPUCTRPERM`: Windows only lets administrators read GPU performance counters by default. No admin or driver settings were changed. To enable it: NVIDIA Control Panel → Desktop → Enable Developer Settings → Developer → Manage GPU Performance Counters → "Allow access to the GPU performance counters to all users", then re-run `scripts/run_m8_advanced_optimizations.py`.

Fallback figures (`ncu_metrics.csv`): registers, shared memory and local memory come from `cudaFuncGetAttributes` (MEASURED, from the compiled kernel); occupancy is the theoretical value from `cudaOccupancyMaxActiveBlocksPerMultiprocessor` (ESTIMATED — not the achieved occupancy Nsight would report).

| Distance kernel | Registers / thread | Static shared memory / block | Theoretical occupancy |
| :--- | ---: | ---: | ---: |
| V0 | 37 | 0 B | 100% |
| V1 | 40 | 2,048 B | 100% |
| V2 | 54 | 8,704 B | 66.7% |
| V3 | 42 | 4,096 B | 83.3% |
| V4 | 34 | 1,024 B | 100% |
| V5 | 78 | 12,288 B | 50% |

The k-NN kernels spill to local memory: 2,048 B per thread for the naive kernel, 256 B for the M5 kernel used by V1–V5. This predates M8 and is a candidate for future work.

## 7. Limitations

- **Single run on one laptop GPU.** Timings depend on power and clock state; GPU run-to-run variation at 25k is high (standard deviation tens to hundreds of ms), which is why medians are used.
- **Low free RAM during CPU runs** (about 2.7–2.9 GB at the start), so CPU times may be inflated by paging. GPU timings are unaffected.
- **No achieved-occupancy, bandwidth or stall metrics** until GPU performance counters are enabled for Nsight.
- **The dense N×N matrix is still the limit.** FP16 moves the ceiling from about 38k to about 54k rows on this GPU but does not remove it; reaching 100k+ still needs a chunked distance + k-NN design.
- **Synthetic dataset** — see README §3.

## 8. Reproduce

```powershell
# Build (from a shell with vcvars64 + CUDA on PATH; see docs/WINDOWS_SETUP_GUIDE.md)
set ARCH=sm_89
scripts\build_cuda_lof_variants.bat

# Measure (PowerShell, CUDA on PATH, laptop plugged in)
python scripts\run_m8_advanced_optimizations.py
python scripts\update_readme_tables.py
pytest -v tests\cuda\test_cuda_lof_variants.py
```

Outputs: `results/m8_advanced_optimizations/` — `variant_benchmarks.csv`, `kernel_breakdown.csv`, `correctness_summary.csv`, `correctness_report.json`, `memory_analysis.csv`, `feasibility_checks.csv`, `ncu_metrics.csv`, `benchmark_metadata.json`, `summary.json`, `report.md`, `charts/m8_01…m8_05`. The dashboard's **Optimization Journey** page reads all of them.
