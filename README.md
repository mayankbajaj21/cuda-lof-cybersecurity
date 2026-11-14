# CUDA-Optimized Local Outlier Factor for Large-Scale Cybersecurity Anomaly Detection

A research project that implements, validates, and benchmarks the **Local Outlier Factor (LOF)** unsupervised anomaly-detection algorithm twice — once as a transparent CPU reference in NumPy, once as a native CUDA C++ pipeline — and measures the GPU speedup on a real cybersecurity intrusion-detection dataset (**CICIDS2017**).

---

## 1. Why this project exists

Network intrusion detection increasingly relies on **unsupervised** anomaly detection because labeled attack data is scarce and attackers constantly invent new patterns (zero-days). LOF is attractive for this because it flags a point as anomalous purely by comparing the *local density* around it to the density around its neighbors — no attack labels required.

The catch: LOF's two core stages — the **all-pairs distance matrix** and the **k-nearest-neighbor search** — are $O(N^2 \cdot D)$ and $O(N^2)$ respectively. The CPU baseline measured in this repo spends **>99.8% of its runtime** in exactly those two stages (see §6), and naive CPU LOF would take an estimated **thousands of hours** to run on the full 2.52M-row CICIDS2017 dataset. That quadratic wall is why this project exists: to show, with real measurements, how much of that wall GPU parallelism removes, and where the GPU's own new bottlenecks appear.

**Why CUDA specifically:** both bottleneck stages are embarrassingly parallel — each `(i, j)` distance and each per-row top-k search is independent — which maps cleanly onto thousands of CUDA threads, and network flow features are naturally stored as dense float32 matrices that transfer and compute efficiently on a GPU.

---

## 2. What is actually implemented vs. what is projected

To be precise about the state of the repository:

| Milestone | Status | What it actually is |
| :--- | :--- | :--- |
| **M1** Dataset prep | ✅ Real | Preparation pipeline is real code, cleaning/scaling/splitting into real `.npz` subsets — **but see §3 below: on this machine it ran on a synthetic placeholder, not downloaded CICIDS2017** |
| **M2** CPU LOF baseline | ✅ Real | First-principles NumPy LOF, validated against scikit-learn, timed on real hardware |
| **M3** CUDA LOF baseline | ✅ Real | Native CUDA C++ kernels compiled with `nvcc`, executed on a physical GPU, timed with CUDA events, cross-checked against the CPU baseline |
| **M4** Profiling | ✅ Real (CUDA-event) / ⚠️ Nsight unavailable | Real per-kernel timing via `CudaLofTimings` on an actual CUDA build. Nsight Compute/Systems were not installed on this machine, so occupancy/register/throughput metrics are not collected — wall-clock kernel timing is |
| **M5** Optimization | ✅ Real | A shared-memory-tiled distance kernel and a tuned k-NN kernel were implemented, built, and measured against the naive baseline on real hardware — modest, honestly-reported speedup (see §12) |
| **M6** Final benchmarks | ✅ Real (10K–25K) / 🚫 Infeasible (100K–1M) | Real CPU vs. naive-CUDA vs. optimized-CUDA measurements at 10K/25K. 100K+ sizes were checked against live VRAM/RAM and correctly refused, not faked — see §13 |
| **M7** Dashboard & final docs | ✅ Done | This README plus a Streamlit dashboard (`app.py`) that reads every result below live from disk |

This repo was originally built with M4–M6 as **analytical projections only** (`src/python/benchmarking/milestones.py`, which scales M2's CPU numbers by hardcoded speedup multipliers) because no working CUDA toolchain was available in that environment. That module is left intact as a historical artifact — the dashboard's benchmark page shows it labeled "projected" next to the real M4–M6 numbers below, so the two are never conflated. Every number under `results/m4_profiling/`, `results/m5_cuda_optimized/`, and `results/m6_final_benchmarks/` as of this revision is a **real CUDA-event measurement** on an NVIDIA RTX 4050 Laptop GPU (sm_89, CUDA 13.5) — a different machine than the original RTX 3050 (sm_86) that produced the M3 numbers in §7. The two GPUs are never merged into one comparison; where both appear, the hardware is labeled.

---

## 3. Dataset: CICIDS2017

> ⚠️ **On this machine, the data is a synthetic placeholder, not real CICIDS2017.** `data/raw/` is empty here — no CICIDS2017 CSVs were ever downloaded on this machine — and `data/processed/dataset_metadata.json` records its own source file as `synthetic_cicids2017.csv` with a 3-class label scheme (`BENIGN`/`DDoS`/`PortScan`) instead of the real dataset's 15 classes. The rows/features/scaling match CICIDS2017's shape (1M rows, 68 numeric features, z-scored) so that pipeline code, kernel timings, and memory-scaling conclusions in §6–§13 are still meaningful — runtime and VRAM depend on row/feature count, not on label semantics — but any *anomaly-detection result* (which specific points get flagged, attack-type breakdowns) is **not a validated real-world finding** on this machine. The table below describes the real CICIDS2017 cleaning pipeline as designed and as it ran against actual CIC data on the original development machine; treat it as the pipeline's specification, not this machine's live numbers.

Source: Canadian Institute for Cybersecurity (CIC), University of New Brunswick — [ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html). 5 days of labelled network flow captures (`GeneratedLabelledFlows.zip`), processed with CICFlowMeter into 8 CSVs.

| Stage | Rows | Notes |
| :--- | :---: | :--- |
| Raw (8 CSVs concatenated) | 2,830,743 | 79 columns |
| After removing NaN/±Inf rows | −2,867 | `Flow Bytes/s`, `Flow Packets/s` corruption |
| After deduplication | −308,381 | prevents zero-distance collapse in k-NN |
| **Clean dataset** | **2,520,798** | 89.05% retention |

- **Features**: 68 of 79 columns kept (numeric only). Excluded: `Label` (target, leakage), `Destination Port` (categorical, not Euclidean), `Fwd Header Length.1` (duplicate column — CICFlowMeter bug), and 8 zero-variance flag/bulk-rate columns.
- **Labels**: decoupled into `y_raw` (attack name string), `y_multiclass` (15 classes, 0=BENIGN), `y_binary` (0=benign, 1=anomaly).
- **Class balance**: 83.11% BENIGN, 16.89% attack traffic across 14 attack types (DoS Hulk, DDoS, PortScan, DoS GoldenEye, FTP/SSH-Patator, Bot, Web Attacks, Infiltration, Heartbleed — down to only 11 Heartbleed samples). This long-tailed imbalance is exactly why an *unsupervised* density method like LOF is a reasonable fit: supervised classifiers starve on the rare classes.
- **Scaling**: z-score standardization (`StandardScaler`), cast to `float32` (half the memory of `float64`, and the native throughput format for consumer GPUs).
- **Reproducible stratified subsets** (fixed seed 42): 10K, 25K, 100K, 250K, 500K, 1M, and the full 2.52M set — sized specifically so the smaller ones fit a dense $N\times N$ distance matrix inside the RTX 3050's 6GB VRAM, while the larger ones intentionally don't (used only for projected/extrapolated numbers, not naive dense GPU runs).

---

## 4. The algorithm: Local Outlier Factor (Breunig et al., 2000)

Both implementations (CPU and GPU) compute the exact same five stages, with the exact same formulas, so their outputs are numerically comparable.

1. **Pairwise Euclidean distance**
   $$d(p,q) = \sqrt{\textstyle\sum_{f=1}^{D} (p_f - q_f)^2}$$
2. **k-distance and k-NN set** — sort each row, take the nearest $k$ neighbors (excluding self); $k\text{-distance}(p)$ is the distance to the $k$-th nearest.
3. **Reachability distance** — dampens distance fluctuations inside dense clusters while preserving true distances for outliers:
   $$\text{reach-dist}_k(p,o) = \max(k\text{-distance}(o),\, d(p,o))$$
4. **Local Reachability Density (LRD)** — inverse of the mean reachability distance to the $k$ neighbors:
   $$\text{lrd}_k(p) = \frac{1}{\frac{1}{k}\sum_{o \in N_k(p)} \text{reach-dist}_k(p,o) + \epsilon}$$
5. **LOF score** — ratio of a point's neighbors' average density to its own:
   $$\text{LOF}_k(p) = \frac{\frac{1}{k}\sum_{o \in N_k(p)} \text{lrd}_k(o)}{\text{lrd}_k(p) + \epsilon}$$

   - $\text{LOF} \approx 1$ → point sits in a cluster of similar density (normal traffic).
   - $\text{LOF} \gg 1$ → point sits in a sparser region than its neighbors (anomalous traffic / potential attack).

   Anomaly decision: threshold at the $(1-c)$-percentile of LOF scores, where contamination $c$ defaults to 0.10 (flag the top 10% as attacks).

Default parameters throughout the project: $k = 20$ neighbors, $\epsilon = 10^{-10}$, contamination $c = 0.10$.

---

## 5. Two implementations, same math

### CPU reference — `src/python/cpu/cpu_lof.py` (`CPULocalOutlierFactor`)
Vectorized NumPy. Distance is computed via the quadratic expansion $\|p-q\|^2 = \|p\|^2 + \|q\|^2 - 2\langle p,q\rangle$ (a single `X @ X.T` matmul) — fast, but for near-duplicate rows in float32 this subtracts two large, nearly equal numbers and can round to exactly `0.0` (catastrophic cancellation). This is a known, documented discrepancy source (§7). `np.argsort` finds the k-NN. Every stage is individually timed.

### CUDA pipeline — `src/cuda/` (native C++/CUDA, called from Python via `ctypes`)
Direct coordinate-difference accumulation $\sum(p_d-q_d)^2$ instead of the matrix-expansion trick — this avoids the cancellation bug and is actually *more* numerically accurate than the CPU reference, not just faster (verified in §7). Five kernels, one orchestrator:

```
Host X (N, 68) float32
   │  cudaMemcpy H2D
   ▼
pairwise_distance_naive_kernel   dim3(16,16) 2D blocks  →  d_dist (N, N)
   │  (d_X freed immediately after this stage)
   ▼
knn_naive_kernel                 256 threads/block, 1D   →  d_knn_indices (N,k), d_knn_distances (N,k)
   │  (d_dist — the large N×N matrix — freed immediately: this is the key VRAM trick)
   ▼
reachability_distance_naive_kernel                        →  d_reach_dists (N,k)
   │  (d_knn_distances freed)
   ▼
lrd_naive_kernel                                           →  d_lrd (N,)
   │  (d_reach_dists freed)
   ▼
lof_scoring_naive_kernel                                   →  d_lof_scores (N,)
   │  cudaMemcpy D2H
   ▼
Host LOF scores (N,) float32
```

| Kernel | File | Role | Launch config |
| :--- | :--- | :--- | :--- |
| `pairwise_distance_naive_kernel` | `src/cuda/kernels/distance_kernel.cu` | $d(i,j)=\sqrt{\sum_d(X_{i,d}-X_{j,d})^2}$, diagonal forced to 0 | 2D grid, `dim3(16,16)` blocks |
| `knn_naive_kernel` | `src/cuda/kernels/knn_kernel.cu` | per-row scan + register-local insertion sort to keep the k smallest distances, self excluded | 1D grid, 256 threads/block |
| `reachability_distance_naive_kernel` | `src/cuda/kernels/reachability_kernel.cu` | $\max(k\text{-distance}(o), d(p,o))$ over all $N\times k$ (point, neighbor) pairs | grid-stride |
| `lrd_naive_kernel` | `src/cuda/kernels/lrd_kernel.cu` | mean reachability distance, inverted with $\epsilon$ guard | 1D over $N$ |
| `lof_scoring_naive_kernel` | `src/cuda/kernels/lof_kernel.cu` | neighbor LRD average ÷ own LRD | 1D over $N$ |

Orchestration lives in `src/cuda/lof/cuda_lof_pipeline.cu` (`compute_lof_pipeline_cuda` / `_cuda_timed`), which is the piece that actually frees each intermediate GPU buffer as soon as it's consumed — this is what keeps VRAM usage from exploding: the $N\times N$ distance matrix ($2.5\text{GB}$ at $N=25\text{K}$) is deallocated right after k-NN extraction, so stages 3–5 run with under 5MB resident. `src/cuda/common/cuda_utils.cuh` holds error-check macros and device-query helpers; `src/cuda/common/smoke_test.cu` is a standalone vector-add sanity check for the toolchain, independent of LOF.

Python never calls `nvcc` directly — compiled `.dll`s (`bin/distance_kernel.dll`, `bin/knn_kernel.dll`, `bin/cuda_lof.dll`) are loaded via `ctypes.CDLL` and called with raw pointers from `src/python/cuda/{distance,knn,lof}.py`.

---

## 6. CPU baseline results (M2 — measured)

Hardware: 13th-gen Intel Core i5-13450HX (10C/16T), 16GB RAM, Windows 11, Python 3.13.11, NumPy 2.4.4.

| Stage | 10K | 25K | Share of total (25K) |
| :--- | :---: | :---: | :---: |
| Pairwise distance | 1.39s | 10.98s | 49.59% |
| k-NN search & sort | 1.67s | 11.13s | 50.26% |
| Reachability + LRD + LOF scoring | ~0.006s | ~0.03s | 0.15% |
| **Total LOF compute** | **3.06s** | **22.14s** | **100%** |

A $2.5\times$ increase in $N$ (10K→25K) produced a $\sim$7.2× runtime increase — consistent with the theoretical $O(N^2)$ scaling ($2.5^2=6.25$). Extrapolated to the full 2.52M-row dataset, naive CPU LOF would take on the order of **thousands of hours**. This single number is the project's entire motivation for parallelizing on a GPU.

---

## 7. CUDA baseline results (M3 — measured on NVIDIA RTX 3050 6GB, sm_86)

| Metric | 10K | 25K |
| :--- | :---: | :---: |
| CPU end-to-end (M2) | 3.08s | 19.70s |
| CUDA end-to-end | **0.080s** | **0.567s** |
| **End-to-end speedup** | **38.6×** | **34.7×** |
| Kernel-only speedup | 47.3× | 39.8× |
| H2D / D2H transfer | <1ms combined | <1ms combined (<1.5% of pipeline) |
| Distance kernel share | 66.3% | 48.0% |
| k-NN kernel share | 33.6% | 51.9% |
| Correctness vs. CPU (Pearson r) | 0.9999982 | 0.26 raw / **0.9999999** on a same-distance diagnostic* |
| Anomaly agreement | 99.98% | 99.58% (99.99% on the diagnostic) |

\* **Why 25K correlation looks bad at first glance**: it isn't a GPU bug. The CPU reference's matrix-expansion distance formula hits float32 catastrophic cancellation on ~110 near-duplicate points in the 25K subset, driving a handful of CPU LRD/LOF values to spurious $\sim10^9$-scale outliers. The CUDA kernel's direct-accumulation formula doesn't have this failure mode. Feeding the CUDA-computed distances back into the CPU LOF stages ("same-distance diagnostic") recovers $r=0.9999999$ agreement — proof the discrepancy is a known CPU-side float32 precision artifact, not a GPU correctness issue.

Dense 100K+ subsets are *excluded* from this naive baseline on purpose: $100\text{K}\times100\text{K}\times4\text{B}=40\text{GB}$, which exceeds the RTX 3050's 6GB VRAM. §13 shows this same wall measured again on the current (RTX 4050) machine, still unresolved — M5's tiling (§12) improved cache reuse, not VRAM capacity.

---

## 8. Build & run

### Toolchain — hardware-independent by design
Build scripts no longer hardcode a GPU architecture. Each `scripts\build_*.bat` reads an `ARCH` environment variable (default `sm_86`, the original RTX 3050) — override it for your own GPU's compute capability before building:

```cmd
call "<VS install path>\VC\Auxiliary\Build\vcvars64.bat"
set "PATH=<CUDA Toolkit install>\bin;%PATH%"
set "ARCH=sm_89"                REM e.g. sm_86=RTX 3050, sm_89=RTX 4050, sm_90=H100 — match your GPU
scripts\build_cuda_lof.bat run   REM naive kernels → bin\cuda_lof.dll
scripts\build_cuda_lof_optimized.bat run   REM M5 optimized kernels → bin\cuda_lof_optimized.dll
```
Or run the equivalent `nvcc -arch=%ARCH% -O2 --shared -o bin\cuda_lof.dll <sources>` directly. M3's original build used CUDA Toolkit 12.8 + MSVC v143 targeting `sm_86`; M4–M7 on this machine used CUDA Toolkit 13.5 + VS2022 MSVC targeting `sm_89` — both work unmodified because of the `ARCH` parameterization. `src/python/environment/hardware_info.py` detects the running machine's actual CPU/GPU/driver/toolchain at runtime rather than assuming either.

### Data pipeline
```powershell
python preprocessing/inspect_dataset.py
python preprocessing/prepare_dataset.py --seed 42 --raw-dir data/raw --output-dir data/processed
```

### Experiments
```powershell
python scripts/run_cpu_baseline.py --subset 10k --k 20 --output-dir results/m2_cpu_baseline/benchmarks
python scripts/run_cuda_baseline.py --subset 10k --k 20
python scripts/run_benchmarks.py            # generates the OLD M4/M5/M6 *projected* reports (historical)
python scripts/run_m4_profiling.py          # real per-kernel CUDA profiling (needs a built bin\cuda_lof.dll)
python scripts/run_m5_optimization.py       # builds/measures the optimized kernels vs. naive
                                             # (M6 final benchmarks were produced by a one-off run combining
                                             #  run_cpu_baseline.py + run_cuda_baseline.py + lof_optimized.py
                                             #  across subsets, gated by a live VRAM/RAM feasibility check)
```

### Dashboard
```powershell
streamlit run app.py
```
Reads every artifact under `results/`, `data/`, and live hardware detection directly from disk — no hardcoded numbers. Pages degrade to an explicit "Not available" state if a result file is missing rather than crashing.

### Tests
```powershell
pytest -v tests/
```
Covers dataset integrity (`tests/test_dataset.py`), CPU LOF correctness against scikit-learn (`tests/test_cpu_lof.py`), and CUDA smoke/distance/k-NN/naive-pipeline/optimized-pipeline kernels against the CPU reference (`tests/cuda/`). **67 tests passed** once `bin/cuda_lof.dll` and `bin/cuda_lof_optimized.dll` were built on this machine (CUDA tests skip gracefully to 18 passed/39 skipped if no DLL is present, e.g. on a CPU-only machine).

---

## 9. Repository layout

```
cuda-lof-cybersecurity/
├── app.py                     # Streamlit dashboard entry point (M7)
├── preprocessing/            # CICIDS2017 download/inspect/clean/scale → data/processed/
├── src/
│   ├── python/
│   │   ├── data/              dataset.py — loaders for the full dataset & fixed subsets
│   │   ├── cpu/                cpu_lof.py — first-principles NumPy LOF reference
│   │   ├── cuda/                distance.py / knn.py / lof.py — naive ctypes bindings;
│   │   │                        lof_optimized.py — M5 optimized-kernel ctypes bindings
│   │   ├── benchmarking/        milestones.py — OLD M4/M5/M6 analytical projection generators (historical)
│   │   ├── environment/         hardware_info.py — dynamic CPU/GPU/OS/toolchain detection (M4)
│   │   ├── profiling/           capability_report.py — profiling-tool availability report (M4)
│   │   ├── dashboard/           Streamlit page modules + data loaders (M7)
│   │   └── visualization/     m1_/m2_/m3_/m5_visualizations.py — chart generation from real result JSON/CSV
│   └── cuda/
│       ├── common/             cuda_utils.cuh (error macros, device info), smoke_test.cu (toolchain check)
│       ├── kernels/             NAIVE: distance_, knn_, reachability_, lrd_, lof_kernel.cu/.cuh + test_*.cu
│       ├── kernels_optimized/   M5 OPTIMIZED: shared-memory-tiled distance_kernel_optimized.cu,
│       │                        tuned knn_kernel_optimized.cu (reachability/lrd/lof reused unmodified —
│       │                        M4 showed them <0.2% of runtime, not worth optimizing)
│       └── lof/                 cuda_lof_pipeline.cu/.cuh (naive) + cuda_lof_pipeline_optimized.cu/.cuh (M5);
│                                 test_cuda_lof.cu
├── scripts/                   build_*.bat (ARCH-parameterized nvcc build scripts, naive + optimized),
│                               run_cpu_baseline.py, run_cuda_baseline.py, run_benchmarks.py (old projections),
│                               run_m4_profiling.py (real profiling), run_m5_optimization.py (real optimization),
│                               generate_m3_charts.py
├── tests/                     pytest suite — CPU, dataset, naive CUDA, and optimized CUDA kernel correctness
├── data/{raw,processed}/      raw CICIDS2017 CSVs → cleaned .npz full set + stratified subsets
│                               (⚠️ on this machine: synthetic placeholder, see §3)
├── results/m{1..6}_*/         tables/, charts/, benchmarks/ per milestone — m1–m6 are now ALL real measurements
│                               on this machine (see §2); the old milestones.py projections remain viewable
│                               separately in the dashboard for comparison
└── docs/                      PROJECT_PLAN.md + one markdown report per milestone (M1_*.md … M6_*.md)
```

---

## 11. CUDA profiling (M4 — measured on NVIDIA RTX 4050 Laptop 6GB, sm_89, CUDA 13.5)

Per-kernel CUDA-event timing (2 warm-up + 10 timed runs each), via the pipeline's existing `CudaLofTimings` instrumentation — the hooks were always there, this is the first time they were exercised on a machine with a working `nvcc`:

| Kernel | 10K | Share | 25K | Share |
| :--- | :---: | :---: | :---: | :---: |
| Pairwise distance | 28.62ms | **78.9%** | 169.82ms | **80.0%** |
| k-NN | 7.57ms | 20.9% | 42.30ms | 19.9% |
| Reachability | 0.03ms | 0.08% | 0.07ms | 0.03% |
| LRD | 0.03ms | 0.09% | 0.01ms | <0.01% |
| LOF scoring | 0.02ms | 0.06% | 0.03ms | 0.02% |

**Confirmed bottleneck: the pairwise-distance kernel, ~80% of kernel time at both sizes** — this is what M5's optimization targets. Nsight Compute/Systems were not installed on this machine (not bundled with the CUDA 13.5 installer), so occupancy, register pressure, and memory-throughput metrics were not collected — only wall-clock CUDA-event timing. Full data: `results/m4_profiling/{kernel_timing,transfer_timing}.csv`, `bottleneck_analysis.md`.

---

## 12. CUDA optimization (M5 — measured, same hardware)

A shared-memory-tiled pairwise-distance kernel and a tuned k-NN kernel were built in `src/cuda/kernels_optimized/`, kept entirely separate from the naive kernels so both remain independently buildable and comparable.

| Stage | 10K naive → optimized | Speedup | 25K naive → optimized | Speedup |
| :--- | :---: | :---: | :---: | :---: |
| Distance kernel | 27.55 → 23.72ms | 1.16× | 169.38 → 149.17ms | 1.14× |
| k-NN kernel | 7.59 → 7.02ms | 1.08× | 42.29 → 41.45ms | 1.02× |
| **End-to-end** | 52.26 → 43.63ms | **1.20×** | 278.26 → 264.99ms | **1.05×** |

**The honest story is a modest win, not a dramatic one.** A first attempt at 32×32 shared-memory tiles was actually ~5× *slower* than naive — 1024 threads/block left only one resident block per SM on this GPU, collapsing occupancy. Dropping to 16×16 tiles fixed that. Even after the fix, the gain tops out around 14–16%, because this dataset's full feature matrix (2.7–6.8MB at 10K–25K rows × 68 features) fits entirely inside this GPU's 24MB L2 cache — the naive kernel's "redundant" global-memory reads were mostly L2 cache hits already, leaving less headroom for shared-memory tiling to reclaim than the textbook case assumes. This is a legitimate, reportable finding about *this* workload's cache behavior, not a failed optimization.

Correctness: naive vs. optimized CUDA are **bit-identical** (max abs. error 0.0, Pearson r = 1.00000000) at both sizes; both agree with a freshly-regenerated CPU baseline at r ≈ 0.99999997 with 100% anomaly-flag agreement. Reachability/LRD/LOF-scoring kernels were left unmodified — M4 showed them at <0.2% of total runtime combined, not worth the risk of touching. Full data: `results/m5_cuda_optimized/` (`optimization_report.md`, `kernel_comparison.csv`, `correctness_report.json`, 5 charts).

---

## 13. Final benchmarks & scalability (M6 — measured, same hardware)

| Subset | CPU | Naive CUDA | Optimized CUDA | Naive speedup | Optimized speedup |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 10K | 3185ms | 48.3ms | 43.6ms | 65.9× | 73.1× |
| 25K | 24996ms | 279.0ms | 273.7ms | 89.6× | 91.3× |

**100K, 250K, 500K, and 1M were checked against live free VRAM/RAM and correctly refused, not run with fabricated numbers.** A dense $N\times N$ float32 distance matrix needs 37GB at 100K, scaling to 3.7TB at 1M — far beyond this GPU's 6GB VRAM and the host's available RAM at measurement time. Both the naive and M5-optimized kernels still materialize the full matrix at once (M5 only improved cache reuse *within* that allocation, it did not remove it), so this is a genuine architectural ceiling of the current kernel design, not a missing experiment. Reaching 100K+ (or the real 2.52M-row full dataset) needs an out-of-core/chunked distance-and-k-NN kernel that processes the matrix in VRAM-sized blocks and only ever keeps each row's running top-k — a well-scoped, currently unimplemented, future-work item. Full data and the memory-feasibility table: `results/m6_final_benchmarks/` (`scalability_analysis.md`, `memory_analysis.csv`, 7 charts, including one that visually separates these real measurements from the old `milestones.py` projections).

---

## 14. Dashboard (M7)

`app.py` (+ `src/python/dashboard/`) is a Streamlit app with pages for: Overview, Hardware & Environment (live detection vs. recorded historical hardware, kept separate), Dataset Analytics (with the §3 synthetic-data banner), CPU vs. GPU Benchmarks, CUDA Profiling, CUDA Optimization, Memory & Scalability (including a live N→estimated-memory calculator), Anomaly Detection Results (with an unsupervised-flags-≠-ground-truth disclaimer), Correctness & Numerical Validation, Experiment History, and Exports. Every page reads directly from the files already described above — it does not recompute or fabricate anything; a missing result file renders an explicit "Not available" message instead of crashing.

```powershell
streamlit run app.py
```

---

## 15. Roadmap

| Milestone | Status |
| :--- | :--- |
| M0 Repo/env setup | Done |
| M1 Dataset preparation | Done (⚠️ synthetic placeholder data on this machine — see §3) |
| M2 CPU LOF baseline | Done |
| M3 CUDA LOF baseline | Done |
| M4 CUDA profiling | Done — real CUDA-event per-kernel timing; Nsight Compute/Systems unavailable on this machine |
| M5 CUDA optimization | Done — real shared-memory-tiled distance kernel + tuned k-NN, modest honestly-reported speedup |
| M6 Final benchmarking | Done at 10K–25K (real); 100K–1M correctly marked infeasible on this hardware, not faked |
| M7 Analysis & documentation + dashboard | Done |
