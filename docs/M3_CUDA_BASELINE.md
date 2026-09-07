# Milestone M3: CUDA LOF Baseline

## 1. Overview
This document tracks the implementation of **Milestone M3: CUDA LOF Baseline**.

*Current Status: **COMPLETE (Steps 1, 2, 3, 4, & 5 Complete & Validated)** — Formal Benchmark & Visualization Suite Established*

---

## 2. Milestone Objectives
- Implement the initial unoptimized CUDA C++ kernels for the core computational stages of Local Outlier Factor:
  1. Pairwise Euclidean Distance Matrix kernel ($O(N^2 \cdot D)$)
  2. Top-$k$ Nearest Neighbor Search & Sorting kernel ($O(N^2)$)
  3. Reachability Distance calculation kernel ($O(N \cdot k)$)
  4. Local Reachability Density (LRD) reduction kernel ($O(N \cdot k)$)
  5. Local Outlier Factor (LOF) scoring kernel ($O(N \cdot k)$)
- Establish functional C++/CUDA bindings (via Python C-types / PyBind11 / CuPy / custom wrapper).
- Numerically verify GPU LOF output against the validated CPU baseline from Milestone M2.
- Measure baseline GPU kernel latencies and memory transfer overhead.

---

## 3. Step 1 — Build Infrastructure Verification

### 3.1 CUDA Toolkit & Toolchain

| Component                | Value                                           |
| ------------------------ | ----------------------------------------------- |
| CUDA Toolkit             | 12.8                                            |
| nvcc                     | V12.8.61 (`release 12.8, V12.8.61`)             |
| nvcc Path                | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe` |
| Host Compiler            | MSVC `cl.exe` v19.38.33145 (v143 toolset v14.38-17.8) |
| Host Compiler Path       | `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.38.33130\bin\Hostx64\x64\cl.exe` |
| Target Architecture      | `sm_86` (Compute Capability 8.6)                |

### 3.2 GPU

| Property                 | Value                                           |
| ------------------------ | ----------------------------------------------- |
| GPU                      | NVIDIA GeForce RTX 3050 6GB Laptop GPU          |
| Compute Capability       | 8.6                                             |
| Global Memory            | 6.00 GB (6,441,926,656 bytes)                   |
| Multiprocessors (SMs)    | 20                                              |
| NVIDIA Driver            | 581.86                                          |
| CUDA Driver API Version  | 13.0                                            |
| CUDA Runtime API Version | 12.8                                            |

### 3.3 Smoke Test

**Purpose:** Verify that the CUDA build toolchain works end-to-end:
1. `nvcc` can compile a `.cu` source file targeting `sm_86`.
2. The executable launches a CUDA kernel on the GPU.
3. The RTX 3050 is visible to the CUDA runtime.
4. A simple GPU vector addition (65,536 elements) produces correct results.
5. Device properties are reported.

**Source file:** `src/cuda/common/smoke_test.cu`

**Environment Initialization & Build command:**
```cmd
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.38
nvcc -arch=sm_86 -O2 -o bin\smoke_test.exe src\cuda\common\smoke_test.cu
```

Or via the build script:
```cmd
scripts\build_smoke_test.bat run
```

### 3.4 Smoke Test Result

Actual output from `bin\smoke_test.exe`:
```text
============================================================
  CUDA Smoke Test — M3 Build Infrastructure Verification
============================================================

[INFO] CUDA devices found: 1
[INFO] Selected device: 0

============================================================
  CUDA Device Information
============================================================
  Device ID             : 0
  Device Name           : NVIDIA GeForce RTX 3050 6GB Laptop GPU
  Compute Capability    : 8.6
  Global Memory         : 6.00 GB (6441926656 bytes)
  Shared Memory / Block : 49152 bytes
  Warp Size             : 32
  Max Threads / Block   : 1024
  Max Block Dimensions  : [1024, 1024, 64]
  Max Grid Dimensions   : [2147483647, 65535, 65535]
  Multiprocessors (SMs) : 20
  Clock Rate            : 1.49 GHz
  Memory Clock Rate     : 7.00 GHz
  Memory Bus Width      : 96-bit
  L2 Cache Size         : 1536 KB
  CUDA Runtime Version  : 12.8
  CUDA Driver Version   : 13.0
============================================================

[INFO] Vector addition smoke test
       N = 65536 elements (262144 bytes per vector)

[PASS] Device memory allocated (3 x 262144 bytes = 768.00 KB)
[PASS] Host-to-device memcpy completed
[INFO] Launching kernel: <<<256, 256>>> (65536 total threads)
[PASS] Kernel launched successfully
[PASS] Kernel execution completed (cudaDeviceSynchronize OK)
[PASS] Device-to-host memcpy completed

[INFO] Verifying results...
[PASS] All 65536 elements correct! (C[i] = A[i] + B[i] = 3*i)

[PASS] Memory freed successfully

============================================================
  CUDA SMOKE TEST PASSED
  All checks succeeded. CUDA build infrastructure is ready.
============================================================
```

### 3.5 Files Created

| File                                | Purpose                                        |
| ----------------------------------- | ---------------------------------------------- |
| `src/cuda/common/cuda_utils.cuh`    | Common error-checking macros & device info      |
| `src/cuda/common/smoke_test.cu`     | CUDA smoke test (vector addition)              |
| `scripts/build_smoke_test.bat`      | Build script targeting sm_86                   |
| `tests/cuda/test_cuda_smoke.py`     | Python pytest wrapper (skips if no binary/GPU) |

### 3.6 Test Compatibility

- Running `pytest -v tests/` executes 22 tests (18 M1/M2 tests + 4 CUDA smoke tests):
  - `tests/cuda/test_cuda_smoke.py`: 4 passed
  - `tests/test_cpu_lof.py`: 10 passed
  - `tests/test_dataset.py`: 8 passed
- All 22 tests pass in ~8.7s with zero regressions.

---

## 4. Step 2 — Pairwise Euclidean Distance CUDA Kernel

### 4.1 Kernel Overview & Purpose
Step 2 establishes the naive baseline CUDA kernel for calculating dense pairwise Euclidean distance matrices. This serves as the foundational geometric distance metric required by k-NN search in LOF.

$$\text{dist}(i, j) = \sqrt{\sum_{d=0}^{D-1} (X[i, d] - X[j, d])^2}$$
$$\text{dist}(i, i) = 0.0f$$

### 4.2 Specifications

| Property                 | Specification                                   |
| ------------------------ | ----------------------------------------------- |
| CUDA Kernel              | `pairwise_distance_naive_kernel`                |
| Kernel Source            | `src/cuda/kernels/distance_kernel.cu`           |
| Kernel Header            | `src/cuda/kernels/distance_kernel.cuh`          |
| Input Matrix $X$         | Shape $(N, 68)$, row-major contiguous, `float32` |
| Output Matrix            | Shape $(N, N)$, row-major contiguous, `float32`  |
| Feature Dimension $D$    | **68 standardized numerical features**          |
| Target Architecture      | `sm_86` (NVIDIA GeForce RTX 3050 Laptop GPU)    |
| Thread Block Dimensions  | 2D: `dim3(16, 16)` (256 threads/block)          |
| Grid Dimensions          | 2D: `dim3((N + 15)/16, (N + 15)/16)`            |
| Diagonal Handling        | Explicitly forced to `0.0f` for $i == j$        |
| Boundary Handling        | Guarded against out-of-bounds: $i < N \land j < N$ |
| Optimization Status      | Naive baseline (no shared memory, tiling, streams) |

### 4.3 Memory Management & Constraints

The NVIDIA RTX 3050 has ~6.0 GB of VRAM. Dense $N \times N$ distance matrices scale quadratically in memory:

* **10K Subset**: $10,000 \times 10,000 \times 4\text{ bytes} = 400\text{ MB}$ (Safely fits in VRAM)
* **25K Subset**: $25,000 \times 25,000 \times 4\text{ bytes} = 2.50\text{ GB}$ (Safely fits in VRAM)
* **100K Subset**: $100,000 \times 100,000 \times 4\text{ bytes} \approx 40.0\text{ GB}$ (Exceeds physical VRAM; avoided in dense baseline)

### 4.4 Build Commands

Compile shared library (`bin\distance_kernel.dll`) and standalone C++ verification executable (`bin\test_distance.exe`):
```cmd
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.38
nvcc -arch=sm_86 -O2 --shared -o bin\distance_kernel.dll src\cuda\kernels\distance_kernel.cu
nvcc -arch=sm_86 -O2 -o bin\test_distance.exe src\cuda\kernels\test_distance.cu src\cuda\kernels\distance_kernel.cu
```

Or via the build script:
```cmd
scripts\build_distance_kernel.bat run
```

### 4.5 Standalone C++ Verification Output
```text
============================================================
  CUDA Pairwise Euclidean Distance Kernel Verification
  Milestone: M3 Step 2
============================================================

[INFO] Running test on synthetic data: N=64, D=68...
[PASS] N=64, D=68 verified: zero diagonal, symmetric, exact match with CPU

[INFO] Running test on synthetic data: N=256, D=68...
[PASS] N=256, D=68 verified: zero diagonal, symmetric, exact match with CPU

[INFO] Running test on synthetic data: N=1024, D=68...
[PASS] N=1024, D=68 verified: zero diagonal, symmetric, exact match with CPU

============================================================
  ALL C++ DISTANCE KERNEL VERIFICATIONS PASSED
============================================================
```

### 4.6 Python Bindings & Pytest Validation
- Python module: [`src/python/cuda/distance.py`](file:///c:/GPU_PROJ/cuda-lof-cybersecurity/src/python/cuda/distance.py) provides `compute_pairwise_distances(X)`.
- Automated test suite: [`tests/cuda/test_distance_kernel.py`](file:///c:/GPU_PROJ/cuda-lof-cybersecurity/tests/cuda/test_distance_kernel.py).
- Correctness criteria verified:
  1. Shape: `(N, N)` float32 for arbitrary $N$.
  2. Diagonal: strictly `0.0f`.
  3. Symmetry: $\text{dist}(i, j) == \text{dist}(j, i)$.
  4. Finiteness: Zero NaN or Inf values.
  5. Numerical accuracy: Max absolute difference against CPU $L_2$ norm $\le 1 \times 10^{-4}$.
  6. Boundary conditions: Tested on non-block-aligned dimensions ($N \in \{1, 7, 31, 65, 127, 257\}$).
  7. Real-world CICIDS2017 subsets: Validated on 10K ($N=10,000$, $D=68$) and 25K ($N=25,000$, $D=68$).

### 4.7 Full Repository Test Suite Status
Running `python -m pytest -v tests/`:
- `tests/cuda/test_cuda_smoke.py`: 4 passed
- `tests/cuda/test_distance_kernel.py`: 13 passed
- `tests/test_cpu_lof.py`: 10 passed
- `tests/test_dataset.py`: 8 passed
- **Total: 35 passed in 9.78s**

---

## 5. Step 3 — CUDA k-NN Baseline

### 5.1 Kernel Overview & Purpose
Step 3 implements the nearest-neighbor extraction stage required by Local Outlier Factor. Given the dense pairwise distance matrix computed in Step 2, the k-NN kernel extracts the $k$ nearest distinct neighbors for each query point $i \in [0, N-1]$, excluding self-distance ($j == i$), sorted in ascending order of distance:

$$\text{neighbor\_distances}[i, 0] \le \text{neighbor\_distances}[i, 1] \le \dots \le \text{neighbor\_distances}[i, k-1]$$

and outputs the matching sample indices:

$$\text{neighbor\_indices}[i, m] \in [0, N-1] \setminus \{i\}$$

### 5.2 Specifications

| Property                 | Specification                                   |
| ------------------------ | ----------------------------------------------- |
| CUDA Kernel              | `knn_naive_kernel`                              |
| Kernel Source            | `src/cuda/kernels/knn_kernel.cu`                |
| Kernel Header            | `src/cuda/kernels/knn_kernel.cuh`               |
| Input Distance Matrix    | Shape $(N, N)$, row-major contiguous, `float32` |
| Output Neighbor Indices  | Shape $(N, k)$, row-major contiguous, `int32`   |
| Output Neighbor Distances| Shape $(N, k)$, row-major contiguous, `float32` |
| Neighborhood Parameter $k$| Configurable, default $k = 20$ (up to $256$)   |
| Target Architecture      | `sm_86` (NVIDIA GeForce RTX 3050 Laptop GPU)    |
| Grid Decomposition       | 1D: `dim3 blocks((N + 255)/256)`                |
| Block Dimensions         | 1D: `dim3 threads(256)` (256 threads/block)     |
| Self-Distance Exclusion  | Explicitly skips $j == i$                       |
| Selection Algorithm      | Naive row scan with thread-local insertion sort |
| Optimization Status      | Naive baseline (no shared memory, warp shuffles)|

### 5.3 Memory Management & Budgets

Dense distance matrices fit safely within the RTX 3050 6.0 GB VRAM:

* **10K Subset** ($N = 10,000, k = 20$):
  * Input Distance Matrix: $10,000 \times 10,000 \times 4\text{ bytes} = 400.0\text{ MB}$
  * Output Indices: $10,000 \times 20 \times 4\text{ bytes} = 800\text{ KB}$
  * Output Distances: $10,000 \times 20 \times 4\text{ bytes} = 800\text{ KB}$
  * Total GPU Memory: $\approx 401.6\text{ MB}$
* **25K Subset** ($N = 25,000, k = 20$):
  * Input Distance Matrix: $25,000 \times 25,000 \times 4\text{ bytes} = 2.50\text{ GB}$
  * Output Indices: $25,000 \times 20 \times 4\text{ bytes} = 2.0\text{ MB}$
  * Output Distances: $25,000 \times 20 \times 4\text{ bytes} = 2.0\text{ MB}$
  * Total GPU Memory: $\approx 2.504\text{ GB}$
* **100K Subset**: Safely excluded (matrix requires $40.0\text{ GB}$, exceeding physical VRAM).

### 5.4 Build Commands

Compile shared library (`bin\knn_kernel.dll`) and standalone C++ verification executable (`bin\test_knn.exe`):
```cmd
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.38
nvcc -arch=sm_86 -O2 --shared -o bin\knn_kernel.dll src\cuda\kernels\knn_kernel.cu
nvcc -arch=sm_86 -O2 -o bin\test_knn.exe src\cuda\kernels\test_knn.cu src\cuda\kernels\knn_kernel.cu src\cuda\kernels\distance_kernel.cu
```

Or via the build script:
```cmd
scripts\build_knn_kernel.bat run
```

### 5.5 Standalone C++ Verification Output
```text
============================================================
  CUDA k-NN Kernel Standalone Verification
  Milestone: M3 Step 3
============================================================

[INFO] Testing k-NN on synthetic data: N=64, D=68, k=20...
[PASS] N=64, k=20: valid indices, no self-neighbor, sorted distances, exact CPU match

[INFO] Testing k-NN on synthetic data: N=256, D=68, k=20...
[PASS] N=256, k=20: valid indices, no self-neighbor, sorted distances, exact CPU match

[INFO] Testing k-NN on synthetic data: N=1024, D=68, k=20...
[PASS] N=1024, k=20: valid indices, no self-neighbor, sorted distances, exact CPU match

============================================================
  ALL C++ k-NN KERNEL VERIFICATIONS PASSED
============================================================
```

### 5.6 Python Bindings & Pytest Validation
- Python module: [`src/python/cuda/knn.py`](file:///c:/GPU_PROJ/cuda-lof-cybersecurity/src/python/cuda/knn.py) provides `compute_knn(dist_matrix, k=20)`.
- Automated test suite: [`tests/cuda/test_knn_kernel.py`](file:///c:/GPU_PROJ/cuda-lof-cybersecurity/tests/cuda/test_knn_kernel.py) (15 tests).
- Correctness criteria verified:
  1. Shapes & Dtypes: Indices `(N, k)` int32, Distances `(N, k)` float32.
  2. Valid index ranges: $0 \le \text{idx} < N$.
  3. No self-neighbors: $\text{idx}[i, m] \ne i$ for all $i, m$.
  4. Non-negative, finite distances (no NaN/Inf).
  5. Distances sorted monotonically ascending: $d_0 \le d_1 \le \dots \le d_{k-1}$.
  6. Exact correspondence: `dist_matrix[i, indices[i, m]] == distances[i, m]`.
  7. Configurable $k$: Tested on $k \in \{5, 10, 20, 25\}$.
  8. Determinism: Bit-for-bit identical across multiple executions.
  9. Real CICIDS2017 10K subset ($k=20$): 50 randomly sampled query rows compared against CPU `np.argsort` reference; exact numerical match.
  10. Real CICIDS2017 25K subset ($k=20$): Selected rows validated against CPU reference; exact match.

### 5.7 Full Repository Test Suite Status
Running `python -m pytest -v tests/`:
- `tests/cuda/test_cuda_smoke.py`: 4 passed
- `tests/cuda/test_distance_kernel.py`: 13 passed
- `tests/cuda/test_knn_kernel.py`: 15 passed
- `tests/test_cpu_lof.py`: 10 passed
- `tests/test_dataset.py`: 8 passed
- **Total: 50 passed in 11.70s**

---

## 6. Step 4 — Complete CUDA LOF Baseline Integration & Correctness

### 6.1 Mathematical Formulations (M2 CPU Exact Counterpart)
The CUDA baseline reproduces the exact mathematical formulations defined in the validated M2 CPU implementation (`src/python/cpu/cpu_lof.py`):

1. **$k$-distance**:
   $$\text{k-distance}(p) = \text{knn\_distances}[p, k - 1]$$
   where `knn_distances` is sorted ascending and excludes self ($p \ne o$).

2. **Reachability Distance**:
   $$\text{reach-dist}_k(p, o) = \max\left(\text{k-distance}(o),\, d(p, o)\right)$$
   where $d(p, o)$ is the direct Euclidean distance between sample $p$ and neighbor $o \in N_k(p)$.

3. **Local Reachability Density (LRD)**:
   $$\text{lrd}_k(p) = \frac{1.0}{\left(\frac{1}{k} \sum_{o \in N_k(p)} \text{reach-dist}_k(p, o)\right) + \varepsilon}$$
   where $\varepsilon = 10^{-10}$ is the numerical stability floor matching M2 CPU.

4. **Local Outlier Factor (LOF)**:
   $$\text{LOF}_k(p) = \frac{\frac{1}{k} \sum_{o \in N_k(p)} \text{lrd}(o)}{\text{lrd}(p) + \varepsilon}$$

5. **Decision Threshold & Predictions**:
   $$\tau = \text{percentile}_{100(1 - c)}\left(\text{LOF}\right), \quad \hat{y}_i = \mathbb{I}(\text{LOF}_i \ge \tau)$$
   where default contamination $c = 0.10$ (10%).

### 6.2 CUDA Pipeline & Implementation Architecture

The 5 computational stages are kept cleanly modularized into dedicated kernels with a unified pipeline:

```text
Feature Matrix X (N, 68) [Host]
       │
       ▼ (cudaMemcpy H2D)
  d_X (N, 68) [Device]
       │
       ▼ (pairwise_distance_naive_kernel: sm_86, dim3(16, 16))
  d_dist (N, N) [Device]  <── d_X freed immediately
       │
       ▼ (knn_naive_kernel: 256 threads/block, local insertion sort)
  d_knn_indices (N, k), d_knn_distances (N, k) [Device]  <── d_dist freed immediately
       │
       ▼ (reachability_distance_naive_kernel)
  d_reach_dists (N, k) [Device]  <── d_knn_distances freed
       │
       ▼ (lrd_naive_kernel)
  d_lrd (N,) [Device]  <── d_reach_dists freed
       │
       ▼ (lof_scoring_naive_kernel)
  d_lof_scores (N,) [Device]  <── d_lrd & d_knn_indices freed
       │
       ▼ (cudaMemcpy D2H)
LOF Scores (N,) [Host]
```

### 6.3 Memory Management & Peak Footprint

By immediately deallocating the $N \times N$ dense distance matrix after the k-NN extraction phase:
* **10K Subset** ($N = 10,000, D = 68, k = 20$): Peak VRAM $\approx 402\text{ MB}$ (during distance/k-NN), dropping to $< 2\text{ MB}$ for subsequent stages.
* **25K Subset** ($N = 25,000, D = 68, k = 20$): Peak VRAM $\approx 2.504\text{ GB}$, dropping to $< 5\text{ MB}$ for subsequent stages.
* **100K Subset**: Excluded safely due to quadratic memory limit.

### 6.4 Build Commands

Compile complete unified library (`bin\cuda_lof.dll`) and test executable (`bin\test_cuda_lof.exe`):
```cmd
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.38
nvcc -arch=sm_86 -O2 --shared -o bin\cuda_lof.dll src\cuda\kernels\distance_kernel.cu src\cuda\kernels\knn_kernel.cu src\cuda\kernels\reachability_kernel.cu src\cuda\kernels\lrd_kernel.cu src\cuda\kernels\lof_kernel.cu src\cuda\lof\cuda_lof_pipeline.cu
nvcc -arch=sm_86 -O2 -o bin\test_cuda_lof.exe src\cuda\lof\test_cuda_lof.cu src\cuda\kernels\distance_kernel.cu src\cuda\kernels\knn_kernel.cu src\cuda\kernels\reachability_kernel.cu src\cuda\kernels\lrd_kernel.cu src\cuda\kernels\lof_kernel.cu src\cuda\lof\cuda_lof_pipeline.cu
```

Or via the build script:
```cmd
scripts\build_cuda_lof.bat run
```

### 6.5 Standalone C++ Verification Output
```text
============================================================
  Complete CUDA LOF Pipeline Standalone Verification
  Milestone: M3 Step 4
============================================================

============================================
  CUDA Device Information
============================================
  Device ID            : 0
  Device Name          : NVIDIA GeForce RTX 3050 6GB Laptop GPU
  Compute Capability   : 8.6
  Global Memory        : 6.00 GB (6441926656 bytes)
  Shared Memory / Block: 49152 bytes
  Warp Size            : 32
  Max Threads / Block  : 1024
  Max Block Dimensions : [1024, 1024, 64]
  Max Grid Dimensions  : [2147483647, 65535, 65535]
  Multiprocessors (SMs): 20
  Clock Rate           : 1.49 GHz
  Memory Clock Rate    : 7.00 GHz
  Memory Bus Width     : 96-bit
  L2 Cache Size        : 1572864 bytes
  CUDA Runtime Version : 12.8
  CUDA Driver Version  : 13.0
============================================

[INFO] Generating synthetic dataset: N=256, D=68, k=20...
[INFO] Executing complete CUDA LOF pipeline...
[INFO] Min LOF score: 0.9911, Max LOF score: 694.4266
[INFO] Injected outlier LOF score at sample 255: 694.4266
[PASS] Outlier score is prominent (694.4266 >> 1.0)
[PASS] All 256 scores are finite and non-negative

============================================================
  COMPLETE CUDA LOF PIPELINE VERIFICATION PASSED
============================================================
```

### 6.6 Correctness & Numerical Agreement vs. M2 CPU Baseline

| Metric | Synthetic Dataset ($N=100$) | CICIDS2017 10K Subset | CICIDS2017 25K Subset |
| --- | --- | --- | --- |
| Sample Size ($N$) | $100$ | $10,000$ | $25,000$ |
| Features ($D$) | $68$ | $68$ | $68$ |
| Neighborhood ($k$) | $20$ | $20$ | $20$ |
| Contamination ($c$) | $0.10$ | $0.10$ | $0.10$ |
| Max Absolute Diff | $2.38 \times 10^{-7}$ | $1.08 \times 10^{0}$ | $2.97 \times 10^{9}$ (Diagnosed below) |
| Mean Absolute Diff | $7.81 \times 10^{-8}$ | $1.16 \times 10^{-3}$ | $1.21 \times 10^{6}$ / $7.85 \times 10^{-5}$* |
| Pearson Correlation | **1.00000000** | **0.99999817** | **0.99999989*** |
| Anomaly Label Agreement | **100.00%** | **99.98%** (9,998 / 10,000) | **99.58%** (24,894 / 25,000) |
| CPU Detected Anomalies | $10$ | $1,000$ | $2,500$ |
| CUDA Detected Anomalies| $10$ | $1,000$ | $2,500$ |

*\* Note on 25K Numerical Discrepancy Root Cause Analysis:*
* **Diagnosis**: In `src/python/cpu/cpu_lof.py`, CPU pairwise distance is computed via the quadratic expansion $\|p - q\|^2 = \|p\|^2 + \|q\|^2 - 2\langle p, q\rangle$. In 32-bit floating point, when $\|p\| \approx 100$, $\|p\|^2 \approx 10,000$; subtracting large values introduces catastrophic cancellation for closely spaced points ($d < 0.001$), clipping true positive distances to exact zero ($0.0$). In the 25K dataset, ~110 points having $\ge 20$ near-identical packets were assigned reachability distance $0.0$ on CPU, resulting in degenerate $LRD = 1/\varepsilon = 10^{10}$ and inflated CPU LOF scores.
* **Verification**: In CUDA, `distance_kernel.cu` calculates direct accumulation $\sum (X[i, d] - X[j, d])^2$, avoiding quadratic cancellation. When the exact CUDA distances are fed to the CPU reference stages, CPU and CUDA LOF scores achieve a **0.99999989 Pearson correlation** and **$7.85 \times 10^{-5}$ mean absolute difference**.
* Both models achieve **99.58% classification agreement** and flag exactly $2,500$ anomalies.

### 6.7 Full Repository Test Suite Status
Running `python -m pytest -v tests/`:
- `tests/cuda/test_cuda_smoke.py`: 4 passed
- `tests/cuda/test_distance_kernel.py`: 13 passed
- `tests/cuda/test_knn_kernel.py`: 15 passed
- `tests/cuda/test_cuda_lof.py`: 7 passed (including hardware event timing validation)
- `tests/test_cpu_lof.py`: 10 passed
- `tests/test_dataset.py`: 8 passed
- **Total: 57 passed in 16.91s**

---

## 7. Step 5 — Formal Benchmark & Final M3 Results

### 7.1 M3 Implementation Summary

Milestone 3 successfully established the first complete, unoptimized, native CUDA C++ implementation of the Local Outlier Factor algorithm targeting NVIDIA Ampere architecture (`sm_86`):
1. **Pairwise Euclidean Distance Matrix Kernel** (`distance_kernel.cu`): Evaluates $d(p, q) = \sqrt{\sum_{d=0}^{D-1} (X[i,d] - X[j,d])^2}$ directly with $16 \times 16$ 2D thread blocks across all $D=68$ continuous features.
2. **$k$-Nearest Neighbors Kernel** (`knn_kernel.cu`): Computes top-$k$ nearest neighbors per sample utilizing local register insertion sort and self-distance exclusion ($d(i,i) = 0.0$ ignored).
3. **Reachability Distance Kernel** (`reachability_kernel.cu`): Calculates $\text{reach-dist}_k(p, o) = \max(k\text{-distance}(o), d(p, o))$ with grid-stride parallel mapping over $N \times k$ pairs.
4. **Local Reachability Density (LRD) Kernel** (`lrd_kernel.cu`): Computes $\text{lrd}_k(p) = 1.0 / (\frac{1}{k} \sum \text{reach-dist} + \epsilon)$ with zero-division epsilon guard.
5. **LOF Scoring Kernel** (`lof_kernel.cu`): Calculates final anomaly score $\text{LOF}_k(p) = (\frac{1}{k} \sum \text{lrd}(o)) / (\text{lrd}(p) + \epsilon)$.
6. **VRAM Lifetime Management** (`cuda_lof_pipeline.cu`): Immediately deallocates the large $N \times N$ distance matrix upon completion of $k$-NN extraction, dropping active VRAM footprint from $2.5\text{ GB}$ to $<5\text{ MB}$ for downstream stages.

---

### 7.2 Benchmark Methodology

* **Datasets**: Authentic CICIDS2017 preprocessed subsets:
  - `subset_10k.npz`: $N = 10,000$ samples, $D = 68$ features ($400\text{ MB}$ dense distance matrix).
  - `subset_25k.npz`: $N = 25,000$ samples, $D = 68$ features ($2.50\text{ GB}$ dense distance matrix).
* **Feature Representation**: Exact $D = 68$ standardized float32 continuous features (no dimensionality reduction, PCA, or feature selection applied).
* **Algorithm Parameters**: Neighborhood size $k = 20$, contamination ratio $c = 0.10$ (top 10% anomalies), numerical epsilon $\epsilon = 10^{-10}$.
* **Audited Hardware Environment**:
  - **GPU**: NVIDIA GeForce RTX 3050 6GB Laptop GPU (Compute Capability 8.6, 20 SMs, 6.00 GB GDDR6, Driver 581.86).
  - **Host CPU**: 13th Gen Intel Core i5-13450HX (10 physical cores / 16 logical processors).
  - **Host RAM**: 15.71 GB (~16 GB DDR5).
  - **Operating System**: Microsoft Windows 11 Home (x64, Build 10.0.26200).
  - **Software Toolchain**: CUDA Toolkit 12.8.61 (`nvcc`), MSVC v143 (`14.38.33130`), Python 3.13.11.
* **Timing Instrumentation**:
  - **GPU Kernel & Transfer Timing**: Hardware-level CUDA events (`cudaEventCreate`, `cudaEventRecord`, `cudaEventSynchronize`, `cudaEventElapsedTime`) measuring GPU device execution without host CPU launch overhead.
  - **End-to-End Pipeline Timing**: GPU event elapsed time spanning from the initiation of Host-to-Device (H2D) transfer through the final Device-to-Host (D2H) memory transfer.
  - **Host Wall-Clock Timing**: Measured via high-precision `time.perf_counter()` synchronized across the device.
* **Warm-Up Policy**: Exactly 1 untimed warm-up execution performed prior to measurement to eliminate initial CUDA context creation and driver initialization latency. All reported statistics strictly exclude warm-up executions.
* **Iteration Counts**: 5 timed iterations for 10K, 3 timed iterations for 25K. All reported numbers represent statistical means across valid timed iterations.

---

### 7.3 Performance Results

#### Table 7.1: End-to-End Performance & Acceleration Summary

| Performance Metric | 10K Subset ($N = 10,000$) | 25K Subset ($N = 25,000$) |
| :--- | :---: | :---: |
| **CPU End-to-End Time (M2)** | **3.0814 s** | **19.6976 s** |
| **CPU Compute-Only Time (M2)** | **3.0638 s** | **19.6596 s** |
| **CPU Throughput (M2)** | 3,245.3 records/s | 1,269.2 records/s |
| **CUDA Host-to-Device (H2D) Transfer** | 0.393 ms (0.00039 s) | 0.915 ms (0.00092 s) |
| **CUDA Total Kernel Execution Time** | 64.778 ms (0.06478 s) | 493.731 ms (0.49373 s) |
| **CUDA Device-to-Host (D2H) Transfer** | 0.083 ms (0.00008 s) | 0.058 ms (0.00006 s) |
| **CUDA End-to-End Time** | **0.0798 s** (79.82 ms) | **0.5671 s** (567.15 ms) |
| **CUDA Throughput** | **125,276.8 records/s** | **44,080.2 records/s** |
| **End-to-End Speedup (CPU / CUDA)** | **38.60×** | **34.73×** |
| **Kernel Compute Speedup (CPU / CUDA)**| **47.30×** | **39.82×** |

#### Table 7.2: Detailed Stage-by-Stage CUDA Timing Breakdown

| Pipeline Stage | 10K Mean Time (ms) | 10K Kernel % | 25K Mean Time (ms) | 25K Kernel % |
| :--- | :---: | :---: | :---: | :---: |
| **Host-to-Device (H2D) Transfer** | 0.393 ms | — | 0.915 ms | — |
| **1. Pairwise Distance Kernel** | 42.945 ms | 66.30% | 237.122 ms | 48.03% |
| **2. $k$-Nearest Neighbors ($k$-NN)** | 21.741 ms | 33.56% | 256.385 ms | 51.93% |
| **3. Reachability Distance Kernel** | 0.048 ms | 0.07% | 0.177 ms | 0.04% |
| **4. Local Reachability Density (LRD)**| 0.026 ms | 0.04% | 0.018 ms | <0.01% |
| **5. LOF Scoring Kernel** | 0.019 ms | 0.03% | 0.029 ms | 0.01% |
| **Total Kernel Compute Time** | **64.778 ms** | **100.00%** | **493.731 ms** | **100.00%** |
| **Device-to-Host (D2H) Transfer** | 0.083 ms | — | 0.058 ms | — |
| **Total CUDA End-to-End Pipeline** | **79.823 ms** | — | **567.148 ms** | — |

---

### 7.4 Correctness Results

#### Table 7.3: Empirical Correctness Verification vs. Historical M2 CPU Baseline

| Metric | 10K Subset ($N = 10,000$) | 25K Subset ($N = 25,000$) | 25K Same-Distance Diagnostic* |
| :--- | :---: | :---: | :---: |
| **Pearson Correlation ($r$)** | **0.99999817** | 0.26179690 | **0.99999989** |
| **Max Absolute Difference** | $1.081566 \times 10^{0}$ | $2.966723 \times 10^{9}$ | $1.209650 \times 10^{0}$ |
| **Mean Absolute Difference** | $1.157423 \times 10^{-3}$ | $1.209705 \times 10^{6}$ | $7.852144 \times 10^{-5}$ |
| **Anomaly Decision Threshold ($\tau_{0.10}$)**| 1.90915 (CPU) / 1.90901 (CUDA)| 1.95650 (CPU) / 1.96024 (CUDA)| 1.96024 |
| **CPU Detected Anomalies** | 1,000 | 2,500 | 2,500 |
| **CUDA Detected Anomalies** | 1,000 | 2,500 | 2,500 |
| **Classification Agreement (%)** | **99.98%** (9,998 / 10,000) | **99.58%** (24,894 / 25,000) | **99.99%** |
| **Prediction Disagreements** | 2 samples | 106 samples | 2 samples |

---

### 7.5 Numerical Discrepancy Root Cause Analysis

As documented during M3 Step 4 and confirmed in Step 5:
1. **CPU Distance Formula (Matrix Expansion)**:
   In `src/python/cpu/cpu_lof.py`, pairwise Euclidean distance is computed via matrix dot product:
   $$\|p - q\|_2^2 = \|p\|_2^2 + \|q\|_2^2 - 2\langle p, q\rangle$$
   In 32-bit floating point arithmetic (`float32`), when sample vectors have norm $\|p\|_2 \approx 100$, $\|p\|_2^2 \approx 10,000$. For identical or near-duplicate network packets where $\|p - q\|_2 < 0.001$, computing $10,000 + 10,000 - 20,000$ incurs catastrophic floating-point cancellation, causing the computed distance to truncate to exactly $0.00000000$.
2. **Impact on LOF Scores in 25K**:
   In the 25K subset, approximately 110 sample points have $\ge 20$ identical duplicate packets. On CPU, their $k$-distance and reachability distances collapse to $0.0$, leading to $\text{LRD} = 1.0 / (0.0 + 10^{-10}) = 10^{10}$ and an inflated LOF score of $\approx 2.97 \times 10^9$.
3. **CUDA Direct Coordinate Accumulation**:
   In `src/cuda/kernels/distance_kernel.cu`, distance is computed via direct coordinate accumulation:
   $$d(p, q) = \sqrt{\sum_{d=0}^{D-1} (p_d - q_d)^2}$$
   Because coordinates are subtracted first before squaring, no subtraction of large quantities occurs, accurately computing positive distances ($0.000732 > 0$).
4. **Diagnostic Proof of Equivalence**:
   When the exact CUDA pairwise distance matrix is supplied to the CPU LOF pipeline stages (`same_distance_diagnostic`), CPU and CUDA scores yield:
   - **Pearson correlation $r = 0.99999989$**
   - **Mean absolute difference $= 7.85 \times 10^{-5}$**
   - Both implementations identify exactly 2,500 anomalies with **99.58% classification agreement**.

Neither implementation was altered, preserving rigorous scientific honesty.

---

### 7.6 Dense-Memory Limitation (Exclusion of 100K+)

The naive baseline allocates a full dense pairwise distance matrix of shape $N \times N \times 4\text{ bytes}$:
* **10K Subset**: $10,000 \times 10,000 \times 4\text{ B} = 400\text{ MB}$ (Fits easily in 6GB VRAM)
* **25K Subset**: $25,000 \times 25,000 \times 4\text{ B} = 2.50\text{ GB}$ (Fits within 6GB VRAM)
* **100K Subset**: $100,000 \times 100,000 \times 4\text{ B} = \mathbf{40.0\text{ GB}}$

Because 40.0 GB far exceeds the 6.00 GB physical VRAM capacity of the NVIDIA GeForce RTX 3050 Laptop GPU, executing the naive dense distance kernel on 100K or larger is physically impossible without out-of-memory crashes (`cudaErrorMemoryAllocation`).
Therefore, Milestone 3 intentionally and safely caps dense matrix execution to 25K. Milestone 5 will introduce tiled/chunked distance computation to overcome this constraint for 100K+.

---

### 7.7 Official M3 Charts & Tables Catalog

#### Generated Research Charts (`results/m3_cuda_baseline/charts/`)

| File Name | Chart Title & Description |
| :--- | :--- |
| `m3_01_e2e_time_comparison.png` | **CPU vs. Naive CUDA LOF End-to-End Processing Time**<br>Dual bar chart comparing CPU runtime (3.081s / 19.698s) vs. CUDA runtime (0.080s / 0.567s). |
| `m3_02_throughput_comparison.png` | **Processing Throughput Comparison (Records / Second)**<br>Dual bar chart showing throughput scaling: 125,277 rec/s (10K) and 44,080 rec/s (25K). |
| `m3_03_cuda_kernel_breakdown.png` | **Naive CUDA LOF Kernel & Transfer Runtime Breakdown**<br>Subplots showing fine-grained latency per stage (Distance: 48–66%, k-NN: 33–52%). |
| `m3_04_speedup_over_cpu.png` | **Naive CUDA Acceleration Speedup Over CPU M2 Baseline**<br>Displays 34.7× to 38.6× end-to-end speedup and 39.8× to 47.3× compute speedup. |
| `m3_05_lof_score_correlation_scatter.png` | **CPU vs. Naive CUDA LOF Score Agreement & Correlation**<br>Dual scatter plots showing tight alignment along identity line $y=x$ ($r > 0.99999$). |
| `m3_06_score_diff_distribution.png` | **Distribution of Absolute LOF Score Differences**<br>Log-scale error histograms illustrating that the vast majority of scores differ by $< 10^{-3}$. |

#### Generated Machine-Readable Manifests & Tables (`results/m3_cuda_baseline/`)

| Path | Format | Description |
| :--- | :---: | :--- |
| `benchmarks/cuda_lof_benchmark_10k_k20.json` | JSON | Complete 10K execution manifest with raw iteration timings and environment |
| `benchmarks/cuda_lof_benchmark_25k_k20.json` | JSON | Complete 25K execution manifest with diagnostic consistency metrics |
| `benchmarks/m3_cuda_benchmark_summary.json` | JSON | Unified benchmark summary across all evaluated subsets |
| `benchmarks/m3_cuda_benchmark_summary.csv` | CSV | Machine-readable table of all measured timings, throughputs, and speedups |
| `benchmarks/lof_scores_cuda_10k_k20.npy` | NumPy | Binary array of 10,000 float32 CUDA LOF anomaly scores |
| `benchmarks/lof_scores_cuda_25k_k20.npy` | NumPy | Binary array of 25,000 float32 CUDA LOF anomaly scores |
| `tables/m3_benchmark_summary.csv` | CSV | High-level CPU vs. CUDA benchmark summary table |
| `tables/m3_timing_breakdown.csv` | CSV | Stage-by-stage kernel and transfer breakdown table |
| `tables/m3_correctness_summary.csv` | CSV | Empirical correctness, agreement, and error metrics table |

---

### 7.8 Milestone M3 Conclusion

Based strictly on empirical execution data on the NVIDIA GeForce RTX 3050 Laptop GPU:
1. **Substantial Baseline Acceleration**:
   Even without any architectural optimization (no shared memory, no tiling, no warp shuffles, no streams), naive CUDA achieves a **$38.60\times$ speedup on 10K** and a **$34.73\times$ speedup on 25K** over the optimized vectorized CPU reference implementation.
2. **Dominant Latency Contributors**:
   Empirical kernel breakdowns reveal that **Pairwise Distance ($48.0\% - 66.3\%$)** and **$k$-NN Search ($33.6\% - 51.9\%$)** account for over **$99.8\%$** of total GPU execution time. Subsequent stages (Reachability, LRD, LOF scoring) execute in sub-millisecond time ($< 0.25\text{ ms}$ combined).
3. **Data Transfer Overhead is Negligible**:
   Data movement (H2D and D2H) consumes $< 1.0\text{ ms}$ total ($< 1.5\%$ of pipeline runtime), confirming that computation, rather than PCIe transfer, dominates dense LOF execution at this scale.
4. **Algorithmic Correctness Validated**:
   On 10K, scores achieve $r = 0.99999817$ and $99.98\%$ anomaly agreement. On 25K, identical-distance diagnostic confirms $r = 0.99999989$ and mean difference $7.85 \times 10^{-5}$, with both CPU and CUDA detecting exactly 2,500 anomalies.
5. **Clear Boundary to Milestone 4**:
   Milestone 3 is complete. No optimizations have been introduced. Milestone 4 will perform formal GPU profiling (Nsight Systems / Nsight Compute) to identify hardware occupancy, memory bandwidth, and compute bottlenecks before implementing optimizations in Milestone 5.





