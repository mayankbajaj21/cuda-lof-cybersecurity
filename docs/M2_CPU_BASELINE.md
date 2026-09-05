# Milestone M2: Naive CPU Local Outlier Factor Baseline

## 1. Objective
The objective of **Milestone M2** is to establish a transparent, CPU-only reference implementation of the **Local Outlier Factor (LOF)** unsupervised anomaly detection algorithm (Breunig et al., 2000) built from first principles.

This baseline serves as:
1. The **correctness ground-truth** against which GPU kernel results (M3 / M5) will be numerically verified.
2. The **performance baseline** against which GPU acceleration speedup ratios will be evaluated in M6.

---

## 2. Mathematical Formulation of Local Outlier Factor

Local Outlier Factor measures the degree to which an object is isolated relative to its surrounding neighborhood. The computation executes through five deterministic stages:

### Stage 1: Pairwise Euclidean Distance Matrix
For a dataset $X = \{x_1, x_2, \dots, x_N\} \subset \mathbb{R}^D$, calculate the full pairwise distance matrix:
$$d(p, q) = \|p - q\|_2 = \sqrt{\sum_{f=1}^D (p_f - q_f)^2}$$
- **Computational Complexity**: $O(N^2 \cdot D)$ operations.
- **Memory Complexity**: $O(N^2)$ storage.

### Stage 2: $k$-Nearest Neighbors ($N_k(p)$) and $k$-distance ($k\text{-dist}(p)$)
For each point $p$, sort distances to all other points $q \neq p$ and identify the $k$ closest neighbors:
- $N_k(p)$: The set of $k$ nearest neighbors to $p$ (excluding $p$).
- $k\text{-distance}(p)$: The distance from $p$ to its $k$-th nearest neighbor:
  $$k\text{-distance}(p) = \max_{o \in N_k(p)} d(p, o)$$
- **Computational Complexity**: $O(N^2 \log N)$ (or $O(N^2)$ with partial selection).

### Stage 3: Reachability Distance ($\text{reach-dist}_k(p, o)$)
The reachability distance of point $p$ with respect to neighbor point $o \in N_k(p)$ is defined as:
$$\text{reach-dist}_k(p, o) = \max(k\text{-distance}(o), d(p, o))$$
- **Purpose**: Dampens statistical fluctuations of distance for points located within dense clusters while preserving true distances for isolated outliers.

### Stage 4: Local Reachability Density ($\text{lrd}_k(p)$)
The local reachability density of $p$ is the inverse of the average reachability distance from $p$ to its $k$-nearest neighbors:
$$\text{lrd}_k(p) = \frac{|N_k(p)|}{\sum_{o \in N_k(p)} \text{reach-dist}_k(p, o)} = \frac{1}{\frac{1}{k} \sum_{o \in N_k(p)} \text{reach-dist}_k(p, o) + \epsilon}$$
where $\epsilon = 10^{-10}$ is a zero-division guard.

### Stage 5: Local Outlier Factor Score ($\text{LOF}_k(p)$)
The LOF score compares the local reachability density of point $p$ to the average LRD of its $k$ neighbors:
$$\text{LOF}_k(p) = \frac{\sum_{o \in N_k(p)} \frac{\text{lrd}_k(o)}{\text{lrd}_k(p)}}{|N_k(p)|} = \frac{\frac{1}{k} \sum_{o \in N_k(p)} \text{lrd}_k(o)}{\text{lrd}_k(p) + \epsilon}$$

#### Interpretation:
- $\text{LOF}(p) \approx 1.0$: Point $p$ has a density comparable to its neighbors (homogeneous cluster / inlier).
- $\text{LOF}(p) < 1.0$: Point $p$ resides in a higher-density region than its neighbors.
- $\text{LOF}(p) > 1.0$: Point $p$ is located in a significantly lower-density region (local anomaly / outlier).

---

## 3. Implementation Architecture

The reference CPU implementation is structured in `src/python/cpu/cpu_lof.py`:

```
src/python/
├── data/
│   └── dataset.py        # Processed dataset & subset loaders
└── cpu/
    └── cpu_lof.py        # CPULocalOutlierFactor class & baseline CLI implementation
```

### Key Class Methods in `CPULocalOutlierFactor`:
- `compute_pairwise_distances(X)`: Vectorized Euclidean distance calculation via matrix dot products: $\|p - q\|^2 = \|p\|^2 + \|q\|^2 - 2\langle p, q \rangle$.
- `compute_k_neighbors(dist_matrix)`: Vectorized neighbor sorting extracting top-$k$ indices, distances, and $k$-distance vectors.
- `compute_reachability_distances(dist_matrix, knn_indices, k_distances)`: Element-wise maximum between neighbor $k$-distances and direct Euclidean distances.
- `compute_lrd(reach_dists)`: Row-wise mean reachability calculation with epsilon-guarded inversion.
- `compute_lof_scores(lrd, knn_indices)`: Neighbor LRD gathering and density ratio averaging.
- `fit(X)` / `fit_predict(X)`: End-to-end pipeline with stage-by-stage runtime profiling and percentile-based anomaly thresholding.

---

## 4. Verification & Correctness Validation

Correctness is validated independently using `scikit-learn`'s `LocalOutlierFactor` in `tests/test_cpu_lof.py`.

### Automated Test Suite (`pytest -v tests/test_cpu_lof.py`)

| Test Name | Validation Purpose | Status |
| :--- | :--- | :---: |
| `test_cpu_lof_output_dimensions` | Validates shapes of all intermediate arrays ($N \times N$, $N \times k$, $N$) | PASSED |
| `test_absence_of_nan_and_inf` | Confirms zero NaNs/Infs and strictly positive LOF scores | PASSED |
| `test_neighbor_index_validity` | Verifies index bounds $[0, N-1]$, monotonic sorting, and no self-loops | PASSED |
| `test_reproducibility` | Confirms bitwise identical output across repeated runs | PASSED |
| `test_numerical_correctness_against_sklearn` | Verifies exact numerical equivalence on synthetic data ($r > 0.9999$) | PASSED |
| `test_numerical_correctness_on_cicids_subset` | Verifies numerical equivalence on actual CICIDS2017 features ($r > 0.9999$) | PASSED |
| `test_configurable_k_parameter[5, 10, 25]` | Validates flexibility across multiple neighbor configurations | PASSED |
| `test_error_handling` | Confirms graceful exception handling for invalid parameters | PASSED |

**All 10 unit tests passed.**

---

## 5. CPU Experimental Results & Performance Breakdown

### 5.1 Hardware & Environment Baseline
All baseline benchmarks were captured on the following audited host system:

| Parameter | Specification | Details |
| :--- | :--- | :--- |
| **CPU Model** | `13th Gen Intel(R) Core(TM) i5-13450HX` | 10 Cores (6 P-cores + 4 E-cores), Base clock 2.40 GHz |
| **Physical Cores** | `10` | Hardware physical execution cores |
| **Logical Threads** | `16` | Hyperthreaded logical processors |
| **Host System RAM** | `15.71 GB` (~16 GB DDR5) | 16,873,545,728 bytes total physical memory |
| **Operating System** | `Microsoft Windows 11 Home` | 64-bit (AMD64), Build `10.0.26200` |
| **Python Runtime** | `Python 3.13.11` | 64-bit CPython (MSC v.1944) |
| **Key Libraries** | `numpy 2.4.4`, `scipy 1.17.1`, `pandas 3.0.2`, `scikit-learn 1.8.0` | Numerical & ML stack |

---

### 5.2 Execution Time Breakdown
Experiments were executed on this baseline CPU environment using the preprocessed CICIDS2017 subsets ($k=20$, contamination $= 0.10$):

| Stage | Subset 10K ($N = 10,000$) | Subset 25K ($N = 25,000$) | Scaling Factor ($2.5\times N$) |
| :--- | :---: | :---: | :---: |
| **Data Loading** | 0.0176 s | 0.0381 s | $2.16\times$ |
| **1. Pairwise Distance Matrix** | 1.3906 s (45.39%) | 8.1257 s (41.33%) | $5.84\times$ |
| **2. $k$-NN Search & Sort** | 1.6674 s (54.42%) | 11.5173 s (58.58%) | $6.91\times$ |
| **3. Reachability Distance** | 0.0042 s (0.14%) | 0.0125 s (0.06%) | $2.98\times$ |
| **4. LRD Computation** | 0.0003 s (0.01%) | 0.0006 s (0.00%) | $2.00\times$ |
| **5. LOF Scoring** | 0.0013 s (0.04%) | 0.0034 s (0.02%) | $2.62\times$ |
| **Total LOF Compute Time** | **3.0638 s** | **19.6596 s** | **$6.42\times$** |
| **Total End-to-End Time** | **3.0814 s** | **19.6976 s** | **$6.39\times$** |

### LOF Score Distribution Summary

| Metric | Subset 10K ($N = 10,000$) | Subset 25K ($N = 25,000$) |
| :--- | :---: | :---: |
| **Minimum LOF Score** | 0.9144 | 0.9069 |
| **Median LOF Score ($p_{50}$)** | 1.0068 | 1.0076 |
| **Mean LOF Score** | 1.7163 | 1.8385 |
| **Standard Deviation** | 11.3284 | 15.6961 |
| **90th Percentile ($p_{90}$)** | 1.9091 | 1.9565 |
| **99th Percentile ($p_{99}$)** | 15.2281 | 16.8928 |
| **Maximum LOF Score** | 1061.4340 | 1373.1235 |
| **Anomaly Decision Threshold ($\tau_{0.10}$)** | 1.9091 | 1.9565 |

---

## 6. Key Bottleneck Findings & Motivation for GPU Acceleration

The experimental profiling on CPU demonstrates the following architectural characteristics:

1. **Quadratic Complexity Dominance**:
   - Pairwise distance calculation + $k$-NN sorting account for **99.81%** of total computation time on 10K samples and **99.91%** on 25K samples.
   - The remaining steps (reachability distance, LRD, and LOF scoring) represent less than **0.2%** of total runtime combined.

2. **Quadratic Scaling Curve**:
   - Increasing dataset size from $N = 10\text{K}$ to $N = 25\text{K}$ ($2.5\times$ increase) caused execution time to scale by **$6.42\times$** (closely matching the theoretical $O(N^2)$ multiplier of $2.5^2 = 6.25$).
   - Extrapolating this CPU baseline to the full dataset ($N \approx 2.52\text{M}$) yields an estimated CPU compute time of thousands of hours without algorithmic partitioning or hardware acceleration.

3. **Memory Footprint**:
   - An $N \times N$ dense distance matrix in `float32` requires $N^2 \times 4$ bytes ($400\text{ MB}$ for $10\text{K}$, $2.5\text{ GB}$ for $25\text{K}$).
   - On CPU, full matrix materialization is bounded by host RAM. On GPU (RTX 3050 6GB VRAM), this confirms the need for chunked/tiled distance kernels in **M3 / M5**.

---

## 7. Artifacts Preserved in `results/m2_cpu_baseline/`

```
results/m2_cpu_baseline/
├── benchmarks/
│   ├── cpu_lof_10k_k20.json       # Full run metadata, parameter config, and timing breakdown for 10K
│   ├── lof_scores_10k_k20.npy     # Raw LOF score array for 10K (N = 10,000)
│   ├── cpu_lof_25k_k20.json       # Full run metadata, parameter config, and timing breakdown for 25K
│   └── lof_scores_25k_k20.npy     # Raw LOF score array for 25K (N = 25,000)
├── tables/                        # CSV tables generated by visualization suite
└── charts/                        # PNG visual figures generated by visualization suite
```

---

## 8. Reproducibility

To run the full test suite:
```powershell
pytest -v tests/
```

To execute a CPU LOF experiment via script runner:
```powershell
python scripts/run_cpu_baseline.py --subset 10k --k 20 --output-dir results/m2_cpu_baseline/benchmarks
python scripts/run_cpu_baseline.py --subset 25k --k 20 --output-dir results/m2_cpu_baseline/benchmarks
```
Alternatively, directly via the module:
```powershell
python src/python/cpu/cpu_lof.py --subset 10k --k 20 --output-dir results/m2_cpu_baseline/benchmarks
```
