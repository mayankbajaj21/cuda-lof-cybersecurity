# CUDA-Optimized Local Outlier Factor for Large-Scale Cybersecurity Anomaly Detection

## Project Objective
The objective of this research project is to design, implement, profile, and optimize a high-performance GPU-accelerated implementation of the **Local Outlier Factor (LOF)** unsupervised anomaly detection algorithm using NVIDIA CUDA, targeting high-throughput cybersecurity intrusion detection on network traffic datasets.

---

## Motivation: Cybersecurity Anomaly Detection
Modern network infrastructures generate massive volumes of high-dimensional network flow data per second. Detecting sophisticated intrusions, zero-day vulnerabilities, and anomalous traffic patterns requires robust unsupervised anomaly detection techniques capable of identifying deviations without relying exclusively on labeled training data.

However, density-based anomaly detection algorithms like Local Outlier Factor (LOF) incur severe computational overhead—primarily due to all-pairs distance computations and $k$-nearest neighbor ($k$-NN) density estimation over millions of records. Standard CPU implementations become major computational bottlenecks in real-time or large-scale forensic cybersecurity pipelines.

---

## Intended Algorithm: Local Outlier Factor (LOF)
Local Outlier Factor (Breunig et al., 2000) measures the local density deviation of a given data point with respect to its surrounding $k$-nearest neighbors. The algorithm comprises three core computational phases:
1. **$k$-Nearest Neighbor ($k$-NN) Search & Distance Matrix Computation**: Calculating pairwise metric distances across the dataset.
2. **Local Reachability Density (LRD)**: Evaluating the reachability distance and local density of each point relative to its $k$-neighbors.
3. **Local Outlier Factor (LOF) Scoring**: Comparing the LRD of each point against the average LRD of its neighbors to generate an anomaly score.

---

## Intended Dataset: CICIDS2017
The evaluation and benchmarking will target the **CICIDS2017** benchmark dataset (Canadian Institute for Cybersecurity), which provides realistic, high-dimensional network flow records containing benign network behavior and common cyber-attack vectors (e.g., DoS/DDoS, Port Scans, Web Attacks, Infiltration, Botnets).

---

## Research Focus: CUDA / GPU Acceleration
The primary research contribution centers on parallelizing and optimizing the computationally intensive stages of LOF on NVIDIA CUDA hardware:
- Accelerating pairwise distance calculations and $k$-NN sorting via massively parallel CUDA kernels.
- Minimizing global memory latency through shared memory tiling, coalesced memory access, and warp-level primitives.
- Utilizing hardware-specific profiling tools (NVIDIA Nsight Compute / Nsight Systems) to identify compute vs. memory bottlenecks.
- Iteratively optimizing kernel configurations to maximize memory bandwidth and compute throughput.

---

## Milestone Roadmap

| Milestone | Description |
| :--- | :--- |
| **M0 — Repository & Environment Setup** | Establish repository architecture, dependencies, project documentation, and environment verification. |
| **M1 — Dataset Preparation** | Acquire, clean, preprocess, and normalize the CICIDS2017 dataset for numerical processing. |
| **M2 — Naive CPU LOF Baseline** | Implement standard CPU-based LOF and validate correctness against reference implementations. |
| **M3 — CUDA LOF Baseline** | Implement the initial baseline CUDA kernels for distance, $k$-NN, LRD, and LOF scoring. |
| **M4 — CUDA Profiling & Bottleneck Analysis** | Profile baseline GPU execution using Nsight tools to analyze memory bandwidth, latency, and warp efficiency. |
| **M5 — CUDA Optimization** | Apply GPU optimization strategies (shared memory tiling, warp reductions, memory coalescing, stream pipelining). |
| **M6 — Final Benchmarking** | Conduct systematic scalability and performance benchmarks across sample sizes and parameter sweeps. |
| **M7 — Analysis & Documentation** | Consolidate experimental results, performance speedup graphs, analysis, and comprehensive documentation. |

---

## Repository Structure

```text
cuda-lof-cybersecurity/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── full/
│       └── subsets/
│           ├── 10k/
│           ├── 25k/
│           ├── 100k/
│           ├── 250k/
│           ├── 500k/
│           └── 1m/
│
├── preprocessing/
│   ├── README.md
│   ├── inspect_dataset.py
│   ├── prepare_dataset.py
│   └── download_dataset.py
│
├── src/
│   ├── python/
│   │   ├── cpu/
│   │   │   └── cpu_lof.py
│   │   ├── data/
│   │   │   └── dataset.py
│   │   └── visualization/
│   │       ├── m1_visualizations.py
│   │       └── m2_visualizations.py
│   │
│   └── cuda/
│       ├── common/
│       ├── kernels/
│       └── lof/
│
├── tests/
│   ├── test_dataset.py
│   ├── test_cpu_lof.py
│   └── cuda/
│
├── experiments/
│   ├── m2_cpu_baseline/
│   ├── m3_cuda_baseline/
│   ├── m4_profiling/
│   ├── m5_cuda_optimized/
│   └── m6_final_benchmarks/
│
├── results/
│   ├── m1_dataset_preparation/
│   │   ├── tables/
│   │   └── charts/
│   │
│   ├── m2_cpu_baseline/
│   │   ├── benchmarks/
│   │   ├── tables/
│   │   └── charts/
│   │
│   ├── m3_cuda_baseline/
│   │   ├── benchmarks/
│   │   ├── tables/
│   │   └── charts/
│   │
│   ├── m4_profiling/
│   │   ├── reports/
│   │   └── profiles/
│   │
│   ├── m5_cuda_optimized/
│   │   ├── benchmarks/
│   │   ├── tables/
│   │   └── charts/
│   │
│   └── m6_final_benchmarks/
│       ├── benchmarks/
│       ├── tables/
│       └── charts/
│
├── docs/
│   ├── PROJECT_PLAN.md
│   ├── M1_DATASET_PREPARATION.md
│   ├── M2_CPU_BASELINE.md
│   ├── M3_CUDA_BASELINE.md
│   ├── M4_PROFILING.md
│   ├── M5_CUDA_OPTIMIZATION.md
│   ├── M6_FINAL_BENCHMARKS.md
│   └── architecture/
│
└── scripts/
    ├── run_cpu_baseline.py
    ├── run_cuda_baseline.py
    └── run_benchmarks.py
```

---

## Current Status
- **Current Milestone**: `M2 — Naive CPU LOF Baseline (Completed & Reorganized for M3)`
- First-principles CPU reference implementation of Local Outlier Factor (LOF) established in `src/python/cpu/cpu_lof.py` and accessible via runner `scripts/run_cpu_baseline.py`.
- Step-by-step algorithm verified (pairwise Euclidean distances, $k$-NN sorting, reachability distance, LRD, LOF scores, anomaly predictions).
- Correctness and numerical equivalence verified against `scikit-learn` reference implementation ($r > 0.9999$).
- Baseline execution timings on CICIDS2017 subsets (10K: ~3.08s total, 25K: ~22.17s total) benchmarked and preserved in `results/m2_cpu_baseline/benchmarks/`.
- M1 and M2 visualization generators located in `src/python/visualization/` producing research tables and charts in `results/m1_dataset_preparation/` and `results/m2_cpu_baseline/`.
- Key bottleneck confirmed: Pairwise distance calculation + $k$-NN sorting consume **>99.8%** of total CPU runtime, motivating GPU acceleration.
- *Note: The repository includes a deterministic synthetic fallback dataset for offline validation when the original CICIDS2017 CSV bundle is unavailable; native CUDA kernels for M3 and profiling/optimization milestones M4-M6 remain GPU-accelerated project extensions that are exercised through the included scaffolding and validation pipeline.*
