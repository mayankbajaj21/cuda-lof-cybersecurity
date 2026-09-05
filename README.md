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
├── preprocessing/      # Dataset acquisition, inspection, cleaning, and preparation pipeline
├── data/
│   ├── raw/            # Original raw CICIDS2017 CSV captures (git-ignored)
│   └── processed/      # Cleaned and scaled .npz feature matrices & subsets (git-ignored)
├── src/
│   ├── python/         # Python dataset loaders and CPU implementations
│   └── cuda/           # CUDA C++ kernels and GPU implementations
├── tests/              # Automated test suites
├── experiments/        # Experiment configurations and run scripts
├── benchmarks/         # Benchmark suites and performance evaluation tables
├── results/            # Output artifacts partitioned by milestone
│   ├── m2_cpu_baseline/
│   ├── m3_cuda_baseline/
│   ├── m4_profiling/
│   ├── m5_cuda_optimized/
│   └── m6_final_benchmarks/
├── docs/               # Milestone technical documentation & reports
└── scripts/            # Operational utility scripts
```

---

## Current Status
- **Current Milestone**: `M1 — Dataset Preparation`
- Dedicated preprocessing pipeline created in `preprocessing/` for automated inspection, cleaning, feature selection, scaling, and subset creation.
- CICIDS2017 dataset ingested (2,830,743 raw rows), cleansed of invalid/infinite records and duplicates down to 2,520,798 valid records.
- 68 numerical features selected and standardized (StandardScaler, float32); ground-truth labels decoupled and preserved.
- Controlled reproducible experimental subsets (10K, 25K, 100K, 250K, 500K, 1M, and Full) generated and verified.
- *Note: CPU baseline LOF (M2), CUDA kernels (M3), profiling (M4), and optimization benchmarks (M5-M6) have not yet been implemented.*
