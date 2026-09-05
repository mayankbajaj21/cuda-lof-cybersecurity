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

## Current Status
- **Current Milestone**: `M0 — Repository & Environment Setup`
- Repository structure, configuration files, and development roadmaps are initialized.
- *Note: CUDA kernels, CPU baseline implementations, and performance benchmarks have not yet been implemented or evaluated.*
