# Project Plan: CUDA-Optimized LOF for Cybersecurity Anomaly Detection

## Overview
This document outlines the phased milestone roadmap for implementing, profiling, and optimizing the Local Outlier Factor (LOF) anomaly detection algorithm on NVIDIA CUDA architectures using the CICIDS2017 cybersecurity dataset.

---

## Milestones

### M0 — Repository & Environment Setup
- Establish repository folder structure, version control configurations, and minimal dependency declarations.
- Inspect local hardware, drivers, and CUDA development toolchains.
- Define project milestones and operational guidelines.

### M1 — Dataset Preparation
- Acquire and structure the CICIDS2017 intrusion detection dataset.
- Clean invalid values (e.g., infinite/NaN entries), select numerical network flow features, and normalize feature distributions.
- Provide data loading and batching utilities for CPU and GPU processing pipelines.

### M2 — Naive CPU LOF Baseline
- Implement a reference CPU-based Local Outlier Factor algorithm from first principles.
- Verify numerical correctness and anomaly score consistency against standard library implementations (e.g., scikit-learn).
- Establish execution time baseline metrics on CPU for future comparative analysis.

### M3 — CUDA LOF Baseline
- Develop the initial functional CUDA kernels for pairwise distance calculations, $k$-nearest neighbor search, local reachability density (LRD), and LOF scoring.
- Verify GPU output numerical equivalence against the CPU baseline implementation.
- Establish functional end-to-end GPU execution.

### M4 — CUDA Profiling & Bottleneck Analysis
- Profile the baseline CUDA implementation using NVIDIA profiling tools (Nsight Compute / Nsight Systems).
- Quantify key performance indicators: compute throughput, memory bandwidth saturation, instruction divergence, and occupancy.
- Identify the primary architectural bottlenecks across each computational phase.

### M5 — CUDA Optimization
- Implement targeted GPU optimization techniques based on profiling insights:
  - Shared memory tiling for pairwise distance matrices.
  - Coalesced global memory access patterns.
  - Warp-level reduction and sorting primitives for top-$k$ selection.
  - CUDA streams for concurrent data transfer and kernel execution.
- Validate numerical accuracy throughout optimization iterations.

### M6 — Final Benchmarking
- Execute comprehensive benchmark suites across varying dataset sizes ($N$) and neighborhood sizes ($k$).
- Measure end-to-end latency, throughput (samples/second), speedup ratios (CPU vs. Baseline GPU vs. Optimized GPU), and memory scaling.

### M7 — Analysis & Documentation
- Synthesize benchmarking metrics, profiling reports, and performance charts.
- Document architectural findings, optimization trade-offs, and research conclusions in final technical reports.
