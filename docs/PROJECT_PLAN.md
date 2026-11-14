# Project Plan: CUDA-Optimized LOF for Cybersecurity Anomaly Detection

## Overview
This document outlines the phased milestone roadmap for implementing, profiling, and optimizing the Local Outlier Factor (LOF) anomaly detection algorithm on NVIDIA CUDA architectures using the CICIDS2017 cybersecurity dataset.

---

## Milestone status

All milestones below are complete with real measurements (see `README.md` §2, §11–§13 for exact numbers and hardware). M4–M6 were originally completed only as analytical projections (`src/python/benchmarking/milestones.py`) on an environment without a working CUDA toolchain; once CUDA Toolkit 13.5 + MSVC were installed on this machine, M4–M6 were redone for real on an NVIDIA RTX 4050 Laptop GPU (sm_89). The dataset present on this machine is a synthetic CICIDS2017-shaped placeholder, not downloaded real data — see `README.md` §3.

| Milestone | Status |
| :--- | :--- |
| M0 | Done |
| M1 | Done (synthetic placeholder dataset on this machine) |
| M2 | Done — real |
| M3 | Done — real |
| M4 | Done — real CUDA-event profiling; Nsight tools not installed |
| M5 | Done — real optimized kernels, modest measured speedup |
| M6 | Done at 10K–25K; 100K–1M correctly marked infeasible (VRAM/RAM wall) |
| M7 | Done — README + Streamlit dashboard (`app.py`) |

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
