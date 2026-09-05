# Milestone M4: CUDA Profiling & Bottleneck Analysis (Roadmap)

## 1. Overview
This document outlines the planned design and methodology for **Milestone M4: CUDA Profiling & Bottleneck Analysis**.

*Current Status: Scheduled following Milestone M3.*

---

## 2. Milestone Objectives
- Profile baseline CUDA kernels using NVIDIA Nsight Compute (`ncu`) and NVIDIA Nsight Systems (`nsys`).
- Measure key GPU performance metrics:
  - Memory Bandwidth Saturation (DRAM vs. L1/L2 cache throughput).
  - Compute vs. Memory bound characterization (Roofline analysis).
  - Warp Execution Efficiency and branch divergence in top-$k$ sorting.
  - Streaming Multiprocessor (SM) Occupancy and register pressure.
- Document specific architectural bottlenecks guiding the optimization roadmap in Milestone M5.
