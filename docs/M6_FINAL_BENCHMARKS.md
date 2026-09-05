# Milestone M6: Final Benchmarking & Evaluation (Roadmap)

## 1. Overview
This document outlines the evaluation protocol for **Milestone M6: Final Benchmarking**.

*Current Status: Scheduled following Milestone M5.*

---

## 2. Milestone Objectives
- Conduct comprehensive benchmarking sweeps across sample scales ($N \in \{10\text{K}, 25\text{K}, 100\text{K}, 250\text{K}, 500\text{K}, 1\text{M}\}$) and neighborhood parameters ($k \in \{10, 20, 50\}$).
- Quantify key performance indicators:
  - Latency comparison: CPU Baseline vs. Naive CUDA vs. Optimized CUDA.
  - Speedup ratios ($S = T_{\text{CPU}} / T_{\text{GPU}}$).
  - Throughput (samples processed per second).
  - VRAM scaling and PCIe transfer overhead.
- Export all benchmark results to `results/m6_final_benchmarks/`.
