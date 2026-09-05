# Milestone M3: CUDA LOF Baseline (Roadmap)

## 1. Overview
This document outlines the planned design and specifications for **Milestone M3: CUDA LOF Baseline**.

*Current Status: Milestone M3 is scheduled after M2 baseline review.*

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
