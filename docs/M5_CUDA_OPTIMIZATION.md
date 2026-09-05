# Milestone M5: CUDA Optimization (Roadmap)

## 1. Overview
This document outlines the planned optimization strategies for **Milestone M5: CUDA Optimization**.

*Current Status: Scheduled following Milestone M4.*

---

## 2. Milestone Objectives
- Implement advanced GPU optimization techniques identified during M4 profiling:
  1. **Shared Memory Tiling**: Block-level caching of feature vectors to minimize global memory traffic.
  2. **Memory Coalescing**: Transposed / aligned memory access patterns for full cache line utilization.
  3. **Warp-Level Primitives**: Fast intra-warp reduction and bitonic sorting networks for top-$k$ selection.
  4. **Tiled / Chunked Distance Kernels**: Partition large datasets to fit within 6GB VRAM bounds.
  5. **Asynchronous Streams & Pipelining**: Overlapping PCIe host-to-device transfers with kernel execution.
- Maintain numerical correctness against the M2 CPU reference implementation.
