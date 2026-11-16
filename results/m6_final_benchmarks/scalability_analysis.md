# M6 Scalability Analysis
Hardware: NVIDIA GeForce RTX 4050 Laptop GPU (5.78 GB free VRAM at run time), 15.31 GB total / 8.40 GB available RAM.

## Benchmarked sizes: 10k, 25k
- **10k** (N=10,000): CPU 5592.95 ms, naive CUDA 112.97 ms (49.5x), optimized CUDA 50.98 ms (109.7x vs CPU, 2.22x vs naive)
- **25k** (N=25,000): CPU 37648.60 ms, naive CUDA 351.87 ms (107.0x), optimized CUDA 315.43 ms (119.4x vs CPU, 1.12x vs naive)

## Unavailable sizes (dense-matrix VRAM/RAM wall)
- **100k** (N=100,000): Dense N x N float32 distance matrix needs 37.25 GB; exceeds both 8.40 GB available RAM and 5.78 GB free VRAM. Requires a tiled/chunked (out-of-core) distance kernel that never materializes the full matrix at once -- not implemented by M5 (M5 only added cache-reuse tiling within VRAM, not chunking across VRAM boundaries). Out of scope here.
- **250k** (N=250,000): Dense N x N float32 distance matrix needs 232.83 GB; exceeds both 8.40 GB available RAM and 5.78 GB free VRAM. Requires a tiled/chunked (out-of-core) distance kernel that never materializes the full matrix at once -- not implemented by M5 (M5 only added cache-reuse tiling within VRAM, not chunking across VRAM boundaries). Out of scope here.
- **500k** (N=500,000): Dense N x N float32 distance matrix needs 931.32 GB; exceeds both 8.40 GB available RAM and 5.78 GB free VRAM. Requires a tiled/chunked (out-of-core) distance kernel that never materializes the full matrix at once -- not implemented by M5 (M5 only added cache-reuse tiling within VRAM, not chunking across VRAM boundaries). Out of scope here.
- **1m** (N=1,000,000): Dense N x N float32 distance matrix needs 3725.29 GB; exceeds both 8.40 GB available RAM and 5.78 GB free VRAM. Requires a tiled/chunked (out-of-core) distance kernel that never materializes the full matrix at once -- not implemented by M5 (M5 only added cache-reuse tiling within VRAM, not chunking across VRAM boundaries). Out of scope here.

## Interpretation

Runtime for both CPU and naive/optimized CUDA grows quadratically with N, as expected from the O(N^2) distance-matrix and k-NN stages (the only stages that scale this way; reachability/LRD/LOF-scoring are O(N*k) and stay negligible at every size tested). The dominant real-world constraint is not compute time but memory: both the naive and M5-optimized GPU kernels, and the CPU matrix-expansion reference, all materialize a dense N x N float32 distance matrix at once. That matrix's size (4*N^2 bytes) crosses this GPU's ~5 GB usable VRAM between N=25,000 (2.5 GB, fits) and N=100,000 (40 GB, does not) -- a 16x jump in N causes a 256x jump in memory, so there is no size in between that was skipped; the wall is simply crossed abruptly in that gap on the sizes this project's preprocessing produced.

Reaching 100K-1M-scale (or the real 2.52M-row CICIDS2017 dataset) would require a genuinely different kernel architecture -- a tiled/chunked/out-of-core pairwise-distance and k-NN computation that processes the N x N matrix in blocks small enough to fit in VRAM, writing only the running top-k per row back to host memory, never allocating the full matrix at once. M5's shared-memory tiling improved *cache reuse within* an already-VRAM-resident matrix; it did not remove the matrix's existence. Implementing true out-of-core chunking is well-scoped future work, not completed here.
