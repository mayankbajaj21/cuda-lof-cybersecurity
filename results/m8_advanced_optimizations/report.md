# M8 - Advanced (outside-syllabus) optimizations: measured results

Run: 2026-11-16T06:03:09.526131+00:00 -> 2026-11-16T06:06:33.895145+00:00 (UTC). k=20, contamination=0.1. GPU: 3 warm-ups + 10 timed reps interleaved across variants; CPU: 1 warm-up + 5 reps.
Power: AC plugged = True, scheme = Power Scheme GUID: 381b4222-f694-41f0-9685-ff5bb260df2e  (Balanced).
Dataset: synthetic CICIDS2017-shaped placeholder (data/processed).

## End-to-end median time and speedups

| Subset | Implementation | Exact? | Distance kernel (ms) | End-to-end median (ms) | vs CPU | vs V0 | vs V1 | Peak GPU mem (MiB) |
|---|---|---|---|---|---|---|---|---|
| 10k | V0 Naive | exact | 28.17 | 48.55 | 61.6x | 1.00x | 0.92x | 384.1 |
| 10k | V1 Syllabus-optimized | exact | 24.66 | 44.47 | 67.2x | 1.09x | 1.00x | 384.1 |
| 10k | V2 Register blocking | exact | 5.82 | 25.46 | 117.4x | 1.91x | 1.75x | 384.1 |
| 10k | V3 Async copy + double buffering | exact | 29.64 | 49.37 | 60.6x | 0.98x | 0.90x | 384.1 |
| 10k | V4 FP16 storage | approximate | 13.90 | 25.68 | 116.4x | 1.89x | 1.73x | 192.3 |
| 10k | V5 Combined | approximate | 4.23 | 16.27 | 183.8x | 2.98x | 2.73x | 192.3 |
| 10k | CPU (NumPy) | exact | 1,117.15 | 2,989.60 | 1.0x | 0.02x | 0.01x | - |
| 25k | V0 Naive | exact | 176.09 | 293.70 | 73.2x | 1.00x | 0.91x | 2,390.7 |
| 25k | V1 Syllabus-optimized | exact | 157.46 | 268.18 | 80.2x | 1.10x | 1.00x | 2,390.7 |
| 25k | V2 Register blocking | exact | 36.07 | 149.66 | 143.7x | 1.96x | 1.79x | 2,390.7 |
| 25k | V3 Async copy + double buffering | exact | 185.61 | 299.52 | 71.8x | 0.98x | 0.90x | 2,390.7 |
| 25k | V4 FP16 storage | approximate | 89.10 | 148.87 | 144.5x | 1.97x | 1.80x | 1,195.9 |
| 25k | V5 Combined | approximate | 26.55 | 88.15 | 243.9x | 3.33x | 3.04x | 1,195.9 |
| 25k | CPU (NumPy) | exact | 8,694.56 | 21,504.27 | 1.0x | 0.01x | 0.01x | - |
| 45k (first 45k rows of 100k subset) | V0 Naive | exact | UNAVAILABLE | UNAVAILABLE | - | - | - | - |
| 45k (first 45k rows of 100k subset) | V1 Syllabus-optimized | exact | UNAVAILABLE | UNAVAILABLE | - | - | - | - |
| 45k (first 45k rows of 100k subset) | V2 Register blocking | exact | UNAVAILABLE | UNAVAILABLE | - | - | - | - |
| 45k (first 45k rows of 100k subset) | V3 Async copy + double buffering | exact | UNAVAILABLE | UNAVAILABLE | - | - | - | - |
| 45k (first 45k rows of 100k subset) | V4 FP16 storage | approximate | 290.27 | 466.16 | n/a | n/a | n/a | 3,869.2 |
| 45k (first 45k rows of 100k subset) | V5 Combined | approximate | 91.53 | 270.45 | n/a | n/a | n/a | 3,869.2 |
| 45k (first 45k rows of 100k subset) | CPU (NumPy) | exact | UNAVAILABLE | UNAVAILABLE | - | - | - | - |

## Not run (UNAVAILABLE)

- **45k (first 45k rows of 100k subset) / V0 Naive**: Needs ~7.56 GiB (+300 MiB margin) but only 5.67 GiB VRAM was free at run time
- **45k (first 45k rows of 100k subset) / V1 Syllabus-optimized**: Needs ~7.56 GiB (+300 MiB margin) but only 5.67 GiB VRAM was free at run time
- **45k (first 45k rows of 100k subset) / V2 Register blocking**: Needs ~7.56 GiB (+300 MiB margin) but only 5.67 GiB VRAM was free at run time
- **45k (first 45k rows of 100k subset) / V3 Async copy + double buffering**: Needs ~7.56 GiB (+300 MiB margin) but only 5.67 GiB VRAM was free at run time
- **45k (first 45k rows of 100k subset) / CPU (NumPy)**: Not run: the CPU reference materialises at least one 45000x45000 float32 matrix (7.5 GiB) plus temporaries; 8.7 GiB RAM was available. O(N^2) extrapolation from the 25k median: ~70 s per run (ESTIMATED).

## What each technique did

**10k** (distance kernel V1 = 24.66 ms)
- V2 Register blocking: distance kernel 5.82 ms (4.24x vs V1, faster); k-NN kernel 7.21 ms; end-to-end 1.75x vs V1.
- V3 Async copy + double buffering: distance kernel 29.64 ms (0.83x vs V1, slower); k-NN kernel 7.21 ms; end-to-end 0.90x vs V1.
- V4 FP16 storage: distance kernel 13.90 ms (1.77x vs V1, faster); k-NN kernel 4.38 ms; end-to-end 1.73x vs V1.
- V5 Combined: distance kernel 4.23 ms (5.83x vs V1, faster); k-NN kernel 4.36 ms; end-to-end 2.73x vs V1.

**25k** (distance kernel V1 = 157.46 ms)
- V2 Register blocking: distance kernel 36.07 ms (4.37x vs V1, faster); k-NN kernel 42.78 ms; end-to-end 1.79x vs V1.
- V3 Async copy + double buffering: distance kernel 185.61 ms (0.85x vs V1, slower); k-NN kernel 42.69 ms; end-to-end 0.90x vs V1.
- V4 FP16 storage: distance kernel 89.10 ms (1.77x vs V1, faster); k-NN kernel 22.31 ms; end-to-end 1.80x vs V1.
- V5 Combined: distance kernel 26.55 ms (5.93x vs V1, faster); k-NN kernel 22.37 ms; end-to-end 3.04x vs V1.

## Correctness

| Subset | Reference | Candidate | Exact? | Max abs err | Pearson r | kNN recall@k | Anomaly agreement |
|---|---|---|---|---|---|---|---|
| 10k | CPU (NumPy) | V0 Naive | exact | 0.000945 | 0.99999997 | 1.0000 | 1.0000 |
| 10k | CPU (NumPy) | V1 Syllabus-optimized | exact | 0.000945 | 0.99999997 | 1.0000 | 1.0000 |
| 10k | CPU (NumPy) | V2 Register blocking | exact | 0.000945 | 0.99999997 | 1.0000 | 1.0000 |
| 10k | CPU (NumPy) | V3 Async copy + double buffering | exact | 0.000945 | 0.99999997 | 1.0000 | 1.0000 |
| 10k | CPU (NumPy) | V4 FP16 storage | approximate | 0.00679 | 0.99988584 | 0.9933 | 0.9990 |
| 10k | CPU (NumPy) | V5 Combined | approximate | 0.00679 | 0.99988584 | 0.9933 | 0.9990 |
| 10k | V0 Naive | V1 Syllabus-optimized | exact | 0 | 1.00000000 | 1.0000 | 1.0000 |
| 10k | V0 Naive | V2 Register blocking | exact | 0 | 1.00000000 | 1.0000 | 1.0000 |
| 10k | V0 Naive | V3 Async copy + double buffering | exact | 0 | 1.00000000 | 1.0000 | 1.0000 |
| 10k | V0 Naive | V4 FP16 storage | approximate | 0.00679 | 0.99988581 | 0.9933 | 0.9990 |
| 10k | V0 Naive | V5 Combined | approximate | 0.00679 | 0.99988581 | 0.9933 | 0.9990 |
| 25k | CPU (NumPy) | V0 Naive | exact | 0.000474 | 1.00000000 | 1.0000 | 1.0000 |
| 25k | CPU (NumPy) | V1 Syllabus-optimized | exact | 0.000474 | 1.00000000 | 1.0000 | 1.0000 |
| 25k | CPU (NumPy) | V2 Register blocking | exact | 0.000474 | 1.00000000 | 1.0000 | 1.0000 |
| 25k | CPU (NumPy) | V3 Async copy + double buffering | exact | 0.000474 | 1.00000000 | 1.0000 | 1.0000 |
| 25k | CPU (NumPy) | V4 FP16 storage | approximate | 0.0102 | 0.99988543 | 0.9936 | 0.9990 |
| 25k | CPU (NumPy) | V5 Combined | approximate | 0.0102 | 0.99988543 | 0.9936 | 0.9990 |
| 25k | V0 Naive | V1 Syllabus-optimized | exact | 0 | 1.00000000 | 1.0000 | 1.0000 |
| 25k | V0 Naive | V2 Register blocking | exact | 0 | 1.00000000 | 1.0000 | 1.0000 |
| 25k | V0 Naive | V3 Async copy + double buffering | exact | 0 | 1.00000000 | 1.0000 | 1.0000 |
| 25k | V0 Naive | V4 FP16 storage | approximate | 0.0102 | 0.99988543 | 0.9936 | 0.9990 |
| 25k | V0 Naive | V5 Combined | approximate | 0.0102 | 0.99988543 | 0.9936 | 0.9990 |
| 45k (first 45k rows of 100k subset) | V4 FP16 storage | V5 Combined | approximate | 0 | 1.00000000 | 1.0000 | 1.0000 |

## Kernel resource usage

Nsight Compute: UNAVAILABLE - ==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters on the target device 0. For instructions on enabling permissions and to get more information see https://developer.nvidia.com/ERR_NVGPUCTRPERM. Hardware performance counters need administrator rights or the driver's 'allow access to GPU performance counters' setting, which this run did not change. Static values below come from the CUDA runtime instead.

| Variant | Kernel | registers_per_thread | static_shared_mem_bytes_per_block | local_mem_bytes_per_thread | max_active_blocks_per_sm | theoretical_occupancy_pct |
|---|---|---|---|---|---|---|
| V0 Naive | distance | 37 | 0 | 0 | 6 | 100 |
| V0 Naive | knn | 40 | 0 | 2048 | 6 | 100 |
| V1 Syllabus-optimized | distance | 40 | 2048 | 0 | 6 | 100 |
| V1 Syllabus-optimized | knn | 40 | 0 | 256 | 6 | 100 |
| V2 Register blocking | distance | 54 | 8704 | 0 | 4 | 66.67 |
| V2 Register blocking | knn | 40 | 0 | 256 | 6 | 100 |
| V3 Async copy + double buffering | distance | 42 | 4096 | 0 | 5 | 83.33 |
| V3 Async copy + double buffering | knn | 40 | 0 | 256 | 6 | 100 |
| V4 FP16 storage | distance | 34 | 1024 | 0 | 6 | 100 |
| V4 FP16 storage | knn | 40 | 0 | 256 | 6 | 100 |
| V5 Combined | distance | 78 | 12288 | 0 | 3 | 50 |
| V5 Combined | knn | 40 | 0 | 256 | 6 | 100 |

theoretical_occupancy_pct is ESTIMATED (resource-based), not achieved occupancy.
