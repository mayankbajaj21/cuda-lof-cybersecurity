"""
CUDA Optimized Local Outlier Factor (LOF) Module
Milestone M5: Profiling-Driven CUDA Optimization

Python bindings to bin/cuda_lof_optimized.dll - a SEPARATE binary from
bin/cuda_lof.dll (the naive M3 baseline). Both remain available so callers
can explicitly choose which implementation to run and compare:

    from src.python.cuda.lof import compute_cuda_lof, benchmark_cuda_lof
    from src.python.cuda.lof_optimized import compute_cuda_lof_optimized, benchmark_cuda_lof_optimized

Optimizations applied (see src/cuda/kernels_optimized/*.cuh for the full
M4-profiling-driven rationale behind each):
  1. Pairwise distance: shared-memory 32x32 tiling (naive kernel was
     78.9-80.0% of kernel time at N=10K/25K per M4).
  2. k-NN: right-sized top-k register buffer (32 vs. the naive kernel's
     256, eliminating local-memory spill) + float4-vectorized row scan
     (naive kernel was 19.9-20.9% of kernel time per M4).
  3. Reachability / LRD / LOF-scoring: UNCHANGED from the naive kernels -
     M4 measured these as sub-millisecond and together under 0.2% of
     kernel time, so no optimization was attempted there.
"""

import os
import ctypes
import numpy as np
from typing import Dict, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DLL_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "bin", "cuda_lof_optimized.dll"),
]

# OPT_KNN_MAX_K from src/cuda/kernels_optimized/knn_kernel_optimized.cuh -
# the optimized k-NN kernel only supports k <= this value.
OPTIMIZED_MAX_K = 32

_lof_opt_lib: Optional[ctypes.CDLL] = None


class CudaLofTimingsOptimized(ctypes.Structure):
    """Mirrors C++ CudaLofTimingsOptimized (identical field layout to the naive CudaLofTimings)."""
    _fields_ = [
        ("h2d_ms", ctypes.c_float),
        ("distance_kernel_ms", ctypes.c_float),
        ("knn_kernel_ms", ctypes.c_float),
        ("reachability_kernel_ms", ctypes.c_float),
        ("lrd_kernel_ms", ctypes.c_float),
        ("lof_kernel_ms", ctypes.c_float),
        ("d2h_ms", ctypes.c_float),
        ("total_kernel_ms", ctypes.c_float),
        ("end_to_end_ms", ctypes.c_float),
    ]

    def to_dict(self) -> Dict[str, float]:
        return {
            "h2d_ms": float(self.h2d_ms),
            "distance_kernel_ms": float(self.distance_kernel_ms),
            "knn_kernel_ms": float(self.knn_kernel_ms),
            "reachability_kernel_ms": float(self.reachability_kernel_ms),
            "lrd_kernel_ms": float(self.lrd_kernel_ms),
            "lof_kernel_ms": float(self.lof_kernel_ms),
            "d2h_ms": float(self.d2h_ms),
            "total_kernel_ms": float(self.total_kernel_ms),
            "end_to_end_ms": float(self.end_to_end_ms),
        }


def get_lof_optimized_library() -> Optional[ctypes.CDLL]:
    """Load and return the compiled optimized CUDA LOF library."""
    global _lof_opt_lib
    if _lof_opt_lib is not None:
        return _lof_opt_lib

    for dll_path in DLL_CANDIDATES:
        if os.path.isfile(dll_path):
            try:
                lib = ctypes.CDLL(dll_path)

                lib.compute_pairwise_distances_cuda_optimized.argtypes = [
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int,
                    ctypes.c_int,
                ]
                lib.compute_pairwise_distances_cuda_optimized.restype = ctypes.c_int

                lib.compute_knn_cuda_optimized.argtypes = [
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_int),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int,
                    ctypes.c_int,
                ]
                lib.compute_knn_cuda_optimized.restype = ctypes.c_int

                lib.compute_lof_pipeline_cuda_optimized.argtypes = [
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_float,
                ]
                lib.compute_lof_pipeline_cuda_optimized.restype = ctypes.c_int

                lib.compute_lof_pipeline_cuda_optimized_timed.argtypes = [
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_float,
                    ctypes.POINTER(CudaLofTimingsOptimized),
                ]
                lib.compute_lof_pipeline_cuda_optimized_timed.restype = ctypes.c_int

                _lof_opt_lib = lib
                return _lof_opt_lib
            except Exception as e:
                print(f"[WARNING] Failed to load optimized CUDA LOF DLL at {dll_path}: {e}")

    return None


def is_lof_optimized_available() -> bool:
    """Return True if the optimized CUDA LOF DLL is compiled and loadable."""
    return get_lof_optimized_library() is not None


def compute_pairwise_distances_optimized(X: np.ndarray) -> np.ndarray:
    """Standalone optimized distance-matrix call, for direct naive-vs-optimized comparison."""
    lib = get_lof_optimized_library()
    if lib is None:
        raise RuntimeError("Optimized CUDA LOF library not available. Build bin/cuda_lof_optimized.dll first.")

    X_contiguous = np.ascontiguousarray(X, dtype=np.float32)
    N, D = X_contiguous.shape
    dist_matrix = np.empty((N, N), dtype=np.float32)

    ret = lib.compute_pairwise_distances_cuda_optimized(
        X_contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        dist_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(N),
        ctypes.c_int(D),
    )
    if ret != 0:
        raise RuntimeError(f"compute_pairwise_distances_cuda_optimized failed with code {ret}")

    return dist_matrix


def compute_knn_optimized(dist_matrix: np.ndarray, k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Standalone optimized k-NN call, for direct naive-vs-optimized comparison."""
    if k > OPTIMIZED_MAX_K:
        raise ValueError(f"Optimized k-NN kernel supports k <= {OPTIMIZED_MAX_K}, got k={k}")

    lib = get_lof_optimized_library()
    if lib is None:
        raise RuntimeError("Optimized CUDA LOF library not available. Build bin/cuda_lof_optimized.dll first.")

    dist_contiguous = np.ascontiguousarray(dist_matrix, dtype=np.float32)
    N = dist_contiguous.shape[0]
    neighbor_indices = np.empty((N, k), dtype=np.int32)
    neighbor_distances = np.empty((N, k), dtype=np.float32)

    ret = lib.compute_knn_cuda_optimized(
        dist_contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        neighbor_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        neighbor_distances.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(N),
        ctypes.c_int(k),
    )
    if ret != 0:
        raise RuntimeError(f"compute_knn_cuda_optimized failed with code {ret}")

    return neighbor_indices, neighbor_distances


def compute_cuda_lof_optimized(X: np.ndarray, k: int = 20, eps: float = 1e-10) -> np.ndarray:
    """Run the full optimized CUDA LOF pipeline, returning only final LOF scores."""
    if k > OPTIMIZED_MAX_K:
        raise ValueError(f"Optimized pipeline supports k <= {OPTIMIZED_MAX_K}, got k={k}")

    lib = get_lof_optimized_library()
    if lib is None:
        raise RuntimeError("Optimized CUDA LOF library not available. Build bin/cuda_lof_optimized.dll first.")

    X_contiguous = np.ascontiguousarray(X, dtype=np.float32)
    N, D = X_contiguous.shape
    lof_scores = np.empty(N, dtype=np.float32)

    ret = lib.compute_lof_pipeline_cuda_optimized(
        X_contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        lof_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(N),
        ctypes.c_int(D),
        ctypes.c_int(k),
        ctypes.c_float(eps),
    )
    if ret != 0:
        raise RuntimeError(f"compute_lof_pipeline_cuda_optimized failed with code {ret}")

    return lof_scores


def benchmark_cuda_lof_optimized(
    X: np.ndarray,
    k: int = 20,
    eps: float = 1e-10
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Run the full optimized CUDA LOF pipeline with CUDA-event hardware timing."""
    if k > OPTIMIZED_MAX_K:
        raise ValueError(f"Optimized pipeline supports k <= {OPTIMIZED_MAX_K}, got k={k}")

    lib = get_lof_optimized_library()
    if lib is None:
        raise RuntimeError("Optimized CUDA LOF library not available. Build bin/cuda_lof_optimized.dll first.")

    X_contiguous = np.ascontiguousarray(X, dtype=np.float32)
    N, D = X_contiguous.shape
    lof_scores = np.empty(N, dtype=np.float32)

    timings_struct = CudaLofTimingsOptimized()
    ret = lib.compute_lof_pipeline_cuda_optimized_timed(
        X_contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        lof_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(N),
        ctypes.c_int(D),
        ctypes.c_int(k),
        ctypes.c_float(eps),
        ctypes.byref(timings_struct),
    )
    if ret != 0:
        raise RuntimeError(f"compute_lof_pipeline_cuda_optimized_timed failed with code {ret}")

    timings = timings_struct.to_dict()
    for stage in ("h2d", "distance_kernel", "knn_kernel", "reachability_kernel",
                  "lrd_kernel", "lof_kernel", "d2h", "total_kernel", "end_to_end"):
        timings[f"{stage}_seconds"] = timings[f"{stage}_ms"] / 1000.0

    return lof_scores, timings
