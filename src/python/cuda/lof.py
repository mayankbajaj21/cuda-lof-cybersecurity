"""
CUDA Local Outlier Factor (LOF) Baseline Module
Milestone M3: CUDA LOF Baseline (Step 4)

Provides Python bindings via ctypes to the native CUDA LOF kernels and full pipeline:
  1. Pairwise Euclidean Distance Matrix (sm_86)
  2. k-Nearest Neighbors (k-NN) with self-distance exclusion
  3. Reachability Distance calculation
  4. Local Reachability Density (LRD)
  5. Local Outlier Factor (LOF) scoring
"""

import os
import time
import ctypes
import numpy as np
from typing import Tuple, Dict, Any, Optional

from .distance import compute_pairwise_distances, get_cuda_library
from .knn import compute_knn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DLL_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "bin", "cuda_lof.dll"),
]

_lof_lib: Optional[ctypes.CDLL] = None


class CudaLofTimings(ctypes.Structure):
    """C ctypes structure mirroring C++ CudaLofTimings for event timings."""
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


def get_lof_library() -> Optional[ctypes.CDLL]:
    """Load and return the compiled CUDA LOF library."""
    global _lof_lib
    if _lof_lib is not None:
        return _lof_lib

    for dll_path in DLL_CANDIDATES:
        if os.path.isfile(dll_path):
            try:
                lib = ctypes.CDLL(dll_path)

                # compute_reachability_distances_cuda
                lib.compute_reachability_distances_cuda.argtypes = [
                    ctypes.POINTER(ctypes.c_int),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int,
                    ctypes.c_int,
                ]
                lib.compute_reachability_distances_cuda.restype = ctypes.c_int

                # compute_lrd_cuda
                lib.compute_lrd_cuda.argtypes = [
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_float,
                ]
                lib.compute_lrd_cuda.restype = ctypes.c_int

                # compute_lof_scores_cuda
                lib.compute_lof_scores_cuda.argtypes = [
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_int),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_float,
                ]
                lib.compute_lof_scores_cuda.restype = ctypes.c_int

                # compute_lof_pipeline_cuda
                lib.compute_lof_pipeline_cuda.argtypes = [
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_float,
                ]
                lib.compute_lof_pipeline_cuda.restype = ctypes.c_int

                # compute_lof_pipeline_cuda_timed
                lib.compute_lof_pipeline_cuda_timed.argtypes = [
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_float,
                    ctypes.POINTER(CudaLofTimings),
                ]
                lib.compute_lof_pipeline_cuda_timed.restype = ctypes.c_int

                _lof_lib = lib
                return _lof_lib
            except Exception as e:
                print(f"[WARNING] Failed to load CUDA LOF DLL at {dll_path}: {e}")

    return None


def is_lof_available() -> bool:
    """Return True if the CUDA LOF DLL is compiled and loadable."""
    return get_lof_library() is not None


def compute_reachability_distances(
    knn_indices: np.ndarray,
    knn_distances: np.ndarray
) -> np.ndarray:
    """
    Compute reachability distance from each point p to all its k neighbors:
      reach-dist_k(p, o) = max(k-distance(o), d(p, o))

    Args:
        knn_indices (np.ndarray): (N, k) int32 array.
        knn_distances (np.ndarray): (N, k) float32 array.

    Returns:
        reach_dists (np.ndarray): (N, k) float32 array.
    """
    lib = get_lof_library()
    if lib is None:
        raise RuntimeError("CUDA LOF library not available. Build bin/cuda_lof.dll first.")

    N, k = knn_indices.shape
    idx_contiguous = np.ascontiguousarray(knn_indices, dtype=np.int32)
    dist_contiguous = np.ascontiguousarray(knn_distances, dtype=np.float32)
    reach_dists = np.empty((N, k), dtype=np.float32)

    ret = lib.compute_reachability_distances_cuda(
        idx_contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        dist_contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        reach_dists.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(N),
        ctypes.c_int(k),
    )
    if ret != 0:
        raise RuntimeError(f"compute_reachability_distances_cuda failed with code {ret}")

    return reach_dists


def compute_lrd(
    reach_dists: np.ndarray,
    k: int = 20,
    eps: float = 1e-10
) -> np.ndarray:
    """
    Compute Local Reachability Density (LRD) for each point:
      lrd_k(p) = 1.0 / ( ( (1/k) * sum_{m=0}^{k-1} reach_dists[p, m] ) + eps )

    Args:
        reach_dists (np.ndarray): (N, k) float32 array.
        k (int): number of neighbors.
        eps (float): numerical stability epsilon.

    Returns:
        lrd (np.ndarray): (N,) float32 array.
    """
    lib = get_lof_library()
    if lib is None:
        raise RuntimeError("CUDA LOF library not available. Build bin/cuda_lof.dll first.")

    N, k_in = reach_dists.shape
    reach_contiguous = np.ascontiguousarray(reach_dists, dtype=np.float32)
    lrd = np.empty(N, dtype=np.float32)

    ret = lib.compute_lrd_cuda(
        reach_contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        lrd.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(N),
        ctypes.c_int(k),
        ctypes.c_float(eps),
    )
    if ret != 0:
        raise RuntimeError(f"compute_lrd_cuda failed with code {ret}")

    return lrd


def compute_lof_scores(
    lrd: np.ndarray,
    knn_indices: np.ndarray,
    k: int = 20,
    eps: float = 1e-10
) -> np.ndarray:
    """
    Compute Local Outlier Factor (LOF) score for each point:
      LOF_k(p) = ( (1/k) * sum_{o in N_k(p)} lrd(o) ) / (lrd(p) + eps)

    Args:
        lrd (np.ndarray): (N,) float32 array.
        knn_indices (np.ndarray): (N, k) int32 array.
        k (int): number of neighbors.
        eps (float): numerical stability epsilon.

    Returns:
        lof_scores (np.ndarray): (N,) float32 array.
    """
    lib = get_lof_library()
    if lib is None:
        raise RuntimeError("CUDA LOF library not available. Build bin/cuda_lof.dll first.")

    N = len(lrd)
    lrd_contiguous = np.ascontiguousarray(lrd, dtype=np.float32)
    idx_contiguous = np.ascontiguousarray(knn_indices, dtype=np.int32)
    lof_scores = np.empty(N, dtype=np.float32)

    ret = lib.compute_lof_scores_cuda(
        lrd_contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        idx_contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        lof_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(N),
        ctypes.c_int(k),
        ctypes.c_float(eps),
    )
    if ret != 0:
        raise RuntimeError(f"compute_lof_scores_cuda failed with code {ret}")

    return lof_scores


def compute_cuda_lof(
    X: np.ndarray,
    k: int = 20,
    eps: float = 1e-10,
    return_intermediates: bool = False
):
    """
    Execute full CUDA LOF Baseline pipeline on feature matrix X.

    Args:
        X (np.ndarray): (N, D) continuous float32 feature matrix.
        k (int): Number of nearest neighbors (default 20).
        eps (float): Numerical stability epsilon (default 1e-10).
        return_intermediates (bool): If True, returns dict of intermediate stages.

    Returns:
        lof_scores (np.ndarray): (N,) float32 LOF scores.
        intermediates (dict, optional): If return_intermediates=True.
    """
    lib = get_lof_library()
    if lib is None:
        raise RuntimeError("CUDA LOF library not available. Build bin/cuda_lof.dll first.")

    X_contiguous = np.ascontiguousarray(X, dtype=np.float32)
    N, D = X_contiguous.shape

    if return_intermediates:
        # Step-by-step pipeline for inspection and fine-grained validation
        t0 = time.perf_counter()
        dist_matrix = compute_pairwise_distances(X_contiguous)
        t_dist = time.perf_counter() - t0

        t0 = time.perf_counter()
        knn_indices, knn_distances = compute_knn(dist_matrix, k=k)
        t_knn = time.perf_counter() - t0

        t0 = time.perf_counter()
        reach_dists = compute_reachability_distances(knn_indices, knn_distances)
        t_reach = time.perf_counter() - t0

        t0 = time.perf_counter()
        lrd = compute_lrd(reach_dists, k=k, eps=eps)
        t_lrd = time.perf_counter() - t0

        t0 = time.perf_counter()
        lof_scores = compute_lof_scores(lrd, knn_indices, k=k, eps=eps)
        t_lof = time.perf_counter() - t0

        intermediates = {
            "dist_matrix": dist_matrix,
            "knn_indices": knn_indices,
            "knn_distances": knn_distances,
            "k_distances": knn_distances[:, -1],
            "reach_dists": reach_dists,
            "lrd": lrd,
            "lof_scores": lof_scores,
            "timings": {
                "distance": t_dist,
                "knn": t_knn,
                "reachability": t_reach,
                "lrd": t_lrd,
                "lof": t_lof,
            }
        }
        return lof_scores, intermediates
    else:
        # Memory-optimized native pipeline (frees distance matrix on GPU immediately)
        lof_scores = np.empty(N, dtype=np.float32)
        h_X_ptr = X_contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        h_lof_ptr = lof_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        ret = lib.compute_lof_pipeline_cuda(
            h_X_ptr,
            h_lof_ptr,
            ctypes.c_int(N),
            ctypes.c_int(D),
            ctypes.c_int(k),
            ctypes.c_float(eps),
        )
        if ret != 0:
            raise RuntimeError(f"compute_lof_pipeline_cuda failed with code {ret}")

        return lof_scores


def benchmark_cuda_lof(
    X: np.ndarray,
    k: int = 20,
    eps: float = 1e-10
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Execute full CUDA LOF Baseline pipeline with GPU hardware event timing.

    Accurately measures kernel-only, transfer, and end-to-end execution times
    via CUDA events synchronized on the GPU device.

    Args:
        X (np.ndarray): (N, D) continuous float32 feature matrix.
        k (int): Number of nearest neighbors (default 20).
        eps (float): Numerical stability epsilon (default 1e-10).

    Returns:
        lof_scores (np.ndarray): (N,) float32 LOF scores.
        timings (dict): Accurate CUDA event timings in milliseconds and seconds.
    """
    lib = get_lof_library()
    if lib is None:
        raise RuntimeError("CUDA LOF library not available. Build bin/cuda_lof.dll first.")

    X_contiguous = np.ascontiguousarray(X, dtype=np.float32)
    N, D = X_contiguous.shape

    lof_scores = np.empty(N, dtype=np.float32)
    h_X_ptr = X_contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    h_lof_ptr = lof_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    timings_struct = CudaLofTimings()
    ret = lib.compute_lof_pipeline_cuda_timed(
        h_X_ptr,
        h_lof_ptr,
        ctypes.c_int(N),
        ctypes.c_int(D),
        ctypes.c_int(k),
        ctypes.c_float(eps),
        ctypes.byref(timings_struct),
    )
    if ret != 0:
        raise RuntimeError(f"compute_lof_pipeline_cuda_timed failed with code {ret}")

    timings = timings_struct.to_dict()
    # Also add second-denominated fields
    timings["h2d_seconds"] = timings["h2d_ms"] / 1000.0
    timings["distance_kernel_seconds"] = timings["distance_kernel_ms"] / 1000.0
    timings["knn_kernel_seconds"] = timings["knn_kernel_ms"] / 1000.0
    timings["reachability_kernel_seconds"] = timings["reachability_kernel_ms"] / 1000.0
    timings["lrd_kernel_seconds"] = timings["lrd_kernel_ms"] / 1000.0
    timings["lof_kernel_seconds"] = timings["lof_kernel_ms"] / 1000.0
    timings["d2h_seconds"] = timings["d2h_ms"] / 1000.0
    timings["total_kernel_seconds"] = timings["total_kernel_ms"] / 1000.0
    timings["end_to_end_seconds"] = timings["end_to_end_ms"] / 1000.0

    return lof_scores, timings


class CUDALocalOutlierFactor:
    """
    Native CUDA Baseline Implementation of Local Outlier Factor (LOF).
    Exact functional and numerical counterpart to CPULocalOutlierFactor.
    """

    def __init__(self, k: int = 20, eps: float = 1e-10, contamination: float = 0.1):
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        if not (0.0 < contamination < 1.0):
            raise ValueError(f"contamination must be between 0 and 1, got {contamination}")
        self.k = k
        self.eps = eps
        self.contamination = contamination

        self.n_samples_: int = 0
        self.n_features_: int = 0
        self.lof_scores_: Optional[np.ndarray] = None
        self.threshold_: Optional[float] = None
        self.predictions_: Optional[np.ndarray] = None
        self.timings_: Dict[str, float] = {}

    def fit(self, X: np.ndarray) -> "CUDALocalOutlierFactor":
        """Fit CUDA LOF on feature matrix X and compute anomaly predictions."""
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X_arr.shape}")

        self.n_samples_, self.n_features_ = X_arr.shape
        t_start = time.perf_counter()

        self.lof_scores_, intermediates = compute_cuda_lof(
            X_arr, k=self.k, eps=self.eps, return_intermediates=True
        )

        total_time = time.perf_counter() - t_start
        self.timings_ = intermediates["timings"]
        self.timings_["total"] = total_time

        # Decision threshold matching M2 CPU convention
        self.threshold_ = float(np.percentile(self.lof_scores_, 100.0 * (1.0 - self.contamination)))
        self.predictions_ = np.where(self.lof_scores_ >= self.threshold_, 1, 0).astype(np.int8)

        return self
