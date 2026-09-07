"""
CUDA k-Nearest Neighbors Module
Milestone M3: CUDA LOF Baseline (Step 3)

Provides Python bindings via ctypes to the native CUDA k-NN kernel:
  For each sample i in [0, N-1], extracts the k nearest distinct neighbors
  (excluding self-distance at j == i) sorted in ascending order of distance:
    neighbor_distances[i, 0] <= neighbor_distances[i, 1] <= ... <= neighbor_distances[i, k-1]
"""

import os
import ctypes
import numpy as np
from typing import Tuple, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DLL_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "bin", "knn_kernel.dll"),
]

_knn_lib: Optional[ctypes.CDLL] = None


def get_knn_library() -> Optional[ctypes.CDLL]:
    """Load and return the compiled CUDA k-NN kernel DLL."""
    global _knn_lib
    if _knn_lib is not None:
        return _knn_lib

    for dll_path in DLL_CANDIDATES:
        if os.path.isfile(dll_path):
            try:
                lib = ctypes.CDLL(dll_path)
                lib.compute_knn_cuda.argtypes = [
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_int),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int,
                    ctypes.c_int,
                ]
                lib.compute_knn_cuda.restype = ctypes.c_int
                _knn_lib = lib
                return _knn_lib
            except Exception as e:
                print(f"[WARNING] Failed to load CUDA DLL at {dll_path}: {e}")

    return None


def is_knn_available() -> bool:
    """Return True if the CUDA k-NN kernel DLL is compiled and loadable."""
    return get_knn_library() is not None


def compute_knn(dist_matrix: np.ndarray, k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute k-nearest neighbors for each sample from pairwise distance matrix using CUDA.

    Args:
        dist_matrix (np.ndarray): Pairwise distance matrix of shape (N, N), float32.
        k (int): Number of nearest neighbors to retrieve (default: 20).

    Returns:
        neighbor_indices (np.ndarray): Integer array of shape (N, k).
        neighbor_distances (np.ndarray): Float32 array of shape (N, k), sorted ascending.
    """
    lib = get_knn_library()
    if lib is None:
        raise RuntimeError(
            "CUDA k-NN kernel DLL not found or failed to load. "
            "Run scripts/build_knn_kernel.bat to compile bin/knn_kernel.dll."
        )

    if not isinstance(dist_matrix, np.ndarray):
        raise TypeError(f"dist_matrix must be a numpy ndarray, got {type(dist_matrix)}")
    if dist_matrix.ndim != 2:
        raise ValueError(f"dist_matrix must be 2D of shape (N, N), got ndim={dist_matrix.ndim}")

    N, N_cols = dist_matrix.shape
    if N != N_cols:
        raise ValueError(f"dist_matrix must be square (N, N), got shape ({N}, {N_cols})")

    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    if k >= N:
        raise ValueError(f"k ({k}) must be strictly less than sample size N ({N})")
    if k > 256:
        raise ValueError(f"k ({k}) exceeds maximum supported neighborhood size (256)")

    # Ensure float32 and contiguous C memory layout
    dist_contiguous = np.ascontiguousarray(dist_matrix, dtype=np.float32)

    neighbor_indices = np.empty((N, k), dtype=np.int32)
    neighbor_distances = np.empty((N, k), dtype=np.float32)

    h_dist_ptr = dist_contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    h_idx_ptr = neighbor_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    h_dists_ptr = neighbor_distances.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    ret = lib.compute_knn_cuda(
        h_dist_ptr,
        h_idx_ptr,
        h_dists_ptr,
        ctypes.c_int(N),
        ctypes.c_int(k)
    )

    if ret != 0:
        raise RuntimeError(f"CUDA compute_knn_cuda returned error code {ret}")

    return neighbor_indices, neighbor_distances
