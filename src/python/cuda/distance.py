"""
CUDA Pairwise Euclidean Distance Module
Milestone M3: CUDA LOF Baseline (Step 2)

Provides Python bindings via ctypes to the native CUDA pairwise Euclidean distance kernel:
  d(i, j) = sqrt( sum_{d=0}^{D-1} (X[i, d] - X[j, d])^2 )
  d(i, i) = 0.0f
"""

import os
import ctypes
import numpy as np
from typing import Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DLL_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "bin", "distance_kernel.dll"),
    os.path.join(PROJECT_ROOT, "bin", "cuda_lof.dll"),
]

_cuda_lib: Optional[ctypes.CDLL] = None


def get_cuda_library() -> Optional[ctypes.CDLL]:
    """Load and return the compiled CUDA distance kernel DLL."""
    global _cuda_lib
    if _cuda_lib is not None:
        return _cuda_lib

    for dll_path in DLL_CANDIDATES:
        if os.path.isfile(dll_path):
            try:
                lib = ctypes.CDLL(dll_path)
                # Set function signature
                lib.compute_pairwise_distances_cuda.argtypes = [
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int,
                    ctypes.c_int,
                ]
                lib.compute_pairwise_distances_cuda.restype = ctypes.c_int
                _cuda_lib = lib
                return _cuda_lib
            except Exception as e:
                print(f"[WARNING] Failed to load CUDA DLL at {dll_path}: {e}")

    return None


def is_cuda_available() -> bool:
    """Return True if the CUDA distance kernel DLL is compiled and loadable."""
    return get_cuda_library() is not None


def compute_pairwise_distances(X: np.ndarray) -> np.ndarray:
    """
    Compute full pairwise Euclidean distance matrix using native CUDA kernel.

    Args:
        X (np.ndarray): 2D array of shape (N, D), float32.
                        In CICIDS2017 baseline, D=68 features.

    Returns:
        np.ndarray: Pairwise distance matrix of shape (N, N), float32.
                    Diagonal entries are guaranteed to be 0.0f.
    """
    lib = get_cuda_library()
    if lib is None:
        raise RuntimeError(
            "CUDA distance kernel DLL not found or failed to load. "
            "Run scripts/build_distance_kernel.bat to compile bin/distance_kernel.dll."
        )

    if not isinstance(X, np.ndarray):
        raise TypeError(f"X must be a numpy ndarray, got {type(X)}")
    if X.ndim != 2:
        raise ValueError(f"X must be a 2D array of shape (N, D), got ndim={X.ndim}")

    N, D = X.shape
    if N <= 0 or D <= 0:
        raise ValueError(f"Invalid dimensions: N={N}, D={D}")

    # Ensure float32 and contiguous C memory layout
    X_contiguous = np.ascontiguousarray(X, dtype=np.float32)
    dist_matrix = np.empty((N, N), dtype=np.float32)

    # Cast pointers
    h_X_ptr = X_contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    h_dist_ptr = dist_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    ret = lib.compute_pairwise_distances_cuda(h_X_ptr, h_dist_ptr, ctypes.c_int(N), ctypes.c_int(D))
    if ret != 0:
        raise RuntimeError(f"CUDA compute_pairwise_distances_cuda kernel returned error code {ret}")

    return dist_matrix
