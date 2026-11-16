"""
CUDA LOF Variants V0..V5 (Milestone M8 - Advanced Optimizations)

Python bindings to bin/cuda_lof_variants.dll, a separate binary from the
M3 (bin/cuda_lof.dll) and M5 (bin/cuda_lof_optimized.dll) libraries.

    V0  naive                                         exact
    V1  syllabus-optimized (M5 kernels)               exact
    V2  V1 + register blocking                        exact
    V3  V1 + cp.async double buffering                exact
    V4  V1 + FP16 storage                             approximate
    V5  register blocking + cp.async + FP16 storage   approximate
"""

import ctypes
import os
from typing import Dict, Optional, Tuple

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DLL_PATH = os.path.join(PROJECT_ROOT, "bin", "cuda_lof_variants.dll")

VARIANTS: Dict[int, Dict[str, str]] = {
    0: {
        "name": "V0 Naive",
        "technique_source": "baseline",
        "techniques": "One thread per distance, direct global-memory reads; per-thread top-k insertion sort",
        "exact_or_approximate": "exact",
    },
    1: {
        "name": "V1 Syllabus-optimized",
        "technique_source": "syllabus",
        "techniques": "Coalesced loads, 16x16 shared-memory tiling, occupancy-tuned block size, loop unrolling, "
                      "divergence removal; register-resident top-k with float4 loads",
        "exact_or_approximate": "exact",
    },
    2: {
        "name": "V2 Register blocking",
        "technique_source": "outside_syllabus",
        "techniques": "V1 + register blocking / thread coarsening (4x4 distances per thread, 64x64 tile per block)",
        "exact_or_approximate": "exact",
    },
    3: {
        "name": "V3 Async copy + double buffering",
        "technique_source": "outside_syllabus",
        "techniques": "V1 + cp.async global->shared copies with double buffering (prefetch tile t+1 during tile t)",
        "exact_or_approximate": "exact",
    },
    4: {
        "name": "V4 FP16 storage",
        "technique_source": "outside_syllabus",
        "techniques": "V1 + FP16 storage of features and distance matrix, FP32 accumulation",
        "exact_or_approximate": "approximate",
    },
    5: {
        "name": "V5 Combined",
        "technique_source": "combined_outside_syllabus",
        "techniques": "Register blocking + cp.async double buffering + FP16 storage",
        "exact_or_approximate": "approximate",
    },
}

FP16_VARIANTS = (4, 5)

_lib: Optional[ctypes.CDLL] = None


class VariantTimings(ctypes.Structure):
    _fields_ = [
        ("h2d_ms", ctypes.c_float),
        ("fp16_convert_ms", ctypes.c_float),
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
        return {name: float(getattr(self, name)) for name, _ in self._fields_}


class VariantMemStats(ctypes.Structure):
    _fields_ = [
        ("tracked_peak_bytes", ctypes.c_ulonglong),
        ("memgetinfo_peak_used_delta_bytes", ctypes.c_ulonglong),
        ("memgetinfo_free_before_bytes", ctypes.c_ulonglong),
        ("memgetinfo_total_bytes", ctypes.c_ulonglong),
        ("dist_matrix_bytes", ctypes.c_ulonglong),
        ("fp16_overflow_count", ctypes.c_uint),
        ("dist_element_bytes", ctypes.c_int),
    ]

    def to_dict(self) -> Dict[str, int]:
        return {name: int(getattr(self, name)) for name, _ in self._fields_}


class VariantKernelAttributes(ctypes.Structure):
    _fields_ = [
        ("num_regs", ctypes.c_int),
        ("static_shared_bytes", ctypes.c_int),
        ("local_bytes", ctypes.c_int),
        ("max_threads_per_block", ctypes.c_int),
        ("block_threads", ctypes.c_int),
        ("max_active_blocks_per_sm", ctypes.c_int),
        ("theoretical_occupancy_pct", ctypes.c_float),
    ]


def get_variants_library() -> Optional[ctypes.CDLL]:
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.isfile(DLL_PATH):
        return None
    try:
        lib = ctypes.CDLL(DLL_PATH)
        lib.compute_lof_pipeline_variant.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.POINTER(VariantTimings),
            ctypes.POINTER(VariantMemStats),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.compute_lof_pipeline_variant.restype = ctypes.c_int
        lib.get_variant_kernel_attributes.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.POINTER(VariantKernelAttributes)
        ]
        lib.get_variant_kernel_attributes.restype = ctypes.c_int
        lib.variant_max_k.argtypes = [ctypes.c_int]
        lib.variant_max_k.restype = ctypes.c_int
    except Exception as e:
        print(f"[WARNING] Failed to load CUDA LOF variants DLL at {DLL_PATH}: {e}")
        return None
    _lib = lib
    return _lib


def is_variants_available() -> bool:
    return get_variants_library() is not None


def _require_lib() -> ctypes.CDLL:
    lib = get_variants_library()
    if lib is None:
        raise RuntimeError("CUDA LOF variants library not available. Build bin/cuda_lof_variants.dll first "
                           "(scripts/build_cuda_lof_variants.bat).")
    return lib


def variant_max_k(variant: int) -> int:
    return int(_require_lib().variant_max_k(int(variant)))


def compute_lof_variant(
    X: np.ndarray,
    k: int = 20,
    variant: int = 1,
    eps: float = 1e-10,
    return_knn_indices: bool = False,
):
    """Run one LOF pipeline variant.

    Returns (scores, timings_dict, peak_mem_dict), plus the (N, k) neighbor
    index array as a fourth element when return_knn_indices is True.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VARIANTS)}, got {variant}")
    lib = _require_lib()
    max_k = int(lib.variant_max_k(int(variant)))
    if k > max_k:
        raise ValueError(f"variant {variant} supports k <= {max_k}, got k={k}")

    Xc = np.ascontiguousarray(X, dtype=np.float32)
    N, D = Xc.shape
    scores = np.empty(N, dtype=np.float32)
    timings = VariantTimings()
    mem = VariantMemStats()
    knn_idx = np.empty((N, k), dtype=np.int32) if return_knn_indices else None
    knn_ptr = knn_idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int)) if knn_idx is not None else None

    ret = lib.compute_lof_pipeline_variant(
        Xc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(N),
        ctypes.c_int(D),
        ctypes.c_int(k),
        ctypes.c_float(eps),
        ctypes.c_int(variant),
        ctypes.byref(timings),
        ctypes.byref(mem),
        knn_ptr,
    )
    if ret != 0:
        raise RuntimeError(f"compute_lof_pipeline_variant(variant={variant}) failed with code {ret}")

    if return_knn_indices:
        return scores, timings.to_dict(), mem.to_dict(), knn_idx
    return scores, timings.to_dict(), mem.to_dict()


def get_kernel_attributes(variant: int, kernel: str) -> Dict[str, float]:
    """Static per-kernel attributes (cudaFuncGetAttributes) and theoretical occupancy
    (cudaOccupancyMaxActiveBlocksPerMultiprocessor). kernel: 'distance' or 'knn'."""
    which = {"distance": 0, "knn": 1}[kernel]
    attrs = VariantKernelAttributes()
    ret = _require_lib().get_variant_kernel_attributes(int(variant), which, ctypes.byref(attrs))
    if ret != 0:
        raise RuntimeError(f"get_variant_kernel_attributes({variant}, {kernel}) failed with code {ret}")
    return {name: getattr(attrs, name) for name, _ in attrs._fields_}


def estimate_variant_gpu_bytes(N: int, D: int, k: int, variant: int) -> int:
    """Peak bytes the pipeline allocates: the distance matrix plus the larger of
    the feature buffer (distance stage) or the k-NN outputs (k-NN stage)."""
    knn_bytes = N * k * 8
    if variant in FP16_VARIANTS:
        dp = ((D + 7) // 8) * 8
        return N * N * 2 + max(N * dp * 2, knn_bytes) + 4
    return N * N * 4 + max(N * D * 4, knn_bytes)
