"""Determines, on the current machine, whether real GPU kernel profiling
(CUDA-event timing, Nsight Compute/Systems) can actually be performed.

Never fabricates a capability. A True/False here is a direct consequence
of what get_full_environment_snapshot() observed on this machine.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "src" / "python"))

from environment.hardware_info import get_full_environment_snapshot  # noqa: E402


def generate_profiling_capability_report(repo_root=None):
    repo_root = str(repo_root) if repo_root is not None else str(_REPO_ROOT)
    snapshot = get_full_environment_snapshot(repo_root=repo_root)

    tools = snapshot["tools"]
    gpu = snapshot["gpu"]
    artifacts = snapshot.get("cuda_artifacts", {"bin_dir_exists": False, "dlls_found": []})

    nvcc_available = tools["nvcc"]["available"]
    dll_available = artifacts["bin_dir_exists"] and len(artifacts["dlls_found"]) > 0
    nvidia_gpu_present = gpu["available"]

    limitations = []
    if not nvidia_gpu_present:
        limitations.append(
            "No NVIDIA GPU detected via nvidia-smi — GPU kernel profiling is impossible on this machine."
        )
    if not nvcc_available:
        limitations.append(
            "nvcc (CUDA Toolkit compiler) not found on PATH — CUDA kernel source cannot be compiled here."
        )
    if not dll_available:
        limitations.append(
            "No compiled CUDA LOF DLL found under bin/ — scripts/run_cuda_baseline.py has nothing to load via ctypes."
        )
    if not tools["ncu"]["available"]:
        limitations.append(
            "Nsight Compute (ncu) not found — per-kernel occupancy/register/throughput metrics cannot be collected."
        )
    if not tools["nsys"]["available"]:
        limitations.append(
            "Nsight Systems (nsys) not found — timeline-level GPU/CPU/transfer tracing cannot be collected."
        )

    can_run_gpu_profiling = nvidia_gpu_present and nvcc_available and dll_available

    report = {
        "generated_for": "M4 GPU Profiling and Performance Analysis",
        "hardware_snapshot": snapshot,
        "profiling_tools": {
            "nvcc": tools["nvcc"],
            "nsys": tools["nsys"],
            "ncu": tools["ncu"],
            "nvidia_smi": tools["nvidia_smi"],
        },
        "cuda_dll_present": dll_available,
        "cuda_dll_files": artifacts["dlls_found"],
        "can_run_gpu_profiling": can_run_gpu_profiling,
        "can_run_cuda_event_timing": can_run_gpu_profiling,
        "can_run_nsight_compute": can_run_gpu_profiling and tools["ncu"]["available"],
        "can_run_nsight_systems": can_run_gpu_profiling and tools["nsys"]["available"],
        "limitations": limitations,
    }
    return report
