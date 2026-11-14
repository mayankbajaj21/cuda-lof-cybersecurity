"""Dynamic hardware and software environment detection.

Every function here degrades gracefully: on any missing tool, missing
dependency, subprocess failure, or unexpected output it returns an
"unavailable" value instead of raising. Nothing here assumes a specific
machine, OS, GPU vendor, or CUDA version.
"""

import os
import platform
import re
import subprocess
import sys

try:
    import psutil
except ImportError:
    psutil = None

try:
    import numpy
except ImportError:
    numpy = None


def _run(cmd, timeout=5.0):
    """Run a command and return (success, stdout) or (False, None)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            return False, None
        return True, result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False, None


def get_cpu_info():
    """CPU manufacturer, model, core counts, frequency."""
    info = {
        "model": platform.processor() or "unavailable",
        "architecture": platform.machine() or "unavailable",
        "physical_cores": None,
        "logical_cores": None,
        "max_frequency_mhz": None,
        "current_frequency_mhz": None,
    }

    if psutil is not None:
        try:
            info["physical_cores"] = psutil.cpu_count(logical=False)
            info["logical_cores"] = psutil.cpu_count(logical=True)
        except Exception:
            pass
        try:
            freq = psutil.cpu_freq()
            if freq is not None:
                info["max_frequency_mhz"] = round(freq.max, 2) if freq.max else None
                info["current_frequency_mhz"] = round(freq.current, 2) if freq.current else None
        except Exception:
            pass

    if platform.system() == "Windows":
        model = os.environ.get("PROCESSOR_IDENTIFIER")
        if model:
            info["model"] = model

    return info


def get_memory_info():
    """Total / available system RAM."""
    info = {
        "total_ram_bytes": None,
        "total_ram_gb": None,
        "available_ram_bytes": None,
        "available_ram_gb": None,
        "used_ram_percent": None,
    }
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            info["total_ram_bytes"] = vm.total
            info["total_ram_gb"] = round(vm.total / (1024 ** 3), 2)
            info["available_ram_bytes"] = vm.available
            info["available_ram_gb"] = round(vm.available / (1024 ** 3), 2)
            info["used_ram_percent"] = vm.percent
        except Exception:
            pass
    return info


def get_gpu_info():
    """GPU vendor/model/memory/driver via nvidia-smi. Unavailable if absent or non-NVIDIA."""
    info = {
        "available": False,
        "vendor": "unavailable",
        "gpu_count": 0,
        "gpus": [],
        "driver_version": None,
        "cuda_runtime_version": None,
        "detection_method": "nvidia-smi",
        "reason_unavailable": None,
    }

    ok, out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.free,memory.used,compute_cap,utilization.gpu,temperature.gpu,power.draw,clocks.sm",
            "--format=csv,noheader,nounits",
        ]
    )
    if not ok or not out:
        info["reason_unavailable"] = "nvidia-smi not found or returned no output (no NVIDIA GPU/driver detected)"
        return info

    gpus = []
    driver_version = None
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 10:
            continue
        name, driver, mem_total, mem_free, mem_used, compute_cap, util, temp, power, clock = parts[:10]
        driver_version = driver
        gpus.append(
            {
                "name": name,
                "total_memory_mb": _to_float(mem_total),
                "free_memory_mb": _to_float(mem_free),
                "used_memory_mb": _to_float(mem_used),
                "compute_capability": compute_cap,
                "utilization_percent": _to_float(util),
                "temperature_celsius": _to_float(temp),
                "power_draw_watts": _to_float(power),
                "sm_clock_mhz": _to_float(clock),
            }
        )

    if not gpus:
        info["reason_unavailable"] = "nvidia-smi ran but returned no parsable GPU rows"
        return info

    info["available"] = True
    info["vendor"] = "NVIDIA"
    info["gpu_count"] = len(gpus)
    info["gpus"] = gpus
    info["driver_version"] = driver_version

    cuda_ok, cuda_out = _run(["nvidia-smi"])
    if cuda_ok and cuda_out:
        match = re.search(r"CUDA Version:\s*([\d.]+)", cuda_out)
        if match:
            info["cuda_runtime_version"] = match.group(1)

    return info


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_os_info():
    return {
        "system": platform.system() or "unavailable",
        "release": platform.release() or "unavailable",
        "version": platform.version() or "unavailable",
        "platform_string": platform.platform() or "unavailable",
        "architecture": platform.machine() or "unavailable",
    }


def get_python_info():
    info = {
        "version": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "numpy_version": "unavailable",
    }
    if numpy is not None:
        info["numpy_version"] = numpy.__version__
    return info


def get_tool_versions():
    """Detect CUDA toolchain / profiling tool availability and versions."""
    tools = {}

    ok, out = _run(["nvcc", "--version"])
    tools["nvcc"] = {
        "available": ok,
        "version": _extract_version(out, r"release\s+([\d.]+)") if ok else None,
    }

    ok, out = _run(["nsys", "--version"])
    tools["nsys"] = {
        "available": ok,
        "version": _extract_version(out, r"[Vv]ersion\s+([\d.]+)") if ok else None,
    }

    ok, out = _run(["ncu", "--version"])
    tools["ncu"] = {
        "available": ok,
        "version": _extract_version(out, r"[Vv]ersion\s+([\d.]+)") if ok else None,
    }

    ok, out = _run(["nvidia-smi", "--version"])
    tools["nvidia_smi"] = {
        "available": ok,
        "version": _extract_version(out, r"version\s*:\s*([\d.]+)") if ok else None,
    }

    return tools


def _extract_version(text, pattern):
    if not text:
        return None
    match = re.search(pattern, text)
    return match.group(1) if match else None


def check_compiled_cuda_artifacts(repo_root):
    """Check whether compiled CUDA DLLs exist under <repo_root>/bin/."""
    bin_dir = os.path.join(repo_root, "bin")
    result = {
        "bin_dir_exists": os.path.isdir(bin_dir),
        "dlls_found": [],
    }
    if result["bin_dir_exists"]:
        try:
            result["dlls_found"] = sorted(
                f for f in os.listdir(bin_dir) if f.lower().endswith(".dll")
            )
        except OSError:
            pass
    return result


def get_full_environment_snapshot(repo_root=None):
    """Single entry point: everything about the current machine, JSON-serializable."""
    snapshot = {
        "cpu": get_cpu_info(),
        "memory": get_memory_info(),
        "gpu": get_gpu_info(),
        "os": get_os_info(),
        "python": get_python_info(),
        "tools": get_tool_versions(),
    }
    if repo_root is not None:
        snapshot["cuda_artifacts"] = check_compiled_cuda_artifacts(repo_root)
    return snapshot
