import sys
import os

import streamlit as st

_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from . import loaders
from .components import not_available, section

REPO_ROOT = loaders.REPO_ROOT


def _get_live_snapshot():
    try:
        from environment.hardware_info import get_full_environment_snapshot

        return get_full_environment_snapshot(repo_root=REPO_ROOT)
    except Exception as exc:
        return {"error": str(exc)}


def render():
    st.title("Hardware & Environment")
    st.caption(
        "Live detection of THIS machine, right now — separate from historical hardware metadata "
        "recorded inside saved benchmark files, which describes whatever machine produced that result."
    )

    section("Live — current machine")
    snapshot = _get_live_snapshot()
    if "error" in snapshot:
        not_available(f"hardware detection failed: {snapshot['error']}")
    else:
        c1, c2, c3 = st.columns(3)
        cpu = snapshot.get("cpu", {})
        with c1:
            st.markdown("**CPU**")
            st.write(cpu.get("model", "unavailable"))
            st.write(f"Physical cores: {cpu.get('physical_cores', 'unavailable')}")
            st.write(f"Logical cores: {cpu.get('logical_cores', 'unavailable')}")

        mem = snapshot.get("memory", {})
        with c2:
            st.markdown("**Memory**")
            st.write(f"Total RAM: {mem.get('total_ram_gb', 'unavailable')} GB")
            st.write(f"Available RAM: {mem.get('available_ram_gb', 'unavailable')} GB")

        gpu = snapshot.get("gpu", {})
        with c3:
            st.markdown("**GPU**")
            if gpu.get("available"):
                g = gpu["gpus"][0]
                st.write(g.get("name", "unavailable"))
                st.write(f"VRAM: {g.get('total_memory_mb')} MB total / {g.get('free_memory_mb')} MB free")
                st.write(f"Compute capability: {g.get('compute_capability')}")
                st.write(f"Driver: {gpu.get('driver_version')} / CUDA runtime: {gpu.get('cuda_runtime_version')}")
            else:
                st.write(f"Not available — {gpu.get('reason_unavailable', 'unknown')}")

        st.markdown("**Operating environment**")
        os_info = snapshot.get("os", {})
        py_info = snapshot.get("python", {})
        tools = snapshot.get("tools", {})
        st.write(f"OS: {os_info.get('platform_string', 'unavailable')}")
        st.write(f"Python {py_info.get('version', 'unavailable')}, NumPy {py_info.get('numpy_version', 'unavailable')}")
        tool_bits = []
        for name, info in tools.items():
            tool_bits.append(f"{name}: {'available (' + str(info.get('version')) + ')' if info.get('available') else 'not found'}")
        st.write(" | ".join(tool_bits))

        artifacts = snapshot.get("cuda_artifacts", {})
        if artifacts.get("dlls_found"):
            st.success(f"Compiled CUDA DLLs found: {', '.join(artifacts['dlls_found'])}")
        else:
            st.info("No compiled CUDA DLLs found under bin/ — GPU execution paths are unavailable until built.")

    st.markdown("---")
    section("Historical — hardware recorded in saved benchmark files")
    st.caption("Never merged with the live snapshot above — each result file speaks for its own run.")

    m2_meta = loaders.load_json("results", "m2_cpu_baseline", "benchmarks_current", "cpu_lof_10k_k20.json")
    if m2_meta:
        st.markdown("**M2 CPU baseline (benchmarks_current)**")
        st.json(m2_meta.get("hardware", m2_meta.get("environment", m2_meta)), expanded=False)
    else:
        not_available("results/m2_cpu_baseline/benchmarks_current/cpu_lof_10k_k20.json not found")

    m4_cap = loaders.load_json("results", "m4_profiling", "profiling_capabilities.json")
    if m4_cap:
        st.markdown("**M4 profiling run hardware snapshot**")
        st.json(m4_cap.get("hardware_snapshot", m4_cap), expanded=False)
    else:
        not_available("results/m4_profiling/profiling_capabilities.json not found")

    m5_meta = loaders.load_json("results", "m5_cuda_optimized", "optimization_metadata.json")
    if m5_meta:
        st.markdown("**M5 optimization run metadata**")
        st.json(m5_meta, expanded=False)
    else:
        not_available("results/m5_cuda_optimized/optimization_metadata.json not found")
