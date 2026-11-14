import streamlit as st

from . import loaders
from .components import not_available, section


def render():
    st.title("Memory & Scalability (M6)")

    section("Live memory estimator")
    st.caption("Client-side N²×4-byte calculation — not a live benchmark run.")
    n = st.number_input("Hypothetical dataset size (N rows)", min_value=1000, max_value=5_000_000, value=100_000, step=10_000)
    k = st.number_input("k (neighbors)", min_value=1, max_value=256, value=20)
    distance_bytes = n * n * 4
    pipeline_bytes = distance_bytes + n * k * 4 * 2 + n * 4 * 2
    st.write(f"Dense N×N distance matrix: **{distance_bytes / (1024**3):.2f} GB**")
    st.write(f"Estimated total pipeline memory: **{pipeline_bytes / (1024**3):.2f} GB**")

    import sys
    import os

    _src = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _src not in sys.path:
        sys.path.insert(0, _src)
    try:
        from environment.hardware_info import get_gpu_info, get_memory_info

        gpu = get_gpu_info()
        mem = get_memory_info()
        free_vram_gb = gpu["gpus"][0]["free_memory_mb"] / 1024 if gpu.get("available") else None
        avail_ram_gb = mem.get("available_ram_gb")
        if free_vram_gb is not None:
            ok = "fits" if distance_bytes / (1024**3) <= free_vram_gb else "DOES NOT FIT"
            st.write(f"Current free VRAM: {free_vram_gb:.2f} GB — this configuration **{ok}**")
        if avail_ram_gb is not None:
            ok = "fits" if distance_bytes / (1024**3) <= avail_ram_gb else "DOES NOT FIT"
            st.write(f"Current available RAM: {avail_ram_gb:.2f} GB — this configuration **{ok}**")
    except Exception:
        st.caption("Live VRAM/RAM comparison unavailable.")

    st.markdown("---")
    section("Measured feasibility table (M6)")
    mem_csv = loaders.load_csv("results", "m6_final_benchmarks", "memory_analysis.csv")
    if mem_csv is not None:
        st.dataframe(mem_csv, use_container_width=True)
    else:
        not_available("results/m6_final_benchmarks/memory_analysis.csv not found")

    section("Charts")
    for name in ["m6_04_memory_vs_vram_ceiling.png", "m6_03_runtime_scaling_loglog.png"]:
        img = loaders.load_image_path("results", "m6_final_benchmarks", "charts", name)
        if img:
            st.image(img, caption=name, use_container_width=True)

    section("Scalability analysis")
    md = loaders.load_text("results", "m6_final_benchmarks", "scalability_analysis.md")
    if md:
        st.markdown(md)
    else:
        not_available("results/m6_final_benchmarks/scalability_analysis.md not found")
