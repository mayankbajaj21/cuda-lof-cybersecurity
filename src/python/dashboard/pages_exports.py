import os

import streamlit as st

from . import loaders

EXPORTS = [
    ("M2 CPU benchmark CSV", ("results", "m2_cpu_baseline", "tables", "m2_cpu_benchmark.csv")),
    ("M4 profiling summary (JSON)", ("results", "m4_profiling", "profiling_summary.json")),
    ("M4 kernel timing (CSV)", ("results", "m4_profiling", "kernel_timing.csv")),
    ("M5 optimization report (Markdown)", ("results", "m5_cuda_optimized", "optimization_report.md")),
    ("M5 correctness report (JSON)", ("results", "m5_cuda_optimized", "correctness_report.json")),
    ("M6 final benchmark results (CSV)", ("results", "m6_final_benchmarks", "final_benchmark_results.csv")),
    ("M6 scalability analysis (Markdown)", ("results", "m6_final_benchmarks", "scalability_analysis.md")),
    ("M1 dataset summary (CSV)", ("results", "m1_dataset_preparation", "tables", "m1_dataset_summary.csv")),
    ("M8 variant benchmarks (CSV)", ("results", "m8_advanced_optimizations", "variant_benchmarks.csv")),
    ("M8 per-kernel breakdown (CSV)", ("results", "m8_advanced_optimizations", "kernel_breakdown.csv")),
    ("M8 correctness summary (CSV)", ("results", "m8_advanced_optimizations", "correctness_summary.csv")),
    ("M8 correctness report (JSON)", ("results", "m8_advanced_optimizations", "correctness_report.json")),
    ("M8 memory analysis (CSV)", ("results", "m8_advanced_optimizations", "memory_analysis.csv")),
    ("M8 feasibility checks (CSV)", ("results", "m8_advanced_optimizations", "feasibility_checks.csv")),
    ("M8 kernel resources / Nsight status (CSV)", ("results", "m8_advanced_optimizations", "ncu_metrics.csv")),
    ("M8 benchmark metadata (JSON)", ("results", "m8_advanced_optimizations", "benchmark_metadata.json")),
    ("M8 summary (JSON)", ("results", "m8_advanced_optimizations", "summary.json")),
    ("M8 report (Markdown)", ("results", "m8_advanced_optimizations", "report.md")),
]


def render():
    st.title("Exports & Reports")
    st.caption("Downloads existing files as-is — nothing is regenerated here.")

    for label, parts in EXPORTS:
        p = loaders.path(*parts)
        if os.path.isfile(p):
            with open(p, "rb") as f:
                st.download_button(label, f.read(), file_name=os.path.basename(p), key=label)
        else:
            st.caption(f"{label}: not available ({'/'.join(parts)})")
