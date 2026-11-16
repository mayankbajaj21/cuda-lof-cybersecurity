import streamlit as st

from . import loaders
from .components import not_available, section

ARTIFACTS = [
    ("M2 CPU baseline (fresh)", ("results", "m2_cpu_baseline", "benchmarks_current", "cpu_lof_10k_k20.json")),
    ("M4 profiling summary", ("results", "m4_profiling", "profiling_summary.json")),
    ("M5 optimization summary", ("results", "m5_cuda_optimized", "optimization_summary.json")),
    ("M6 final benchmark summary", ("results", "m6_final_benchmarks", "final_benchmark_summary.json")),
    ("M8 advanced optimizations (same-session run)", ("results", "m8_advanced_optimizations", "benchmark_metadata.json")),
]


def render():
    st.title("Experiment History")
    st.caption("Timestamps are read from each file's own metadata where present, otherwise its last-modified time (labeled).")

    rows = []
    for label, parts in ARTIFACTS:
        data = loaders.load_json(*parts)
        mtime = loaders.file_mtime(*parts)
        if data is None:
            rows.append({"artifact": label, "timestamp": "not available", "source": "n/a"})
            continue
        ts = None
        for key in ("generated_at_utc", "timestamp", "generated_at", "finished_utc", "started_utc"):
            if isinstance(data, dict) and key in data:
                ts = data[key]
                break
        if ts:
            rows.append({"artifact": label, "timestamp": ts, "source": "recorded in file"})
        else:
            rows.append({"artifact": label, "timestamp": str(mtime), "source": "file mtime (fallback)"})

    section("Timeline")
    st.dataframe(rows, use_container_width=True)
