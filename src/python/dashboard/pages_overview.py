import streamlit as st

from . import loaders
from .components import metric_row, not_available, synthetic_dataset_banner

MILESTONES = [
    ("M0", "Repo/environment setup", "Done"),
    ("M1", "Dataset preparation", "Done (synthetic placeholder on this machine)"),
    ("M2", "CPU LOF baseline", "Done — real"),
    ("M3", "CUDA LOF baseline", "Done — real"),
    ("M4", "CUDA profiling", "Done — real (Nsight tools unavailable)"),
    ("M5", "CUDA optimization", "Done — real"),
    ("M6", "Final benchmarks", "Done at 10K-25K; 100K-1M infeasible (VRAM/RAM wall)"),
    ("M7", "Analysis & dashboard", "Done (this page)"),
]


def render():
    st.title("CUDA-Optimized Local Outlier Factor")
    st.caption(
        "Local Outlier Factor anomaly detection for cybersecurity, implemented as a CPU "
        "NumPy reference and a native CUDA pipeline, benchmarked against each other on real hardware."
    )
    synthetic_dataset_banner()

    st.markdown(
        "**Why LOF**: unsupervised density-based anomaly detection — a point is flagged by comparing "
        "its local density to its neighbors', no attack labels required. **Why GPU**: LOF's pairwise-"
        "distance and k-NN stages are O(N²), dominating >99% of CPU runtime; both are embarrassingly "
        "parallel and map cleanly onto CUDA threads."
    )

    st.markdown("---")
    st.subheader("Key measured metrics")

    m6 = loaders.load_csv("results", "m6_final_benchmarks", "final_benchmark_results.csv")
    m1_summary = loaders.load_csv("results", "m1_dataset_preparation", "tables", "m1_dataset_summary.csv")

    items = []
    if m1_summary is not None:
        try:
            row = m1_summary[m1_summary["Metric"] == "Cleaned Samples"]
            dataset_rows = row["Result"].iloc[0] if not row.empty else "n/a"
        except Exception:
            dataset_rows = "n/a"
        items.append(("Dataset rows (cleaned)", str(dataset_rows), "From results/m1_dataset_preparation"))
    else:
        items.append(("Dataset rows (cleaned)", "n/a", None))

    if m6 is not None and not m6.empty:
        try:
            cpu_25k = m6[(m6["subset"] == "25k") & (m6["implementation"] == "cpu")]["end_to_end_mean_ms"].iloc[0]
            opt_25k = m6[(m6["subset"] == "25k") & (m6["implementation"] == "optimized_cuda")][
                "end_to_end_mean_ms"
            ].iloc[0]
            best_speedup = cpu_25k / opt_25k
            items.append(("Best measured speedup (25K)", f"{best_speedup:.1f}x", "Optimized CUDA vs. CPU"))
        except Exception:
            items.append(("Best measured speedup", "n/a", None))
        largest_feasible = m6["n_samples"].max()
        items.append(("Largest feasible benchmark size", f"{largest_feasible:,}", "VRAM/RAM-limited"))
    else:
        items.append(("Best measured speedup", "n/a", "results/m6_final_benchmarks not found"))
        items.append(("Largest feasible benchmark size", "n/a", None))

    gpu_info = loaders.load_json("results", "m4_profiling", "profiling_capabilities.json")
    backends = 1
    if gpu_info and gpu_info.get("hardware", {}).get("gpu", {}).get("available"):
        backends = 2
    items.append(("Execution backends on this machine", str(backends), "CPU always available; GPU if CUDA DLL built"))

    metric_row(items)

    st.markdown("---")
    st.subheader("Milestone status")
    st.table(
        {
            "Milestone": [m[0] for m in MILESTONES],
            "Description": [m[1] for m in MILESTONES],
            "Status": [m[2] for m in MILESTONES],
        }
    )
