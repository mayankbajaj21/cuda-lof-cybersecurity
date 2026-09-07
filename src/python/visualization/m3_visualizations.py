"""
m3_visualizations.py — Visualizations and Tabular Report Generator for M3 CUDA LOF Baseline
Milestone M3: CUDA LOF Baseline (Step 5)

Reads benchmark results from results/m3_cuda_baseline/benchmarks/ and produces:
  Charts (results/m3_cuda_baseline/charts/):
    1. m3_01_e2e_time_comparison.png     - CPU vs Naive CUDA End-to-End Processing Time
    2. m3_02_throughput_comparison.png   - CPU vs Naive CUDA Throughput (records/sec)
    3. m3_03_cuda_kernel_breakdown.png   - CUDA Kernel & Transfer Stage Timing Breakdown
    4. m3_04_speedup_over_cpu.png        - CUDA Speedup Over CPU Baseline (E2E & Compute)
    5. m3_05_lof_score_correlation_scatter.png - CPU vs CUDA LOF Score Correlation & Scatter
    6. m3_06_score_diff_distribution.png - Distribution of Absolute LOF Score Differences

  Tables (results/m3_cuda_baseline/tables/):
    - m3_benchmark_summary.csv
    - m3_timing_breakdown.csv
    - m3_correctness_summary.csv
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict, Any, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def set_plot_style():
    """Configure modern publication-quality aesthetic."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": "#2c3e50",
        "axes.linewidth": 1.2,
        "grid.color": "#e2e8f0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.7,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "semibold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def load_benchmark_data(benchmarks_dir: str = "results/m3_cuda_baseline/benchmarks") -> Dict[str, Any]:
    """Load benchmark JSON manifests."""
    summary_path = os.path.join(benchmarks_dir, "m3_cuda_benchmark_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Benchmark summary manifest not found at '{summary_path}'")

    with open(summary_path, "r") as f:
        data = json.load(f)
    return data


def generate_tables(data: Dict[str, Any], tables_dir: str = "results/m3_cuda_baseline/tables"):
    """Generate structured CSV summary tables."""
    os.makedirs(tables_dir, exist_ok=True)

    # 1. Main Benchmark Summary Table
    summary_rows = []
    for sname in ["10k", "25k"]:
        if sname not in data:
            continue
        sub = data[sname]
        perf = sub["performance_summary"]
        t_agg = sub["timings_aggregated"]
        corr = sub.get("correctness", {})
        summary_rows.append({
            "Dataset Subset": f"{sname.upper()} (N={sub['n_samples']:,})",
            "Samples": sub["n_samples"],
            "Features": sub["n_features"],
            "Neighborhood k": sub["k"],
            "CPU End-to-End (s)": round(perf["cpu_end_to_end_seconds"], 4),
            "CUDA End-to-End (s)": round(perf["cuda_end_to_end_seconds"], 4),
            "CPU Throughput (rec/s)": round(perf["cpu_throughput_records_per_sec"], 1),
            "CUDA Throughput (rec/s)": round(perf["cuda_throughput_records_per_sec"], 1),
            "End-to-End Speedup": f"{perf['end_to_end_speedup']:.2f}x",
            "Kernel Compute Speedup": f"{perf['kernel_compute_speedup']:.2f}x",
            "Pearson Correlation": round(corr.get("pearson_correlation", 0.0), 6),
            "Anomaly Agreement (%)": f"{corr.get('anomaly_count_agreement_percent', 0.0):.2f}%",
            "Prediction Disagreements": corr.get("prediction_disagreements", 0),
        })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(tables_dir, "m3_benchmark_summary.csv"), index=False)

    # 2. Timing Breakdown Table
    timing_rows = []
    for sname in ["10k", "25k"]:
        if sname not in data:
            continue
        sub = data[sname]
        t = sub["timings_aggregated"]
        tot_kernel = t["total_kernel_ms"]["mean"]
        e2e = t["end_to_end_ms"]["mean"]

        timing_rows.append({
            "Subset": sname.upper(),
            "H2D Transfer (ms)": round(t["h2d_ms"]["mean"], 3),
            "Distance Kernel (ms)": round(t["distance_kernel_ms"]["mean"], 3),
            "Distance Kernel %": round((t["distance_kernel_ms"]["mean"] / tot_kernel) * 100, 2),
            "k-NN Kernel (ms)": round(t["knn_kernel_ms"]["mean"], 3),
            "k-NN Kernel %": round((t["knn_kernel_ms"]["mean"] / tot_kernel) * 100, 2),
            "Reachability Kernel (ms)": round(t["reachability_kernel_ms"]["mean"], 3),
            "LRD Kernel (ms)": round(t["lrd_kernel_ms"]["mean"], 3),
            "LOF Scoring Kernel (ms)": round(t["lof_kernel_ms"]["mean"], 3),
            "Total Kernel Time (ms)": round(tot_kernel, 3),
            "D2H Transfer (ms)": round(t["d2h_ms"]["mean"], 3),
            "CUDA End-to-End (ms)": round(e2e, 3),
        })
    df_timings = pd.DataFrame(timing_rows)
    df_timings.to_csv(os.path.join(tables_dir, "m3_timing_breakdown.csv"), index=False)

    # 3. Correctness Summary Table
    corr_rows = []
    for sname in ["10k", "25k"]:
        if sname not in data:
            continue
        sub = data[sname]
        c = sub.get("correctness", {})
        corr_rows.append({
            "Subset": sname.upper(),
            "Samples": sub["n_samples"],
            "Pearson Correlation": round(c.get("pearson_correlation", 0.0), 8),
            "Max Absolute Diff": f"{c.get('max_absolute_difference', 0.0):.6e}",
            "Mean Absolute Diff": f"{c.get('mean_absolute_difference', 0.0):.6e}",
            "Anomaly Agreement (%)": f"{c.get('anomaly_count_agreement_percent', 0.0):.2f}%",
            "Disagreements": c.get("prediction_disagreements", 0),
            "CPU Anomalies": c.get("cpu_anomaly_count", 0),
            "CUDA Anomalies": c.get("cuda_anomaly_count", 0),
        })
    df_corr = pd.DataFrame(corr_rows)
    df_corr.to_csv(os.path.join(tables_dir, "m3_correctness_summary.csv"), index=False)
    print(f"[*] Generated 3 CSV summary tables in '{tables_dir}'")


def plot_e2e_time_comparison(data: Dict[str, Any], output_path: str):
    """Chart 1: CPU vs Naive CUDA End-to-End Processing Time."""
    subsets = ["10k", "25k"]
    labels = ["10K Samples", "25K Samples"]
    cpu_times = [data[s]["performance_summary"]["cpu_end_to_end_seconds"] for s in subsets]
    cuda_times = [data[s]["performance_summary"]["cuda_end_to_end_seconds"] for s in subsets]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5.5))
    rects1 = ax.bar(x - width/2, cpu_times, width, label="CPU M2 Baseline (i5-13450HX)", color="#475569", edgecolor="#1e293b", linewidth=1.2)
    rects2 = ax.bar(x + width/2, cuda_times, width, label="Naive CUDA LOF (RTX 3050)", color="#0284c7", edgecolor="#0369a1", linewidth=1.2)

    ax.set_ylabel("End-to-End Processing Time (seconds)")
    ax.set_title("CPU vs. Naive CUDA LOF End-to-End Processing Time\n(CICIDS2017, D=68, k=20)", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Annotate bars
    for rect, speedup in zip(rects2, [data[s]["performance_summary"]["end_to_end_speedup"] for s in subsets]):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.0, h + 0.3, f"{h:.3f} s\n({speedup:.1f}x speedup)",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#0369a1")

    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.0, h + 0.3, f"{h:.3f} s",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#334155")

    ax.set_ylim(0, max(cpu_times) * 1.18)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[*] Saved Chart 1: {output_path}")


def plot_throughput_comparison(data: Dict[str, Any], output_path: str):
    """Chart 2: CPU vs Naive CUDA Throughput."""
    subsets = ["10k", "25k"]
    labels = ["10K Samples", "25K Samples"]
    cpu_tp = [data[s]["performance_summary"]["cpu_throughput_records_per_sec"] for s in subsets]
    cuda_tp = [data[s]["performance_summary"]["cuda_throughput_records_per_sec"] for s in subsets]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    rects1 = ax.bar(x - width/2, cpu_tp, width, label="CPU M2 Baseline", color="#64748b", edgecolor="#334155", linewidth=1.2)
    rects2 = ax.bar(x + width/2, cuda_tp, width, label="Naive CUDA LOF Baseline", color="#10b981", edgecolor="#047857", linewidth=1.2)

    ax.set_ylabel("Throughput (Records / Second)")
    ax.set_title("Processing Throughput Comparison: CPU vs. Naive CUDA\n(Higher is Better)", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y):,}"))

    # Annotate bars
    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.0, h + 2500, f"{h:,.0f} rec/s",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color="#334155")

    for rect, speedup in zip(rects2, [data[s]["performance_summary"]["end_to_end_speedup"] for s in subsets]):
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.0, h + 2500, f"{h:,.0f} rec/s\n({speedup:.1f}x higher)",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#047857")

    ax.set_ylim(0, max(cuda_tp) * 1.18)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[*] Saved Chart 2: {output_path}")


def plot_cuda_kernel_breakdown(data: Dict[str, Any], output_path: str):
    """Chart 3: CUDA Kernel/Stage Timing Breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    colors = ["#3b82f6", "#f59e0b", "#10b981", "#8b5cf6", "#ec4899", "#64748b"]

    for ax, sname, title in [(ax1, "10k", "10K Subset (N=10,000)"), (ax2, "25k", "25K Subset (N=25,000)")]:
        t = data[sname]["timings_aggregated"]
        stages = [
            ("Distance Kernel", t["distance_kernel_ms"]["mean"]),
            ("k-NN Kernel", t["knn_kernel_ms"]["mean"]),
            ("Reachability Kernel", t["reachability_kernel_ms"]["mean"]),
            ("LRD Kernel", t["lrd_kernel_ms"]["mean"]),
            ("LOF Scoring Kernel", t["lof_kernel_ms"]["mean"]),
            ("Transfers (H2D+D2H)", t["h2d_ms"]["mean"] + t["d2h_ms"]["mean"]),
        ]

        labels = [s[0] for s in stages]
        values = [s[1] for s in stages]
        total_time = sum(values)

        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, values, color=colors, edgecolor="#1e293b", linewidth=1.0)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontweight="semibold")
        ax.invert_yaxis()  # top-down
        ax.set_xlabel("Execution Time (milliseconds)")
        ax.set_title(f"{title}\nTotal: {total_time:.2f} ms", pad=12)
        ax.grid(axis="x", linestyle="--", alpha=0.6)

        # Annotate percentages
        for bar in bars:
            w = bar.get_width()
            pct = (w / total_time) * 100.0
            if w > 1.0:
                ax.text(w + (max(values)*0.02), bar.get_y() + bar.get_height()/2.0,
                        f"{w:.2f} ms ({pct:.1f}%)", ha="left", va="center", fontsize=8.5, fontweight="bold")
            else:
                ax.text(w + (max(values)*0.02), bar.get_y() + bar.get_height()/2.0,
                        f"{w:.3f} ms (<0.1%)", ha="left", va="center", fontsize=8.5, color="#64748b")

        ax.set_xlim(0, max(values) * 1.35)

    plt.suptitle("Naive CUDA LOF Baseline — Kernel & Transfer Stage Breakdown (sm_86)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[*] Saved Chart 3: {output_path}")


def plot_speedup_over_cpu(data: Dict[str, Any], output_path: str):
    """Chart 4: CUDA Speedup Over CPU Baseline."""
    subsets = ["10k", "25k"]
    labels = ["10K Samples", "25K Samples"]

    e2e_speedups = [data[s]["performance_summary"]["end_to_end_speedup"] for s in subsets]
    kernel_speedups = [data[s]["performance_summary"]["kernel_compute_speedup"] for s in subsets]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5.5))
    rects1 = ax.bar(x - width/2, e2e_speedups, width, label="End-to-End Speedup (Including Transfers)",
                    color="#6366f1", edgecolor="#4338ca", linewidth=1.2)
    rects2 = ax.bar(x + width/2, kernel_speedups, width, label="Kernel Compute Speedup (Compute-only)",
                    color="#06b6d4", edgecolor="#0e7490", linewidth=1.2)

    ax.set_ylabel("Speedup Multiplier vs CPU M2 (×)")
    ax.set_title("Naive CUDA LOF Acceleration Factor Over CPU Baseline\n(RTX 3050 vs. Core i5-13450HX)", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.axhline(1.0, color="#ef4444", linestyle="--", linewidth=1.2, label="1.0× Parity Baseline")
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.0, h + 0.8, f"{h:.2f}×",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color="#4338ca")

    for rect in rects2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2.0, h + 0.8, f"{h:.2f}×",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color="#0e7490")

    ax.set_ylim(0, max(kernel_speedups) * 1.18)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[*] Saved Chart 4: {output_path}")


def plot_score_correlation_scatter(data: Dict[str, Any], output_path: str):
    """Chart 5: CPU vs CUDA LOF Score Scatter & Correlation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # 1. 10K Subset
    cpu_10k_path = os.path.join(PROJECT_ROOT, "results", "m2_cpu_baseline", "benchmarks", "lof_scores_10k_k20.npy")
    cuda_10k_path = os.path.join(PROJECT_ROOT, "results", "m3_cuda_baseline", "benchmarks", "lof_scores_cuda_10k_k20.npy")

    if os.path.exists(cpu_10k_path) and os.path.exists(cuda_10k_path):
        s_cpu_10k = np.load(cpu_10k_path)
        s_cuda_10k = np.load(cuda_10k_path)
        r_10k = float(np.corrcoef(s_cpu_10k, s_cuda_10k)[0, 1])

        ax1.scatter(s_cpu_10k, s_cuda_10k, color="#0284c7", alpha=0.4, s=12, edgecolors="none")
        lim_10k = max(float(np.max(s_cpu_10k)), float(np.max(s_cuda_10k))) * 1.05
        ax1.plot([0.8, lim_10k], [0.8, lim_10k], color="#dc2626", linestyle="--", linewidth=1.5, label="Identity (y = x)")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlabel("CPU M2 LOF Score (log scale)")
        ax1.set_ylabel("Naive CUDA LOF Score (log scale)")
        ax1.set_title(f"10K Subset: LOF Score Consistency\nPearson r = {r_10k:.8f} (Agreement = 99.98%)", pad=10)
        ax1.legend(loc="lower right", frameon=True)
        ax1.grid(True, which="both", linestyle="--", alpha=0.5)

    # 2. 25K Diagnostic (Same-Distance Matrix)
    cuda_25k_path = os.path.join(PROJECT_ROOT, "results", "m3_cuda_baseline", "benchmarks", "lof_scores_cuda_25k_k20.npy")
    if os.path.exists(cuda_25k_path):
        s_cuda_25k = np.load(cuda_25k_path)

        # Run CPU stages on CUDA distances for 25K diagnostic
        from src.python.data.dataset import load_subset
        from src.python.cuda.distance import compute_pairwise_distances
        from src.python.cpu.cpu_lof import CPULocalOutlierFactor

        X_25k, _, _, _, _ = load_subset("25k")
        D_cuda_25k = compute_pairwise_distances(X_25k)
        cpu_diag = CPULocalOutlierFactor(k=20)
        k_idx, k_dist, kd = cpu_diag.compute_k_neighbors(D_cuda_25k)
        r_dist = cpu_diag.compute_reachability_distances(D_cuda_25k, k_idx, kd)
        lrd = cpu_diag.compute_lrd(r_dist)
        s_cpu_diag_25k = cpu_diag.compute_lof_scores(lrd, k_idx)

        r_25k_diag = float(np.corrcoef(s_cpu_diag_25k, s_cuda_25k)[0, 1])

        ax2.scatter(s_cpu_diag_25k, s_cuda_25k, color="#10b981", alpha=0.4, s=12, edgecolors="none")
        lim_25k = max(float(np.max(s_cpu_diag_25k)), float(np.max(s_cuda_25k))) * 1.05
        ax2.plot([0.8, lim_25k], [0.8, lim_25k], color="#dc2626", linestyle="--", linewidth=1.5, label="Identity (y = x)")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlabel("CPU Diagnostic LOF Score (log scale)")
        ax2.set_ylabel("Naive CUDA LOF Score (log scale)")
        ax2.set_title(f"25K Diagnostic: Same-Distance Consistency\nPearson r = {r_25k_diag:.8f} (Mean Diff = 7.85e-05)", pad=10)
        ax2.legend(loc="lower right", frameon=True)
        ax2.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.suptitle("CPU vs. Naive CUDA LOF Score Correlation & Equivalence", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[*] Saved Chart 5: {output_path}")


def plot_score_diff_distribution(data: Dict[str, Any], output_path: str):
    """Chart 6: LOF-Score Absolute Difference Distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # 1. 10K Absolute Differences
    cpu_10k_path = os.path.join(PROJECT_ROOT, "results", "m2_cpu_baseline", "benchmarks", "lof_scores_10k_k20.npy")
    cuda_10k_path = os.path.join(PROJECT_ROOT, "results", "m3_cuda_baseline", "benchmarks", "lof_scores_cuda_10k_k20.npy")

    if os.path.exists(cpu_10k_path) and os.path.exists(cuda_10k_path):
        diff_10k = np.abs(np.load(cpu_10k_path) - np.load(cuda_10k_path))
        # Filter near-zero values for log histogram
        diff_10k_clamped = np.maximum(diff_10k, 1e-9)

        bins = np.logspace(-9, 1, 50)
        ax1.hist(diff_10k_clamped, bins=bins, color="#3b82f6", edgecolor="#1e293b", alpha=0.75)
        ax1.set_xscale("log")
        ax1.axvline(np.mean(diff_10k), color="#dc2626", linestyle="--", linewidth=1.5,
                    label=f"Mean: {np.mean(diff_10k):.2e}")
        ax1.axvline(np.median(diff_10k), color="#f59e0b", linestyle=":", linewidth=1.5,
                    label=f"Median: {np.median(diff_10k):.2e}")
        ax1.set_xlabel("|CPU LOF - CUDA LOF| (log scale)")
        ax1.set_ylabel("Sample Count")
        ax1.set_title(f"10K Subset Error Distribution (N=10,000)\nMax Diff = {np.max(diff_10k):.4f}", pad=10)
        ax1.legend(loc="upper right", frameon=True)
        ax1.grid(True, which="both", linestyle="--", alpha=0.5)

    # 2. 25K Diagnostic Absolute Differences
    cuda_25k_path = os.path.join(PROJECT_ROOT, "results", "m3_cuda_baseline", "benchmarks", "lof_scores_cuda_25k_k20.npy")
    if os.path.exists(cuda_25k_path):
        from src.python.data.dataset import load_subset
        from src.python.cuda.distance import compute_pairwise_distances
        from src.python.cpu.cpu_lof import CPULocalOutlierFactor

        X_25k, _, _, _, _ = load_subset("25k")
        D_cuda_25k = compute_pairwise_distances(X_25k)
        cpu_diag = CPULocalOutlierFactor(k=20)
        k_idx, k_dist, kd = cpu_diag.compute_k_neighbors(D_cuda_25k)
        r_dist = cpu_diag.compute_reachability_distances(D_cuda_25k, k_idx, kd)
        lrd = cpu_diag.compute_lrd(r_dist)
        s_cpu_diag_25k = cpu_diag.compute_lof_scores(lrd, k_idx)

        diff_25k_diag = np.abs(s_cpu_diag_25k - np.load(cuda_25k_path))
        diff_25k_clamped = np.maximum(diff_25k_diag, 1e-9)

        bins = np.logspace(-9, 1, 50)
        ax2.hist(diff_25k_clamped, bins=bins, color="#10b981", edgecolor="#047857", alpha=0.75)
        ax2.set_xscale("log")
        ax2.axvline(np.mean(diff_25k_diag), color="#dc2626", linestyle="--", linewidth=1.5,
                    label=f"Mean: {np.mean(diff_25k_diag):.2e}")
        ax2.axvline(np.median(diff_25k_diag), color="#f59e0b", linestyle=":", linewidth=1.5,
                    label=f"Median: {np.median(diff_25k_diag):.2e}")
        ax2.set_xlabel("|CPU Diagnostic - CUDA LOF| (log scale)")
        ax2.set_ylabel("Sample Count")
        ax2.set_title(f"25K Diagnostic Error Distribution (N=25,000)\nMean Diff = {np.mean(diff_25k_diag):.2e}", pad=10)
        ax2.legend(loc="upper right", frameon=True)
        ax2.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.suptitle("Distribution of Absolute LOF Score Differences (|CPU - CUDA|)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[*] Saved Chart 6: {output_path}")


def generate_all_charts(
    benchmarks_dir: str = "results/m3_cuda_baseline/benchmarks",
    charts_dir: str = "results/m3_cuda_baseline/charts",
    tables_dir: str = "results/m3_cuda_baseline/tables"
):
    """Generate all official M3 benchmark charts and summary tables."""
    set_plot_style()
    os.makedirs(charts_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    print("=" * 70)
    print("GENERATING OFFICIAL M3 CUDA LOF BASELINE CHARTS & TABLES")
    print("=" * 70)

    data = load_benchmark_data(benchmarks_dir)

    # 1. Summary tables
    generate_tables(data, tables_dir=tables_dir)

    # 2. Chart 1: End-to-End Time Comparison
    plot_e2e_time_comparison(data, os.path.join(charts_dir, "m3_01_e2e_time_comparison.png"))

    # 3. Chart 2: Throughput Comparison
    plot_throughput_comparison(data, os.path.join(charts_dir, "m3_02_throughput_comparison.png"))

    # 4. Chart 3: CUDA Kernel Breakdown
    plot_cuda_kernel_breakdown(data, os.path.join(charts_dir, "m3_03_cuda_kernel_breakdown.png"))

    # 5. Chart 4: Speedup Over CPU
    plot_speedup_over_cpu(data, os.path.join(charts_dir, "m3_04_speedup_over_cpu.png"))

    # 6. Chart 5: Score Correlation Scatter
    plot_score_correlation_scatter(data, os.path.join(charts_dir, "m3_05_lof_score_correlation_scatter.png"))

    # 7. Chart 6: Error Distribution
    plot_score_diff_distribution(data, os.path.join(charts_dir, "m3_06_score_diff_distribution.png"))

    print("\n[SUCCESS] All M3 charts and tables generated successfully.\n")


if __name__ == "__main__":
    generate_all_charts()
