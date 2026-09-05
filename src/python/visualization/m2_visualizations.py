"""
Milestone M2 Visualizations & Tabular Results Generator
Milestone M2: Naive CPU LOF Baseline

Reads actual CPU LOF experiment results from results/m2_cpu_baseline/ and generates:
- Table 4: CPU LOF Benchmark (CSV)
- Table 5: LOF Score Statistics (CSV)
- Table 6: CPU LOF Runtime Breakdown (CSV)
- Chart 4: CPU LOF Execution Time (PNG)
- Chart 5: CPU LOF Runtime Breakdown — 25K (PNG)
- Chart 6: CPU LOF Scaling (PNG)
- Chart 7: Pairwise Distance vs k-NN Search Time (PNG)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict, Any, Tuple


def set_plot_style():
    """Configure clean publication-quality aesthetic."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
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


def load_m2_results(results_dir: str = "results/m2_cpu_baseline") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load actual M2 JSON experiment result manifests."""
    candidates_10k = [
        os.path.join(results_dir, "benchmarks", "cpu_lof_10k_k20.json"),
        os.path.join(results_dir, "cpu_lof_10k_k20.json"),
    ]
    candidates_25k = [
        os.path.join(results_dir, "benchmarks", "cpu_lof_25k_k20.json"),
        os.path.join(results_dir, "cpu_lof_25k_k20.json"),
    ]

    path_10k = next((p for p in candidates_10k if os.path.exists(p)), None)
    path_25k = next((p for p in candidates_25k if os.path.exists(p)), None)

    if not path_10k:
        raise FileNotFoundError(f"Missing M2 result file for 10k in: {candidates_10k}")
    if not path_25k:
        raise FileNotFoundError(f"Missing M2 result file for 25k in: {candidates_25k}")

    with open(path_10k, "r") as f:
        res_10k = json.load(f)
    with open(path_25k, "r") as f:
        res_25k = json.load(f)

    return res_10k, res_25k


def generate_m2_tables(res_10k: Dict[str, Any], res_25k: Dict[str, Any], output_dir: str) -> Dict[str, pd.DataFrame]:
    """Generate CSV tables from actual M2 experiment outputs."""
    os.makedirs(output_dir, exist_ok=True)

    t_10k = res_10k["timings"]
    t_25k = res_25k["timings"]
    s_10k = res_10k["score_statistics"]
    s_25k = res_25k["score_statistics"]

    # -------------------------------------------------------------
    # TABLE 4: CPU LOF Benchmark
    # -------------------------------------------------------------
    env = res_10k.get("environment", {})
    cpu = env.get("cpu", {})
    mem = env.get("memory", {})
    os_data = env.get("os", {})
    py = env.get("python", {})

    table4_data = [
        {"Metric": "CPU Model", "10K": cpu.get("model", "N/A"), "25K": cpu.get("model", "N/A")},
        {"Metric": "Physical CPU Cores", "10K": str(cpu.get("physical_cores", "N/A")), "25K": str(cpu.get("physical_cores", "N/A"))},
        {"Metric": "Logical Processors", "10K": str(cpu.get("logical_processors", "N/A")), "25K": str(cpu.get("logical_processors", "N/A"))},
        {"Metric": "Host RAM (GB)", "10K": f"{mem.get('total_ram_gb', 'N/A')}", "25K": f"{mem.get('total_ram_gb', 'N/A')}"},
        {"Metric": "Operating System", "10K": f"{os_data.get('system', 'Windows')} {os_data.get('release', '')}", "25K": f"{os_data.get('system', 'Windows')} {os_data.get('release', '')}"},
        {"Metric": "Python Version", "10K": py.get("version", "N/A"), "25K": py.get("version", "N/A")},
        {"Metric": "Samples (N)", "10K": f"{res_10k['n_samples']:,}", "25K": f"{res_25k['n_samples']:,}"},
        {"Metric": "Features (D)", "10K": str(res_10k["n_features"]), "25K": str(res_25k["n_features"])},
        {"Metric": "k (Nearest Neighbors)", "10K": str(res_10k["k"]), "25K": str(res_25k["k"])},
        {"Metric": "Contamination", "10K": str(res_10k["contamination"]), "25K": str(res_25k["contamination"])},
        {"Metric": "Data Loading Time (s)", "10K": f"{t_10k['data_loading_seconds']:.5f}", "25K": f"{t_25k['data_loading_seconds']:.5f}"},
        {"Metric": "Pairwise Distance Matrix (s)", "10K": f"{t_10k['distance_matrix_seconds']:.5f}", "25K": f"{t_25k['distance_matrix_seconds']:.5f}"},
        {"Metric": "k-NN Search & Sort (s)", "10K": f"{t_10k['knn_search_seconds']:.5f}", "25K": f"{t_25k['knn_search_seconds']:.5f}"},
        {"Metric": "Reachability Distance (s)", "10K": f"{t_10k['reachability_distance_seconds']:.5f}", "25K": f"{t_25k['reachability_distance_seconds']:.5f}"},
        {"Metric": "LRD Computation (s)", "10K": f"{t_10k['lrd_computation_seconds']:.5f}", "25K": f"{t_25k['lrd_computation_seconds']:.5f}"},
        {"Metric": "LOF Scoring (s)", "10K": f"{t_10k['lof_scoring_seconds']:.5f}", "25K": f"{t_25k['lof_scoring_seconds']:.5f}"},
        {"Metric": "Total LOF Compute Time (s)", "10K": f"{t_10k['total_lof_compute_seconds']:.5f}", "25K": f"{t_25k['total_lof_compute_seconds']:.5f}"},
        {"Metric": "Total End-to-End Time (s)", "10K": f"{t_10k['total_execution_seconds']:.5f}", "25K": f"{t_25k['total_execution_seconds']:.5f}"}
    ]
    df_table4 = pd.DataFrame(table4_data)
    t4_path = os.path.join(output_dir, "m2_cpu_benchmark.csv")
    df_table4.to_csv(t4_path, index=False)
    print(f"[*] Generated Table 4: {t4_path}")

    # -------------------------------------------------------------
    # TABLE 5: LOF Score Statistics
    # -------------------------------------------------------------
    table5_data = [
        {"Metric": "Minimum LOF Score", "10K": f"{s_10k['min']:.5f}", "25K": f"{s_25k['min']:.5f}"},
        {"Metric": "Maximum LOF Score", "10K": f"{s_10k['max']:.5f}", "25K": f"{s_25k['max']:.5f}"},
        {"Metric": "Mean LOF Score", "10K": f"{s_10k['mean']:.5f}", "25K": f"{s_25k['mean']:.5f}"},
        {"Metric": "Standard Deviation", "10K": f"{s_10k['std']:.5f}", "25K": f"{s_25k['std']:.5f}"},
        {"Metric": "Median (p50)", "10K": f"{s_10k['median']:.5f}", "25K": f"{s_25k['median']:.5f}"},
        {"Metric": "25th Percentile (p25)", "10K": f"{s_10k['p25']:.5f}", "25K": f"{s_25k['p25']:.5f}"},
        {"Metric": "75th Percentile (p75)", "10K": f"{s_10k['p75']:.5f}", "25K": f"{s_25k['p75']:.5f}"},
        {"Metric": "90th Percentile (p90)", "10K": f"{s_10k['p90']:.5f}", "25K": f"{s_25k['p90']:.5f}"},
        {"Metric": "95th Percentile (p95)", "10K": f"{s_10k['p95']:.5f}", "25K": f"{s_25k['p95']:.5f}"},
        {"Metric": "99th Percentile (p99)", "10K": f"{s_10k['p99']:.5f}", "25K": f"{s_25k['p99']:.5f}"},
        {"Metric": "Decision Threshold (tau)", "10K": f"{res_10k['threshold']:.5f}", "25K": f"{res_25k['threshold']:.5f}"},
        {"Metric": "Predicted Inliers", "10K": f"{res_10k['predicted_inliers']:,}", "25K": f"{res_25k['predicted_inliers']:,}"},
        {"Metric": "Predicted Anomalies", "10K": f"{res_10k['predicted_anomalies']:,}", "25K": f"{res_25k['predicted_anomalies']:,}"}
    ]
    df_table5 = pd.DataFrame(table5_data)
    t5_path = os.path.join(output_dir, "m2_lof_statistics.csv")
    df_table5.to_csv(t5_path, index=False)
    print(f"[*] Generated Table 5: {t5_path}")

    # -------------------------------------------------------------
    # TABLE 6: CPU LOF Runtime Breakdown
    # -------------------------------------------------------------
    tot_10k = t_10k["total_lof_compute_seconds"]
    tot_25k = t_25k["total_lof_compute_seconds"]

    stages = [
        ("Pairwise Distance Matrix", t_10k["distance_matrix_seconds"], t_25k["distance_matrix_seconds"]),
        ("k-NN Search & Sort", t_10k["knn_search_seconds"], t_25k["knn_search_seconds"]),
        ("Reachability Distance", t_10k["reachability_distance_seconds"], t_25k["reachability_distance_seconds"]),
        ("LRD Computation", t_10k["lrd_computation_seconds"], t_25k["lrd_computation_seconds"]),
        ("LOF Scoring", t_10k["lof_scoring_seconds"], t_25k["lof_scoring_seconds"]),
    ]

    table6_data = []
    for name, v10, v25 in stages:
        pct10 = (v10 / tot_10k) * 100
        pct25 = (v25 / tot_25k) * 100
        table6_data.append({
            "Stage": name,
            "10K Time (s)": f"{v10:.5f}",
            "10K %": f"{pct10:.2f}%",
            "25K Time (s)": f"{v25:.5f}",
            "25K %": f"{pct25:.2f}%"
        })

    table6_data.append({
        "Stage": "Total LOF Compute",
        "10K Time (s)": f"{tot_10k:.5f}",
        "10K %": "100.00%",
        "25K Time (s)": f"{tot_25k:.5f}",
        "25K %": "100.00%"
    })

    df_table6 = pd.DataFrame(table6_data)
    t6_path = os.path.join(output_dir, "m2_runtime_breakdown.csv")
    df_table6.to_csv(t6_path, index=False)
    print(f"[*] Generated Table 6: {t6_path}")

    return {"benchmark": df_table4, "statistics": df_table5, "breakdown": df_table6}


def generate_m2_charts(res_10k: Dict[str, Any], res_25k: Dict[str, Any], output_dir: str):
    """Generate presentation-quality PNG charts for M2."""
    os.makedirs(output_dir, exist_ok=True)
    set_plot_style()

    t_10k = res_10k["timings"]
    t_25k = res_25k["timings"]

    # =============================================================
    # CHART 4: CPU LOF Execution Time (10K vs 25K)
    # =============================================================
    fig, ax = plt.subplots(figsize=(7, 5.5))

    subsets = ["10K Subset\n(N = 10,000)", "25K Subset\n(N = 25,000)"]
    compute_times = [t_10k["total_lof_compute_seconds"], t_25k["total_lof_compute_seconds"]]
    colors = ["#2b5c8f", "#c0392b"]

    bars = ax.bar(subsets, compute_times, color=colors, width=0.45, edgecolor="#222222", linewidth=1.1)
    ax.set_ylabel("Compute Time (seconds)", fontsize=11)
    ax.set_title("Naive CPU LOF Execution Time Comparison (k = 20)", fontsize=13, pad=12)
    ax.grid(axis="y", alpha=0.4)
    ax.set_ylim(0, max(compute_times) * 1.25)

    scaling_ratio = t_25k["total_lof_compute_seconds"] / t_10k["total_lof_compute_seconds"]

    for bar, val in zip(bars, compute_times):
        height = bar.get_height()
        ax.annotate(
            f"{val:.3f} s",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    # Scaling annotation
    ax.text(
        0.05, 0.90,
        f"Dataset Scaling: 2.5x increase in samples\nCPU Runtime Scaling: {scaling_ratio:.2f}x increase\nEmpirical Scaling: ~O(N^2)",
        transform=ax.transAxes,
        fontsize=9.5,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.9)
    )

    chart4_path = os.path.join(output_dir, "m2_cpu_execution_time.png")
    plt.savefig(chart4_path)
    plt.close()
    print(f"[*] Saved Chart 4: {chart4_path}")

    # =============================================================
    # CHART 5: CPU LOF Runtime Breakdown — 25K (Horizontal Bar / Donut)
    # =============================================================
    fig, ax = plt.subplots(figsize=(9, 5))

    stages = [
        "LOF Scoring",
        "LRD Computation",
        "Reachability Distance",
        "Pairwise Distance Matrix",
        "k-NN Search & Sort"
    ]
    raw_times_25k = [
        t_25k["lof_scoring_seconds"],
        t_25k["lrd_computation_seconds"],
        t_25k["reachability_distance_seconds"],
        t_25k["distance_matrix_seconds"],
        t_25k["knn_search_seconds"]
    ]
    tot_compute_25k = t_25k["total_lof_compute_seconds"]
    pcts_25k = [(v / tot_compute_25k) * 100 for v in raw_times_25k]

    colors = ["#95a5a6", "#7f8c8d", "#e67e22", "#2980b9", "#d35400"]
    bars = ax.barh(stages, pcts_25k, color=colors, edgecolor="#222222", height=0.55, linewidth=0.9)

    ax.set_xlabel("Percentage of Total Compute Time (%)", fontsize=11)
    ax.set_title("CPU LOF Runtime Breakdown — 25K Subset (Total: 22.14s)", fontsize=13, pad=12)
    ax.set_xlim(0, 70)
    ax.grid(axis="x", alpha=0.4)

    for bar, val, raw_s in zip(bars, pcts_25k, raw_times_25k):
        width = bar.get_width()
        label_text = f"{val:.2f}%  ({raw_s:.4f} s)" if val >= 0.05 else f"<0.05%  ({raw_s:.4f} s)"
        ax.text(
            width + 1.0,
            bar.get_y() + bar.get_height() / 2,
            label_text,
            va="center",
            ha="left",
            fontsize=9,
            fontweight="semibold"
        )

    # Bottleneck highlight box
    dist_knn_pct = (t_25k["distance_matrix_seconds"] + t_25k["knn_search_seconds"]) / tot_compute_25k * 100
    ax.text(
        0.55, 0.35,
        f"CRITICAL BOTTLENECK IDENTIFIED:\nDistance Matrix + k-NN Sort = {dist_knn_pct:.2f}%\nof total CPU execution time.\nPrimary target for CUDA optimization.",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff3cd", edgecolor="#ffeeba", alpha=0.95)
    )

    chart5_path = os.path.join(output_dir, "m2_runtime_breakdown_25k.png")
    plt.savefig(chart5_path)
    plt.close()
    print(f"[*] Saved Chart 5: {chart5_path}")

    # =============================================================
    # CHART 6: CPU LOF Scaling Curve (10K vs 25K Measured)
    # =============================================================
    fig, ax = plt.subplots(figsize=(8, 5.5))

    measured_n = [10_000, 25_000]
    measured_time = [t_10k["total_lof_compute_seconds"], t_25k["total_lof_compute_seconds"]]

    ax.plot(measured_n, measured_time, marker="o", markersize=9, color="#c0392b", linewidth=2.4, label="Measured CPU LOF Runtime")

    for n, t in zip(measured_n, measured_time):
        ax.annotate(
            f"N={n//1000}K: {t:.3f} s",
            xy=(n, t),
            xytext=(15, -10 if n == 25000 else 10),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0)
        )

    ax.set_xlabel("Sample Size (N)", fontsize=11)
    ax.set_ylabel("LOF Compute Time (seconds)", fontsize=11)
    ax.set_title("CPU LOF Empirical Scaling (Measured Baseline)", fontsize=13, pad=12)
    ax.set_xticks([10_000, 15_000, 20_000, 25_000])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y):,}"))
    ax.set_ylim(0, 26)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper left", framealpha=0.95)

    ax.text(
        0.05, 0.65,
        f"Measured Data Points:\n- N = 10,000 -> {measured_time[0]:.3f} s\n- N = 25,000 -> {measured_time[1]:.3f} s\nEmpirical scaling factor: {measured_time[1]/measured_time[0]:.2f}x\nTheoretical O(N^2) factor: 6.25x",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.9)
    )

    chart6_path = os.path.join(output_dir, "m2_cpu_scaling.png")
    plt.savefig(chart6_path)
    plt.close()
    print(f"[*] Saved Chart 6: {chart6_path}")

    # =============================================================
    # CHART 7: Pairwise Distance vs k-NN Search Time
    # =============================================================
    fig, ax = plt.subplots(figsize=(8, 5.5))

    subsets = ["10K Subset", "25K Subset"]
    dist_times = [t_10k["distance_matrix_seconds"], t_25k["distance_matrix_seconds"]]
    knn_times = [t_10k["knn_search_seconds"], t_25k["knn_search_seconds"]]

    x = np.arange(len(subsets))
    width = 0.35

    bars_dist = ax.bar(x - width/2, dist_times, width, label="Pairwise Distance Matrix", color="#2980b9", edgecolor="#222222", linewidth=1.0)
    bars_knn = ax.bar(x + width/2, knn_times, width, label="k-NN Search & Sort", color="#d35400", edgecolor="#222222", linewidth=1.0)

    ax.set_ylabel("Execution Time (seconds)", fontsize=11)
    ax.set_title("Dominant CPU Stages: Pairwise Distance vs. k-NN Search", fontsize=13, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(subsets, fontweight="bold")
    ax.grid(axis="y", alpha=0.4)
    ax.set_ylim(0, max(knn_times) * 1.25)
    ax.legend(loc="upper left", framealpha=0.95)

    for bar, val in zip(bars_dist, dist_times):
        height = bar.get_height()
        ax.annotate(f"{val:.2f} s", xy=(bar.get_x() + bar.get_width()/2, height), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    for bar, val in zip(bars_knn, knn_times):
        height = bar.get_height()
        ax.annotate(f"{val:.2f} s", xy=(bar.get_x() + bar.get_width()/2, height), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    chart7_path = os.path.join(output_dir, "m2_pairwise_vs_knn.png")
    plt.savefig(chart7_path)
    plt.close()
    print(f"[*] Saved Chart 7: {chart7_path}")


def run_m2_visualizations():
    results_dir = "results/m2_cpu_baseline"
    tables_dir = "results/m2_cpu_baseline/tables"
    charts_dir = "results/m2_cpu_baseline/charts"

    print("\n" + "=" * 65)
    print("GENERATING MILESTONE M2 TABULAR RESULTS & CHARTS")
    print("=" * 65)

    res_10k, res_25k = load_m2_results(results_dir)
    generate_m2_tables(res_10k, res_25k, tables_dir)
    generate_m2_charts(res_10k, res_25k, charts_dir)

    print("=" * 65)
    print("M2 VISUALIZATION GENERATION COMPLETE")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_m2_visualizations()
