"""
m5_visualizations.py — Chart Generator for M5 CUDA Optimization

Reads the real measured results from results/m5_cuda_optimized/ (written by
scripts/run_m5_optimization.py) and produces:
  1. m5_01_naive_vs_optimized_runtime.png - End-to-end naive vs optimized runtime
  2. m5_02_per_kernel_before_after.png     - Per-kernel before/after bar chart
  3. m5_03_speedup_by_size.png             - Speedup vs dataset size
  4. m5_04_kernel_time_distribution.png    - Kernel time share before vs after
  5. m5_05_correctness_agreement.png       - CPU/naive/optimized score agreement (Pearson r)
"""

import os
import sys
import json
import csv
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "m5_cuda_optimized")
CHARTS_DIR = os.path.join(RESULTS_DIR, "charts")

STAGE_LABELS = {
    "distance_kernel": "Distance",
    "knn_kernel": "k-NN",
    "reachability_kernel": "Reachability",
    "lrd_kernel": "LRD",
    "lof_kernel": "LOF scoring",
}


def set_plot_style():
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
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def load_summary():
    with open(os.path.join(RESULTS_DIR, "optimization_summary.json"), encoding="utf-8") as f:
        return json.load(f)


def load_kernel_comparison():
    rows = []
    with open(os.path.join(RESULTS_DIR, "kernel_comparison.csv"), encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_correctness():
    with open(os.path.join(RESULTS_DIR, "correctness_report.json"), encoding="utf-8") as f:
        return json.load(f)


def plot_naive_vs_optimized_runtime(summary, output_path):
    subsets = list(summary["subsets"].keys())
    naive_ms = [summary["subsets"][s]["naive_end_to_end_ms"]["mean_ms"] for s in subsets]
    opt_ms = [summary["subsets"][s]["optimized_end_to_end_ms"]["mean_ms"] for s in subsets]

    x = np.arange(len(subsets))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.bar(x - width / 2, naive_ms, width, label="Naive CUDA (M3)", color="#5b8def")
    ax.bar(x + width / 2, opt_ms, width, label="Optimized CUDA (M5)", color="#2ecc71")
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in subsets])
    ax.set_ylabel("End-to-end time (ms)")
    ax.set_title("Naive vs. Optimized CUDA LOF — End-to-End Runtime")
    ax.legend()
    ax.grid(axis="y")
    for i, (n, o) in enumerate(zip(naive_ms, opt_ms)):
        ax.text(i - width / 2, n, f"{n:.1f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + width / 2, o, f"{o:.1f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_per_kernel_before_after(rows, output_path):
    stages = [s for s in STAGE_LABELS if any(r["stage"] == s for r in rows)]
    subsets = sorted(set(r["subset"] for r in rows))

    fig, axes = plt.subplots(1, len(subsets), figsize=(7 * len(subsets), 5.5))
    if len(subsets) == 1:
        axes = [axes]

    for ax, subset in zip(axes, subsets):
        naive_vals = []
        opt_vals = []
        for stage in stages:
            r = next(r for r in rows if r["subset"] == subset and r["stage"] == stage)
            naive_vals.append(float(r["naive_mean_ms"]))
            opt_vals.append(float(r["optimized_mean_ms"]))

        x = np.arange(len(stages))
        width = 0.35
        ax.bar(x - width / 2, naive_vals, width, label="Naive", color="#5b8def")
        ax.bar(x + width / 2, opt_vals, width, label="Optimized", color="#2ecc71")
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([STAGE_LABELS[s] for s in stages], rotation=20, ha="right")
        ax.set_ylabel("Mean time (ms, log scale)")
        ax.set_title(f"Per-Kernel Before/After — {subset.upper()}")
        ax.legend()
        ax.grid(axis="y", which="both")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_speedup_by_size(summary, output_path):
    subsets = list(summary["subsets"].keys())
    n_samples = [summary["subsets"][s]["n_samples"] for s in subsets]
    speedups = [summary["subsets"][s]["end_to_end_speedup"] for s in subsets]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(n_samples, speedups, marker="o", linewidth=2, color="#2ecc71")
    ax.axhline(1.0, color="#999", linestyle="--", linewidth=1, label="No speedup (1.0x)")
    for n, s in zip(n_samples, speedups):
        ax.annotate(f"{s:.2f}x", (n, s), textcoords="offset points", xytext=(0, 10), ha="center")
    ax.set_xlabel("Dataset size (N)")
    ax.set_ylabel("End-to-end speedup (naive / optimized)")
    ax.set_title("Optimized CUDA Speedup vs. Dataset Size")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_kernel_time_distribution(summary, output_path):
    subsets = list(summary["subsets"].keys())
    fig, axes = plt.subplots(1, len(subsets) * 2, figsize=(5 * len(subsets) * 2, 5.5))

    idx = 0
    for subset in subsets:
        for impl, key in (("Naive", "naive_stage_means_ms"), ("Optimized", "optimized_stage_means_ms")):
            stage_means = summary["subsets"][subset][key]
            labels = [STAGE_LABELS.get(s, s) for s in stage_means if s in STAGE_LABELS]
            values = [stage_means[s] for s in stage_means if s in STAGE_LABELS]
            ax = axes[idx]
            ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90,
                   colors=["#5b8def", "#2ecc71", "#f39c12", "#9b59b6", "#e74c3c"])
            ax.set_title(f"{subset.upper()} — {impl}")
            idx += 1

    plt.suptitle("Kernel Time Share — Naive vs. Optimized", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_correctness_agreement(correctness, output_path):
    subsets = list(correctness.keys())
    comparisons = ["cpu_vs_naive_cuda", "cpu_vs_optimized_cuda", "naive_cuda_vs_optimized_cuda"]
    comp_labels = ["CPU vs Naive", "CPU vs Optimized", "Naive vs Optimized"]

    x = np.arange(len(subsets))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i, (comp, label) in enumerate(zip(comparisons, comp_labels)):
        rs = [correctness[s][comp].get("pearson_r", 0.0) or 0.0 for s in subsets]
        ax.bar(x + (i - 1) * width, rs, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in subsets])
    ax.set_ylabel("Pearson correlation (r)")
    ax.set_ylim(0.999, 1.0005)
    ax.set_title("LOF Score Agreement — CPU vs. Naive vs. Optimized CUDA")
    ax.legend()
    ax.grid(axis="y")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    set_plot_style()
    os.makedirs(CHARTS_DIR, exist_ok=True)

    summary = load_summary()
    rows = load_kernel_comparison()
    correctness = load_correctness()

    plot_naive_vs_optimized_runtime(summary, os.path.join(CHARTS_DIR, "m5_01_naive_vs_optimized_runtime.png"))
    plot_per_kernel_before_after(rows, os.path.join(CHARTS_DIR, "m5_02_per_kernel_before_after.png"))
    plot_speedup_by_size(summary, os.path.join(CHARTS_DIR, "m5_03_speedup_by_size.png"))
    plot_kernel_time_distribution(summary, os.path.join(CHARTS_DIR, "m5_04_kernel_time_distribution.png"))
    plot_correctness_agreement(correctness, os.path.join(CHARTS_DIR, "m5_05_correctness_agreement.png"))

    print(f"Wrote 5 charts to {CHARTS_DIR}")


if __name__ == "__main__":
    main()
