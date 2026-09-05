"""
Milestone M1 Visualizations & Tabular Results Generator
Milestone M1: Dataset Preparation

Reads actual CICIDS2017 preprocessing metadata and generates:
- Table 1: Dataset Preparation Summary (CSV)
- Table 2: Processed Dataset Subsets (CSV)
- Table 3: Final Class Distribution (CSV)
- Chart 1: Dataset Reduction (PNG)
- Chart 2: Final CICIDS2017 Class Distribution (PNG)
- Chart 3: Processed Dataset Subsets (PNG)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict, Any


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


def generate_m1_tables(metadata_path: str, output_dir: str) -> Dict[str, pd.DataFrame]:
    """Generate CSV tables from actual M1 dataset metadata."""
    os.makedirs(output_dir, exist_ok=True)
    with open(metadata_path, "r") as f:
        meta = json.load(f)

    clean_meta = meta["cleaning_summary"]
    feat_meta = meta["feature_summary"]
    label_meta = meta["label_summary"]
    subsets_meta = meta["subsets"]

    # -------------------------------------------------------------
    # TABLE 1: Dataset Preparation Summary
    # -------------------------------------------------------------
    table1_data = [
        {"Metric": "Raw CSV Files", "Result": str(len(clean_meta["file_stats"]))},
        {"Metric": "Raw Samples", "Result": f"{clean_meta['raw_total_rows']:,}"},
        {"Metric": "Raw Features", "Result": str(clean_meta["raw_total_cols"])},
        {"Metric": "Numerical Features (Raw)", "Result": str(clean_meta["raw_total_cols"] - 1)},
        {"Metric": "Non-Numerical Features (Raw)", "Result": "1 (Label)"},
        {"Metric": "NaN Rows Removed", "Result": f"{clean_meta['nan_rows_removed']:,}"},
        {"Metric": "Inf Rows Removed", "Result": f"{clean_meta['inf_rows_removed']:,}"},
        {"Metric": "Total Invalid Rows Removed", "Result": f"{clean_meta['total_invalid_rows_removed']:,}"},
        {"Metric": "Duplicate Rows Removed", "Result": f"{clean_meta['duplicate_rows_removed']:,}"},
        {"Metric": "Cleaned Samples", "Result": f"{clean_meta['cleaned_rows']:,}"},
        {"Metric": "Retention Rate", "Result": f"{(clean_meta['cleaned_rows'] / clean_meta['raw_total_rows']) * 100:.2f}%"},
        {"Metric": "Final Numerical Features", "Result": str(feat_meta["total_selected_features"])},
        {"Metric": "Excluded Features Total", "Result": str(feat_meta["total_excluded_columns"])},
        {"Metric": "Removed Zero-Variance Features", "Result": "8"},
        {"Metric": "Removed Identifiers / Redundant", "Result": "3 (Label, Destination Port, Fwd Header Length.1)"},
        {"Metric": "Final Benign Samples", "Result": f"{label_meta['total_benign']:,} ({label_meta['distribution']['BENIGN']['percentage']}%)"},
        {"Metric": "Final Anomaly Samples", "Result": f"{label_meta['total_anomalies']:,} ({label_meta['anomaly_ratio'] * 100:.2f}%)"},
        {"Metric": "Scaling Method", "Result": feat_meta["scaler_type"]},
        {"Metric": "Data Type", "Result": feat_meta["data_type"]},
        {"Metric": "Random Seed", "Result": str(meta["random_seed"])}
    ]
    df_table1 = pd.DataFrame(table1_data)
    t1_path = os.path.join(output_dir, "m1_dataset_summary.csv")
    df_table1.to_csv(t1_path, index=False)
    print(f"[*] Generated Table 1: {t1_path}")

    # -------------------------------------------------------------
    # TABLE 2: Processed Dataset Subsets
    # -------------------------------------------------------------
    table2_rows = []
    # Add subset rows
    for key, sub in subsets_meta.items():
        size_name = f"{sub['size'] // 1000}K" if sub['size'] < 1_000_000 else f"{sub['size'] // 1_000_000}M"
        table2_rows.append({
            "Subset": size_name,
            "Samples": sub["size"],
            "File": sub["filename"],
            "File Size (MB)": sub["file_size_mb"],
            "Benign Count": sub["benign_count"],
            "Anomaly Count": sub["anomaly_count"],
            "Anomaly %": f"{sub['anomaly_ratio'] * 100:.2f}%"
        })
    # Add Full dataset row
    full_info = meta["full_dataset_file"]
    table2_rows.append({
        "Subset": "Full",
        "Samples": full_info["total_samples"],
        "File": full_info["filename"],
        "File Size (MB)": full_info["file_size_mb"],
        "Benign Count": label_meta["total_benign"],
        "Anomaly Count": label_meta["total_anomalies"],
        "Anomaly %": f"{label_meta['anomaly_ratio'] * 100:.2f}%"
    })
    df_table2 = pd.DataFrame(table2_rows)
    t2_path = os.path.join(output_dir, "m1_subset_summary.csv")
    df_table2.to_csv(t2_path, index=False)
    print(f"[*] Generated Table 2: {t2_path}")

    # -------------------------------------------------------------
    # TABLE 3: Final Class Distribution
    # -------------------------------------------------------------
    table3_rows = []
    dist = label_meta["distribution"]
    for cls_name, info in sorted(dist.items(), key=lambda x: x[1]["count"], reverse=True):
        table3_rows.append({
            "Class ID": info["class_id"],
            "Class Name": cls_name,
            "Type": "Normal (Benign)" if info["is_anomaly"] == 0 else "Anomaly (Attack)",
            "Samples": info["count"],
            "Percentage": f"{info['percentage']:.4f}%"
        })
    df_table3 = pd.DataFrame(table3_rows)
    t3_path = os.path.join(output_dir, "m1_class_distribution.csv")
    df_table3.to_csv(t3_path, index=False)
    print(f"[*] Generated Table 3: {t3_path}")

    return {"summary": df_table1, "subsets": df_table2, "distribution": df_table3}


def generate_m1_charts(metadata_path: str, output_dir: str):
    """Generate presentation-quality PNG charts for M1."""
    os.makedirs(output_dir, exist_ok=True)
    set_plot_style()

    with open(metadata_path, "r") as f:
        meta = json.load(f)

    clean_meta = meta["cleaning_summary"]
    label_meta = meta["label_summary"]
    subsets_meta = meta["subsets"]

    # =============================================================
    # CHART 1: Dataset Reduction (Waterfall / Stage Breakdown)
    # =============================================================
    fig, ax = plt.subplots(figsize=(8, 5.5))
    
    stages = ["Raw Ingested", "Invalid (NaN/Inf)", "Duplicates", "Cleaned Dataset"]
    counts = [
        clean_meta["raw_total_rows"],
        -clean_meta["total_invalid_rows_removed"],
        -clean_meta["duplicate_rows_removed"],
        clean_meta["cleaned_rows"]
    ]
    colors = ["#2b5c8f", "#d9534f", "#f0ad4e", "#2e7d32"]
    
    # Absolute bar representation
    plot_values = [
        clean_meta["raw_total_rows"],
        clean_meta["total_invalid_rows_removed"],
        clean_meta["duplicate_rows_removed"],
        clean_meta["cleaned_rows"]
    ]
    bars = ax.bar(stages, [v / 1e6 for v in plot_values], color=colors, width=0.55, edgecolor="#222222", linewidth=1.1)
    
    ax.set_ylabel("Records (Millions)", fontsize=11)
    ax.set_title("CICIDS2017 Dataset Cleansing & Reduction Breakdown", fontsize=13, pad=12)
    ax.set_ylim(0, 3.3)
    ax.grid(axis="y", alpha=0.4)

    # Annotate bar values
    for bar, val in zip(bars, plot_values):
        height = bar.get_height()
        ax.annotate(
            f"{val:,}\n({val / clean_meta['raw_total_rows'] * 100:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold"
        )

    # Add descriptive callout
    ax.text(
        0.98, 0.94,
        f"Raw Total: {clean_meta['raw_total_rows']:,}\nCleaned: {clean_meta['cleaned_rows']:,} (89.05% retained)\nInvalid Removed: {clean_meta['total_invalid_rows_removed']:,}\nDuplicates Removed: {clean_meta['duplicate_rows_removed']:,}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.9)
    )

    chart1_path = os.path.join(output_dir, "m1_dataset_reduction.png")
    plt.savefig(chart1_path)
    plt.close()
    print(f"[*] Saved Chart 1: {chart1_path}")

    # =============================================================
    # CHART 2: Final CICIDS2017 Class Distribution (Horizontal Bar)
    # =============================================================
    fig, ax = plt.subplots(figsize=(10, 7.5))

    dist = label_meta["distribution"]
    sorted_classes = sorted(dist.items(), key=lambda x: x[1]["count"], reverse=False)
    
    class_names = [item[0] for item in sorted_classes]
    counts = [item[1]["count"] for item in sorted_classes]
    is_anom = [item[1]["is_anomaly"] for item in sorted_classes]
    percentages = [item[1]["percentage"] for item in sorted_classes]

    bar_colors = ["#c0392b" if anom == 1 else "#1b4f72" for anom in is_anom]
    bars = ax.barh(class_names, counts, color=bar_colors, edgecolor="#222222", height=0.65, linewidth=0.8)

    ax.set_xscale("log")
    ax.set_xlabel("Number of Samples (Log Scale)", fontsize=11)
    ax.set_title("Cleaned CICIDS2017 Class Distribution (N = 2,520,798)", fontsize=13, pad=12)
    ax.grid(axis="x", which="both", alpha=0.4)
    ax.set_xlim(5, 5e6)

    # Format log x-axis
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{int(y):,}"))

    # Annotations next to bars
    for bar, count, pct in zip(bars, counts, percentages):
        width = bar.get_width()
        ax.text(
            width * 1.25,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,} ({pct:.2f}%)",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="semibold",
            color="#222222"
        )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1b4f72", edgecolor="#222222", label=f"Benign: 2,095,057 ({label_meta['distribution']['BENIGN']['percentage']}%)"),
        Patch(facecolor="#c0392b", edgecolor="#222222", label=f"Anomaly: 425,741 ({label_meta['anomaly_ratio']*100:.2f}%)")
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.95, fontsize=9.5)

    chart2_path = os.path.join(output_dir, "m1_class_distribution.png")
    plt.savefig(chart2_path)
    plt.close()
    print(f"[*] Saved Chart 2: {chart2_path}")

    # =============================================================
    # CHART 3: Processed Dataset Subsets
    # =============================================================
    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    sub_names = ["10K", "25K", "100K", "250K", "500K", "1M"]
    sub_sizes = [10_000, 25_000, 100_000, 250_000, 500_000, 1_000_000]
    sub_files = [subsets_meta[f"subset_{k.lower()}.npz"] for k in sub_names]
    sub_sizes_mb = [s["file_size_mb"] for s in sub_files]

    x = np.arange(len(sub_names))
    width = 0.45

    bars1 = ax1.bar(x, [s / 1000 for s in sub_sizes], width, color="#2c3e50", edgecolor="#222222", label="Sample Count (Thousands)", linewidth=1.0)
    ax1.set_ylabel("Samples (Thousands)", color="#2c3e50", fontsize=11)
    ax1.set_xlabel("Subset Identifier", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sub_names, fontweight="bold")
    ax1.set_title("Generated Experimental Subsets for Controlled Scaling (Seed = 42)", fontsize=13, pad=12)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, 1200)

    # Annotate samples & file size
    for bar, size_val, mb_val in zip(bars1, sub_sizes, sub_sizes_mb):
        height = bar.get_height()
        ax1.annotate(
            f"{size_val:,}\n({mb_val:.2f} MB)",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    # Info box
    ax1.text(
        0.03, 0.92,
        "Sampling Strategy: Stratified (Proportional Multiclass)\nFeature Dimension: 68 float32 Features\nAnomaly Ratio Preserved: ~16.89%",
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffffff", edgecolor="#cccccc", alpha=0.9)
    )

    chart3_path = os.path.join(output_dir, "m1_processed_subsets.png")
    plt.savefig(chart3_path)
    plt.close()
    print(f"[*] Saved Chart 3: {chart3_path}")


def run_m1_visualizations():
    metadata_path = "data/processed/dataset_metadata.json"
    tables_dir = "results/m1_dataset_preparation/tables"
    charts_dir = "results/m1_dataset_preparation/charts"

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found at '{metadata_path}'. Run preprocessing/prepare_dataset.py first.")

    print("\n" + "=" * 65)
    print("GENERATING MILESTONE M1 TABULAR RESULTS & CHARTS")
    print("=" * 65)

    generate_m1_tables(metadata_path, tables_dir)
    generate_m1_charts(metadata_path, charts_dir)

    print("=" * 65)
    print("M1 VISUALIZATION GENERATION COMPLETE")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_m1_visualizations()
