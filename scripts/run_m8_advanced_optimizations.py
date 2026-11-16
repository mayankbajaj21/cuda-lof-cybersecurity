"""
M8 - Advanced (outside-syllabus) optimizations: unified benchmark.

Measures, in ONE session on the same data with k=20 and contamination=0.10:
    CPU reference, V0 naive, V1 syllabus-optimized, V2 register blocking,
    V3 cp.async double buffering, V4 FP16 storage, V5 combined.

GPU timed repetitions are interleaved across variants (rep 1 of every
variant, then rep 2, ...) so slow drift in laptop clocks/temperature is
spread evenly over all variants instead of penalising whichever runs last.

Writes everything to results/m8_advanced_optimizations/. Does not touch any
earlier milestone's results.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import psutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.python.cpu.cpu_lof import CPULocalOutlierFactor  # noqa: E402
from src.python.cuda.lof_variants import (  # noqa: E402
    FP16_VARIANTS,
    VARIANTS,
    compute_lof_variant,
    estimate_variant_gpu_bytes,
    get_kernel_attributes,
    is_variants_available,
)
from src.python.data.dataset import load_subset  # noqa: E402
from src.python.environment.hardware_info import get_full_environment_snapshot  # noqa: E402

OUT_DIR = os.path.join(PROJECT_ROOT, "results", "m8_advanced_optimizations")
CHART_DIR = os.path.join(OUT_DIR, "charts")

K = 20
CONTAMINATION = 0.10
GPU_WARMUP = 3
GPU_REPEATS = 10
CPU_WARMUP = 1
CPU_REPEATS = 5
VRAM_SAFETY_MARGIN_BYTES = 300 * 1024 * 1024
SLICE_45K_ROWS = 45000
SLICE_45K_LABEL = "45k (first 45k rows of 100k subset)"

NCU_EXE = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.4.0\target\windows-desktop-win7-x64\ncu.exe"
NCU_METRICS = [
    "launch__registers_per_thread",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__shared_mem_per_block_static",
    "launch__shared_mem_per_block_dynamic",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
]

GPU_STAGES = [
    ("h2d", "h2d_ms"),
    ("fp16_convert", "fp16_convert_ms"),
    ("distance", "distance_kernel_ms"),
    ("knn", "knn_kernel_ms"),
    ("reachability", "reachability_kernel_ms"),
    ("lrd", "lrd_kernel_ms"),
    ("lof", "lof_kernel_ms"),
    ("d2h", "d2h_ms"),
]
CPU_STAGES = [
    ("distance", "distance_matrix_seconds"),
    ("knn", "knn_search_seconds"),
    ("reachability", "reachability_distance_seconds"),
    ("lrd", "lrd_computation_seconds"),
    ("lof", "lof_scoring_seconds"),
]


def log(msg):
    print(f"[M8 {datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min": float(a.min()),
        "max": float(a.max()),
    }


def live_free_vram_bytes():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return int(float(out.stdout.strip().splitlines()[0])) * 1024 * 1024
    except Exception:
        return None


def power_status():
    status = {"ac_power_plugged": None, "battery_percent": None, "active_power_scheme": None, "warning": None}
    try:
        batt = psutil.sensors_battery()
        if batt is not None:
            status["ac_power_plugged"] = bool(batt.power_plugged)
            status["battery_percent"] = float(batt.percent)
    except Exception:
        pass
    try:
        out = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, timeout=10)
        status["active_power_scheme"] = out.stdout.strip() or None
    except Exception:
        pass
    if status["ac_power_plugged"] is False:
        status["warning"] = "Laptop was on battery during the run; GPU/CPU clocks may be reduced."
    return status


def anomaly_flags(scores):
    threshold = np.percentile(scores, 100.0 * (1.0 - CONTAMINATION))
    return scores >= threshold


def knn_recall(ref_idx, idx):
    k = ref_idx.shape[1]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(ref_idx, idx)]))


def compare(ref_scores, scores, ref_idx, idx):
    ref = np.asarray(ref_scores, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    diff = np.abs(ref - s)
    return {
        "max_abs_error": float(diff.max()),
        "mean_abs_error": float(diff.mean()),
        "max_rel_error": float(np.max(diff / np.maximum(np.abs(ref), 1e-12))),
        "pearson_r": float(np.corrcoef(ref, s)[0, 1]),
        "knn_recall_at_k": knn_recall(ref_idx, idx),
        "knn_indices_identical": bool(np.array_equal(ref_idx, idx)),
        "anomaly_flag_agreement": float(np.mean(anomaly_flags(ref) == anomaly_flags(s))),
    }


def max_feasible_n(variant, d, free_bytes):
    if free_bytes is None:
        return None
    budget = free_bytes - VRAM_SAFETY_MARGIN_BYTES
    lo, hi = 1, 1_000_000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_variant_gpu_bytes(mid, d, K, variant) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------
def run_gpu_variants(X, variants):
    for v in variants:
        for _ in range(GPU_WARMUP):
            compute_lof_variant(X, K, v)

    timings = {v: [] for v in variants}
    mem = {}
    for _ in range(GPU_REPEATS):
        for v in variants:
            _, t, m = compute_lof_variant(X, K, v)
            timings[v].append(t)
            mem[v] = m

    outputs = {}
    for v in variants:
        scores, _, _, idx = compute_lof_variant(X, K, v, return_knn_indices=True)
        outputs[v] = (scores, idx)
    return timings, mem, outputs


def run_cpu(X):
    for _ in range(CPU_WARMUP):
        CPULocalOutlierFactor(k=K, contamination=CONTAMINATION).fit(X)
    walls, stage_runs = [], []
    model = None
    for _ in range(CPU_REPEATS):
        model = None
        t0 = time.perf_counter()
        model = CPULocalOutlierFactor(k=K, contamination=CONTAMINATION).fit(X)
        walls.append((time.perf_counter() - t0) * 1000.0)
        stage_runs.append(dict(model.timings_))
    scores = np.asarray(model.lof_scores_, dtype=np.float32)
    idx = np.asarray(model.knn_indices_)
    del model
    return walls, stage_runs, scores, idx


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------
def base_row(subset, n, d, variant_id, name, source, techniques, exactness):
    return {
        "subset": subset,
        "n_samples": n,
        "n_features": d,
        "variant_id": variant_id,
        "variant_name": name,
        "technique_source": source,
        "techniques": techniques,
        "exact_or_approximate": exactness,
    }


STAGE_COLUMNS = [f"{s}_mean_ms" for s, _ in GPU_STAGES]
TIMING_COLUMNS = STAGE_COLUMNS + [
    "kernel_only_mean_ms",
    "end_to_end_mean_ms", "end_to_end_median_ms", "end_to_end_std_ms", "end_to_end_min_ms", "end_to_end_max_ms",
]


def empty_timing_fields():
    return {c: np.nan for c in TIMING_COLUMNS}


def gpu_row(subset, n, d, v, timing_runs, mem):
    info = VARIANTS[v]
    row = base_row(subset, n, d, v, info["name"], info["technique_source"], info["techniques"],
                   info["exact_or_approximate"])
    for stage, key in GPU_STAGES:
        row[f"{stage}_mean_ms"] = float(np.mean([t[key] for t in timing_runs]))
    row["kernel_only_mean_ms"] = float(np.mean([t["total_kernel_ms"] for t in timing_runs]))
    e2e = stats([t["end_to_end_ms"] for t in timing_runs])
    for k_, val in e2e.items():
        row[f"end_to_end_{k_}_ms"] = val
    row.update({
        "repetitions": GPU_REPEATS,
        "warmups": GPU_WARMUP,
        "peak_gpu_mem_bytes_tracked": mem["tracked_peak_bytes"],
        "peak_gpu_mem_bytes_memgetinfo": mem["memgetinfo_peak_used_delta_bytes"],
        "mem_method": "tracked = running sum of the pipeline's own cudaMalloc bytes (exact); "
                      "memgetinfo = device used-memory delta from cudaMemGetInfo after each allocation "
                      "(driver page granularity, includes allocator rounding)",
        "fp16_overflow_count": mem["fp16_overflow_count"],
        "timing_method": "CUDA events per stage; end-to-end event spans allocation, transfers and all kernels; "
                         f"{GPU_WARMUP} warm-ups discarded, {GPU_REPEATS} timed reps interleaved across variants",
        "classification": "MEASURED",
        "unavailable_reason": "",
    })
    return row


def cpu_row(subset, n, d, walls, stage_runs):
    row = base_row(subset, n, d, -1, "CPU (NumPy)", "cpu", "Vectorised NumPy reference (src/python/cpu/cpu_lof.py)",
                   "exact")
    row.update({c: np.nan for c in STAGE_COLUMNS})
    for stage, key in CPU_STAGES:
        row[f"{stage}_mean_ms"] = float(np.mean([r[key] for r in stage_runs]) * 1000.0)
    row["kernel_only_mean_ms"] = float(np.mean([r["total_lof_compute_seconds"] for r in stage_runs]) * 1000.0)
    e2e = stats(walls)
    for k_, val in e2e.items():
        row[f"end_to_end_{k_}_ms"] = val
    row.update({
        "repetitions": CPU_REPEATS,
        "warmups": CPU_WARMUP,
        "peak_gpu_mem_bytes_tracked": np.nan,
        "peak_gpu_mem_bytes_memgetinfo": np.nan,
        "mem_method": "not applicable (CPU)",
        "fp16_overflow_count": np.nan,
        "timing_method": f"time.perf_counter around fit(); {CPU_WARMUP} warm-up discarded, {CPU_REPEATS} timed reps; "
                         "per-stage columns from the model's own perf_counter timings",
        "classification": "MEASURED",
        "unavailable_reason": "",
    })
    return row


def unavailable_row(subset, n, d, variant_id, reason):
    if variant_id < 0:
        row = base_row(subset, n, d, -1, "CPU (NumPy)", "cpu", "Vectorised NumPy reference", "exact")
    else:
        info = VARIANTS[variant_id]
        row = base_row(subset, n, d, variant_id, info["name"], info["technique_source"], info["techniques"],
                       info["exact_or_approximate"])
    row.update(empty_timing_fields())
    row.update({
        "repetitions": 0, "warmups": 0,
        "peak_gpu_mem_bytes_tracked": np.nan, "peak_gpu_mem_bytes_memgetinfo": np.nan,
        "mem_method": "", "fp16_overflow_count": np.nan, "timing_method": "",
        "classification": "UNAVAILABLE",
        "unavailable_reason": reason,
    })
    return row


# ---------------------------------------------------------------------------
# Nsight Compute + static kernel attributes
# ---------------------------------------------------------------------------
def collect_kernel_metrics():
    rows = []
    ncu_error = None
    if os.path.isfile(NCU_EXE):
        driver = os.path.join(PROJECT_ROOT, "scripts", "m8_ncu_driver.py")
        try:
            proc = subprocess.run(
                [NCU_EXE, "--target-processes", "all", "--kernel-name", "regex:pairwise_distance|knn",
                 "--launch-count", "2", "--metrics", ",".join(NCU_METRICS), "--csv",
                 sys.executable, driver, "1"],
                capture_output=True, text=True, timeout=600, cwd=PROJECT_ROOT,
            )
            text = (proc.stdout or "") + (proc.stderr or "")
            if proc.returncode != 0 or "ERR_NVGPUCTRPERM" in text:
                err_lines = [ln for ln in text.splitlines() if "ERROR" in ln]
                ncu_error = err_lines[0].strip() if err_lines else f"ncu exited with code {proc.returncode}"
        except Exception as e:
            ncu_error = f"ncu could not be run: {e}"
    else:
        ncu_error = f"Nsight Compute not found at {NCU_EXE}"

    if ncu_error is None:
        ncu_error = "ncu ran but per-variant collection is not implemented beyond the probe; see report.md"

    for v in VARIANTS:
        for kernel in ("distance", "knn"):
            for metric in NCU_METRICS:
                rows.append({
                    "variant_id": v, "variant_name": VARIANTS[v]["name"], "kernel": kernel,
                    "metric": metric, "value": np.nan, "source": "Nsight Compute (ncu)",
                    "classification": "UNAVAILABLE", "note": ncu_error,
                })
            try:
                attrs = get_kernel_attributes(v, kernel)
            except Exception as e:
                rows.append({
                    "variant_id": v, "variant_name": VARIANTS[v]["name"], "kernel": kernel,
                    "metric": "cudaFuncGetAttributes", "value": np.nan, "source": "CUDA runtime",
                    "classification": "UNAVAILABLE", "note": str(e),
                })
                continue
            static = [
                ("registers_per_thread", attrs["num_regs"]),
                ("static_shared_mem_bytes_per_block", attrs["static_shared_bytes"]),
                ("local_mem_bytes_per_thread", attrs["local_bytes"]),
                ("threads_per_block", attrs["block_threads"]),
            ]
            for metric, value in static:
                rows.append({
                    "variant_id": v, "variant_name": VARIANTS[v]["name"], "kernel": kernel,
                    "metric": metric, "value": value, "source": "cudaFuncGetAttributes (compiled kernel)",
                    "classification": "MEASURED", "note": "",
                })
            for metric, value in (("max_active_blocks_per_sm", attrs["max_active_blocks_per_sm"]),
                                  ("theoretical_occupancy_pct", round(attrs["theoretical_occupancy_pct"], 2))):
                rows.append({
                    "variant_id": v, "variant_name": VARIANTS[v]["name"], "kernel": kernel,
                    "metric": metric, "value": value,
                    "source": "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
                    "classification": "ESTIMATED",
                    "note": "Theoretical occupancy from resource usage, not achieved occupancy",
                })
    return rows, ncu_error


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
LADDER_ORDER = [-1, 0, 1, 2, 3, 4, 5]
LADDER_LABELS = {-1: "CPU", 0: "V0 Naive", 1: "V1 Syllabus", 2: "V2 RegBlock", 3: "V3 Async", 4: "V4 FP16",
                 5: "V5 Combined"}
LADDER_COLORS = {-1: "#7f7f7f", 0: "#d62728", 1: "#1f77b4", 2: "#2ca02c", 3: "#9467bd", 4: "#ff7f0e",
                 5: "#17becf"}


def chart_stage_ladder(df):
    measured = df[df["classification"] == "MEASURED"]
    subsets = [s for s in ["10k", "25k"] if s in set(measured["subset"])]
    fig, axes = plt.subplots(1, len(subsets), figsize=(7 * len(subsets), 5), squeeze=False)
    for ax, subset in zip(axes[0], subsets):
        sub = measured[measured["subset"] == subset].set_index("variant_id")
        ids = [v for v in LADDER_ORDER if v in sub.index]
        vals = [sub.loc[v, "end_to_end_median_ms"] for v in ids]
        bars = ax.bar([LADDER_LABELS[v] for v in ids], vals, color=[LADDER_COLORS[v] for v in ids])
        ax.set_yscale("log")
        ax.set_ylabel("End-to-end median time (ms, log scale)")
        ax.set_title(f"{subset}: CPU -> naive GPU -> syllabus -> outside-syllabus")
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, val * 1.08, f"{val:,.1f}", ha="center", fontsize=8)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "m8_01_stage_ladder.png"), dpi=150)
    plt.close(fig)


def chart_kernel_breakdown(df):
    measured = df[(df["classification"] == "MEASURED") & (df["variant_id"] >= 0)]
    subsets = [s for s in ["10k", "25k", SLICE_45K_LABEL] if s in set(measured["subset"])]
    fig, axes = plt.subplots(1, len(subsets), figsize=(6 * len(subsets), 5), squeeze=False)
    parts = [("distance_mean_ms", "Distance kernel", "#1f77b4"),
             ("knn_mean_ms", "k-NN kernel", "#ff7f0e"),
             ("other_kernels", "Reach + LRD + LOF kernels", "#2ca02c"),
             ("transfers", "H2D + D2H + FP16 convert", "#9467bd"),
             ("overhead", "Alloc/free + sync overhead", "#c7c7c7")]
    for ax, subset in zip(axes[0], subsets):
        sub = measured[measured["subset"] == subset].sort_values("variant_id")
        labels = [LADDER_LABELS[v] for v in sub["variant_id"]]
        comp = {
            "distance_mean_ms": sub["distance_mean_ms"].values,
            "knn_mean_ms": sub["knn_mean_ms"].values,
            "other_kernels": (sub["reachability_mean_ms"] + sub["lrd_mean_ms"] + sub["lof_mean_ms"]).values,
            "transfers": (sub["h2d_mean_ms"] + sub["d2h_mean_ms"] + sub["fp16_convert_mean_ms"]).values,
        }
        accounted = sum(comp.values())
        comp["overhead"] = np.clip(sub["end_to_end_mean_ms"].values - accounted, 0, None)
        bottom = np.zeros(len(sub))
        for key, label, color in parts:
            ax.bar(labels, comp[key], bottom=bottom, label=label, color=color)
            bottom += comp[key]
        ax.set_ylabel("Mean time (ms)")
        ax.set_title(f"Where the time goes - {subset}")
        ax.tick_params(axis="x", rotation=30)
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "m8_02_kernel_breakdown.png"), dpi=150)
    plt.close(fig)


def chart_speedup_vs_v1(df):
    measured = df[df["classification"] == "MEASURED"]
    subsets = [s for s in ["10k", "25k"] if s in set(measured["subset"])]
    techniques = [2, 3, 4, 5]
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.2
    x = np.arange(len(techniques))
    for i, subset in enumerate(subsets):
        sub = measured[measured["subset"] == subset].set_index("variant_id")
        dist = [sub.loc[1, "distance_mean_ms"] / sub.loc[v, "distance_mean_ms"] for v in techniques]
        e2e = [sub.loc[1, "end_to_end_median_ms"] / sub.loc[v, "end_to_end_median_ms"] for v in techniques]
        ax.bar(x + (2 * i) * width - 1.5 * width, dist, width, label=f"{subset} distance kernel")
        ax.bar(x + (2 * i + 1) * width - 1.5 * width, e2e, width, label=f"{subset} end-to-end", alpha=0.75)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([LADDER_LABELS[v] for v in techniques])
    ax.set_ylabel("Speedup vs V1 syllabus-optimized (x)")
    ax.set_title("What each outside-syllabus technique adds on top of V1 (below 1.0 = slower)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "m8_03_speedup_vs_v1.png"), dpi=150)
    plt.close(fig)


def chart_peak_memory(df, free_vram):
    gpu = df[df["variant_id"] >= 0]
    subsets = [s for s in ["10k", "25k", SLICE_45K_LABEL] if s in set(gpu["subset"])]
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.13
    x = np.arange(len(subsets))
    for v in range(6):
        vals, unavailable = [], []
        for subset in subsets:
            r = gpu[(gpu["subset"] == subset) & (gpu["variant_id"] == v)]
            if r.empty or r.iloc[0]["classification"] != "MEASURED":
                vals.append(0.0)
                unavailable.append(True)
            else:
                vals.append(r.iloc[0]["peak_gpu_mem_bytes_tracked"] / 1024 ** 3)
                unavailable.append(False)
        pos = x + (v - 2.5) * width
        bars = ax.bar(pos, vals, width, label=LADDER_LABELS[v], color=LADDER_COLORS[v])
        for b, un in zip(bars, unavailable):
            if un:
                ax.text(b.get_x() + b.get_width() / 2, 0.1, "won't fit", rotation=90, ha="center", fontsize=7)
    if free_vram:
        ax.axhline(free_vram / 1024 ** 3, color="red", linestyle="--", linewidth=1,
                   label=f"Free VRAM at run time ({free_vram / 1024 ** 3:.2f} GiB)")
    ax.set_xticks(x)
    ax.set_xticklabels(subsets)
    ax.set_ylabel("Peak GPU memory allocated by pipeline (GiB)")
    ax.set_title("Peak GPU memory per variant (tracked cudaMalloc bytes)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "m8_04_peak_memory.png"), dpi=150)
    plt.close(fig)


def chart_fp16_accuracy(corr_rows):
    df = pd.DataFrame(corr_rows)
    df = df[(df["reference"] == "V0 Naive") & (df["candidate_id"].isin([4, 5]))]
    if df.empty:
        return
    metrics = [("pearson_r", "LOF score Pearson r"), ("knn_recall_at_k", "k-NN recall@k"),
               ("anomaly_flag_agreement", "Anomaly-flag agreement")]
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [f"{r.candidate_name} / {r.subset}" for r in df.itertuples()]
    x = np.arange(len(df))
    width = 0.25
    for i, (key, label) in enumerate(metrics):
        ax.bar(x + (i - 1) * width, df[key].values, width, label=label)
    ax.set_ylim(min(0.9, float(df[[m for m, _ in metrics]].min().min()) - 0.01), 1.001)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Agreement with exact V0 (1.0 = identical)")
    ax.set_title("FP16 variants (approximate) vs exact fp32 result")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "m8_05_fp16_accuracy.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not is_variants_available():
        raise SystemExit("bin/cuda_lof_variants.dll not available - build it with scripts/build_cuda_lof_variants.bat")
    os.makedirs(CHART_DIR, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    power = power_status()
    if power["warning"]:
        log(f"WARNING: {power['warning']}")

    datasets = []
    for name in ("10k", "25k"):
        X = load_subset(name)[0]
        datasets.append((name, X, True))
    X100 = load_subset("100k")[0]
    datasets.append((SLICE_45K_LABEL, np.ascontiguousarray(X100[:SLICE_45K_ROWS]), False))
    del X100

    rows, breakdown, corr_rows, feasibility = [], [], [], []
    cpu_25k_median_ms = None
    free_vram_by_subset = {}
    ram_available_by_subset = {}

    for subset, X, run_cpu_here in datasets:
        n, d = X.shape
        free = live_free_vram_bytes()
        free_vram_by_subset[subset] = free
        feasible, infeasible = [], {}
        for v in VARIANTS:
            need = estimate_variant_gpu_bytes(n, d, K, v)
            ok = free is not None and need + VRAM_SAFETY_MARGIN_BYTES <= free
            feasibility.append({
                "subset": subset, "variant_id": v, "variant_name": VARIANTS[v]["name"],
                "estimated_peak_bytes": need, "live_free_vram_bytes": free,
                "safety_margin_bytes": VRAM_SAFETY_MARGIN_BYTES, "feasible": ok,
            })
            if ok:
                feasible.append(v)
            else:
                infeasible[v] = (f"Needs ~{need / 1024 ** 3:.2f} GiB (+{VRAM_SAFETY_MARGIN_BYTES / 1024 ** 2:.0f} MiB "
                                 f"margin) but only {free / 1024 ** 3:.2f} GiB VRAM was free at run time"
                                 if free is not None else "Free VRAM could not be determined")

        log(f"{subset}: GPU variants {feasible} (skipping {sorted(infeasible)})")
        timings, mem, outputs = run_gpu_variants(X, feasible) if feasible else ({}, {}, {})
        for v in VARIANTS:
            if v in infeasible:
                rows.append(unavailable_row(subset, n, d, v, infeasible[v]))
                continue
            rows.append(gpu_row(subset, n, d, v, timings[v], mem[v]))
            for stage, key in GPU_STAGES + [("total_kernel", "total_kernel_ms"), ("end_to_end", "end_to_end_ms")]:
                s = stats([t[key] for t in timings[v]])
                breakdown.append({"subset": subset, "variant_id": v, "variant_name": VARIANTS[v]["name"],
                                  "stage": stage, "mean_ms": s["mean"], "median_ms": s["median"],
                                  "std_ms": s["std"], "repetitions": GPU_REPEATS})

        cpu_scores = cpu_idx = None
        if run_cpu_here:
            ram_avail = psutil.virtual_memory().available
            ram_available_by_subset[subset] = ram_avail
            log(f"{subset}: CPU reference ({CPU_WARMUP} warm-up + {CPU_REPEATS} reps), "
                f"{ram_avail / 1024 ** 3:.1f} GiB RAM available")
            try:
                walls, stage_runs, cpu_scores, cpu_idx = run_cpu(X)
            except MemoryError:
                cpu_scores = cpu_idx = None
                rows.append(unavailable_row(
                    subset, n, d, -1,
                    f"MemoryError during the CPU run ({ram_avail / 1024 ** 3:.1f} GiB RAM available at start)"))
            else:
                row = cpu_row(subset, n, d, walls, stage_runs)
                row["ram_available_bytes_at_start"] = ram_avail
                rows.append(row)
                if subset == "25k":
                    cpu_25k_median_ms = row["end_to_end_median_ms"]
                for stage, key in CPU_STAGES + [("total_compute", "total_lof_compute_seconds")]:
                    s = stats([r[key] * 1000.0 for r in stage_runs])
                    breakdown.append({"subset": subset, "variant_id": -1, "variant_name": "CPU (NumPy)",
                                      "stage": stage, "mean_ms": s["mean"], "median_ms": s["median"],
                                      "std_ms": s["std"], "repetitions": CPU_REPEATS})
        else:
            est_s = (cpu_25k_median_ms / 1000.0) * (n / 25000) ** 2 if cpu_25k_median_ms else None
            dist_gb = n * n * 4 / 1024 ** 3
            avail_gb = psutil.virtual_memory().available / 1024 ** 3
            reason = (f"Not run: the CPU reference materialises at least one {n}x{n} float32 matrix "
                      f"({dist_gb:.1f} GiB) plus temporaries; {avail_gb:.1f} GiB RAM was available. "
                      + (f"O(N^2) extrapolation from the 25k median: ~{est_s:.0f} s per run (ESTIMATED)."
                         if est_s else ""))
            rows.append(unavailable_row(subset, n, d, -1, reason))

        if cpu_scores is not None:
            for v, (scores, idx) in outputs.items():
                c = compare(cpu_scores, scores, cpu_idx, idx)
                corr_rows.append({"subset": subset, "reference": "CPU (NumPy)", "candidate_id": v,
                                  "candidate_name": VARIANTS[v]["name"],
                                  "exact_or_approximate": VARIANTS[v]["exact_or_approximate"], **c})
        if 0 in outputs:
            ref_scores, ref_idx = outputs[0]
            for v, (scores, idx) in outputs.items():
                if v == 0:
                    continue
                c = compare(ref_scores, scores, ref_idx, idx)
                corr_rows.append({"subset": subset, "reference": "V0 Naive", "candidate_id": v,
                                  "candidate_name": VARIANTS[v]["name"],
                                  "exact_or_approximate": VARIANTS[v]["exact_or_approximate"], **c})
        elif 4 in outputs and 5 in outputs:
            c = compare(outputs[4][0], outputs[5][0], outputs[4][1], outputs[5][1])
            corr_rows.append({"subset": subset, "reference": "V4 FP16 storage", "candidate_id": 5,
                              "candidate_name": VARIANTS[5]["name"], "exact_or_approximate": "approximate",
                              "note": "No fp32 GPU reference fits in VRAM at this size", **c})
        del X

    df = pd.DataFrame(rows)

    for subset in df["subset"].unique():
        sub = df[df["subset"] == subset].set_index("variant_id")

        def ref_ms(vid):
            if vid in sub.index and sub.loc[vid, "classification"] == "MEASURED":
                return sub.loc[vid, "end_to_end_median_ms"]
            return np.nan

        cpu_ms, v0_ms, v1_ms = ref_ms(-1), ref_ms(0), ref_ms(1)
        mask = df["subset"] == subset
        df.loc[mask, "speedup_vs_cpu"] = cpu_ms / df.loc[mask, "end_to_end_median_ms"]
        df.loc[mask, "speedup_vs_v0"] = v0_ms / df.loc[mask, "end_to_end_median_ms"]
        df.loc[mask, "speedup_vs_v1"] = v1_ms / df.loc[mask, "end_to_end_median_ms"]

    column_order = [
        "subset", "n_samples", "n_features", "variant_id", "variant_name", "technique_source", "techniques",
        "exact_or_approximate", *TIMING_COLUMNS, "repetitions", "warmups", "peak_gpu_mem_bytes_tracked",
        "peak_gpu_mem_bytes_memgetinfo", "mem_method", "fp16_overflow_count", "speedup_vs_cpu", "speedup_vs_v0",
        "speedup_vs_v1", "ram_available_bytes_at_start", "timing_method", "classification", "unavailable_reason",
    ]
    df = df.reindex(columns=column_order)
    df.to_csv(os.path.join(OUT_DIR, "variant_benchmarks.csv"), index=False)
    pd.DataFrame(breakdown).to_csv(os.path.join(OUT_DIR, "kernel_breakdown.csv"), index=False)

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(os.path.join(OUT_DIR, "correctness_summary.csv"), index=False)
    with open(os.path.join(OUT_DIR, "correctness_report.json"), "w") as f:
        json.dump({
            "k": K, "contamination": CONTAMINATION,
            "metrics": {
                "max_abs_error / mean_abs_error / max_rel_error": "LOF score differences vs the reference",
                "pearson_r": "Pearson correlation of LOF scores",
                "knn_recall_at_k": "Mean fraction of the reference's k neighbor indices also found by the candidate",
                "anomaly_flag_agreement": f"Fraction of rows with the same flag at the top-{int(CONTAMINATION * 100)}% "
                                          "threshold",
            },
            "fp16_note": "V4 and V5 store features and distances in FP16 and are approximate by design.",
            "comparisons": corr_rows,
        }, f, indent=2, default=float)

    free_now = live_free_vram_bytes()
    mem_rows = []
    for n_label, n in (("10k", 10000), ("25k", 25000), ("45k", 45000), ("50k", 50000), ("100k", 100000)):
        for v in VARIANTS:
            tracked = df[(df["n_samples"] == n) & (df["variant_id"] == v)]["peak_gpu_mem_bytes_tracked"]
            measured = tracked.iloc[0] if len(tracked) and not pd.isna(tracked.iloc[0]) else np.nan
            mem_rows.append({
                "n_samples": n, "n_label": n_label, "variant_id": v, "variant_name": VARIANTS[v]["name"],
                "dist_matrix_bytes_fp32": n * n * 4, "dist_matrix_bytes_fp16": n * n * 2,
                "estimated_peak_bytes": estimate_variant_gpu_bytes(n, 68, K, v),
                "measured_tracked_peak_bytes": measured,
                "classification": "MEASURED" if not pd.isna(measured) else "ESTIMATED",
                "live_free_vram_bytes": free_now,
                "max_feasible_n_this_gpu": max_feasible_n(v, 68, free_now),
            })
    pd.DataFrame(mem_rows).to_csv(os.path.join(OUT_DIR, "memory_analysis.csv"), index=False)
    pd.DataFrame(feasibility).to_csv(os.path.join(OUT_DIR, "feasibility_checks.csv"), index=False)

    log("Collecting kernel attributes / Nsight Compute probe")
    ncu_rows, ncu_error = collect_kernel_metrics()
    pd.DataFrame(ncu_rows).to_csv(os.path.join(OUT_DIR, "ncu_metrics.csv"), index=False)

    snapshot = get_full_environment_snapshot(PROJECT_ROOT)
    metadata = {
        "milestone": "M8",
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_and_software": snapshot,
        "power": power,
        "build": {
            "dll": "bin/cuda_lof_variants.dll",
            "script": "scripts/build_cuda_lof_variants.bat",
            "nvcc_flags": "-arch=%ARCH% -O2 -lineinfo --shared (ARCH=sm_89 for this run)",
        },
        "parameters": {"k": K, "contamination": CONTAMINATION, "gpu_warmups": GPU_WARMUP,
                       "gpu_repetitions": GPU_REPEATS, "cpu_warmups": CPU_WARMUP, "cpu_repetitions": CPU_REPEATS,
                       "gpu_rep_order": "interleaved across variants", "vram_safety_margin_bytes":
                           VRAM_SAFETY_MARGIN_BYTES},
        "datasets": {"10k": "data/processed/subsets/10k/subset_10k.npz",
                     "25k": "data/processed/subsets/25k/subset_25k.npz",
                     SLICE_45K_LABEL: "first 45,000 rows of data/processed/subsets/100k/subset_100k.npz "
                                      "(in memory only; no file written)",
                     "dataset_note": "data/processed is a synthetic CICIDS2017-shaped placeholder"},
        "free_vram_bytes_before_each_subset": free_vram_by_subset,
        "ram_available_bytes_before_each_cpu_run": ram_available_by_subset,
        "cpu_ram_note": "Low available RAM can force paging and inflate CPU times; see ram_available_bytes_at_start.",
        "nsight_compute": {"exe": NCU_EXE, "status": "UNAVAILABLE", "error": ncu_error,
                           "fallback": "cudaFuncGetAttributes + cudaOccupancyMaxActiveBlocksPerMultiprocessor"},
    }
    with open(os.path.join(OUT_DIR, "benchmark_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    chart_stage_ladder(df)
    chart_kernel_breakdown(df)
    chart_speedup_vs_v1(df)
    chart_peak_memory(df, free_vram_by_subset.get(SLICE_45K_LABEL))
    chart_fp16_accuracy(corr_rows)

    summary = build_summary(df, corr_df, ncu_rows, ncu_error)
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    with open(os.path.join(OUT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write(build_report(df, corr_df, ncu_rows, ncu_error, metadata))
    log("Done")


def build_summary(df, corr_df, ncu_rows, ncu_error):
    out = {"status": "measured", "per_subset": {}}
    for subset in df["subset"].unique():
        sub = df[df["subset"] == subset]
        entries = {}
        for r in sub.itertuples():
            entries[r.variant_name] = {
                "classification": r.classification,
                "end_to_end_median_ms": None if pd.isna(r.end_to_end_median_ms) else r.end_to_end_median_ms,
                "distance_mean_ms": None if pd.isna(r.distance_mean_ms) else r.distance_mean_ms,
                "speedup_vs_cpu": None if pd.isna(r.speedup_vs_cpu) else r.speedup_vs_cpu,
                "speedup_vs_v0": None if pd.isna(r.speedup_vs_v0) else r.speedup_vs_v0,
                "speedup_vs_v1": None if pd.isna(r.speedup_vs_v1) else r.speedup_vs_v1,
                "peak_gpu_mem_bytes_tracked": None if pd.isna(r.peak_gpu_mem_bytes_tracked)
                else int(r.peak_gpu_mem_bytes_tracked),
                "unavailable_reason": r.unavailable_reason or None,
            }
        out["per_subset"][subset] = entries
    out["nsight_compute"] = {"status": "UNAVAILABLE", "error": ncu_error}
    return out


def fmt(x, digits=2, suffix=""):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{digits}f}{suffix}"


def build_report(df, corr_df, ncu_rows, ncu_error, metadata):
    lines = ["# M8 - Advanced (outside-syllabus) optimizations: measured results", ""]
    lines.append(f"Run: {metadata['started_utc']} -> {metadata['finished_utc']} (UTC). "
                 f"k={K}, contamination={CONTAMINATION}. GPU: {GPU_WARMUP} warm-ups + {GPU_REPEATS} timed reps "
                 f"interleaved across variants; CPU: {CPU_WARMUP} warm-up + {CPU_REPEATS} reps.")
    p = metadata["power"]
    lines.append(f"Power: AC plugged = {p['ac_power_plugged']}, scheme = {p['active_power_scheme']}.")
    lines.append("Dataset: synthetic CICIDS2017-shaped placeholder (data/processed).")
    lines.append("")
    lines.append("## End-to-end median time and speedups")
    lines.append("")
    lines.append("| Subset | Implementation | Exact? | Distance kernel (ms) | End-to-end median (ms) | vs CPU | vs V0 | vs V1 | Peak GPU mem (MiB) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in df.itertuples():
        if r.classification != "MEASURED":
            lines.append(f"| {r.subset} | {r.variant_name} | {r.exact_or_approximate} | UNAVAILABLE | "
                         f"UNAVAILABLE | - | - | - | - |")
            continue
        mem = "-" if pd.isna(r.peak_gpu_mem_bytes_tracked) else f"{r.peak_gpu_mem_bytes_tracked / 1024 ** 2:,.1f}"
        lines.append(f"| {r.subset} | {r.variant_name} | {r.exact_or_approximate} | {fmt(r.distance_mean_ms)} | "
                     f"{fmt(r.end_to_end_median_ms)} | {fmt(r.speedup_vs_cpu, 1, 'x')} | "
                     f"{fmt(r.speedup_vs_v0, 2, 'x')} | {fmt(r.speedup_vs_v1, 2, 'x')} | {mem} |")
    lines.append("")
    unavailable = df[df["classification"] != "MEASURED"]
    if not unavailable.empty:
        lines.append("## Not run (UNAVAILABLE)")
        lines.append("")
        for r in unavailable.itertuples():
            lines.append(f"- **{r.subset} / {r.variant_name}**: {r.unavailable_reason}")
        lines.append("")

    lines.append("## What each technique did")
    lines.append("")
    for subset in ("10k", "25k"):
        sub = df[(df["subset"] == subset) & (df["classification"] == "MEASURED")].set_index("variant_id")
        if 1 not in sub.index:
            continue
        v1_dist = sub.loc[1, "distance_mean_ms"]
        lines.append(f"**{subset}** (distance kernel V1 = {fmt(v1_dist)} ms)")
        for v in (2, 3, 4, 5):
            if v not in sub.index:
                continue
            ratio = v1_dist / sub.loc[v, "distance_mean_ms"]
            verdict = "faster" if ratio > 1.02 else ("slower" if ratio < 0.98 else "no meaningful change")
            lines.append(f"- {VARIANTS[v]['name']}: distance kernel {fmt(sub.loc[v, 'distance_mean_ms'])} ms "
                         f"({ratio:.2f}x vs V1, {verdict}); k-NN kernel {fmt(sub.loc[v, 'knn_mean_ms'])} ms; "
                         f"end-to-end {fmt(sub.loc[v, 'speedup_vs_v1'], 2, 'x')} vs V1.")
        lines.append("")

    lines.append("## Correctness")
    lines.append("")
    lines.append("| Subset | Reference | Candidate | Exact? | Max abs err | Pearson r | kNN recall@k | Anomaly agreement |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in corr_df.itertuples():
        lines.append(f"| {r.subset} | {r.reference} | {r.candidate_name} | {r.exact_or_approximate} | "
                     f"{r.max_abs_error:.3g} | {r.pearson_r:.8f} | {r.knn_recall_at_k:.4f} | "
                     f"{r.anomaly_flag_agreement:.4f} |")
    lines.append("")

    lines.append("## Kernel resource usage")
    lines.append("")
    lines.append(f"Nsight Compute: UNAVAILABLE - {ncu_error}. Hardware performance counters need "
                 "administrator rights or the driver's 'allow access to GPU performance counters' setting, "
                 "which this run did not change. Static values below come from the CUDA runtime instead.")
    lines.append("")
    att = pd.DataFrame(ncu_rows)
    att = att[att["classification"] != "UNAVAILABLE"]
    if not att.empty:
        pivot = att.pivot_table(index=["variant_name", "kernel"], columns="metric", values="value", aggfunc="first")
        cols = [c for c in ("registers_per_thread", "static_shared_mem_bytes_per_block", "local_mem_bytes_per_thread",
                            "max_active_blocks_per_sm", "theoretical_occupancy_pct") if c in pivot.columns]
        lines.append("| Variant | Kernel | " + " | ".join(cols) + " |")
        lines.append("|---|---|" + "---|" * len(cols))
        for (name, kernel), row in pivot[cols].iterrows():
            lines.append(f"| {name} | {kernel} | " + " | ".join(f"{row[c]:g}" for c in cols) + " |")
        lines.append("")
        lines.append("theoretical_occupancy_pct is ESTIMATED (resource-based), not achieved occupancy.")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
