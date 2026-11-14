"""
run_m5_optimization.py — Milestone M5: Profiling-Driven CUDA Optimization

Drives the real, measured naive-vs-optimized CUDA LOF comparison on the
10K/25K subsets, using the same warm-up + repeated-timing methodology as
M4 (2 discarded warm-up runs, 10 timed runs, CUDA-event timing via
CudaLofTimings / CudaLofTimingsOptimized). Nothing here is estimated.

Correctness ground truth: the FRESH CPU baseline in
results/m2_cpu_baseline/benchmarks_current/ (regenerated against the
CURRENT data/processed/subsets/*.npz, fixing the stale-baseline mismatch
flagged during M4).
"""

import csv
import json
import os
import statistics
import sys
import platform
from datetime import datetime, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "python"))

import numpy as np  # noqa: E402

from src.python.data.dataset import load_subset  # noqa: E402
from src.python.cuda.lof import benchmark_cuda_lof, is_lof_available  # noqa: E402
from src.python.cuda.lof_optimized import (  # noqa: E402
    benchmark_cuda_lof_optimized,
    compute_pairwise_distances_optimized,
    compute_knn_optimized,
    is_lof_optimized_available,
)
from src.python.cuda.distance import compute_pairwise_distances  # noqa: E402
from src.python.cuda.knn import compute_knn  # noqa: E402
from profiling.capability_report import generate_profiling_capability_report  # noqa: E402

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "m5_cuda_optimized")
CPU_BASELINE_DIR = os.path.join(PROJECT_ROOT, "results", "m2_cpu_baseline", "benchmarks_current")

STAGES = [
    "h2d", "distance_kernel", "knn_kernel", "reachability_kernel",
    "lrd_kernel", "lof_kernel", "d2h",
]
STAGE_KEY = {s: f"{s}_ms" for s in STAGES}

K = 20
CONTAMINATION = 0.10
N_WARMUP = 2
N_REPEATS = 10
SUBSETS = ("10k", "25k")


def ensure_dirs():
    for sub in ("", "charts", "tables"):
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)


def time_pipeline(bench_fn, X, k, n_warmup, n_repeats):
    """Run bench_fn (benchmark_cuda_lof or benchmark_cuda_lof_optimized) with
    warm-up + repeats, return (last_scores, {stage: [ms,...]}, end_to_end_list_ms)."""
    per_stage = {s: [] for s in STAGES}
    e2e = []
    scores = None
    for i in range(n_warmup + n_repeats):
        scores, timings = bench_fn(X, k=k)
        if i < n_warmup:
            continue
        for s in STAGES:
            per_stage[s].append(timings[STAGE_KEY[s]])
        e2e.append(timings["end_to_end_ms"])
    return scores, per_stage, e2e


def summarize(values):
    return {
        "mean_ms": round(statistics.mean(values), 5),
        "median_ms": round(statistics.median(values), 5),
        "min_ms": round(min(values), 5),
        "max_ms": round(max(values), 5),
        "std_ms": round(statistics.stdev(values), 5) if len(values) > 1 else 0.0,
        "repetitions": len(values),
    }


def anomaly_flags(scores, contamination):
    threshold = np.percentile(scores, 100.0 * (1.0 - contamination))
    return (scores >= threshold).astype(np.int8), float(threshold)


def load_fresh_cpu_scores(subset):
    path = os.path.join(CPU_BASELINE_DIR, f"lof_scores_{subset}_k{K}.npy")
    if not os.path.isfile(path):
        return None
    return np.load(path)


def correctness_block(cpu_scores, naive_scores, opt_scores, naive_knn, opt_knn):
    def errstats(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        diff = np.abs(a - b)
        corr = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None
        return {
            "max_abs_error": float(diff.max()),
            "mean_abs_error": float(diff.mean()),
            "pearson_r": corr,
        }

    def anomaly_agreement(a, b):
        flags_a, _ = anomaly_flags(a, CONTAMINATION)
        flags_b, _ = anomaly_flags(b, CONTAMINATION)
        return float(np.mean(flags_a == flags_b))

    block = {
        "cpu_vs_naive_cuda": {
            **errstats(cpu_scores, naive_scores),
            "anomaly_agreement": anomaly_agreement(cpu_scores, naive_scores),
        } if cpu_scores is not None else {"status": "UNAVAILABLE", "reason": "fresh CPU baseline not found"},
        "cpu_vs_optimized_cuda": {
            **errstats(cpu_scores, opt_scores),
            "anomaly_agreement": anomaly_agreement(cpu_scores, opt_scores),
        } if cpu_scores is not None else {"status": "UNAVAILABLE", "reason": "fresh CPU baseline not found"},
        "naive_cuda_vs_optimized_cuda": {
            **errstats(naive_scores, opt_scores),
            "anomaly_agreement": anomaly_agreement(naive_scores, opt_scores),
        },
    }

    knn_idx_match = float(np.mean(naive_knn[0] == opt_knn[0]))
    knn_dist_max_err = float(np.max(np.abs(naive_knn[1].astype(np.float64) - opt_knn[1].astype(np.float64))))
    block["naive_vs_optimized_knn"] = {
        "index_agreement_fraction": knn_idx_match,
        "distance_max_abs_error": knn_dist_max_err,
    }
    return block


def main():
    ensure_dirs()
    print("=" * 70)
    print("M5 — Profiling-Driven CUDA Optimization")
    print("=" * 70)

    if not (is_lof_available() and is_lof_optimized_available()):
        print("[FATAL] Both bin/cuda_lof.dll and bin/cuda_lof_optimized.dll must be built. Aborting.")
        sys.exit(1)

    capability_report = generate_profiling_capability_report(repo_root=PROJECT_ROOT)
    gpu_info = capability_report["hardware_snapshot"]["gpu"]

    optimization_metadata = {
        "milestone": "M5",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": gpu_info,
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "k": K,
        "contamination": CONTAMINATION,
        "n_warmup": N_WARMUP,
        "n_repeats": N_REPEATS,
        "optimizations_applied": [
            "Pairwise distance: 32x32 shared-memory tiled kernel (pairwise_distance_tiled_kernel) "
            "replacing the naive per-thread global-memory kernel. Justified by M4: 78.9-80.0% of "
            "kernel time at N=10K/25K.",
            "k-NN: right-sized top-k register buffer (OPT_KNN_MAX_K=32 vs. naive KNN_MAX_K=256) plus "
            "float4-vectorized row scan. Justified by M4: 19.9-20.9% of kernel time at N=10K/25K.",
        ],
        "optimizations_not_attempted": [
            "Reachability distance, LRD, LOF-scoring kernels left unmodified - M4 measured these as "
            "sub-millisecond and together under 0.2% of total kernel time at both 10K and 25K, so "
            "optimizing them would not move end-to-end performance measurably.",
        ],
        "optimized_knn_constraint": "Optimized k-NN/pipeline kernels support k <= 32 (project default k=20 is within range).",
    }

    kernel_comparison_rows = []
    optimization_benchmarks_rows = []
    correctness_report = {}
    per_subset_summary = {}

    for subset in SUBSETS:
        print(f"\n--- Subset: {subset} ---")
        X, _y_bin, _y_multi, _y_raw, _indices = load_subset(subset)
        n_samples, n_features = X.shape

        print("Timing naive pipeline...")
        naive_scores, naive_stage_times, naive_e2e = time_pipeline(
            benchmark_cuda_lof, X, K, N_WARMUP, N_REPEATS
        )
        print("Timing optimized pipeline...")
        opt_scores, opt_stage_times, opt_e2e = time_pipeline(
            benchmark_cuda_lof_optimized, X, K, N_WARMUP, N_REPEATS
        )

        naive_summary = {s: summarize(naive_stage_times[s]) for s in STAGES}
        opt_summary = {s: summarize(opt_stage_times[s]) for s in STAGES}
        naive_e2e_summary = summarize(naive_e2e)
        opt_e2e_summary = summarize(opt_e2e)

        for s in STAGES:
            n_mean = naive_summary[s]["mean_ms"]
            o_mean = opt_summary[s]["mean_ms"]
            speedup = (n_mean / o_mean) if o_mean > 0 else None
            kernel_comparison_rows.append([
                subset, s, n_mean, o_mean,
                round(speedup, 3) if speedup else "",
                round(100.0 * (n_mean - o_mean) / n_mean, 2) if n_mean > 0 else "",
            ])
            optimization_benchmarks_rows.append([
                subset, s, "naive", n_mean, naive_summary[s]["std_ms"], naive_summary[s]["repetitions"]
            ])
            optimization_benchmarks_rows.append([
                subset, s, "optimized", o_mean, opt_summary[s]["std_ms"], opt_summary[s]["repetitions"]
            ])

        e2e_speedup = naive_e2e_summary["mean_ms"] / opt_e2e_summary["mean_ms"]
        print(f"  Naive end-to-end:     {naive_e2e_summary['mean_ms']:.4f} ms")
        print(f"  Optimized end-to-end: {opt_e2e_summary['mean_ms']:.4f} ms")
        print(f"  End-to-end speedup:   {e2e_speedup:.2f}x")

        # Correctness: standalone distance/kNN comparison + full-pipeline score comparison
        print("Running standalone naive-vs-optimized distance/k-NN correctness check...")
        naive_dist = compute_pairwise_distances(X)
        opt_dist = compute_pairwise_distances_optimized(X)
        dist_max_err = float(np.max(np.abs(naive_dist.astype(np.float64) - opt_dist.astype(np.float64))))

        naive_knn = compute_knn(naive_dist, k=K)
        opt_knn = compute_knn_optimized(opt_dist, k=K)

        cpu_scores = load_fresh_cpu_scores(subset)
        corr_block = correctness_block(cpu_scores, naive_scores, opt_scores, naive_knn, opt_knn)
        corr_block["distance_matrix_naive_vs_optimized_max_abs_error"] = dist_max_err
        correctness_report[subset] = corr_block

        per_subset_summary[subset] = {
            "n_samples": n_samples,
            "n_features": n_features,
            "naive_end_to_end_ms": naive_e2e_summary,
            "optimized_end_to_end_ms": opt_e2e_summary,
            "end_to_end_speedup": round(e2e_speedup, 3),
            "naive_stage_means_ms": {s: naive_summary[s]["mean_ms"] for s in STAGES},
            "optimized_stage_means_ms": {s: opt_summary[s]["mean_ms"] for s in STAGES},
        }

    # --- write outputs ---
    with open(os.path.join(OUTPUT_DIR, "kernel_comparison.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subset", "stage", "naive_mean_ms", "optimized_mean_ms", "speedup", "pct_improvement"])
        w.writerows(kernel_comparison_rows)

    with open(os.path.join(OUTPUT_DIR, "optimization_benchmarks.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subset", "stage", "implementation", "mean_ms", "std_ms", "repetitions"])
        w.writerows(optimization_benchmarks_rows)

    with open(os.path.join(OUTPUT_DIR, "correctness_report.json"), "w", encoding="utf-8") as f:
        json.dump(correctness_report, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "optimization_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(optimization_metadata, f, indent=2)

    summary = {
        "milestone": "M5",
        "status": "measured",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "subsets": per_subset_summary,
    }
    with open(os.path.join(OUTPUT_DIR, "optimization_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote results to {OUTPUT_DIR}")
    print("\n" + "=" * 70)
    print("M5 status: measured")
    print("=" * 70)


if __name__ == "__main__":
    main()
