"""
run_cuda_baseline.py — Formal Reproducible Benchmark for Naive CUDA LOF Baseline
Milestone M3: CUDA LOF Baseline (Step 5)

Executes formal benchmark across verified dense-matrix subsets (10K, 25K):
  - Hardware event timings for each kernel stage (Distance, k-NN, Reachability, LRD, LOF)
  - Data transfer timings (H2D, D2H)
  - End-to-end wall-clock & GPU event timings
  - Statistical aggregation (mean, std, min, max across iterations)
  - Warm-up policy (un-timed warm-up run excluded from benchmark statistics)
  - Correctness verification against historical M2 CPU baseline
  - Summary export to JSON, CSV, and NumPy arrays

Hardware Target: NVIDIA GeForce RTX 3050 6GB Laptop GPU (sm_86, CC 8.6)
Dataset: CICIDS2017 (D=68 standardized continuous features, k=20)
"""

import os
import sys
import json
import time
import csv
import argparse
import platform
import subprocess
import ctypes
from typing import Dict, Any, List, Tuple
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.python.data.dataset import load_subset
from src.python.cuda.lof import benchmark_cuda_lof, is_lof_available
from src.python.cpu.cpu_lof import CPULocalOutlierFactor


HISTORICAL_M2_BASELINE = {
    "10k": {
        "n_samples": 10000,
        "n_features": 68,
        "k": 20,
        "contamination": 0.1,
        "threshold": 1.90915,
        "distance_matrix_seconds": 1.39061,
        "knn_search_seconds": 1.66738,
        "reachability_distance_seconds": 0.00417,
        "lrd_computation_seconds": 0.00028,
        "lof_scoring_seconds": 0.00134,
        "total_lof_compute_seconds": 3.06379,
        "data_loading_seconds": 0.01756,
        "total_execution_seconds": 3.08135,
        "throughput_records_per_sec": 10000.0 / 3.08135,
        "predicted_anomalies": 1000,
    },
    "25k": {
        "n_samples": 25000,
        "n_features": 68,
        "k": 20,
        "contamination": 0.1,
        "threshold": 1.9565,
        "distance_matrix_seconds": 8.1257,
        "knn_search_seconds": 11.5173,
        "reachability_distance_seconds": 0.0125,
        "lrd_computation_seconds": 0.0006,
        "lof_scoring_seconds": 0.0034,
        "total_lof_compute_seconds": 19.6596,
        "data_loading_seconds": 0.0381,
        "total_execution_seconds": 19.6976,
        "throughput_records_per_sec": 25000.0 / 19.6976,
        "predicted_anomalies": 2500,
    }
}


def get_hardware_environment() -> Dict[str, Any]:
    """Capture full host CPU, memory, OS, Python, and GPU hardware metadata."""
    env = {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "architecture": platform.machine(),
        },
        "python": {
            "version": platform.python_version(),
            "compiler": platform.python_compiler(),
            "implementation": platform.python_implementation(),
        },
        "cpu": {
            "model": "13th Gen Intel(R) Core(TM) i5-13450HX",
            "physical_cores": 10,
            "logical_processors": 16,
        },
        "gpu": {
            "model": "NVIDIA GeForce RTX 3050 6GB Laptop GPU",
            "compute_capability": "8.6",
            "sm_count": 20,
            "total_vram_bytes": 6441926656,
            "total_vram_gb": 6.0,
            "cuda_toolkit": "12.8",
            "nvcc_version": "V12.8.61",
            "driver_version": "581.86",
            "msvc_version": "v143 / 14.38.33130",
        },
    }
    return env


def compute_correctness_diagnostics(
    cpu_scores: np.ndarray,
    cuda_scores: np.ndarray,
    contamination: float = 0.1
) -> Dict[str, Any]:
    """
    Compute rigorous correctness metrics between CPU and CUDA LOF scores:
      - Max absolute difference
      - Mean absolute difference
      - Pearson correlation
      - Anomaly classification agreement at contamination threshold
    """
    abs_diff = np.abs(cpu_scores - cuda_scores)
    max_abs_diff = float(np.max(abs_diff))
    mean_abs_diff = float(np.mean(abs_diff))

    # Pearson correlation
    corr_matrix = np.corrcoef(cpu_scores, cuda_scores)
    pearson_corr = float(corr_matrix[0, 1])

    # Thresholds at given contamination (e.g. 10%)
    cpu_thresh = float(np.percentile(cpu_scores, 100.0 * (1.0 - contamination)))
    cuda_thresh = float(np.percentile(cuda_scores, 100.0 * (1.0 - contamination)))

    cpu_preds = np.where(cpu_scores >= cpu_thresh, 1, 0).astype(np.int8)
    cuda_preds = np.where(cuda_scores >= cuda_thresh, 1, 0).astype(np.int8)

    disagreements = int(np.sum(cpu_preds != cuda_preds))
    agreement_ratio = float(np.mean(cpu_preds == cuda_preds))

    return {
        "max_absolute_difference": max_abs_diff,
        "mean_absolute_difference": mean_abs_diff,
        "pearson_correlation": pearson_corr,
        "cpu_threshold": cpu_thresh,
        "cuda_threshold": cuda_thresh,
        "cpu_anomaly_count": int(np.sum(cpu_preds == 1)),
        "cuda_anomaly_count": int(np.sum(cuda_preds == 1)),
        "anomaly_count_agreement_ratio": agreement_ratio,
        "anomaly_count_agreement_percent": agreement_ratio * 100.0,
        "prediction_disagreements": disagreements,
    }


def run_benchmark_for_subset(
    subset_name: str,
    k: int = 20,
    contamination: float = 0.1,
    num_iterations: int = 5,
    num_warmup: int = 1,
    output_dir: str = "results/m3_cuda_baseline/benchmarks"
) -> Dict[str, Any]:
    """Execute formal multi-iteration benchmark for a single dataset subset."""
    print("=" * 75)
    print(f"BENCHMARKING NAIVE CUDA LOF BASELINE — Subset: {subset_name.upper()} (k={k})")
    print(f"Iterations: {num_iterations} timed (+ {num_warmup} warmup)")
    print("=" * 75)

    # 1. Load Data
    t0_load = time.perf_counter()
    X, y_bin, y_multi, y_raw, indices = load_subset(subset_name)
    load_time_s = time.perf_counter() - t0_load
    N, D = X.shape
    print(f"[*] Loaded dataset {subset_name}: N={N:,}, D={D} features in {load_time_s:.4f}s")

    # 2. Warm-up executions (strictly excluded from reported timings)
    for w in range(num_warmup):
        print(f"[*] Executing warm-up run {w+1}/{num_warmup} (excluded from timings)...")
        _, _ = benchmark_cuda_lof(X, k=k)
    print("[*] Warm-up completed.")

    # 3. Timed Iterations
    iteration_records = []
    final_lof_scores = None

    for it in range(num_iterations):
        t0_wall = time.perf_counter()
        scores, timings = benchmark_cuda_lof(X, k=k)
        wall_time_s = time.perf_counter() - t0_wall
        final_lof_scores = scores

        timings["host_wall_time_seconds"] = wall_time_s
        iteration_records.append(timings)

        print(f"    [Iteration {it+1}/{num_iterations}] "
              f"Kernel: {timings['total_kernel_ms']:.2f} ms | "
              f"E2E: {timings['end_to_end_ms']:.2f} ms | "
              f"Wall: {wall_time_s*1000.0:.2f} ms")

    # 4. Statistical Aggregation
    timing_keys = [
        "h2d_ms", "distance_kernel_ms", "knn_kernel_ms",
        "reachability_kernel_ms", "lrd_kernel_ms", "lof_kernel_ms",
        "d2h_ms", "total_kernel_ms", "end_to_end_ms"
    ]

    aggregated_timings = {}
    for key in timing_keys:
        vals = [r[key] for r in iteration_records]
        sec_vals = [v / 1000.0 for v in vals]
        aggregated_timings[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
        aggregated_timings[key.replace("_ms", "_seconds")] = {
            "mean": float(np.mean(sec_vals)),
            "std": float(np.std(sec_vals)),
            "min": float(np.min(sec_vals)),
            "max": float(np.max(sec_vals)),
        }

    # Mean end-to-end and throughput
    mean_e2e_ms = aggregated_timings["end_to_end_ms"]["mean"]
    mean_e2e_s = mean_e2e_ms / 1000.0
    mean_kernel_ms = aggregated_timings["total_kernel_ms"]["mean"]
    mean_kernel_s = mean_kernel_ms / 1000.0
    throughput = float(N / mean_e2e_s)

    # 5. Compare with Historical M2 CPU Baseline
    m2_ref = HISTORICAL_M2_BASELINE.get(subset_name, {})
    cpu_e2e_s = m2_ref.get("total_execution_seconds", 0.0)
    cpu_compute_s = m2_ref.get("total_lof_compute_seconds", 0.0)
    cpu_throughput = m2_ref.get("throughput_records_per_sec", 0.0)

    e2e_speedup = float(cpu_e2e_s / mean_e2e_s) if mean_e2e_s > 0 else 0.0
    kernel_speedup = float(cpu_compute_s / mean_kernel_s) if mean_kernel_s > 0 else 0.0

    # 6. Load Historical CPU Scores and Compute Correctness
    cpu_scores_path = os.path.join(
        PROJECT_ROOT, "results", "m2_cpu_baseline", "benchmarks", f"lof_scores_{subset_name}_k20.npy"
    )
    correctness = {}
    if os.path.isfile(cpu_scores_path):
        cpu_scores = np.load(cpu_scores_path)
        correctness = compute_correctness_diagnostics(cpu_scores, final_lof_scores, contamination=contamination)
        print(f"[*] Correctness vs Historical M2 CPU:")
        print(f"    - Pearson Correlation       : {correctness['pearson_correlation']:.8f}")
        print(f"    - Max Absolute Difference   : {correctness['max_absolute_difference']:.6e}")
        print(f"    - Mean Absolute Difference  : {correctness['mean_absolute_difference']:.6e}")
        print(f"    - Anomaly Agreement         : {correctness['anomaly_count_agreement_percent']:.2f}% "
              f"({N - correctness['prediction_disagreements']}/{N})")
        print(f"    - CPU Anomalies             : {correctness['cpu_anomaly_count']}")
        print(f"    - CUDA Anomalies            : {correctness['cuda_anomaly_count']}")

        # For 25K, also run the same-distance diagnostic to isolate distance formulation differences
        if subset_name == "25k":
            print("[*] Running 25K same-distance-matrix numerical diagnostic...")
            from src.python.cuda.distance import compute_pairwise_distances
            D_cuda = compute_pairwise_distances(X)
            cpu_diag = CPULocalOutlierFactor(k=k)
            knn_idx, knn_dist, k_dist = cpu_diag.compute_k_neighbors(D_cuda)
            r_dist = cpu_diag.compute_reachability_distances(D_cuda, knn_idx, k_dist)
            lrd_diag = cpu_diag.compute_lrd(r_dist)
            cpu_lof_diag = cpu_diag.compute_lof_scores(lrd_diag, knn_idx)

            diag_corr = float(np.corrcoef(cpu_lof_diag, final_lof_scores)[0, 1])
            diag_max = float(np.max(np.abs(cpu_lof_diag - final_lof_scores)))
            diag_mean = float(np.mean(np.abs(cpu_lof_diag - final_lof_scores)))

            correctness["same_distance_diagnostic"] = {
                "description": (
                    "Diagnostic numerical consistency check executing CPU LOF stages directly on "
                    "CUDA pairwise distance matrix to isolate distance formulation differences. "
                    "This is a correctness diagnostic and NOT the CPU performance baseline."
                ),
                "pearson_correlation": diag_corr,
                "max_absolute_difference": diag_max,
                "mean_absolute_difference": diag_mean,
                "is_performance_baseline": False,
            }
            print(f"    [Diagnostic] Same-Distance Pearson Correlation : {diag_corr:.8f}")
            print(f"    [Diagnostic] Same-Distance Max Abs Diff        : {diag_max:.6e}")
            print(f"    [Diagnostic] Same-Distance Mean Abs Diff       : {diag_mean:.6e}")
    else:
        print(f"[WARNING] Historical CPU score file not found at '{cpu_scores_path}'")

    # Score statistics for CUDA LOF
    scores_stats = {
        "min": float(np.min(final_lof_scores)),
        "max": float(np.max(final_lof_scores)),
        "mean": float(np.mean(final_lof_scores)),
        "std": float(np.std(final_lof_scores)),
        "median": float(np.median(final_lof_scores)),
        "p25": float(np.percentile(final_lof_scores, 25)),
        "p75": float(np.percentile(final_lof_scores, 75)),
        "p90": float(np.percentile(final_lof_scores, 90)),
        "p95": float(np.percentile(final_lof_scores, 95)),
        "p99": float(np.percentile(final_lof_scores, 99)),
    }

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    benchmark_data = {
        "dataset_name": "CICIDS2017",
        "subset_name": subset_name,
        "n_samples": N,
        "n_features": D,
        "k": k,
        "contamination": contamination,
        "num_iterations": num_iterations,
        "num_warmup": num_warmup,
        "warmup_policy": "1 initial warm-up execution strictly excluded from reported statistics",
        "timing_methodology": "CUDA Hardware Events (cudaEventElapsedTime) with host synchronization",
        "hardware": get_hardware_environment(),
        "timings_aggregated": aggregated_timings,
        "iteration_raw_timings": iteration_records,
        "performance_summary": {
            "cuda_h2d_seconds": aggregated_timings["h2d_seconds"]["mean"],
            "cuda_distance_kernel_seconds": aggregated_timings["distance_kernel_seconds"]["mean"],
            "cuda_knn_kernel_seconds": aggregated_timings["knn_kernel_seconds"]["mean"],
            "cuda_reachability_kernel_seconds": aggregated_timings["reachability_kernel_seconds"]["mean"],
            "cuda_lrd_kernel_seconds": aggregated_timings["lrd_kernel_seconds"]["mean"],
            "cuda_lof_kernel_seconds": aggregated_timings["lof_kernel_seconds"]["mean"],
            "cuda_total_kernel_seconds": mean_kernel_s,
            "cuda_d2h_seconds": aggregated_timings["d2h_seconds"]["mean"],
            "cuda_end_to_end_seconds": mean_e2e_s,
            "cuda_throughput_records_per_sec": throughput,
            "cpu_end_to_end_seconds": cpu_e2e_s,
            "cpu_compute_seconds": cpu_compute_s,
            "cpu_throughput_records_per_sec": cpu_throughput,
            "end_to_end_speedup": e2e_speedup,
            "kernel_compute_speedup": kernel_speedup,
        },
        "score_statistics": scores_stats,
        "correctness": correctness,
        "memory_limit_note": (
            "Dense 100K+ baseline is excluded because a dense float32 distance matrix "
            "requires O(N^2 * 4 bytes). At N=100,000 this is 40.0 GB, which exceeds the physical "
            "6 GB VRAM of the NVIDIA RTX 3050 Laptop GPU."
        )
    }

    # Write individual JSON
    json_path = os.path.join(output_dir, f"cuda_lof_benchmark_{subset_name}_k{k}.json")
    with open(json_path, "w") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"[*] Benchmark JSON manifest saved to: '{json_path}'")

    # Save CUDA LOF scores
    scores_path = os.path.join(output_dir, f"lof_scores_cuda_{subset_name}_k{k}.npy")
    np.save(scores_path, final_lof_scores)
    print(f"[*] CUDA LOF scores saved to: '{scores_path}'")

    return benchmark_data


def run_all_benchmarks(
    subsets: List[str] = ["10k", "25k"],
    k: int = 20,
    contamination: float = 0.1,
    output_dir: str = "results/m3_cuda_baseline/benchmarks"
) -> Dict[str, Any]:
    """Execute complete formal M3 benchmark suite across subsets and generate summary files."""
    if not is_lof_available():
        raise RuntimeError("CUDA LOF shared library (bin/cuda_lof.dll) is not available. Please build it first.")

    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for subset in subsets:
        # Iteration configuration: 5 iterations for 10k, 3 iterations for 25k
        n_iters = 5 if subset == "10k" else 3
        res = run_benchmark_for_subset(
            subset_name=subset,
            k=k,
            contamination=contamination,
            num_iterations=n_iters,
            num_warmup=1,
            output_dir=output_dir
        )
        all_results[subset] = res

    # Write unified summary JSON
    summary_json_path = os.path.join(output_dir, "m3_cuda_benchmark_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[*] Unified M3 Benchmark JSON summary saved to: '{summary_json_path}'")

    # Write summary CSV
    summary_csv_path = os.path.join(output_dir, "m3_cuda_benchmark_summary.csv")
    fieldnames = [
        "subset_name", "n_samples", "n_features", "k", "num_iterations",
        "cuda_h2d_ms", "cuda_distance_kernel_ms", "cuda_knn_kernel_ms",
        "cuda_reachability_kernel_ms", "cuda_lrd_kernel_ms", "cuda_lof_kernel_ms",
        "cuda_total_kernel_ms", "cuda_d2h_ms", "cuda_end_to_end_ms",
        "cuda_end_to_end_seconds", "cuda_throughput_rec_sec",
        "cpu_end_to_end_seconds", "cpu_throughput_rec_sec",
        "end_to_end_speedup", "kernel_compute_speedup",
        "pearson_correlation", "max_absolute_difference", "mean_absolute_difference",
        "anomaly_agreement_percent", "prediction_disagreements"
    ]

    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sname, res in all_results.items():
            t_agg = res["timings_aggregated"]
            perf = res["performance_summary"]
            corr = res.get("correctness", {})
            row = {
                "subset_name": sname,
                "n_samples": res["n_samples"],
                "n_features": res["n_features"],
                "k": res["k"],
                "num_iterations": res["num_iterations"],
                "cuda_h2d_ms": t_agg["h2d_ms"]["mean"],
                "cuda_distance_kernel_ms": t_agg["distance_kernel_ms"]["mean"],
                "cuda_knn_kernel_ms": t_agg["knn_kernel_ms"]["mean"],
                "cuda_reachability_kernel_ms": t_agg["reachability_kernel_ms"]["mean"],
                "cuda_lrd_kernel_ms": t_agg["lrd_kernel_ms"]["mean"],
                "cuda_lof_kernel_ms": t_agg["lof_kernel_ms"]["mean"],
                "cuda_total_kernel_ms": t_agg["total_kernel_ms"]["mean"],
                "cuda_d2h_ms": t_agg["d2h_ms"]["mean"],
                "cuda_end_to_end_ms": t_agg["end_to_end_ms"]["mean"],
                "cuda_end_to_end_seconds": perf["cuda_end_to_end_seconds"],
                "cuda_throughput_rec_sec": perf["cuda_throughput_records_per_sec"],
                "cpu_end_to_end_seconds": perf["cpu_end_to_end_seconds"],
                "cpu_throughput_rec_sec": perf["cpu_throughput_records_per_sec"],
                "end_to_end_speedup": perf["end_to_end_speedup"],
                "kernel_compute_speedup": perf["kernel_compute_speedup"],
                "pearson_correlation": corr.get("pearson_correlation", 0.0),
                "max_absolute_difference": corr.get("max_absolute_difference", 0.0),
                "mean_absolute_difference": corr.get("mean_absolute_difference", 0.0),
                "anomaly_agreement_percent": corr.get("anomaly_count_agreement_percent", 0.0),
                "prediction_disagreements": corr.get("prediction_disagreements", 0),
            }
            writer.writerow(row)
    print(f"[*] Unified M3 Benchmark CSV summary saved to: '{summary_csv_path}'")

    # Print final console table
    print("\n" + "=" * 90)
    print("M3 NAIVE CUDA LOF BASELINE BENCHMARK SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Metric':<35} | {'10K Subset':<22} | {'25K Subset':<22}")
    print("-" * 90)
    for s in ["10k", "25k"]:
        if s not in all_results:
            continue
    res10 = all_results.get("10k", {})
    res25 = all_results.get("25k", {})

    def fmt_val(d, key, subkey="mean", unit=""):
        if not d:
            return "N/A"
        val = d["timings_aggregated"].get(key, {}).get(subkey, 0.0)
        return f"{val:.3f} {unit}".strip()

    def fmt_perf(d, key, unit=""):
        if not d:
            return "N/A"
        val = d["performance_summary"].get(key, 0.0)
        return f"{val:.3f} {unit}".strip()

    def fmt_corr(d, key, fmt=".6f"):
        if not d:
            return "N/A"
        val = d.get("correctness", {}).get(key, 0.0)
        return f"{val:{fmt}}"

    print(f"{'H2D Transfer Time':<35} | {fmt_val(res10, 'h2d_ms', unit='ms'):<22} | {fmt_val(res25, 'h2d_ms', unit='ms'):<22}")
    print(f"{'Distance Kernel Time':<35} | {fmt_val(res10, 'distance_kernel_ms', unit='ms'):<22} | {fmt_val(res25, 'distance_kernel_ms', unit='ms'):<22}")
    print(f"{'k-NN Kernel Time':<35} | {fmt_val(res10, 'knn_kernel_ms', unit='ms'):<22} | {fmt_val(res25, 'knn_kernel_ms', unit='ms'):<22}")
    print(f"{'Reachability Kernel Time':<35} | {fmt_val(res10, 'reachability_kernel_ms', unit='ms'):<22} | {fmt_val(res25, 'reachability_kernel_ms', unit='ms'):<22}")
    print(f"{'LRD Kernel Time':<35} | {fmt_val(res10, 'lrd_kernel_ms', unit='ms'):<22} | {fmt_val(res25, 'lrd_kernel_ms', unit='ms'):<22}")
    print(f"{'LOF Scoring Kernel Time':<35} | {fmt_val(res10, 'lof_kernel_ms', unit='ms'):<22} | {fmt_val(res25, 'lof_kernel_ms', unit='ms'):<22}")
    print(f"{'D2H Transfer Time':<35} | {fmt_val(res10, 'd2h_ms', unit='ms'):<22} | {fmt_val(res25, 'd2h_ms', unit='ms'):<22}")
    print(f"{'Total CUDA Kernel Time':<35} | {fmt_val(res10, 'total_kernel_ms', unit='ms'):<22} | {fmt_val(res25, 'total_kernel_ms', unit='ms'):<22}")
    print(f"{'CUDA End-to-End Time':<35} | {fmt_perf(res10, 'cuda_end_to_end_seconds', unit='s'):<22} | {fmt_perf(res25, 'cuda_end_to_end_seconds', unit='s'):<22}")
    print(f"{'CUDA Throughput':<35} | {fmt_perf(res10, 'cuda_throughput_records_per_sec', unit='rec/s'):<22} | {fmt_perf(res25, 'cuda_throughput_records_per_sec', unit='rec/s'):<22}")
    print("-" * 90)
    print(f"{'CPU End-to-End Time (M2)':<35} | {fmt_perf(res10, 'cpu_end_to_end_seconds', unit='s'):<22} | {fmt_perf(res25, 'cpu_end_to_end_seconds', unit='s'):<22}")
    print(f"{'CPU Throughput (M2)':<35} | {fmt_perf(res10, 'cpu_throughput_records_per_sec', unit='rec/s'):<22} | {fmt_perf(res25, 'cpu_throughput_records_per_sec', unit='rec/s'):<22}")
    print(f"{'End-to-End Speedup (CPU / CUDA)':<35} | {fmt_perf(res10, 'end_to_end_speedup', unit='x'):<22} | {fmt_perf(res25, 'end_to_end_speedup', unit='x'):<22}")
    print(f"{'Kernel Speedup (Compute)':<35} | {fmt_perf(res10, 'kernel_compute_speedup', unit='x'):<22} | {fmt_perf(res25, 'kernel_compute_speedup', unit='x'):<22}")
    print("-" * 90)
    print(f"{'Pearson Correlation':<35} | {fmt_corr(res10, 'pearson_correlation', '.8f'):<22} | {fmt_corr(res25, 'pearson_correlation', '.8f'):<22}")
    print(f"{'Anomaly Agreement':<35} | {fmt_corr(res10, 'anomaly_count_agreement_percent', '.2f') + '%' :<22} | {fmt_corr(res25, 'anomaly_count_agreement_percent', '.2f') + '%' :<22}")
    print(f"{'Prediction Disagreements':<35} | {str(res10.get('correctness', {}).get('prediction_disagreements', 'N/A')):<22} | {str(res25.get('correctness', {}).get('prediction_disagreements', 'N/A')):<22}")
    print("=" * 90 + "\n")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Formal Naive CUDA LOF Benchmark Runner (M3 Step 5).")
    parser.add_argument("--subsets", nargs="+", default=["10k", "25k"], help="Subsets to benchmark (default: 10k 25k)")
    parser.add_argument("--k", type=int, default=20, help="k nearest neighbors (default: 20)")
    parser.add_argument("--contamination", type=float, default=0.1, help="Contamination threshold (default: 0.1)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "results", "m3_cuda_baseline", "benchmarks"),
        help="Directory to save benchmark results"
    )
    args = parser.parse_args()

    run_all_benchmarks(
        subsets=args.subsets,
        k=args.k,
        contamination=args.contamination,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
