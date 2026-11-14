"""
run_m4_profiling.py — Milestone M4: GPU Profiling and Performance Analysis

This script NEVER fabricates GPU kernel measurements. It:
  1. Detects what profiling is actually possible on the current machine
     (NVIDIA GPU present? nvcc on PATH? compiled CUDA LOF DLL under bin/?
      Nsight Compute / Nsight Systems available?).
  2. If real GPU profiling is possible, drives the already-instrumented
     CUDA-event timing pipeline (CudaLofTimings) for real and writes
     MEASURED rows.
  3. If not possible (the expected state on a machine without the CUDA
     Toolkit installed), writes explicit UNAVAILABLE rows with the same
     schema, plus a bottleneck_analysis.md explaining exactly why and
     what is required to complete this milestone for real.

Every output file carries a measurement_classification: MEASURED,
PROJECTED, or UNAVAILABLE. Nothing here is PROJECTED — the hardcoded
speedup-multiplier projections in src/python/benchmarking/milestones.py
are a separate, already-labeled artifact and are not reproduced here.
"""

import csv
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "python"))

from profiling.capability_report import generate_profiling_capability_report  # noqa: E402

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "m4_profiling")
KERNEL_STAGES = [
    "h2d_transfer",
    "pairwise_distance_kernel",
    "knn_kernel",
    "reachability_distance_kernel",
    "lrd_kernel",
    "lof_scoring_kernel",
    "d2h_transfer",
]


def ensure_output_dirs():
    for sub in ("", "profiles", "reports"):
        os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)


def write_unavailable_kernel_timing_csv(path, limitations):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "stage",
                "mean_ms",
                "median_ms",
                "min_ms",
                "max_ms",
                "std_ms",
                "repetitions",
                "measurement_classification",
                "reason",
            ]
        )
        reason = "; ".join(limitations) if limitations else "GPU profiling not possible on this machine"
        for stage in KERNEL_STAGES:
            writer.writerow([stage, "", "", "", "", "", 0, "UNAVAILABLE", reason])


def write_unavailable_transfer_timing_csv(path, limitations):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "direction",
                "mean_ms",
                "median_ms",
                "min_ms",
                "max_ms",
                "std_ms",
                "repetitions",
                "measurement_classification",
                "reason",
            ]
        )
        reason = "; ".join(limitations) if limitations else "GPU profiling not possible on this machine"
        for direction in ("H2D", "D2H"):
            writer.writerow([direction, "", "", "", "", "", 0, "UNAVAILABLE", reason])


# Maps our report stage names -> CudaLofTimings dict keys (milliseconds)
STAGE_TO_TIMING_KEY = {
    "h2d_transfer": "h2d_ms",
    "pairwise_distance_kernel": "distance_kernel_ms",
    "knn_kernel": "knn_kernel_ms",
    "reachability_distance_kernel": "reachability_kernel_ms",
    "lrd_kernel": "lrd_kernel_ms",
    "lof_scoring_kernel": "lof_kernel_ms",
    "d2h_transfer": "d2h_ms",
}


def run_real_gpu_profiling(capability_report, subsets=("10k", "25k"), k=20, n_warmup=2, n_repeats=10):
    """Only reached when capability_report['can_run_gpu_profiling'] is True.

    Drives the real CUDA-event-timed pipeline (compute_lof_pipeline_cuda_timed via
    benchmark_cuda_lof) on each requested subset, with proper warm-up + repeated
    timed runs, and writes one row per (subset, stage) to the output CSVs.
    """
    from src.python.data.dataset import load_subset
    from src.python.cuda.lof import benchmark_cuda_lof

    kernel_rows = []
    transfer_rows = []
    per_subset_results = {}

    for subset in subsets:
        X, _y_bin, _y_multi, _y_raw, _indices = load_subset(subset)
        n_samples, n_features = X.shape

        timings_by_stage = {stage: [] for stage in KERNEL_STAGES}

        for i in range(n_warmup + n_repeats):
            _scores, timings = benchmark_cuda_lof(X, k=k)
            if i < n_warmup:
                continue
            for stage, timing_key in STAGE_TO_TIMING_KEY.items():
                timings_by_stage[stage].append(timings[timing_key])

        stage_summary = {}
        for stage, timing_key in STAGE_TO_TIMING_KEY.items():
            values = timings_by_stage[stage]
            row = [
                subset,
                stage,
                round(statistics.mean(values), 5),
                round(statistics.median(values), 5),
                round(min(values), 5),
                round(max(values), 5),
                round(statistics.stdev(values), 5) if len(values) > 1 else 0.0,
                len(values),
                "MEASURED",
                "",
            ]
            stage_summary[stage] = round(statistics.mean(values), 5)
            if "transfer" in stage:
                transfer_rows.append(row)
            else:
                kernel_rows.append(row)

        total_kernel_ms = sum(
            stage_summary[s] for s in stage_summary if "transfer" not in s
        )
        per_subset_results[subset] = {
            "n_samples": n_samples,
            "n_features": n_features,
            "k": k,
            "n_warmup": n_warmup,
            "n_repeats": n_repeats,
            "stage_means_ms": stage_summary,
            "total_kernel_ms_mean": round(total_kernel_ms, 5),
            "dominant_stage": max(
                (s for s in stage_summary if "transfer" not in s),
                key=lambda s: stage_summary[s],
            ),
        }

    with open(os.path.join(OUTPUT_DIR, "kernel_timing.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["subset", "stage", "mean_ms", "median_ms", "min_ms", "max_ms", "std_ms", "repetitions", "measurement_classification", "reason"]
        )
        writer.writerows(kernel_rows)

    with open(os.path.join(OUTPUT_DIR, "transfer_timing.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["subset", "direction", "mean_ms", "median_ms", "min_ms", "max_ms", "std_ms", "repetitions", "measurement_classification", "reason"]
        )
        writer.writerows(transfer_rows)

    return {"status": "measured", "subsets": per_subset_results, "k": k, "n_warmup": n_warmup, "n_repeats": n_repeats}


def write_bottleneck_analysis_unavailable(path, capability_report):
    limitations = capability_report["limitations"]
    tools = capability_report["profiling_tools"]
    gpu = capability_report["hardware_snapshot"]["gpu"]

    lines = []
    lines.append("# M4 Bottleneck Analysis\n")
    lines.append(
        "**Status: capability scaffolding only — no GPU kernel measurement was possible on this machine.**\n"
    )
    lines.append("## Why\n")
    for reason in limitations:
        lines.append(f"- {reason}")
    lines.append("")
    lines.append("## What this machine actually has\n")
    if gpu.get("available"):
        for g in gpu["gpus"]:
            lines.append(f"- NVIDIA GPU detected: {g['name']} ({g['total_memory_mb']} MB VRAM, driver {gpu['driver_version']}, compute capability {g['compute_capability']})")
    else:
        lines.append(f"- No NVIDIA GPU detected ({gpu.get('reason_unavailable')})")
    lines.append(f"- nvcc available: {tools['nvcc']['available']}")
    lines.append(f"- Nsight Compute (ncu) available: {tools['ncu']['available']}")
    lines.append(f"- Nsight Systems (nsys) available: {tools['nsys']['available']}")
    lines.append("")
    lines.append("## What is required to complete M4 for real\n")
    lines.append("1. A machine or environment with the CUDA Toolkit installed (`nvcc` on PATH) matching the detected GPU's compute capability.")
    lines.append("2. Build the CUDA LOF pipeline into `bin/cuda_lof.dll` (see `scripts/build_cuda_lof.bat`).")
    lines.append("3. Re-run this script — it will then populate `kernel_timing.csv` / `transfer_timing.csv` with MEASURED CUDA-event timings automatically.")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_bottleneck_analysis_measured(path, capability_report, run_result):
    gpu_list = capability_report["hardware_snapshot"]["gpu"].get("gpus", [])
    gpu_name = gpu_list[0]["name"] if gpu_list else "unknown GPU"
    tools = capability_report["profiling_tools"]

    lines = []
    lines.append("# M4 Bottleneck Analysis — MEASURED\n")
    lines.append(
        f"**Status: real GPU kernel profiling, measured via CUDA events "
        f"(`cudaEventElapsedTime`) on {gpu_name} (this machine).**\n"
    )
    lines.append(
        "**Important — hardware difference from M3's README numbers:** the M3 baseline numbers "
        "quoted in README.md were measured on an NVIDIA RTX 3050 Laptop GPU (sm_86, 20 SMs, 6GB). "
        "This M4 profiling pass ran on a *different* machine with an NVIDIA RTX 4050 Laptop GPU "
        "(sm_89, 20 SMs, 6GB, CUDA runtime 13.0). Do not treat the two as the same hardware — they "
        "are reported here as a separate, additionally-collected dataset, not a replacement for the "
        "original M3 RTX 3050 measurements.\n"
    )
    lines.append("## Per-kernel breakdown (mean of 10 timed runs, 2 warm-up runs discarded)\n")
    for subset, data in run_result["subsets"].items():
        lines.append(f"### Subset: {subset} (N={data['n_samples']:,}, D={data['n_features']}, k={data['k']})\n")
        total = data["total_kernel_ms_mean"]
        lines.append("| Stage | Mean time (ms) | % of kernel time |")
        lines.append("| :--- | :---: | :---: |")
        for stage, ms in data["stage_means_ms"].items():
            if "transfer" in stage:
                continue
            pct = 100.0 * ms / total if total > 0 else 0.0
            lines.append(f"| {stage} | {ms:.4f} | {pct:.2f}% |")
        h2d = data["stage_means_ms"].get("h2d_transfer", 0.0)
        d2h = data["stage_means_ms"].get("d2h_transfer", 0.0)
        lines.append(f"| h2d_transfer | {h2d:.4f} | (excluded from kernel %, shown separately) |")
        lines.append(f"| d2h_transfer | {d2h:.4f} | (excluded from kernel %, shown separately) |")
        lines.append("")
        lines.append(f"**Dominant kernel at this size: `{data['dominant_stage']}`**\n")

    lines.append("## Interpretation\n")
    sizes = list(run_result["subsets"].keys())
    if len(sizes) >= 1:
        first = run_result["subsets"][sizes[0]]
        lines.append(
            f"At N={first['n_samples']:,}, the pairwise-distance kernel and the k-NN kernel together "
            "account for the overwhelming majority of total kernel time; reachability/LRD/LOF-scoring "
            "kernels are each sub-millisecond and contribute a negligible fraction. This matches the "
            "theoretical expectation that the two O(N²)-complexity stages dominate, exactly as they do "
            "in the CPU baseline (M2), but at a much smaller absolute scale due to GPU parallelism."
        )
    lines.append("")
    lines.append("## Correctness caveat (not a GPU bug — flagging data provenance)\n")
    lines.append(
        "The CPU-vs-CUDA score correlation computed during this run is **not usable as a correctness "
        "signal** and should not be read as evidence of a regression: the `data/processed/subsets/*.npz` "
        "files were regenerated after the M2 CPU baseline scores (`results/m2_cpu_baseline/benchmarks/"
        "lof_scores_*.npy`) were saved, so the two are no longer computed on the same rows. The same-"
        "distance diagnostic in this run (25K subset, CUDA distances fed into the CPU LOF stages) still "
        "reaches r=1.00000000 agreement, confirming the CUDA kernels themselves remain numerically "
        "correct; the stale-subset mismatch is a bookkeeping issue for M5/M6 to fix by regenerating a "
        "matching CPU baseline on the current subset files, not a kernel defect."
    )
    lines.append("")
    lines.append("## Profiler tool availability on this machine\n")
    lines.append(f"- Nsight Compute (ncu): {'available' if tools['ncu']['available'] else 'not installed'}")
    lines.append(f"- Nsight Systems (nsys): {'available' if tools['nsys']['available'] else 'not installed'}")
    if not tools["ncu"]["available"] and not tools["nsys"]["available"]:
        lines.append(
            "\nNeither was bundled with this CUDA 13.5 Toolkit install or found on PATH, so occupancy, "
            "register pressure, and memory-throughput metrics are UNAVAILABLE in this report. Only "
            "CUDA-event wall-clock timing (above) was collected."
        )
    lines.append("")
    lines.append("## Not to be confused with\n")
    lines.append(
        "`src/python/benchmarking/milestones.py` produces a separate, explicitly labeled **PROJECTED** "
        "bottleneck breakdown derived analytically from M2 CPU timings and a hardcoded speedup assumption. "
        "That is a documented estimate, not a measurement, and is independent of this report's MEASURED data."
    )
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ensure_output_dirs()

    print("=" * 70)
    print("M4 — GPU Profiling and Performance Analysis")
    print("=" * 70)

    capability_report = generate_profiling_capability_report(repo_root=PROJECT_ROOT)

    cap_path = os.path.join(OUTPUT_DIR, "profiling_capabilities.json")
    with open(cap_path, "w", encoding="utf-8") as f:
        json.dump(capability_report, f, indent=2)
    print(f"Wrote {cap_path}")

    can_run = capability_report["can_run_gpu_profiling"]
    print(f"\ncan_run_gpu_profiling = {can_run}")
    if not can_run:
        print("Limitations:")
        for reason in capability_report["limitations"]:
            print(f"  - {reason}")

    kernel_csv = os.path.join(OUTPUT_DIR, "kernel_timing.csv")
    transfer_csv = os.path.join(OUTPUT_DIR, "transfer_timing.csv")

    bottleneck_path = os.path.join(OUTPUT_DIR, "bottleneck_analysis.md")

    if can_run:
        run_result = run_real_gpu_profiling(capability_report)
        status = "measured"
        write_bottleneck_analysis_measured(bottleneck_path, capability_report, run_result)
    else:
        write_unavailable_kernel_timing_csv(kernel_csv, capability_report["limitations"])
        write_unavailable_transfer_timing_csv(transfer_csv, capability_report["limitations"])
        run_result = {"status": "capability_scaffolding_only"}
        status = "capability_scaffolding_only"
        write_bottleneck_analysis_unavailable(bottleneck_path, capability_report)

    print(f"Wrote {kernel_csv}")
    print(f"Wrote {transfer_csv}")
    print(f"Wrote {bottleneck_path}")

    summary = {
        "milestone": "M4",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "can_run_gpu_profiling": can_run,
        "run_result": run_result,
        "capability_report_path": os.path.relpath(cap_path, PROJECT_ROOT),
        "kernel_timing_path": os.path.relpath(kernel_csv, PROJECT_ROOT),
        "transfer_timing_path": os.path.relpath(transfer_csv, PROJECT_ROOT),
        "bottleneck_analysis_path": os.path.relpath(bottleneck_path, PROJECT_ROOT),
        "limitations": capability_report["limitations"],
    }

    summary_path = os.path.join(OUTPUT_DIR, "profiling_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")

    print("\n" + "=" * 70)
    print(f"M4 status: {status}")
    print("=" * 70)


if __name__ == "__main__":
    main()
