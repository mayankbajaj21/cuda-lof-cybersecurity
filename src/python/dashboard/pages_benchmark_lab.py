"""Interactive Benchmark Lab — run a real, small LOF experiment from the UI.

Safety model:
  - No execution on page load; only on explicit button click.
  - A feasibility check (live free VRAM / available RAM vs. the dense N x N
    distance-matrix requirement, same formula as results/m6_final_benchmarks/
    memory_analysis.csv) runs BEFORE the run button is enabled.
  - Large/slow configurations require an extra confirmation checkbox.
  - Every run is saved as its own new, uniquely-named JSON record under
    results/benchmark_lab/experiments/ — existing milestone results under
    results/m2_cpu_baseline/ .. results/m6_final_benchmarks/ are never touched.
"""

import json
import os
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from . import loaders
from .components import not_available, section, synthetic_dataset_banner

_SRC_PYTHON = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC_PYTHON not in sys.path:
    sys.path.insert(0, _SRC_PYTHON)

EXPERIMENTS_DIR = loaders.path("results", "benchmark_lab", "experiments")

AVAILABLE_SUBSETS = ["10k", "25k", "100k", "250k", "500k", "1m"]
SUBSET_N = {"10k": 10_000, "25k": 25_000, "100k": 100_000, "250k": 250_000, "500k": 500_000, "1m": 1_000_000}

# Rough CPU O(N^2) scaling anchors from real M2/M6 measurements (10k -> 25k).
_CPU_ANCHOR_N = 10_000
_CPU_ANCHOR_SECONDS = 3.1


def _available_subsets():
    found = []
    for name in AVAILABLE_SUBSETS:
        subset_dir = loaders.path("data", "processed", "subsets", name)
        if os.path.isdir(subset_dir) and any(f.endswith(".npz") for f in os.listdir(subset_dir)):
            found.append(name)
    return found


def _live_capability():
    """Reuse the same hardware/capability detection built for M4."""
    try:
        from profiling.capability_report import generate_profiling_capability_report

        report = generate_profiling_capability_report()
        return report
    except Exception as exc:
        return {"can_run_gpu_profiling": False, "limitations": [f"Capability check failed: {exc}"]}


def _estimate_memory_gb(n, k=20, bytes_per_entry=4):
    """Same formula as results/m6_final_benchmarks/memory_analysis.csv; FP16 variants use 2 bytes/entry."""
    distance_bytes = n * n * bytes_per_entry
    pipeline_bytes = distance_bytes + n * k * 4 * 2 + n * 4 * 2
    return distance_bytes / (1024 ** 3), pipeline_bytes / (1024 ** 3)


# Label -> variant id for the M8 variants offered in the lab (V0/V1 are already
# covered by the "Naive CUDA" / "Optimized CUDA" options above).
VARIANT_OPTIONS = {
    "V2 Register blocking (outside syllabus)": 2,
    "V3 Async copy + double buffering (outside syllabus)": 3,
    "V4 FP16 storage (outside syllabus, approximate)": 4,
    "V5 Combined (outside syllabus, approximate)": 5,
}


def _run_variant(X, k, contamination, repetitions, warmup, status_placeholder, variant_id, label):
    from cuda.lof_variants import compute_lof_variant

    rep_times = []
    scores = None
    for i in range(warmup + repetitions):
        phase = f"Warm-up {i + 1}/{warmup}" if i < warmup else f"Repetition {i - warmup + 1}/{repetitions}"
        status_placeholder.text(f"Running {label} — {phase}...")
        lof_scores, timings, _mem = compute_lof_variant(X, k=k, variant=variant_id, eps=1e-10)
        if i >= warmup:
            rep_times.append(timings["end_to_end_ms"] / 1000.0)
            scores = lof_scores
    threshold = float(np.percentile(scores, 100.0 * (1.0 - contamination)))
    predictions = np.where(scores >= threshold, 1, 0).astype(np.int8)
    return scores, predictions, rep_times


def _live_free_vram_gb(capability_report):
    gpu = capability_report.get("hardware_snapshot", {}).get("gpu", {})
    if gpu.get("available") and gpu.get("gpus"):
        return gpu["gpus"][0].get("free_memory_mb", 0) / 1024.0
    return None


def _live_available_ram_gb():
    try:
        import psutil

        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        return None


def _estimate_cpu_seconds(n):
    """O(N^2) extrapolation from a real measured anchor point. Labeled as an estimate."""
    return _CPU_ANCHOR_SECONDS * (n / _CPU_ANCHOR_N) ** 2


def _run_cpu(X, k, contamination, repetitions, warmup, status_placeholder):
    from cpu.cpu_lof import CPULocalOutlierFactor

    rep_times = []
    last_model = None
    total_iters = warmup + repetitions
    for i in range(total_iters):
        label = f"Warm-up {i + 1}/{warmup}" if i < warmup else f"Repetition {i - warmup + 1}/{repetitions}"
        status_placeholder.text(f"Running CPU LOF — {label}...")
        t0 = time.perf_counter()
        model = CPULocalOutlierFactor(k=k, contamination=contamination)
        model.fit(X)
        elapsed = time.perf_counter() - t0
        if i >= warmup:
            rep_times.append(elapsed)
            last_model = model
    return last_model.lof_scores_, last_model.predictions_, rep_times


def _run_naive_cuda(X, k, contamination, repetitions, warmup, status_placeholder):
    from cuda.lof import benchmark_cuda_lof

    rep_times = []
    scores = None
    total_iters = warmup + repetitions
    for i in range(total_iters):
        label = f"Warm-up {i + 1}/{warmup}" if i < warmup else f"Repetition {i - warmup + 1}/{repetitions}"
        status_placeholder.text(f"Running Naive CUDA LOF — {label}...")
        lof_scores, timings = benchmark_cuda_lof(X, k=k, eps=1e-10)
        if i >= warmup:
            rep_times.append(timings["end_to_end_ms"] / 1000.0)
            scores = lof_scores
    threshold = float(np.percentile(scores, 100.0 * (1.0 - contamination)))
    predictions = np.where(scores >= threshold, 1, 0).astype(np.int8)
    return scores, predictions, rep_times


def _run_optimized_cuda(X, k, contamination, repetitions, warmup, status_placeholder):
    from cuda.lof_optimized import benchmark_cuda_lof_optimized

    rep_times = []
    scores = None
    total_iters = warmup + repetitions
    for i in range(total_iters):
        label = f"Warm-up {i + 1}/{warmup}" if i < warmup else f"Repetition {i - warmup + 1}/{repetitions}"
        status_placeholder.text(f"Running Optimized CUDA LOF — {label}...")
        lof_scores, timings = benchmark_cuda_lof_optimized(X, k=k, eps=1e-10)
        if i >= warmup:
            rep_times.append(timings["end_to_end_ms"] / 1000.0)
            scores = lof_scores
    threshold = float(np.percentile(scores, 100.0 * (1.0 - contamination)))
    predictions = np.where(scores >= threshold, 1, 0).astype(np.int8)
    return scores, predictions, rep_times


_BASE_VARIANT_IDS = {"cpu": -1, "naive_cuda": 0, "optimized_cuda": 1}


def _variant_fields(impl_key, variant_id):
    vid = variant_id if variant_id is not None else _BASE_VARIANT_IDS.get(impl_key)
    if vid == -1:
        return {
            "variant_id": -1,
            "variant_name": "CPU (NumPy)",
            "technique_source": "cpu",
            "exact_or_approximate": "exact",
        }
    try:
        from cuda.lof_variants import VARIANTS

        meta = VARIANTS[vid]
        return {
            "variant_id": vid,
            "variant_name": meta["name"],
            "technique_source": meta["technique_source"],
            "exact_or_approximate": meta["exact_or_approximate"],
        }
    except Exception:
        return {"variant_id": vid}


def _save_experiment(record):
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
    exp_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
    record["experiment_id"] = exp_id
    out_path = os.path.join(EXPERIMENTS_DIR, f"{exp_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    return exp_id, out_path


def render():
    st.title("Interactive Benchmark Lab")
    st.caption(
        "Run a real, small LOF experiment on this machine right now — CPU, naive CUDA, "
        "syllabus-optimized CUDA, or one of the outside-syllabus variants V2–V5 — and save the "
        "result as a new measured record."
    )
    synthetic_dataset_banner()

    capability_report = _live_capability()
    can_gpu = capability_report.get("can_run_gpu_profiling", False)

    try:
        from cuda.lof import is_lof_available
        from cuda.lof_optimized import is_lof_optimized_available

        naive_cuda_ok = can_gpu and is_lof_available()
        optimized_cuda_ok = can_gpu and is_lof_optimized_available()
    except Exception:
        naive_cuda_ok = False
        optimized_cuda_ok = False

    try:
        from cuda.lof_variants import is_variants_available

        variants_ok = can_gpu and is_variants_available()
    except Exception:
        variants_ok = False

    subsets = _available_subsets()
    if not subsets:
        not_available("No prepared dataset subsets found under data/processed/subsets/.")
        return

    section("1. Configure experiment")

    impl_options = ["CPU (NumPy reference)"]
    if naive_cuda_ok:
        impl_options.append("Naive CUDA")
    else:
        st.caption("Naive CUDA option hidden/disabled — no compiled bin/cuda_lof.dll or no NVIDIA GPU detected.")
    if optimized_cuda_ok:
        impl_options.append("Optimized CUDA")
    else:
        st.caption("Optimized CUDA option hidden/disabled — no compiled bin/cuda_lof_optimized.dll or no NVIDIA GPU detected.")
    if variants_ok:
        impl_options.extend(VARIANT_OPTIONS.keys())
    else:
        st.caption(
            "V2–V5 options hidden/disabled — no compiled bin/cuda_lof_variants.dll "
            "(build with scripts\\build_cuda_lof_variants.bat) or no NVIDIA GPU detected."
        )

    col1, col2 = st.columns(2)
    with col1:
        implementation = st.selectbox("Implementation", impl_options)
        subset_name = st.selectbox("Dataset size", subsets, index=0)
        k = st.slider("k (neighbors)", min_value=5, max_value=50, value=20)
    with col2:
        contamination = st.slider("Contamination", min_value=0.01, max_value=0.5, value=0.10, step=0.01)
        repetitions = st.slider("Timed repetitions", min_value=1, max_value=10, value=3)
        warmup = st.slider("Warm-up runs (discarded)", min_value=0, max_value=3, value=1)

    n = SUBSET_N.get(subset_name, 0)
    is_cpu = implementation.startswith("CPU")
    variant_id = VARIANT_OPTIONS.get(implementation)
    is_fp16 = variant_id in (4, 5)
    variant_k_limit = None
    if variant_id is not None:
        try:
            from cuda.lof_variants import variant_max_k

            variant_k_limit = variant_max_k(variant_id)
        except Exception:
            variant_k_limit = None

    section("2. Feasibility check (live, before you run)")
    distance_gb, pipeline_gb = _estimate_memory_gb(n, k, bytes_per_entry=2 if is_fp16 else 4)
    precision = "FP16, 2 bytes/entry" if is_fp16 else "FP32, 4 bytes/entry"
    st.write(f"Estimated dense {n:,}×{n:,} distance matrix ({precision}): **{distance_gb:.2f} GB**")
    st.write(f"Estimated total pipeline memory: **{pipeline_gb:.2f} GB**")

    feasible = True
    reasons = []

    if is_cpu:
        avail_ram_gb = _live_available_ram_gb()
        if avail_ram_gb is not None:
            st.write(f"Live available system RAM: **{avail_ram_gb:.2f} GB**")
            if pipeline_gb > avail_ram_gb:
                feasible = False
                reasons.append(
                    f"Estimated requirement ({pipeline_gb:.2f} GB) exceeds live available RAM ({avail_ram_gb:.2f} GB)."
                )
        else:
            st.caption("Could not read live available RAM (psutil unavailable) — proceeding on the estimate alone.")
    else:
        free_vram_gb = _live_free_vram_gb(capability_report)
        if free_vram_gb is not None:
            st.write(f"Live free VRAM: **{free_vram_gb:.2f} GB**")
            if pipeline_gb > free_vram_gb:
                feasible = False
                reasons.append(
                    f"Estimated requirement ({pipeline_gb:.2f} GB) exceeds live free VRAM ({free_vram_gb:.2f} GB)."
                )
        else:
            feasible = False
            reasons.append("Could not read live free VRAM (no NVIDIA GPU detected via nvidia-smi).")

    if variant_k_limit is not None and k > variant_k_limit:
        feasible = False
        reasons.append(f"{implementation} supports k ≤ {variant_k_limit}; lower k to run it.")

    smaller_sizes = [s for s in subsets if SUBSET_N[s] < n]
    if not feasible:
        msg = " ".join(reasons)
        if smaller_sizes:
            msg += f" Try a smaller size instead: {', '.join(smaller_sizes)}."
        st.error(msg)
    else:
        st.success("This configuration fits in available memory on this machine right now.")

    long_running = False
    confirm_long_run = True
    if is_cpu and n >= 100_000:
        est_seconds = _estimate_cpu_seconds(n)
        long_running = True
        st.warning(
            f"CPU LOF is O(N²). Extrapolating from the real measured 10k baseline (~{_CPU_ANCHOR_SECONDS:.1f}s), "
            f"this run is roughly estimated at **~{est_seconds / 60:.1f} minutes** per repetition "
            f"(this is an estimate, not an exact prediction)."
        )
        confirm_long_run = st.checkbox("I understand this may take several minutes and want to proceed anyway.")

    section("3. Run")
    run_enabled = feasible and (confirm_long_run if long_running else True)
    run_clicked = st.button("Run Experiment", disabled=not run_enabled)

    if not feasible:
        st.caption("Run button is disabled because this configuration is not feasible on this machine.")
    elif long_running and not confirm_long_run:
        st.caption("Run button is disabled until you confirm the long-running checkbox above.")

    if run_clicked:
        status_placeholder = st.empty()
        try:
            from data.dataset import load_subset

            with st.spinner("Loading dataset subset..."):
                X, y_bin, _, _, _ = load_subset(subset_name)

            if implementation.startswith("CPU"):
                impl_key = "cpu"
                scores, predictions, rep_times = _run_cpu(X, k, contamination, repetitions, warmup, status_placeholder)
            elif implementation == "Naive CUDA":
                impl_key = "naive_cuda"
                scores, predictions, rep_times = _run_naive_cuda(X, k, contamination, repetitions, warmup, status_placeholder)
            elif variant_id is not None:
                impl_key = f"variant_v{variant_id}"
                scores, predictions, rep_times = _run_variant(
                    X, k, contamination, repetitions, warmup, status_placeholder, variant_id, implementation
                )
            else:
                impl_key = "optimized_cuda"
                scores, predictions, rep_times = _run_optimized_cuda(X, k, contamination, repetitions, warmup, status_placeholder)

            status_placeholder.text("Done.")

            mean_t = statistics.mean(rep_times)
            median_t = statistics.median(rep_times)
            std_t = statistics.stdev(rep_times) if len(rep_times) > 1 else 0.0
            min_t, max_t = min(rep_times), max(rep_times)
            anomaly_count = int(np.sum(predictions == 1))

            section("Results")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Mean runtime", f"{mean_t * 1000:.2f} ms")
            r2.metric("Median runtime", f"{median_t * 1000:.2f} ms")
            r3.metric("Std dev", f"{std_t * 1000:.2f} ms")
            r4.metric("Min / Max", f"{min_t * 1000:.1f} / {max_t * 1000:.1f} ms")

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("Score min", f"{float(np.min(scores)):.4f}")
            s2.metric("Score max", f"{float(np.max(scores)):.4f}")
            s3.metric("Score mean", f"{float(np.mean(scores)):.4f}")
            s4.metric("Anomalies flagged", f"{anomaly_count:,} / {n:,}")

            fig = go.Figure(data=[go.Histogram(x=scores, nbinsx=60)])
            fig.update_layout(
                title="LOF score distribution",
                xaxis_title="LOF score",
                yaxis_title="Count",
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

            record = {
                "measurement_classification": "MEASURED",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "implementation": impl_key,
                "dataset_subset": subset_name,
                "n_samples": n,
                "k": k,
                "contamination": contamination,
                "repetitions": repetitions,
                "warmup_runs": warmup,
                "per_repetition_seconds": rep_times,
                "runtime_summary_seconds": {
                    "mean": mean_t,
                    "median": median_t,
                    "std": std_t,
                    "min": min_t,
                    "max": max_t,
                },
                "score_summary": {
                    "min": float(np.min(scores)),
                    "max": float(np.max(scores)),
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores)),
                },
                "predicted_anomalies": anomaly_count,
                "predicted_inliers": int(n - anomaly_count),
                "hardware_snapshot": capability_report.get("hardware_snapshot"),
                "source": "benchmark_lab (interactive dashboard run)",
            }
            record.update(_variant_fields(impl_key, variant_id))
            exp_id, out_path = _save_experiment(record)
            st.success(f"Saved experiment `{exp_id}` to `{os.path.relpath(out_path, loaders.REPO_ROOT)}`")

        except Exception as exc:
            st.error(f"Experiment failed: {exc}")
            st.caption(
                "This is a run-time error from this specific experiment, not a sign that the "
                "project's underlying measured results are invalid."
            )

    section("Past experiments")
    if os.path.isdir(EXPERIMENTS_DIR):
        files = sorted(
            (f for f in os.listdir(EXPERIMENTS_DIR) if f.endswith(".json")),
            reverse=True,
        )
        if files:
            st.caption(f"{len(files)} experiment(s) saved in this lab so far.")
            with st.expander("Show saved experiment files"):
                for f in files[:50]:
                    st.text(f)
        else:
            not_available("No experiments run yet in this session's lab.")
    else:
        not_available("No experiments run yet in this session's lab.")
