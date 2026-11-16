"""CUDA LOF Cybersecurity — Streamlit dashboard (M7).

Reads every result directly from `results/`, `data/`, and live hardware
detection. Launch: `streamlit run app.py`.
"""

import os
import sys

import streamlit as st

_SRC_PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "python")
if _SRC_PYTHON not in sys.path:
    sys.path.insert(0, _SRC_PYTHON)

from dashboard import (
    pages_anomalies,
    pages_benchmark_lab,
    pages_benchmarks,
    pages_correctness,
    pages_dataset,
    pages_exports,
    pages_hardware,
    pages_history,
    pages_memory,
    pages_optimization,
    pages_optimization_journey,
    pages_overview,
    pages_profiling,
)

st.set_page_config(page_title="CUDA LOF Cybersecurity", layout="wide")

PAGES = {
    "Overview": pages_overview,
    "Hardware & Environment": pages_hardware,
    "Dataset Analytics": pages_dataset,
    "CPU vs. GPU Benchmarks": pages_benchmarks,
    "Optimization Journey": pages_optimization_journey,
    "Benchmark Lab": pages_benchmark_lab,
    "CUDA Profiling": pages_profiling,
    "CUDA Optimization": pages_optimization,
    "Memory & Scalability": pages_memory,
    "Anomaly Detection Results": pages_anomalies,
    "Correctness & Validation": pages_correctness,
    "Experiment History": pages_history,
    "Exports": pages_exports,
}

st.sidebar.title("CUDA LOF Cybersecurity")
st.sidebar.caption("Local Outlier Factor — CPU vs. CUDA, measured on real hardware")
selection = st.sidebar.radio("Navigate", list(PAGES.keys()))

try:
    PAGES[selection].render()
except Exception as exc:
    st.error(f"This page failed to render: {exc}")
    st.caption("This is a dashboard-rendering error, not a sign that the underlying result data is invalid.")
