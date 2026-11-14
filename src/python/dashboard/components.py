"""Shared UI helpers for dashboard pages."""

import streamlit as st

SYNTHETIC_BANNER = (
    "⚠️ **Synthetic placeholder dataset.** `data/raw/` is empty on this machine — no real "
    "CICIDS2017 CSVs were downloaded. `data/processed/dataset_metadata.json` records its own "
    "source as `synthetic_cicids2017.csv` with a 3-class label scheme (BENIGN/DDoS/PortScan), "
    "not the real dataset's 15 classes. Row/feature counts and memory/timing scaling are still "
    "representative (they depend on shape, not label semantics), but anomaly-detection results "
    "shown here are **not a validated real-world finding**."
)


def not_available(reason="Result file not found on disk."):
    st.info(f"Not available — {reason}")


def section(title):
    st.markdown(f"### {title}")


def synthetic_dataset_banner():
    st.warning(SYNTHETIC_BANNER)


def metric_row(items):
    """items: list of (label, value, help_text_or_None)."""
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        col.metric(label, value, help=help_text)
