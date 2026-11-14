import numpy as np
import plotly.express as px
import streamlit as st

from . import loaders
from .components import not_available, section, synthetic_dataset_banner


def render():
    st.title("Anomaly Detection Results")
    synthetic_dataset_banner()
    st.warning(
        "**Unsupervised flags ≠ ground truth.** LOF scores reflect local density deviation, not "
        "verified attacks. Even on real CICIDS2017 data, a high LOF score is not the same claim as "
        "a labeled attack row — treat the numbers below as density-anomaly output, not a validated "
        "intrusion-detection result."
    )

    size = st.selectbox("Subset", ["10k", "25k"])
    scores = loaders.load_npy("results", "m2_cpu_baseline", "benchmarks_current", f"lof_scores_{size}_k20.npy")
    if scores is None:
        not_available(f"results/m2_cpu_baseline/benchmarks_current/lof_scores_{size}_k20.npy not found")
        return

    section("LOF score distribution")
    fig = px.histogram(scores, nbins=60, labels={"value": "LOF score"})
    st.plotly_chart(fig, use_container_width=True)

    contamination = st.slider("Contamination (top % flagged)", 1, 50, 10) / 100.0
    threshold = np.quantile(scores, 1 - contamination)
    n_flagged = int((scores >= threshold).sum())
    st.write(f"Threshold at {1-contamination:.0%} percentile: **{threshold:.4f}** — {n_flagged:,} of {len(scores):,} points flagged")

    section("Top anomalous points")
    order = np.argsort(scores)[::-1][:20]
    st.dataframe({"index": order, "lof_score": scores[order]}, use_container_width=True)
