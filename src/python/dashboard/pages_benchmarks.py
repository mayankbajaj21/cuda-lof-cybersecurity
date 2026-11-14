import plotly.express as px
import streamlit as st

from . import loaders
from .components import not_available, section


def render():
    st.title("CPU vs. GPU Benchmarks")
    st.caption("Real measured end-to-end and kernel-only timing — M6 final benchmark results.")

    df = loaders.load_csv("results", "m6_final_benchmarks", "final_benchmark_results.csv")
    if df is None or df.empty:
        not_available("results/m6_final_benchmarks/final_benchmark_results.csv not found")
        return

    impls = sorted(df["implementation"].unique())
    sizes = sorted(df["subset"].unique(), key=lambda s: df[df["subset"] == s]["n_samples"].iloc[0])

    c1, c2 = st.columns(2)
    sel_sizes = c1.multiselect("Dataset size", sizes, default=sizes)
    sel_impls = c2.multiselect("Implementation", impls, default=impls)

    filtered = df[df["subset"].isin(sel_sizes) & df["implementation"].isin(sel_impls)]

    section("End-to-end runtime (mean, ms)")
    fig = px.bar(
        filtered,
        x="subset",
        color="implementation",
        y="end_to_end_mean_ms",
        barmode="group",
        log_y=True,
        labels={"end_to_end_mean_ms": "End-to-end time (ms, log scale)", "subset": "Subset"},
        error_y="end_to_end_std_ms",
    )
    st.plotly_chart(fig, use_container_width=True)

    section("Kernel-only vs. end-to-end")
    fig2 = px.bar(
        filtered,
        x="subset",
        y=["kernel_only_ms", "end_to_end_mean_ms"],
        color="implementation",
        barmode="group",
        log_y=True,
    )
    st.plotly_chart(fig2, use_container_width=True)

    section("Speedup vs. CPU")
    try:
        pivot = filtered.pivot_table(index="subset", columns="implementation", values="end_to_end_mean_ms")
        if "cpu" in pivot.columns:
            speedup = pivot.apply(lambda col: pivot["cpu"] / col)
            speedup = speedup.drop(columns=["cpu"])
            st.dataframe(speedup.round(2), use_container_width=True)
        else:
            st.caption("CPU implementation not in the current filter selection.")
    except Exception:
        st.caption("Speedup table unavailable for the current filter selection.")

    section("Full comparison table")
    st.dataframe(filtered, use_container_width=True)
