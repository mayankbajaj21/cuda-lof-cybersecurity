import plotly.express as px
import streamlit as st

from . import loaders
from .components import not_available, section


def render():
    st.title("CUDA Profiling (M4)")

    cap = loaders.load_json("results", "m4_profiling", "profiling_capabilities.json")
    if cap:
        tools = cap.get("hardware_snapshot", {}).get("tools", {})
        missing = [name for name, info in tools.items() if not info.get("available") and name in ("nsys", "ncu")]
        if missing:
            st.info(
                f"Nsight tool(s) not installed on this machine: {', '.join(missing)}. "
                "Only CUDA-event wall-clock kernel timing is available below — no occupancy/"
                "register/throughput metrics."
            )

    kt = loaders.load_csv("results", "m4_profiling", "kernel_timing.csv")
    if kt is None or kt.empty:
        not_available("results/m4_profiling/kernel_timing.csv not found")
        return

    section("Per-kernel mean time by dataset size")
    fig = px.bar(
        kt, x="stage", y="mean_ms", color="subset", barmode="group", error_y="std_ms",
        labels={"mean_ms": "Mean kernel time (ms)", "stage": "Kernel"},
    )
    st.plotly_chart(fig, use_container_width=True)

    section("Percentage contribution of each kernel")
    kt = kt.copy()
    kt["pct"] = kt.groupby("subset")["mean_ms"].transform(lambda s: 100 * s / s.sum())
    fig2 = px.bar(
        kt, x="subset", y="pct", color="stage", barmode="stack",
        labels={"pct": "% of total kernel time"},
    )
    st.plotly_chart(fig2, use_container_width=True)

    dominant = kt.loc[kt.groupby("subset")["mean_ms"].idxmax()]
    for _, row in dominant.iterrows():
        st.success(f"**{row['subset']}**: bottleneck kernel is `{row['stage']}` ({row['pct']:.1f}% of kernel time)")

    section("Transfer time vs. kernel time")
    tt = loaders.load_csv("results", "m4_profiling", "transfer_timing.csv")
    if tt is not None and not tt.empty:
        st.dataframe(tt, use_container_width=True)
    else:
        not_available("results/m4_profiling/transfer_timing.csv not found")

    section("Bottleneck analysis report")
    md = loaders.load_text("results", "m4_profiling", "bottleneck_analysis.md")
    if md:
        with st.expander("Read full report"):
            st.markdown(md)
    else:
        not_available("results/m4_profiling/bottleneck_analysis.md not found")
