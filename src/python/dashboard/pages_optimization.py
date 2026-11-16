import streamlit as st

from . import loaders
from .components import metric_row, not_available, section


def render():
    st.title("CUDA Optimization (M5)")
    st.info(
        "These M5 numbers come from an earlier, separate run. For a same-session comparison including the "
        "outside-syllabus techniques (V2–V5), see the **Optimization Journey** page."
    )

    kc = loaders.load_csv("results", "m5_cuda_optimized", "kernel_comparison.csv")
    if kc is None or kc.empty:
        not_available("results/m5_cuda_optimized/kernel_comparison.csv not found")
    else:
        section("Before / after, per kernel")
        st.dataframe(kc, use_container_width=True)

        for size in sorted(kc["subset"].unique()):
            sub = kc[(kc["subset"] == size) & (kc["stage"].str.contains("kernel"))]
            if sub.empty:
                continue
            best = sub.sort_values("speedup", ascending=False).iloc[0]
            metric_row(
                [
                    (f"{size} — top speedup", f"{best['speedup']:.2f}x", best["stage"]),
                ]
            )

    section("Correctness agreement")
    corr = loaders.load_json("results", "m5_cuda_optimized", "correctness_report.json")
    if corr:
        for size, comparisons in corr.items():
            st.markdown(f"**{size}**")
            rows = []
            for name, metrics in comparisons.items():
                if isinstance(metrics, dict) and "pearson_r" in metrics:
                    rows.append(
                        {
                            "comparison": name,
                            "max_abs_error": metrics.get("max_abs_error"),
                            "pearson_r": metrics.get("pearson_r"),
                            "anomaly_agreement": metrics.get("anomaly_agreement"),
                        }
                    )
            if rows:
                st.dataframe(rows, use_container_width=True)
    else:
        not_available("results/m5_cuda_optimized/correctness_report.json not found")

    section("Charts")
    chart_names = [
        "m5_01_naive_vs_optimized_runtime.png",
        "m5_02_per_kernel_before_after.png",
        "m5_03_speedup_by_size.png",
        "m5_04_kernel_time_distribution.png",
        "m5_05_correctness_agreement.png",
    ]
    cols = st.columns(2)
    for i, name in enumerate(chart_names):
        img = loaders.load_image_path("results", "m5_cuda_optimized", "charts", name)
        if img:
            cols[i % 2].image(img, caption=name, use_container_width=True)

    section("Full optimization report")
    md = loaders.load_text("results", "m5_cuda_optimized", "optimization_report.md")
    if md:
        with st.expander("Read full report (includes the honest 5x-slower-then-fixed tiling story)"):
            st.markdown(md)
    else:
        not_available("results/m5_cuda_optimized/optimization_report.md not found")
