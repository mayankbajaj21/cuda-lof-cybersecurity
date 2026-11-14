import streamlit as st

from . import loaders
from .components import not_available, section


def render():
    st.title("Correctness & Numerical Validation")

    st.markdown(
        "The CPU reference computes pairwise distance via the matrix-expansion trick "
        "(`||p-q||^2 = ||p||^2 + ||q||^2 - 2<p,q>`), which can suffer float32 catastrophic "
        "cancellation on near-duplicate rows. The CUDA kernels use direct coordinate-difference "
        "accumulation, which doesn't have this failure mode — so CUDA is, if anything, *more* "
        "numerically faithful to the true Euclidean distance, not just faster."
    )

    section("M5: CPU vs. naive vs. optimized CUDA")
    corr = loaders.load_json("results", "m5_cuda_optimized", "correctness_report.json")
    if corr:
        for size, comparisons in corr.items():
            st.markdown(f"**{size}**")
            rows = []
            for name, metrics in comparisons.items():
                if isinstance(metrics, dict) and "pearson_r" in metrics:
                    rows.append({"comparison": name, **metrics})
            if rows:
                st.dataframe(rows, use_container_width=True)
    else:
        not_available("results/m5_cuda_optimized/correctness_report.json not found")

    section("M6: final-benchmark correctness summary")
    summary = loaders.load_csv("results", "m6_final_benchmarks", "correctness_summary.csv")
    if summary is not None:
        st.dataframe(summary, use_container_width=True)
    else:
        not_available("results/m6_final_benchmarks/correctness_summary.csv not found")

    img = loaders.load_image_path("results", "m6_final_benchmarks", "charts", "m6_05_correctness_agreement.png")
    if img:
        st.image(img, use_container_width=True)
