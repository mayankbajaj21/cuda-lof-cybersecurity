"""Optimization Journey (M8): CPU -> naive GPU -> syllabus-optimized GPU ->
each outside-syllabus technique -> all combined, from one same-session run."""

import plotly.express as px
import streamlit as st

from . import journey_data as jd
from . import loaders
from .components import metric_row, not_available, section

SOURCE_COLORS = {
    "CPU reference": "#7f7f7f",
    "GPU baseline": "#1f77b4",
    "Syllabus techniques": "#2ca02c",
    "Outside syllabus": "#ff7f0e",
    "Outside syllabus (combined)": "#d62728",
}

SYLLABUS_TECHNIQUES = [
    ("Module 4.2", "Coalesced global-memory access — neighbouring threads read neighbouring addresses"),
    ("Module 4.3", "Shared-memory tiling — 16×16 tiles of the feature matrix reused by every thread in a block"),
    ("Module 4.6", "Occupancy tuning — 32×32 tiles collapsed occupancy (1 block/SM, ~5× slower); 16×16 fixed it"),
    ("Module 4.5", "Warp-divergence removal — redundant bounds-check branch taken out of the inner loop"),
    ("Module 4.7", "Loop unrolling — `#pragma unroll` on the feature loop"),
    ("Module 2.7 / 4.6", "Register-local top-k in the k-NN kernel"),
    ("Module 3.5", "Early buffer freeing — each large buffer released once the next stage has used it"),
]

OUTSIDE_CARDS = {
    2: {
        "what": (
            "Each thread computes a 4×4 block of 16 distances instead of one. Partial sums stay in "
            "registers, and each value loaded from shared memory is reused 4 times instead of once."
        ),
        "code": (
            "float acc[RB_MICRO][RB_MICRO];          // 4x4 partial sums in registers\n"
            "for (int a = 0; a < RB_MICRO; ++a) av[a] = As[ty + a*RB_THREADS][t];\n"
            "for (int b = 0; b < RB_MICRO; ++b) bv[b] = Bs[tx + b*RB_THREADS][t];\n"
            "for a, b:  diff = av[a] - bv[b];  acc[a][b] += diff * diff;"
        ),
        "why": (
            "The tiled kernel spends most of its instructions loading from shared memory. Reusing "
            "each load across 4 outputs cuts load instructions per distance by about 4×, so the "
            "kernel becomes compute-bound instead of load-bound."
        ),
    },
    3: {
        "what": (
            "While threads compute on the current tile, the hardware copies the next tile from "
            "global memory into a second shared-memory buffer in the background (Ampere+ `cp.async`)."
        ),
        "code": (
            "__pipeline_memcpy_async(dst, src, 4);   // prefetch tile t+1\n"
            "__pipeline_commit();\n"
            "... compute on tile t ...\n"
            "__pipeline_wait_prior(1);               // tile t+1 ready, swap buffers"
        ),
        "why": (
            "It hides global-memory latency behind computation. It only pays off when the kernel is "
            "actually waiting on memory."
        ),
    },
    4: {
        "what": (
            "The feature matrix and the N×N distance matrix are stored as 16-bit floats instead of "
            "32-bit. All arithmetic still accumulates in 32-bit."
        ),
        "code": (
            "Xh[idx] = __float2half_rn(X[i*D + d]);          // store features as FP16\n"
            "float diff = __half2float(row_tile[ty][t])\n"
            "           - __half2float(col_tile[tx][t]);     // compute in FP32"
        ),
        "why": (
            "Halves the bytes moved and the VRAM needed, so the distance matrix — the reason large "
            "datasets don't fit — takes half the space."
        ),
    },
}


def _fmt(v, suffix="", digits=2):
    try:
        if v is None or v != v:
            return "n/a"
        return f"{float(v):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "n/a"


def _ladder(vb, subset):
    df = jd.ladder_frame(vb, subset)
    if df.empty:
        not_available(f"No rows for subset {subset}.")
        return
    meas = jd.measured(df)
    if not meas.empty:
        fig = px.bar(
            meas,
            x="end_to_end_median_ms",
            y="bar_label",
            orientation="h",
            color="source_label",
            color_discrete_map=SOURCE_COLORS,
            log_x=True,
            text=meas["end_to_end_median_ms"].map(lambda v: f"{v:,.1f} ms"),
            hover_data={"techniques": True, "exact_or_approximate": True, "speedup_vs_cpu": ":.1f"},
            labels={
                "end_to_end_median_ms": "End-to-end median time (ms, log scale)",
                "bar_label": "",
                "source_label": "Technique source",
            },
            category_orders={"bar_label": list(meas["bar_label"])},
        )
        fig.update_layout(height=420)
        fig.update_xaxes(dtick=1)
        st.plotly_chart(fig, use_container_width=True)

    table = df[
        [
            "variant_name", "techniques", "source_label", "exact_or_approximate", "classification",
            "end_to_end_median_ms", "speedup_vs_cpu", "speedup_vs_v0", "speedup_vs_v1", "unavailable_reason",
        ]
    ].rename(
        columns={
            "variant_name": "Variant",
            "techniques": "Techniques",
            "source_label": "Source",
            "exact_or_approximate": "Exact / approx.",
            "classification": "Classification",
            "end_to_end_median_ms": "Median ms",
            "speedup_vs_cpu": "× vs CPU",
            "speedup_vs_v0": "× vs V0",
            "speedup_vs_v1": "× vs V1",
            "unavailable_reason": "Why unavailable",
        }
    )
    st.dataframe(table.round(2), use_container_width=True, hide_index=True)

    unav = jd.unavailable(df)
    if not unav.empty:
        st.caption(
            "Not measured at this size (shown in the table, not as bars): "
            + ", ".join(unav["variant_name"])
        )

    with st.expander("Timing spread (std / min / max)"):
        st.dataframe(
            meas[
                ["variant_name", "repetitions", "warmups", "end_to_end_median_ms", "end_to_end_mean_ms",
                 "end_to_end_std_ms", "end_to_end_min_ms", "end_to_end_max_ms"]
            ].round(2),
            use_container_width=True,
            hide_index=True,
        )


def _result_line(vb, kb, subset, variant_id):
    row = jd.variant_row(vb, subset, variant_id)
    if row is None or row["classification"] != "MEASURED":
        st.caption(f"No measured result for this variant at {subset}.")
        return
    dist = jd.distance_speedup_vs_v1(kb, subset)
    dist_x = None
    if not dist.empty:
        d = dist[dist["variant_id"] == variant_id]
        if not d.empty:
            dist_x = d["distance_speedup_vs_v1"].iloc[0]
    metric_row(
        [
            ("End-to-end median", _fmt(row["end_to_end_median_ms"], " ms"), None),
            ("vs V1 (end-to-end)", _fmt(row["speedup_vs_v1"], "×"), "Above 1× = faster than syllabus-optimized"),
            ("vs V1 (distance kernel)", _fmt(dist_x, "×"), "From kernel_breakdown.csv medians"),
            ("vs CPU", _fmt(row["speedup_vs_cpu"], "×", 1), None),
        ]
    )


def _accuracy_line(corr, subset, variant_id):
    if corr is None or corr.empty:
        return
    r = corr[(corr["subset"] == subset) & (corr["reference"] == "V0 Naive") & (corr["candidate_id"] == variant_id)]
    if r.empty:
        return
    r = r.iloc[0]
    st.markdown(
        f"Accuracy vs exact V0 at {subset}: Pearson r **{_fmt(r['pearson_r'], '', 6)}**, "
        f"k-NN recall **{_fmt(r['knn_recall_at_k'], '', 4)}**, anomaly-flag agreement "
        f"**{_fmt(r['anomaly_flag_agreement'], '', 4)}**, max score error **{_fmt(r['max_abs_error'], '', 4)}**."
    )


def _technique_cards(vb, kb, corr, subset):
    with st.container(border=True):
        st.markdown("#### V1 — Syllabus-optimized `[Syllabus]`")
        st.markdown("\n".join(f"- **{m}**: {t}" for m, t in SYLLABUS_TECHNIQUES))
        _result_line(vb, kb, subset, 1)

    for vid, title in [
        (2, "V2 — Register blocking"),
        (3, "V3 — Async copy + double buffering"),
        (4, "V4 — FP16 storage"),
    ]:
        card = OUTSIDE_CARDS[vid]
        with st.container(border=True):
            tag = "`[Outside syllabus]` `[APPROXIMATE]`" if vid == 4 else "`[Outside syllabus]`"
            st.markdown(f"#### {title} {tag}")
            st.markdown(f"**What it does.** {card['what']}")
            st.markdown("**Key code change** (`src/cuda/kernels_variants/variant_kernels.cu`)")
            st.code(card["code"], language="cpp")
            st.markdown(f"**Why it should help.** {card['why']}")
            st.markdown("**Measured result (built on top of V1)**")
            _result_line(vb, kb, subset, vid)
            row = jd.variant_row(vb, subset, vid)
            if vid == 2:
                st.success(
                    "Verdict: the biggest single win. Exact results (bit-identical to V0). It uses more "
                    "registers per thread, so fewer threads fit on each SM, but each thread does 16× more "
                    "independent work, which more than compensates."
                )
            elif vid == 3:
                st.error(
                    "Verdict: **slower than V1 (about 10%)**, reported as measured. The async copies are "
                    "only 4 bytes each, the second shared-memory buffer lowers occupancy, and the feature "
                    "matrix already sits in the 24 MB L2 cache, so there is almost no memory latency to hide."
                )
            elif vid == 4:
                _accuracy_line(corr, subset, 4)
                overflow = None if row is None else row.get("fp16_overflow_count")
                st.warning(
                    "Verdict: **approximate method.** Faster and half the memory, at a small accuracy cost "
                    f"(FP16 overflow count in this run: {_fmt(overflow, '', 0)}). It is the only variant "
                    "(with V5) that fits the 45k slice on this GPU."
                )

    with st.container(border=True):
        st.markdown("#### V5 — All three combined `[Outside syllabus]` `[APPROXIMATE]`")
        st.markdown("Register blocking + async double-buffered copies + FP16 storage in one distance kernel.")
        _result_line(vb, kb, subset, 5)
        _accuracy_line(corr, subset, 5)
        st.caption("V5 inherits V4's FP16 approximation; its scores are identical to V4's in this run.")


def _kernel_breakdown(kb, subset):
    sb = jd.stage_breakdown(kb, subset)
    if sb.empty:
        not_available("kernel_breakdown.csv has no GPU stages for this subset.")
        return
    fig = px.bar(
        sb,
        x="variant_name",
        y="median_ms",
        color="stage",
        labels={"median_ms": "Median time (ms)", "variant_name": "", "stage": "Stage"},
        category_orders={"stage": jd.STAGE_ORDER},
    )
    fig.update_layout(height=420, barmode="stack")
    st.plotly_chart(fig, use_container_width=True)

    dist = jd.distance_speedup_vs_v1(kb, subset)
    if not dist.empty:
        fig2 = px.bar(
            dist,
            x="variant_name",
            y="distance_speedup_vs_v1",
            text=dist["distance_speedup_vs_v1"].map(lambda v: f"{v:.2f}×"),
            labels={"distance_speedup_vs_v1": "Distance-kernel speedup vs V1 (×)", "variant_name": ""},
        )
        fig2.add_hline(y=1.0, line_dash="dash", annotation_text="V1 = 1×")
        fig2.update_layout(height=360)
        st.plotly_chart(fig2, use_container_width=True)


def _memory(vb, mem, feas):
    if vb is not None and not vb.empty:
        m = jd.measured(vb)
        m = m[m["peak_gpu_mem_bytes_tracked"].notna()].copy()
        if not m.empty:
            m["peak_MiB"] = m["peak_gpu_mem_bytes_tracked"] / (1024 ** 2)
            fig = px.bar(
                m,
                x="subset",
                y="peak_MiB",
                color="variant_name",
                barmode="group",
                labels={"peak_MiB": "Peak GPU memory (MiB, tracked in-pipeline)", "subset": "Subset"},
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Measured inside the pipeline (running total of allocated bytes), cross-checked against "
                "cudaMemGetInfo — agreement within ~0.5%."
            )

    mf = jd.max_feasible_n(mem)
    if not mf.empty:
        st.markdown("**Largest dataset that fits on this GPU** (from live free VRAM at benchmark time)")
        st.dataframe(mf, use_container_width=True, hide_index=True)
        try:
            fp32_n = int(mf[mf["variant_id"] == 1]["max_feasible_n_this_gpu"].iloc[0])
            fp16_n = int(mf[mf["variant_id"] == 4]["max_feasible_n_this_gpu"].iloc[0])
            st.info(
                f"FP16 storage raises the limit from about **{fp32_n:,}** rows (FP32) to about "
                f"**{fp16_n:,}** rows, because the N×N distance matrix takes 2 bytes per entry instead of 4."
            )
        except (IndexError, ValueError):
            pass
    else:
        not_available("memory_analysis.csv not found")

    if mem is not None and not mem.empty:
        with st.expander("Theoretical vs measured memory per size (memory_analysis.csv)"):
            st.dataframe(mem, use_container_width=True, hide_index=True)

    if feas is not None and not feas.empty:
        f45 = feas[feas["subset"].astype(str).str.startswith("45k")]
        if not f45.empty:
            st.markdown("**45k feasibility check** (run live before any 45k attempt)")
            gib = 1024 ** 3
            st.dataframe(
                f45.assign(
                    estimated_peak_gib=(f45["estimated_peak_bytes"] / gib).round(2),
                    live_free_vram_gib=(f45["live_free_vram_bytes"] / gib).round(2),
                    safety_margin_gib=(f45["safety_margin_bytes"] / gib).round(2),
                )[["variant_name", "estimated_peak_gib", "live_free_vram_gib", "safety_margin_gib", "feasible"]],
                use_container_width=True,
                hide_index=True,
            )


def _correctness(corr):
    if corr is None or corr.empty:
        not_available("correctness_summary.csv not found")
        return
    st.markdown(
        "- **V1, V2, V3** are **bit-identical** to V0: zero score difference and identical neighbour indices.\n"
        "- **V4, V5** are **approximate** (FP16 storage): very high but not perfect agreement.\n"
        "- CPU comparisons differ by tiny amounts for every GPU variant because the CPU reference "
        "computes distances with a different formula (documented float32 cancellation issue)."
    )
    st.dataframe(corr, use_container_width=True, hide_index=True)


def _kernel_resources(ncu):
    st.warning(
        "**Nsight Compute counters: UNAVAILABLE** on this machine (`ERR_NVGPUCTRPERM` — Windows blocks "
        "access to GPU performance counters without admin permission: NVIDIA Control Panel → Desktop → "
        "Enable Developer Settings → Developer → Manage GPU Performance Counters → Allow access to all users). "
        "The values below come instead from `cudaFuncGetAttributes` (MEASURED from the compiled kernel) and "
        "`cudaOccupancyMaxActiveBlocksPerMultiprocessor` (theoretical occupancy — ESTIMATED, not achieved)."
    )
    if ncu is None or ncu.empty:
        not_available("ncu_metrics.csv not found")
        return
    avail = ncu[ncu["classification"] != "UNAVAILABLE"]
    if not avail.empty:
        wide = avail.pivot_table(
            index=["variant_id", "variant_name", "kernel"], columns="metric", values="value", aggfunc="first"
        ).reset_index()
        st.dataframe(wide, use_container_width=True, hide_index=True)
        st.caption(
            "Source per metric: registers / shared memory / local memory / threads per block = "
            "cudaFuncGetAttributes (MEASURED); max active blocks and theoretical occupancy = occupancy "
            "calculator (ESTIMATED)."
        )
        knn_spill = avail[(avail["kernel"] == "knn") & (avail["metric"] == "local_mem_bytes_per_thread")]
        if not knn_spill.empty and (knn_spill["value"] > 0).any():
            st.caption(
                "Note: the k-NN kernels spill to local memory (pre-existing from M3/M5, not introduced in M8) "
                "— a candidate for future optimization."
            )
    with st.expander("Raw ncu_metrics.csv (including UNAVAILABLE Nsight rows and their error text)"):
        st.dataframe(ncu, use_container_width=True, hide_index=True)


def render():
    st.title("Optimization Journey")
    st.caption(
        "CPU → naive GPU → GPU with syllabus techniques → each outside-syllabus technique → all combined. "
        "Every number on this page comes from **one benchmark session** (M8), so the steps are directly comparable."
    )

    vb = jd.load_m8_csv("variant_benchmarks.csv")
    if vb is None or vb.empty:
        not_available(
            "results/m8_advanced_optimizations/variant_benchmarks.csv not found — run "
            "scripts/run_m8_advanced_optimizations.py (see docs/WINDOWS_SETUP_GUIDE.md)."
        )
        return
    kb = jd.load_m8_csv("kernel_breakdown.csv")
    corr = jd.load_m8_csv("correctness_summary.csv")
    mem = jd.load_m8_csv("memory_analysis.csv")
    feas = jd.load_m8_csv("feasibility_checks.csv")
    ncu = jd.load_m8_csv("ncu_metrics.csv")

    try:
        gpu_reps = int(vb[vb["variant_id"] >= 0]["repetitions"].max())
        gpu_warm = int(vb[vb["variant_id"] >= 0]["warmups"].max())
        cpu_reps = int(vb[vb["variant_id"] == -1]["repetitions"].max())
        st.info(
            f"GPU timings are medians of {gpu_reps} runs after {gpu_warm} discarded warm-ups; CPU timings are "
            f"medians of {cpu_reps} runs. GPU run-to-run variation is high at 25k (see the timing-spread table), "
            "so medians are used throughout."
        )
    except (ValueError, KeyError):
        pass
    ram_gb = jd.ram_caveat(vb)
    if ram_gb is not None:
        st.warning(
            f"Only about **{ram_gb:.1f} GiB** of system RAM was free when the CPU runs started, so the CPU times "
            "may be inflated by paging to disk. GPU timings are unaffected."
        )

    options = jd.subsets(vb)
    subset = st.radio("Dataset size", options, index=min(1, len(options) - 1), horizontal=True)

    section("1. The ladder: each step in order")
    _ladder(vb, subset)

    section("2. Technique cards")
    _technique_cards(vb, kb, corr, subset)

    section("3. Where the time goes (per kernel)")
    _kernel_breakdown(kb, subset)

    section("4. Memory and largest feasible dataset")
    _memory(vb, mem, feas)

    section("5. Accuracy and correctness")
    _correctness(corr)

    section("6. Kernel resources")
    _kernel_resources(ncu)

    section("7. Charts and full report")
    with st.expander("Saved charts (results/m8_advanced_optimizations/charts/)"):
        names = [
            "m8_01_stage_ladder.png",
            "m8_02_kernel_breakdown.png",
            "m8_03_speedup_vs_v1.png",
            "m8_04_peak_memory.png",
            "m8_05_fp16_accuracy.png",
        ]
        cols = st.columns(2)
        shown = 0
        for name in names:
            img = loaders.load_image_path(*jd.M8_DIR, "charts", name)
            if img:
                cols[shown % 2].image(img, caption=name, use_container_width=True)
                shown += 1
        if not shown:
            not_available("No chart PNGs found.")
    md = loaders.load_text(*jd.M8_DIR, "report.md")
    with st.expander("Full M8 report (report.md)"):
        if md:
            st.markdown(md)
        else:
            not_available("report.md not found")
