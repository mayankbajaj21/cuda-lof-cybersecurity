"""Streamlit-free data preparation for the Optimization Journey page (M8).

Kept separate from the page so it can be unit-tested without a running app.
Every function tolerates missing files/columns and returns None or an empty
frame instead of raising.
"""

import pandas as pd

from . import loaders

M8_DIR = ("results", "m8_advanced_optimizations")

LADDER_ORDER = [-1, 0, 1, 2, 3, 4, 5]
STAGE_ORDER = ["h2d", "fp16_convert", "distance", "knn", "reachability", "lrd", "lof", "d2h"]

SOURCE_LABELS = {
    "cpu": "CPU reference",
    "baseline": "GPU baseline",
    "syllabus": "Syllabus techniques",
    "outside_syllabus": "Outside syllabus",
    "combined_outside_syllabus": "Outside syllabus (combined)",
}


def load_m8_csv(name):
    return loaders.load_csv(*M8_DIR, name)


def subsets(vb):
    if vb is None or vb.empty or "subset" not in vb.columns:
        return []
    order = vb.drop_duplicates("subset").sort_values("n_samples")
    return list(order["subset"])


def ladder_frame(vb, subset):
    """Rows for one subset in ladder order, with a readable source label."""
    if vb is None or vb.empty:
        return pd.DataFrame()
    df = vb[vb["subset"] == subset].copy()
    if df.empty:
        return df
    df["ladder_pos"] = df["variant_id"].apply(lambda v: LADDER_ORDER.index(v) if v in LADDER_ORDER else 99)
    df = df.sort_values("ladder_pos")
    df["source_label"] = df["technique_source"].map(SOURCE_LABELS).fillna(df["technique_source"])
    df["bar_label"] = df["variant_name"] + " — " + df["source_label"]
    return df


def measured(df):
    if df is None or df.empty or "classification" not in df.columns:
        return pd.DataFrame()
    return df[df["classification"] == "MEASURED"]


def unavailable(df):
    if df is None or df.empty or "classification" not in df.columns:
        return pd.DataFrame()
    return df[df["classification"] == "UNAVAILABLE"]


def stage_breakdown(kb, subset):
    """Long-format GPU per-stage medians for one subset (excludes totals and CPU)."""
    if kb is None or kb.empty:
        return pd.DataFrame()
    df = kb[(kb["subset"] == subset) & (kb["variant_id"] >= 0) & (kb["stage"].isin(STAGE_ORDER))].copy()
    df["stage"] = pd.Categorical(df["stage"], STAGE_ORDER, ordered=True)
    return df.sort_values(["variant_id", "stage"])


def distance_speedup_vs_v1(kb, subset):
    """Distance-kernel median speedup of each GPU variant relative to V1."""
    if kb is None or kb.empty:
        return pd.DataFrame()
    d = kb[(kb["subset"] == subset) & (kb["stage"] == "distance") & (kb["variant_id"] >= 0)]
    v1 = d[d["variant_id"] == 1]
    if v1.empty:
        return pd.DataFrame()
    base = float(v1["median_ms"].iloc[0])
    out = d[["variant_id", "variant_name", "median_ms"]].copy()
    out["distance_speedup_vs_v1"] = base / out["median_ms"]
    return out.sort_values("variant_id")


def variant_row(vb, subset, variant_id):
    if vb is None or vb.empty:
        return None
    r = vb[(vb["subset"] == subset) & (vb["variant_id"] == variant_id)]
    return None if r.empty else r.iloc[0]


def ram_caveat(vb):
    """Smallest RAM available at the start of any CPU run, in GiB, or None."""
    if vb is None or vb.empty or "ram_available_bytes_at_start" not in vb.columns:
        return None
    cpu = vb[(vb["variant_id"] == -1) & vb["ram_available_bytes_at_start"].notna()]
    if cpu.empty:
        return None
    return float(cpu["ram_available_bytes_at_start"].min()) / (1024 ** 3)


def max_feasible_n(mem):
    if mem is None or mem.empty:
        return pd.DataFrame()
    return mem.drop_duplicates("variant_id")[["variant_id", "variant_name", "max_feasible_n_this_gpu"]]
