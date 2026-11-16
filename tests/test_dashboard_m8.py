"""Data-prep tests for the Optimization Journey page (M8). No Streamlit runtime needed."""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "python"))

from dashboard import journey_data as jd  # noqa: E402
from dashboard import loaders  # noqa: E402


def _toy_benchmarks():
    rows = []
    for vid, name, src in [
        (3, "V3", "outside_syllabus"),
        (-1, "CPU", "cpu"),
        (1, "V1", "syllabus"),
        (0, "V0", "baseline"),
    ]:
        rows.append(
            {
                "subset": "10k",
                "n_samples": 10000,
                "variant_id": vid,
                "variant_name": name,
                "technique_source": src,
                "classification": "MEASURED",
                "end_to_end_median_ms": 10.0,
                "ram_available_bytes_at_start": 2 * 1024**3 if vid == -1 else None,
            }
        )
    rows.append({**rows[0], "subset": "45k", "n_samples": 45000, "classification": "UNAVAILABLE"})
    return pd.DataFrame(rows)


def test_ladder_is_ordered_cpu_first():
    df = jd.ladder_frame(_toy_benchmarks(), "10k")
    assert list(df["variant_id"]) == [-1, 0, 1, 3]
    assert df["bar_label"].iloc[0] == "CPU — CPU reference"


def test_subsets_sorted_by_size_and_unavailable_split():
    vb = _toy_benchmarks()
    assert jd.subsets(vb) == ["10k", "45k"]
    ladder_45k = jd.ladder_frame(vb, "45k")
    assert jd.measured(ladder_45k).empty
    assert len(jd.unavailable(ladder_45k)) == 1


def test_ram_caveat_uses_cpu_rows():
    assert jd.ram_caveat(_toy_benchmarks()) == pytest.approx(2.0)


def test_helpers_tolerate_missing_data():
    assert jd.subsets(None) == []
    assert jd.ladder_frame(None, "10k").empty
    assert jd.stage_breakdown(None, "10k").empty
    assert jd.distance_speedup_vs_v1(None, "10k").empty
    assert jd.max_feasible_n(None).empty
    assert jd.ram_caveat(None) is None
    assert loaders.load_csv("results", "does_not_exist", "nope.csv") is None


def test_distance_speedup_vs_v1():
    kb = pd.DataFrame(
        [
            {"subset": "10k", "variant_id": 1, "variant_name": "V1", "stage": "distance", "median_ms": 20.0},
            {"subset": "10k", "variant_id": 2, "variant_name": "V2", "stage": "distance", "median_ms": 5.0},
            {"subset": "10k", "variant_id": -1, "variant_name": "CPU", "stage": "distance", "median_ms": 999.0},
        ]
    )
    out = jd.distance_speedup_vs_v1(kb, "10k")
    assert list(out["variant_id"]) == [1, 2]
    assert out[out["variant_id"] == 2]["distance_speedup_vs_v1"].iloc[0] == pytest.approx(4.0)


@pytest.mark.skipif(
    jd.load_m8_csv("variant_benchmarks.csv") is None, reason="M8 results not present on this machine"
)
def test_real_m8_csv_has_expected_shape():
    vb = jd.load_m8_csv("variant_benchmarks.csv")
    for col in ["subset", "variant_id", "technique_source", "classification", "end_to_end_median_ms"]:
        assert col in vb.columns
    for subset in jd.subsets(vb):
        ladder = jd.ladder_frame(vb, subset)
        assert not ladder.empty
        positions = list(ladder["ladder_pos"])
        assert positions == sorted(positions)
