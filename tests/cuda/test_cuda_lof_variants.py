"""
test_cuda_lof_variants.py - Correctness tests for the M8 LOF variants V0..V5
(bin/cuda_lof_variants.dll). No timing assertions.

Exact variants (V1-V3) must reproduce V0. FP16 variants (V4, V5) are
approximate by design: tolerances below are documented, not tuned to pass.
"""

import numpy as np
import pytest

from src.python.cpu.cpu_lof import CPULocalOutlierFactor
from src.python.cuda.lof_variants import (
    VARIANTS,
    compute_lof_variant,
    estimate_variant_gpu_bytes,
    get_kernel_attributes,
    is_variants_available,
    variant_max_k,
)

pytestmark = pytest.mark.skipif(
    not is_variants_available(), reason="bin/cuda_lof_variants.dll not built or GPU unavailable"
)

EXACT_VARIANTS = (1, 2, 3)
FP16_VARIANTS = (4, 5)

# FP16 storage keeps ~3 significant digits per feature and distance; on
# standard-normal data that moves LOF scores by well under 0.1 and swaps a
# small fraction of near-tied neighbors.
FP16_LOF_ATOL = 0.1
FP16_MIN_KNN_RECALL = 0.97


def _knn_recall(ref_idx, idx):
    k = ref_idx.shape[1]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(ref_idx, idx)]))


def _data(n, d, seed=42, outlier=True):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d), dtype=np.float32)
    if outlier:
        X[-1] += 8.0
    return X


@pytest.fixture(scope="module")
def reference():
    X = _data(1024, 68)
    scores, _, _, idx = compute_lof_variant(X, 20, 0, return_knn_indices=True)
    return X, scores, idx


@pytest.mark.parametrize("variant", EXACT_VARIANTS)
def test_exact_variants_match_v0(reference, variant):
    X, ref_scores, ref_idx = reference
    scores, _, _, idx = compute_lof_variant(X, 20, variant, return_knn_indices=True)
    np.testing.assert_allclose(scores, ref_scores, rtol=1e-4, atol=1e-6)
    assert np.array_equal(idx, ref_idx)


@pytest.mark.parametrize("variant", FP16_VARIANTS)
def test_fp16_variants_close_to_v0(reference, variant):
    X, ref_scores, ref_idx = reference
    scores, _, mem, idx = compute_lof_variant(X, 20, variant, return_knn_indices=True)
    assert np.max(np.abs(scores - ref_scores)) < FP16_LOF_ATOL
    assert _knn_recall(ref_idx, idx) >= FP16_MIN_KNN_RECALL
    assert mem["fp16_overflow_count"] == 0
    assert mem["dist_element_bytes"] == 2


def test_v4_and_v5_agree(reference):
    X, _, _ = reference
    s4 = compute_lof_variant(X, 20, 4)[0]
    s5 = compute_lof_variant(X, 20, 5)[0]
    np.testing.assert_allclose(s4, s5, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("n,d", [(1000, 67), (777, 5), (130, 68), (1000, 68)])
@pytest.mark.parametrize("variant", range(1, 6))
def test_odd_shapes(n, d, variant):
    X = _data(n, d, seed=n + d)
    ref, _, _, ref_idx = compute_lof_variant(X, 20, 0, return_knn_indices=True)
    scores, _, _, idx = compute_lof_variant(X, 20, variant, return_knn_indices=True)
    assert np.all(np.isfinite(scores))
    if variant in EXACT_VARIANTS:
        np.testing.assert_allclose(scores, ref, rtol=1e-4, atol=1e-6)
        assert np.array_equal(idx, ref_idx)
    else:
        assert np.max(np.abs(scores - ref)) < FP16_LOF_ATOL
        assert _knn_recall(ref_idx, idx) >= FP16_MIN_KNN_RECALL


@pytest.mark.parametrize("variant", range(6))
def test_self_excluded_and_outlier_highest(variant):
    X = _data(600, 68, seed=3)
    scores, _, _, idx = compute_lof_variant(X, 20, variant, return_knn_indices=True)
    for i in range(idx.shape[0]):
        assert i not in idx[i]
    assert int(np.argmax(scores)) == X.shape[0] - 1


def test_exact_variants_match_cpu_reference():
    X = _data(800, 68, seed=11)
    cpu_scores = CPULocalOutlierFactor(k=20).fit(X).lof_scores_
    for v in EXACT_VARIANTS:
        scores = compute_lof_variant(X, 20, v)[0]
        assert np.corrcoef(cpu_scores, scores)[0, 1] > 0.999


def test_timings_and_memory_reported(reference):
    X, _, _ = reference
    for v in range(6):
        _, t, m = compute_lof_variant(X, 20, v)
        assert t["end_to_end_ms"] > 0 and t["distance_kernel_ms"] > 0 and t["knn_kernel_ms"] > 0
        assert (t["fp16_convert_ms"] > 0) == (v in FP16_VARIANTS)
        assert m["tracked_peak_bytes"] == estimate_variant_gpu_bytes(X.shape[0], X.shape[1], 20, v)


def test_k_limits():
    X = _data(200, 8)
    assert variant_max_k(0) == 256
    for v in range(1, 6):
        assert variant_max_k(v) == 32
        with pytest.raises(ValueError):
            compute_lof_variant(X, 33, v)


def test_kernel_attributes_available():
    for v in VARIANTS:
        for kernel in ("distance", "knn"):
            attrs = get_kernel_attributes(v, kernel)
            assert attrs["num_regs"] > 0
