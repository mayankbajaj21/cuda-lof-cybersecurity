"""
test_cuda_lof_optimized.py - Test Suite for Optimized CUDA LOF Pipeline
Milestone M5: Profiling-Driven CUDA Optimization

Validates the OPTIMIZED pipeline (bin/cuda_lof_optimized.dll) for correctness
against the CPU reference and the naive CUDA baseline. Does not test speed
(no timing assertions) - speed is covered by scripts/run_m5_optimization.py
and reported in results/m5_cuda_optimized/.
"""

import pytest
import numpy as np

from src.python.cpu.cpu_lof import CPULocalOutlierFactor
from src.python.cuda.lof import compute_cuda_lof, is_lof_available
from src.python.cuda.lof_optimized import (
    compute_cuda_lof_optimized,
    compute_pairwise_distances_optimized,
    compute_knn_optimized,
    is_lof_optimized_available,
    OPTIMIZED_MAX_K,
)
from src.python.cuda.distance import compute_pairwise_distances
from src.python.cuda.knn import compute_knn
from src.python.data.dataset import load_subset

pytestmark = pytest.mark.skipif(
    not (is_lof_available() and is_lof_optimized_available()),
    reason="Naive and/or optimized CUDA LOF library not compiled or GPU unavailable"
)


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    return rng.standard_normal((256, 68), dtype=np.float32)


class TestOptimizedDistanceCorrectness:
    def test_matches_naive_distance_matrix_exactly(self, synthetic_data):
        naive = compute_pairwise_distances(synthetic_data)
        optimized = compute_pairwise_distances_optimized(synthetic_data)
        assert naive.shape == optimized.shape
        np.testing.assert_allclose(naive, optimized, atol=1e-5)

    def test_diagonal_is_zero(self, synthetic_data):
        optimized = compute_pairwise_distances_optimized(synthetic_data)
        assert np.allclose(np.diag(optimized), 0.0)

    def test_arbitrary_feature_count_not_multiple_of_tile(self):
        """D=68 is not a multiple of the 16-wide tile - exercises the bounds-checked tail chunk."""
        rng = np.random.default_rng(7)
        X = rng.standard_normal((64, 37), dtype=np.float32)  # D=37, also not a multiple of 16
        naive = compute_pairwise_distances(X)
        optimized = compute_pairwise_distances_optimized(X)
        np.testing.assert_allclose(naive, optimized, atol=1e-5)


class TestOptimizedKNNCorrectness:
    def test_matches_naive_knn_indices_and_distances(self, synthetic_data):
        dist_matrix = compute_pairwise_distances_optimized(synthetic_data)
        naive_idx, naive_dist = compute_knn(dist_matrix, k=20)
        opt_idx, opt_dist = compute_knn_optimized(dist_matrix, k=20)

        assert np.array_equal(naive_idx, opt_idx)
        np.testing.assert_allclose(naive_dist, opt_dist, atol=1e-5)

    def test_self_excluded(self, synthetic_data):
        dist_matrix = compute_pairwise_distances_optimized(synthetic_data)
        opt_idx, _ = compute_knn_optimized(dist_matrix, k=20)
        for i in range(opt_idx.shape[0]):
            assert i not in opt_idx[i]

    def test_k_above_max_raises(self, synthetic_data):
        dist_matrix = compute_pairwise_distances_optimized(synthetic_data)
        with pytest.raises(ValueError):
            compute_knn_optimized(dist_matrix, k=OPTIMIZED_MAX_K + 1)


class TestOptimizedPipelineCorrectness:
    def test_matches_naive_pipeline_exactly(self, synthetic_data):
        naive_scores = compute_cuda_lof(synthetic_data, k=20)
        opt_scores = compute_cuda_lof_optimized(synthetic_data, k=20)
        np.testing.assert_allclose(naive_scores, opt_scores, atol=1e-4)

    def test_outlier_detection_sensitivity(self, synthetic_data):
        X = synthetic_data.copy()
        X[-1] = 100.0  # inject obvious outlier
        scores = compute_cuda_lof_optimized(X, k=20)
        assert np.all(np.isfinite(scores))
        assert scores[-1] > 5.0 * np.median(scores)

    def test_matches_cpu_reference_on_10k_subset(self):
        X, _y_bin, _y_multi, _y_raw, _indices = load_subset("10k")
        cpu_model = CPULocalOutlierFactor(k=20, contamination=0.1)
        cpu_model.fit(X)

        opt_scores = compute_cuda_lof_optimized(X, k=20)

        r = np.corrcoef(cpu_model.lof_scores_, opt_scores)[0, 1]
        assert r > 0.999

    def test_matches_naive_on_25k_subset(self):
        X, _y_bin, _y_multi, _y_raw, _indices = load_subset("25k")
        naive_scores = compute_cuda_lof(X, k=20)
        opt_scores = compute_cuda_lof_optimized(X, k=20)
        np.testing.assert_allclose(naive_scores, opt_scores, atol=1e-3)
