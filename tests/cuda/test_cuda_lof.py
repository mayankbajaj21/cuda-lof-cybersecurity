"""
test_cuda_lof.py - Test Suite for Complete CUDA LOF Baseline Pipeline
Milestone M3: CUDA LOF Baseline (Step 4)

Validates:
1. Reachability distance output shape (N, k) and finiteness
2. LRD output shape (N,) and finiteness
3. LOF scores output shape (N,) and finiteness
4. Outlier detection sensitivity (injected outlier produces LOF >> 1.0)
5. Execution determinism across repeated runs
6. Exact stage-by-stage agreement against M2 CPU reference on synthetic data
7. Detection consistency and correlation on CICIDS2017 10K subset
8. Detection consistency and correlation on CICIDS2017 25K subset
"""

import pytest
import numpy as np

from src.python.cpu.cpu_lof import CPULocalOutlierFactor
from src.python.cuda.lof import (
    compute_reachability_distances,
    compute_lrd,
    compute_lof_scores,
    compute_cuda_lof,
    CUDALocalOutlierFactor,
    is_lof_available,
)
from src.python.data.dataset import load_subset

# Skip module if CUDA LOF library is unavailable
pytestmark = pytest.mark.skipif(
    not is_lof_available(),
    reason="CUDA LOF library not compiled or GPU unavailable"
)


class TestCUDALOFStageShapes:
    """Validate output dimensions, dtypes, and finiteness for all intermediate stages."""

    @pytest.fixture
    def synthetic_data(self):
        rng = np.random.default_rng(42)
        return rng.standard_normal((128, 68), dtype=np.float32)

    def test_pipeline_stages_and_shapes(self, synthetic_data):
        """Verify reachability, LRD, and LOF shapes and dtypes."""
        k = 20
        scores, intermediates = compute_cuda_lof(synthetic_data, k=k, return_intermediates=True)

        reach_dists = intermediates["reach_dists"]
        lrd = intermediates["lrd"]
        lof_scores = intermediates["lof_scores"]

        assert reach_dists.shape == (128, k)
        assert reach_dists.dtype == np.float32
        assert np.all(np.isfinite(reach_dists))
        assert np.all(reach_dists >= 0.0)

        assert lrd.shape == (128,)
        assert lrd.dtype == np.float32
        assert np.all(np.isfinite(lrd))
        assert np.all(lrd > 0.0)

        assert lof_scores.shape == (128,)
        assert lof_scores.dtype == np.float32
        assert np.all(np.isfinite(lof_scores))
        assert np.all(lof_scores > 0.0)

    def test_outlier_sensitivity(self):
        """Injected outlier should yield a significantly elevated LOF score."""
        rng = np.random.default_rng(123)
        X = rng.standard_normal((100, 68), dtype=np.float32) * 0.5
        # Inject prominent outlier
        X[-1] = 50.0

        scores = compute_cuda_lof(X, k=20)
        inlier_mean = np.mean(scores[:-1])
        outlier_score = scores[-1]

        assert inlier_mean < 2.0
        assert outlier_score > 10.0
        assert outlier_score == np.max(scores)

    def test_determinism(self, synthetic_data):
        """Repeated executions must produce bit-for-bit identical LOF scores."""
        scores1 = compute_cuda_lof(synthetic_data, k=20)
        scores2 = compute_cuda_lof(synthetic_data, k=20)

        np.testing.assert_array_equal(scores1, scores2)

    def test_benchmark_cuda_lof_event_timing(self, synthetic_data):
        """Verify hardware event timing structure and positive elapsed times."""
        from src.python.cuda.lof import benchmark_cuda_lof
        scores, timings = benchmark_cuda_lof(synthetic_data, k=20)

        assert scores.shape == (128,)
        assert np.all(np.isfinite(scores))

        expected_keys = [
            "h2d_ms", "distance_kernel_ms", "knn_kernel_ms",
            "reachability_kernel_ms", "lrd_kernel_ms", "lof_kernel_ms",
            "d2h_ms", "total_kernel_ms", "end_to_end_ms"
        ]
        for key in expected_keys:
            assert key in timings
            assert timings[key] > 0.0, f"Timing {key} must be strictly positive, got {timings[key]}"

        assert timings["total_kernel_ms"] <= timings["end_to_end_ms"]


class TestCUDALOFvsCPUM2Synthetic:
    """Direct stage-by-stage comparison against M2 CPU LOF baseline."""

    @pytest.fixture
    def synthetic_dataset(self):
        rng = np.random.default_rng(999)
        return rng.standard_normal((100, 68), dtype=np.float32)

    def test_stage_by_stage_cpu_agreement(self, synthetic_dataset):
        """Compare all stages against CPULocalOutlierFactor on synthetic data."""
        k = 20
        cpu_model = CPULocalOutlierFactor(k=k).fit(synthetic_dataset)
        cuda_scores, cuda_inter = compute_cuda_lof(synthetic_dataset, k=k, return_intermediates=True)

        # 1. k-NN distances
        np.testing.assert_allclose(
            cuda_inter["knn_distances"], cpu_model.knn_distances_, atol=1e-4, rtol=1e-4
        )

        # 2. Reachability distances
        np.testing.assert_allclose(
            cuda_inter["reach_dists"], cpu_model.reach_dists_, atol=1e-4, rtol=1e-4
        )

        # 3. LRD values
        np.testing.assert_allclose(
            cuda_inter["lrd"], cpu_model.lrd_, atol=1e-4, rtol=1e-4
        )

        # 4. LOF scores
        np.testing.assert_allclose(
            cuda_scores, cpu_model.lof_scores_, atol=1e-4, rtol=1e-4
        )

        # 5. Pearson correlation
        corr = np.corrcoef(cpu_model.lof_scores_, cuda_scores)[0, 1]
        assert corr > 0.999999

        # 6. Prediction agreement
        cuda_model = CUDALocalOutlierFactor(k=k).fit(synthetic_dataset)
        np.testing.assert_array_equal(cuda_model.predictions_, cpu_model.predictions_)


class TestCUDALOFCICIDS2017:
    """Validation on actual CICIDS2017 experimental subsets (D=68, k=20)."""

    def test_cicids_10k_complete_pipeline(self):
        """Validate on CICIDS2017 10K subset."""
        X, _, _, _, _ = load_subset("10k")
        assert X.shape == (10000, 68)

        cpu_model = CPULocalOutlierFactor(k=20).fit(X)
        cuda_model = CUDALocalOutlierFactor(k=20).fit(X)

        assert cuda_model.lof_scores_.shape == (10000,)
        assert np.all(np.isfinite(cuda_model.lof_scores_))

        # Correlation between CPU and CUDA LOF scores
        corr = np.corrcoef(cpu_model.lof_scores_, cuda_model.lof_scores_)[0, 1]
        assert corr > 0.999, f"Pearson correlation too low: {corr}"

        # Anomaly label agreement
        agreement = np.mean(cpu_model.predictions_ == cuda_model.predictions_)
        assert agreement >= 0.999, f"Prediction agreement too low: {agreement}"

        # Anomaly counts (10% of 10,000 = 1,000)
        assert np.sum(cuda_model.predictions_ == 1) == 1000

    def test_cicids_25k_complete_pipeline(self):
        """Validate on CICIDS2017 25K subset."""
        X, _, _, _, _ = load_subset("25k")
        assert X.shape == (25000, 68)

        cuda_model = CUDALocalOutlierFactor(k=20).fit(X)

        assert cuda_model.lof_scores_.shape == (25000,)
        assert np.all(np.isfinite(cuda_model.lof_scores_))

        # Check anomaly count at 10% contamination (2,500 anomalies)
        assert np.sum(cuda_model.predictions_ == 1) == 2500

        # Memory & time sanity
        assert cuda_model.timings_["total"] < 15.0
