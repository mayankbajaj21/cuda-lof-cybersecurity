"""
test_distance_kernel.py - Test Suite for CUDA Pairwise Euclidean Distance Kernel
Milestone M3: CUDA LOF Baseline (Step 2)

Validates:
1. Correct output shape (N, N) for arbitrary N
2. Diagonal entries are strictly 0.0f
3. Distance matrix symmetry (dist[i, j] == dist[j, i])
4. Absence of NaN and Inf values
5. Numerical accuracy against independent CPU Euclidean calculation
6. Small synthetic datasets (D=68)
7. Non-block-aligned dimensions (boundary conditions)
8. Actual CICIDS2017 experimental subsets (10K and 25K with 68 standardized features)
"""

import os
import pytest
import numpy as np

from src.python.cuda.distance import compute_pairwise_distances, is_cuda_available
from src.python.data.dataset import load_subset

# Automatically skip this module if CUDA library is unavailable
pytestmark = pytest.mark.skipif(
    not is_cuda_available(),
    reason="CUDA distance kernel DLL not compiled or GPU unavailable"
)


class TestCUDADistanceKernelSynthetic:
    """Correctness tests using synthetic feature data with D=68 features."""

    @pytest.fixture
    def synthetic_data_68d(self):
        """Generate deterministic synthetic feature matrix (N=128, D=68)."""
        rng = np.random.default_rng(42)
        return rng.standard_normal((128, 68), dtype=np.float32)

    def test_output_shape_and_dtype(self, synthetic_data_68d):
        """Kernel output should have shape (N, N) and dtype float32."""
        N, D = synthetic_data_68d.shape
        assert D == 68
        dist = compute_pairwise_distances(synthetic_data_68d)
        assert dist.shape == (N, N)
        assert dist.dtype == np.float32

    def test_diagonal_is_strictly_zero(self, synthetic_data_68d):
        """Self-distance d(i, i) must be exactly 0.0f."""
        dist = compute_pairwise_distances(synthetic_data_68d)
        np.testing.assert_array_equal(np.diag(dist), np.zeros(len(synthetic_data_68d), dtype=np.float32))

    def test_symmetry(self, synthetic_data_68d):
        """Pairwise distance matrix must be symmetric: d(i, j) == d(j, i)."""
        dist = compute_pairwise_distances(synthetic_data_68d)
        np.testing.assert_allclose(dist, dist.T, atol=1e-5)

    def test_no_nan_or_inf(self, synthetic_data_68d):
        """Output must not contain NaN or infinite values."""
        dist = compute_pairwise_distances(synthetic_data_68d)
        assert np.all(np.isfinite(dist))

    def test_numerical_correctness_against_cpu(self, synthetic_data_68d):
        """GPU distance matrix must match independent CPU calculation within float32 tolerance."""
        dist_gpu = compute_pairwise_distances(synthetic_data_68d)

        # Independent CPU Euclidean distance calculation
        diff = synthetic_data_68d[:, np.newaxis, :] - synthetic_data_68d[np.newaxis, :, :]
        dist_cpu = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dist_cpu, 0.0)

        np.testing.assert_allclose(dist_gpu, dist_cpu, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("N", [1, 7, 31, 65, 127, 257])
    def test_arbitrary_n_boundary_handling(self, N):
        """Kernel must correctly handle arbitrary N not divisible by block size (16x16)."""
        rng = np.random.default_rng(100 + N)
        X = rng.standard_normal((N, 68), dtype=np.float32)
        dist = compute_pairwise_distances(X)

        assert dist.shape == (N, N)
        assert np.all(np.diag(dist) == 0.0)
        assert np.all(np.isfinite(dist))

        # Check random pair against CPU
        if N > 1:
            i, j = 0, N - 1
            expected = np.linalg.norm(X[i] - X[j])
            assert abs(dist[i, j] - expected) < 1e-4


class TestCUDADistanceKernelCICIDS2017:
    """Correctness tests using actual CICIDS2017 preprocessed subsets (D=68 features)."""

    def test_cicids_10k_subset_properties_and_correctness(self):
        """Test on actual 10k subset: shape, zero diagonal, finiteness, and sample accuracy."""
        X, _, _, _, _ = load_subset("10k")
        N, D = X.shape
        assert N == 10000
        assert D == 68

        dist = compute_pairwise_distances(X)

        # 1. Output dimensions & dtype
        assert dist.shape == (10000, 10000)
        assert dist.dtype == np.float32

        # 2. Diagonal entries zero
        np.testing.assert_array_equal(np.diag(dist), np.zeros(N, dtype=np.float32))

        # 3. No NaN / Inf
        assert np.all(np.isfinite(dist[:500, :500]))

        # 4. Symmetry on sampled submatrix
        sample_slice = slice(0, 500)
        np.testing.assert_allclose(dist[sample_slice, sample_slice], dist[sample_slice, sample_slice].T, atol=1e-4)

        # 5. Numerical correctness on 50 sampled pairs against CPU reference
        rng = np.random.default_rng(2026)
        i_indices = rng.integers(0, N, size=50)
        j_indices = rng.integers(0, N, size=50)

        for i, j in zip(i_indices, j_indices):
            expected = 0.0 if i == j else float(np.linalg.norm(X[i] - X[j]))
            actual = float(dist[i, j])
            assert abs(actual - expected) < 1e-4, (
                f"Mismatch at ({i}, {j}): actual={actual}, expected={expected}"
            )

    def test_cicids_25k_subset_correctness(self):
        """Test on actual 25k subset (25,000 x 25,000 matrix ≈ 2.5 GB)."""
        X, _, _, _, _ = load_subset("25k")
        N, D = X.shape
        assert N == 25000
        assert D == 68

        dist = compute_pairwise_distances(X)

        assert dist.shape == (25000, 25000)
        assert dist.dtype == np.float32

        # Sampled checks to verify 25k computation
        assert dist[0, 0] == 0.0
        assert dist[N - 1, N - 1] == 0.0
        assert dist[N // 2, N // 2] == 0.0

        # Sampled CPU verification on 20 pairs
        rng = np.random.default_rng(777)
        i_indices = rng.integers(0, N, size=20)
        j_indices = rng.integers(0, N, size=20)

        for i, j in zip(i_indices, j_indices):
            expected = 0.0 if i == j else float(np.linalg.norm(X[i] - X[j]))
            actual = float(dist[i, j])
            assert abs(actual - expected) < 1e-4, (
                f"Mismatch at ({i}, {j}): actual={actual}, expected={expected}"
            )
