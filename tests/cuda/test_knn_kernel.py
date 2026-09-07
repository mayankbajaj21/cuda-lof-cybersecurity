"""
test_knn_kernel.py - Test Suite for CUDA k-Nearest Neighbors (k-NN) Kernel
Milestone M3: CUDA LOF Baseline (Step 3)

Validates:
1. Output shapes (N, k) and data types (int32 for indices, float32 for distances)
2. Valid neighbor indices (0 <= idx < N)
3. Exclusion of self-neighbor (idx != i)
4. Distances are non-negative and finite (no NaN or Inf)
5. Neighbor distances sorted ascending (d_0 <= d_1 <= ... <= d_{k-1})
6. Correspondence between recorded distances and actual distance matrix values
7. Exact numerical match against independent CPU k-NN reference
8. Arbitrary N and configurable k parameter (k=5, 10, 20, 25)
9. Determinism across repeated executions
10. Actual CICIDS2017 experimental subsets (10K and 25K with D=68 features, k=20)
"""

import pytest
import numpy as np

from src.python.cuda.distance import compute_pairwise_distances, is_cuda_available
from src.python.cuda.knn import compute_knn, is_knn_available
from src.python.data.dataset import load_subset

# Automatically skip this module if CUDA library is unavailable
pytestmark = pytest.mark.skipif(
    not is_cuda_available() or not is_knn_available(),
    reason="CUDA distance or k-NN kernel DLL not compiled or GPU unavailable"
)


def cpu_knn_reference(dist_matrix: np.ndarray, k: int, row_indices=None):
    """
    Independent CPU reference calculation for k-NN from a distance matrix.
    Excludes self-distance (diagonal) and returns sorted (indices, distances).
    """
    N = dist_matrix.shape[0]
    if row_indices is None:
        row_indices = range(N)

    ref_indices = []
    ref_distances = []

    for i in row_indices:
        row = dist_matrix[i].copy()
        row[i] = np.inf  # Exclude self
        sorted_idx = np.argsort(row)[:k]
        sorted_d = row[sorted_idx]
        ref_indices.append(sorted_idx)
        ref_distances.append(sorted_d)

    return np.array(ref_indices, dtype=np.int32), np.array(ref_distances, dtype=np.float32)


class TestCUDAKNNSynthetic:
    """Correctness tests using synthetic feature data with D=68 features."""

    @pytest.fixture
    def synthetic_dist_68d(self):
        """Generate deterministic synthetic distance matrix from D=68 features (N=128)."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((128, 68), dtype=np.float32)
        return compute_pairwise_distances(X)

    def test_output_shapes_and_dtypes(self, synthetic_dist_68d):
        """Output shapes must be (N, k) and dtypes int32 and float32."""
        N = synthetic_dist_68d.shape[0]
        k = 20
        indices, distances = compute_knn(synthetic_dist_68d, k=k)

        assert indices.shape == (N, k)
        assert indices.dtype == np.int32
        assert distances.shape == (N, k)
        assert distances.dtype == np.float32

    def test_valid_indices(self, synthetic_dist_68d):
        """Neighbor indices must be in range [0, N-1]."""
        N = synthetic_dist_68d.shape[0]
        indices, _ = compute_knn(synthetic_dist_68d, k=20)

        assert np.all(indices >= 0)
        assert np.all(indices < N)

    def test_no_self_neighbor(self, synthetic_dist_68d):
        """No sample should have itself as a neighbor."""
        N = synthetic_dist_68d.shape[0]
        indices, _ = compute_knn(synthetic_dist_68d, k=20)

        self_indices = np.arange(N)[:, np.newaxis]
        assert np.all(indices != self_indices)

    def test_distances_finite(self, synthetic_dist_68d):
        """Distances must be non-negative, finite, and free of NaN/Inf."""
        _, distances = compute_knn(synthetic_dist_68d, k=20)

        assert np.all(np.isfinite(distances))
        assert np.all(distances >= 0.0)

    def test_neighbor_distances_sorted(self, synthetic_dist_68d):
        """Neighbor distances must be monotonically non-decreasing along each row."""
        _, distances = compute_knn(synthetic_dist_68d, k=20)

        diffs = np.diff(distances, axis=1)
        assert np.all(diffs >= -1e-6)

    def test_indices_distances_correspondence(self, synthetic_dist_68d):
        """Recorded neighbor distances must match dist_matrix[i, idx[i, m]]."""
        N = synthetic_dist_68d.shape[0]
        k = 20
        indices, distances = compute_knn(synthetic_dist_68d, k=k)

        for i in range(N):
            for m in range(k):
                actual_d = synthetic_dist_68d[i, indices[i, m]]
                assert abs(distances[i, m] - actual_d) < 1e-5

    def test_numerical_match_against_cpu_reference(self, synthetic_dist_68d):
        """GPU k-NN distances and indices must match CPU reference calculation."""
        k = 20
        gpu_indices, gpu_distances = compute_knn(synthetic_dist_68d, k=k)
        cpu_indices, cpu_distances = cpu_knn_reference(synthetic_dist_68d, k=k)

        np.testing.assert_allclose(gpu_distances, cpu_distances, atol=1e-5)
        np.testing.assert_array_equal(gpu_indices, cpu_indices)

    @pytest.mark.parametrize("k", [5, 10, 20, 25])
    def test_configurable_k_parameter(self, synthetic_dist_68d, k):
        """Kernel must support arbitrary configurable k parameters."""
        N = synthetic_dist_68d.shape[0]
        indices, distances = compute_knn(synthetic_dist_68d, k=k)

        assert indices.shape == (N, k)
        assert distances.shape == (N, k)
        assert np.all(indices != np.arange(N)[:, np.newaxis])
        assert np.all(np.diff(distances, axis=1) >= -1e-6)

    def test_determinism(self, synthetic_dist_68d):
        """Repeated executions on the same input must produce bit-for-bit identical results."""
        idx1, dist1 = compute_knn(synthetic_dist_68d, k=20)
        idx2, dist2 = compute_knn(synthetic_dist_68d, k=20)

        np.testing.assert_array_equal(idx1, idx2)
        np.testing.assert_array_equal(dist1, dist2)


class TestCUDAKNNCICIDS2017:
    """Correctness tests using actual CICIDS2017 preprocessed subsets (D=68, k=20)."""

    def test_cicids_10k_knn_correctness(self):
        """Strong correctness validation on the 10k subset."""
        X, _, _, _, _ = load_subset("10k")
        N, D = X.shape
        assert N == 10000
        assert D == 68
        k = 20

        dist = compute_pairwise_distances(X)
        gpu_indices, gpu_distances = compute_knn(dist, k=k)

        # 1. Output dimensions & dtypes
        assert gpu_indices.shape == (10000, 20)
        assert gpu_distances.shape == (10000, 20)
        assert gpu_indices.dtype == np.int32
        assert gpu_distances.dtype == np.float32

        # 2. No self-neighbor
        self_indices = np.arange(N)[:, np.newaxis]
        assert np.all(gpu_indices != self_indices)

        # 3. Monotonically non-decreasing
        assert np.all(np.diff(gpu_distances, axis=1) >= -1e-5)

        # 4. Strong correctness: 50 randomly sampled rows compared against CPU reference
        rng = np.random.default_rng(2026)
        sampled_rows = rng.integers(0, N, size=50)

        cpu_idx, cpu_dist = cpu_knn_reference(dist, k=k, row_indices=sampled_rows)
        np.testing.assert_allclose(gpu_distances[sampled_rows], cpu_dist, atol=1e-4)

    def test_cicids_25k_knn_correctness(self):
        """Correctness validation on 25k subset (25,000 x 25,000 matrix)."""
        X, _, _, _, _ = load_subset("25k")
        N, D = X.shape
        assert N == 25000
        assert D == 68
        k = 20

        dist = compute_pairwise_distances(X)
        gpu_indices, gpu_distances = compute_knn(dist, k=k)

        # 1. Output dimensions
        assert gpu_indices.shape == (25000, 20)
        assert gpu_distances.shape == (25000, 20)

        # 2. No self-neighbor
        self_indices = np.arange(N)[:, np.newaxis]
        assert np.all(gpu_indices != self_indices)

        # 3. Monotonically non-decreasing
        assert np.all(np.diff(gpu_distances, axis=1) >= -1e-5)

        # 4. Validate selected rows against CPU reference
        test_rows = [0, 1234, 12500, 24999]
        cpu_idx, cpu_dist = cpu_knn_reference(dist, k=k, row_indices=test_rows)
        np.testing.assert_allclose(gpu_distances[test_rows], cpu_dist, atol=1e-4)

    def test_cicids_10k_determinism(self):
        """Verify determinism across repeated executions on 10k dataset."""
        X, _, _, _, _ = load_subset("10k")
        dist = compute_pairwise_distances(X[:1000])  # 1000 samples for swift execution

        idx1, dist1 = compute_knn(dist, k=20)
        idx2, dist2 = compute_knn(dist, k=20)

        np.testing.assert_array_equal(idx1, idx2)
        np.testing.assert_array_equal(dist1, dist2)
