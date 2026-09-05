"""
Unit Tests for Naive CPU Local Outlier Factor Baseline
Milestone M2: Naive CPU LOF Baseline

Verifies:
- Output array dimensions and shapes
- Absence of NaN and Inf values
- Neighbor index validity (no self-loops, bounded in [0, N-1])
- Deterministic reproducibility
- Numerical equivalence against reference scikit-learn LOF on synthetic and real sample data
- Parameter configurability and error handling
"""

import os
import sys
import pytest
import numpy as np
from sklearn.neighbors import LocalOutlierFactor as SklearnLOF

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.python.cpu.cpu_lof import CPULocalOutlierFactor
from src.python.data.dataset import load_subset


@pytest.fixture(scope="module")
def sample_synthetic_data():
    """Create a controlled synthetic dataset with distinct clusters and known outliers."""
    rng = np.random.default_rng(42)
    # Cluster 1 (dense)
    c1 = rng.normal(loc=0.0, scale=0.5, size=(100, 5))
    # Cluster 2 (moderate density)
    c2 = rng.normal(loc=5.0, scale=1.0, size=(100, 5))
    # Outliers (sparse)
    outliers = rng.uniform(low=15.0, high=25.0, size=(10, 5))
    X = np.vstack([c1, c2, outliers]).astype(np.float32)
    return X


@pytest.fixture(scope="module")
def cicids_10k_subset():
    """Load the preprocessed 10k dataset subset."""
    X, y_bin, y_multi, y_raw, indices = load_subset("10k")
    return X[:500]  # Use first 500 samples for fast test execution


def test_cpu_lof_output_dimensions(sample_synthetic_data):
    X = sample_synthetic_data
    N, D = X.shape
    k = 15
    lof = CPULocalOutlierFactor(k=k)
    lof.fit(X)

    assert lof.n_samples_ == N
    assert lof.n_features_ == D
    assert lof.dist_matrix_.shape == (N, N)
    assert lof.knn_indices_.shape == (N, k)
    assert lof.knn_distances_.shape == (N, k)
    assert lof.k_distances_.shape == (N,)
    assert lof.reach_dists_.shape == (N, k)
    assert lof.lrd_.shape == (N,)
    assert lof.lof_scores_.shape == (N,)
    assert lof.predictions_.shape == (N,)


def test_absence_of_nan_and_inf(sample_synthetic_data):
    X = sample_synthetic_data
    lof = CPULocalOutlierFactor(k=20)
    lof.fit(X)

    assert not np.isnan(lof.dist_matrix_).any(), "Distance matrix contains NaN"
    assert not np.isinf(lof.dist_matrix_).any(), "Distance matrix contains Inf"
    assert not np.isnan(lof.lrd_).any(), "LRD contains NaN"
    assert not np.isinf(lof.lrd_).any(), "LRD contains Inf"
    assert not np.isnan(lof.lof_scores_).any(), "LOF scores contain NaN"
    assert not np.isinf(lof.lof_scores_).any(), "LOF scores contain Inf"
    assert np.all(lof.lof_scores_ > 0), "All LOF scores must be strictly positive"


def test_neighbor_index_validity(sample_synthetic_data):
    X = sample_synthetic_data
    N = X.shape[0]
    k = 20
    lof = CPULocalOutlierFactor(k=k)
    lof.fit(X)

    # 1. Neighbor indices must be strictly within [0, N-1]
    assert np.all(lof.knn_indices_ >= 0)
    assert np.all(lof.knn_indices_ < N)

    # 2. No point should be its own neighbor (no self-loops)
    for i in range(N):
        assert i not in lof.knn_indices_[i], f"Sample {i} includes itself as a nearest neighbor"

    # 3. Neighbor distances must be monotonically non-decreasing along each row
    for i in range(N):
        dists = lof.knn_distances_[i]
        assert np.all(np.diff(dists) >= -1e-6), f"Neighbor distances for sample {i} are not sorted"


def test_reproducibility(sample_synthetic_data):
    X = sample_synthetic_data
    lof1 = CPULocalOutlierFactor(k=20).fit(X)
    lof2 = CPULocalOutlierFactor(k=20).fit(X)

    np.testing.assert_array_equal(lof1.lof_scores_, lof2.lof_scores_)
    np.testing.assert_array_equal(lof1.knn_indices_, lof2.knn_indices_)
    np.testing.assert_array_equal(lof1.predictions_, lof2.predictions_)


def test_numerical_correctness_against_sklearn(sample_synthetic_data):
    """
    Validate that our first-principles implementation matches scikit-learn's
    LocalOutlierFactor reference values.
    Note: scikit-learn stores negative LOF scores (-LOF) in negative_outlier_factor_.
    """
    X = sample_synthetic_data
    k = 20

    # 1. First-principles CPU LOF
    our_lof = CPULocalOutlierFactor(k=k)
    our_lof.fit(X)
    our_scores = our_lof.lof_scores_

    # 2. Reference Scikit-Learn LOF
    sklearn_lof = SklearnLOF(n_neighbors=k, metric="euclidean", algorithm="brute")
    sklearn_lof.fit(X)
    sklearn_scores = -sklearn_lof.negative_outlier_factor_

    # Compare values (should match very closely)
    np.testing.assert_allclose(our_scores, sklearn_scores, rtol=1e-4, atol=1e-4)

    # Verify rank correlation
    correlation = np.corrcoef(our_scores, sklearn_scores)[0, 1]
    assert correlation > 0.9999, f"Rank correlation {correlation} is too low"


def test_numerical_correctness_on_cicids_subset(cicids_10k_subset):
    """Validate numerical equivalence against scikit-learn on actual CICIDS2017 features."""
    X = cicids_10k_subset
    k = 20

    our_lof = CPULocalOutlierFactor(k=k).fit(X)
    sklearn_lof = SklearnLOF(n_neighbors=k, metric="euclidean", algorithm="brute").fit(X)

    our_scores = our_lof.lof_scores_
    sklearn_scores = -sklearn_lof.negative_outlier_factor_

    # Float32 precision numerical equivalence within 1e-3 tolerance
    np.testing.assert_allclose(our_scores, sklearn_scores, rtol=1e-3, atol=1e-3)
    correlation = np.corrcoef(our_scores, sklearn_scores)[0, 1]
    assert correlation > 0.9999, f"Rank correlation {correlation} is too low"


@pytest.mark.parametrize("k", [5, 10, 25])
def test_configurable_k_parameter(sample_synthetic_data, k):
    X = sample_synthetic_data
    lof = CPULocalOutlierFactor(k=k).fit(X)
    assert lof.knn_indices_.shape == (X.shape[0], k)
    assert lof.reach_dists_.shape == (X.shape[0], k)
    assert not np.isnan(lof.lof_scores_).any()


def test_error_handling(sample_synthetic_data):
    X = sample_synthetic_data
    N = X.shape[0]

    # k <= 0
    with pytest.raises(ValueError, match="k must be at least 1"):
        CPULocalOutlierFactor(k=0)

    # k >= N
    with pytest.raises(ValueError, match="must be strictly less than"):
        CPULocalOutlierFactor(k=N + 10).fit(X)

    # invalid contamination
    with pytest.raises(ValueError, match="contamination must be between 0 and 1"):
        CPULocalOutlierFactor(k=20, contamination=1.5)

    # 1D array input
    with pytest.raises(ValueError, match="X must be a 2D array"):
        CPULocalOutlierFactor(k=20).fit(np.array([1.0, 2.0, 3.0]))
