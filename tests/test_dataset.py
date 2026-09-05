"""
Unit tests for CICIDS2017 Dataset Preparation & Preprocessing Pipeline
Milestone M1: Dataset Preparation

Verifies:
- Data integrity of full cleaned dataset and subsets
- Zero NaNs, zero Infs, correct float32 dtype
- Feature dimensionality (68 numerical features)
- Label array alignments and binary/multiclass consistency
- Stratified sampling fidelity and reproducibility
"""

import os
import json
import pytest
import numpy as np
import sys

# Ensure project root and src/python are in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.python.data.dataset import load_metadata, load_full_dataset, load_subset, get_project_root


@pytest.fixture(scope="module")
def metadata():
    return load_metadata()


def test_metadata_structure(metadata):
    assert metadata["dataset_name"] == "CICIDS2017"
    assert metadata["milestone"] == "M1"
    assert metadata["random_seed"] == 42
    assert "cleaning_summary" in metadata
    assert "feature_summary" in metadata
    assert "label_summary" in metadata
    assert "subsets" in metadata
    assert metadata["feature_summary"]["total_selected_features"] == 68
    assert len(metadata["feature_summary"]["feature_names"]) == 68


def test_full_dataset_integrity():
    X, y_binary, y_multiclass, y_raw = load_full_dataset()
    
    # Check shapes
    assert X.ndim == 2
    assert X.shape[1] == 68
    assert len(y_binary) == len(X)
    assert len(y_multiclass) == len(X)
    assert len(y_raw) == len(X)
    
    # Check data types
    assert X.dtype == np.float32
    assert y_binary.dtype == np.int8
    assert y_multiclass.dtype == np.int16
    
    # Check no NaN or Inf
    assert not np.isnan(X).any(), "Full dataset feature matrix contains NaN values!"
    assert not np.isinf(X).any(), "Full dataset feature matrix contains Inf values!"
    
    # Check standardization properties (mean ~ 0, std ~ 1)
    feature_means = np.mean(X, axis=0)
    feature_stds = np.std(X, axis=0)
    np.testing.assert_allclose(feature_means, np.zeros(68), atol=1e-2)
    np.testing.assert_allclose(feature_stds, np.ones(68), atol=1e-2)
    
    # Check binary label values
    unique_bin = set(np.unique(y_binary))
    assert unique_bin == {0, 1}
    
    # Check label alignment
    benign_mask = (y_raw == "BENIGN")
    assert np.all(y_binary[benign_mask] == 0)
    assert np.all(y_binary[~benign_mask] == 1)
    assert np.all(y_multiclass[benign_mask] == 0)


@pytest.mark.parametrize("subset_name,expected_size", [
    ("10k", 10_000),
    ("25k", 25_000),
    ("100k", 100_000),
    ("250k", 250_000),
    ("500k", 500_000),
    ("1m", 1_000_000),
])
def test_subset_integrity(subset_name, expected_size):
    X, y_bin, y_multi, y_raw, indices = load_subset(subset_name)
    
    # Shape & type checks
    assert X.shape == (expected_size, 68)
    assert len(y_bin) == expected_size
    assert len(y_multi) == expected_size
    assert len(y_raw) == expected_size
    assert len(indices) == expected_size
    
    assert X.dtype == np.float32
    assert not np.isnan(X).any()
    assert not np.isinf(X).any()
    
    # Indices are sorted and within bounds
    assert np.all(np.diff(indices) > 0), "Subset indices are not strictly increasing!"
    
    # Check anomaly ratio roughly matches full dataset (~16.89% anomalies)
    anomaly_ratio = np.mean(y_bin == 1)
    assert 0.14 <= anomaly_ratio <= 0.20, f"Subset {subset_name} anomaly ratio {anomaly_ratio:.4f} diverges from population!"
