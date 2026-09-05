"""
Dataset Loading Utilities for CICIDS2017
Milestone M1: Dataset Preparation

Provides helper utilities for loading preprocessed numerical feature matrices,
labels, and experimental subsets generated during M1.
"""

import os
import json
import numpy as np
from typing import Tuple, Dict, Any, Optional


def get_project_root() -> str:
    """Return absolute path to repository root."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "..", ".."))


def load_metadata(processed_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load dataset metadata JSON manifest."""
    if processed_dir is None:
        processed_dir = os.path.join(get_project_root(), "data", "processed")
    metadata_path = os.path.join(processed_dir, "dataset_metadata.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at '{metadata_path}'. Run preprocessing/prepare_dataset.py first.")
    with open(metadata_path, "r") as f:
        return json.load(f)


def load_full_dataset(
    processed_dir: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the complete cleaned and scaled CICIDS2017 dataset.

    Returns:
        X (np.ndarray): Scaled numerical feature matrix (N, 68), float32
        y_binary (np.ndarray): Binary labels (0: BENIGN, 1: Anomaly), int8
        y_multiclass (np.ndarray): Multi-class integer encoded labels, int16
        y_raw (np.ndarray): Original string attack class names
    """
    if processed_dir is None:
        processed_dir = os.path.join(get_project_root(), "data", "processed")
    data_path = os.path.join(processed_dir, "cicids2017_clean_full.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Processed dataset not found at '{data_path}'. Run preprocessing/prepare_dataset.py first.")
    
    data = np.load(data_path, allow_pickle=True)
    return data["X"], data["y_binary"], data["y_multiclass"], data["y_raw"]


def load_subset(
    subset_name_or_size: str,
    processed_dir: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a pre-generated stratified experimental subset.

    Args:
        subset_name_or_size: e.g. "10k", "25k", "100k", "250k", "500k", "1m", or "subset_100k.npz"
        processed_dir: Optional path to processed data directory

    Returns:
        X (np.ndarray): Scaled feature matrix (N, 68), float32
        y_binary (np.ndarray): Binary labels, int8
        y_multiclass (np.ndarray): Multi-class encoded labels, int16
        y_raw (np.ndarray): Original string labels
        indices (np.ndarray): Sample indices relative to the full clean dataset
    """
    if processed_dir is None:
        processed_dir = os.path.join(get_project_root(), "data", "processed")
    
    filename = subset_name_or_size
    if not filename.endswith(".npz"):
        if not filename.startswith("subset_"):
            filename = f"subset_{filename}.npz"
        else:
            filename = f"{filename}.npz"
            
    subset_path = os.path.join(processed_dir, "subsets", filename)
    if not os.path.exists(subset_path):
        raise FileNotFoundError(f"Subset file not found at '{subset_path}'")
    
    data = np.load(subset_path, allow_pickle=True)
    return data["X"], data["y_binary"], data["y_multiclass"], data["y_raw"], data["indices"]
