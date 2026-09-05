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
    # src/python/data -> src/python -> src -> root
    return os.path.abspath(os.path.join(current_dir, "..", "..", ".."))


def load_metadata(processed_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load dataset metadata JSON manifest."""
    if processed_dir is None:
        processed_dir = os.path.join(get_project_root(), "data", "processed")
    
    candidates = [
        os.path.join(processed_dir, "dataset_metadata.json"),
        os.path.join(processed_dir, "full", "dataset_metadata.json")
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)

    raise FileNotFoundError(f"Metadata file not found in '{processed_dir}'. Run preprocessing/prepare_dataset.py first.")


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
    
    candidates = [
        os.path.join(processed_dir, "full", "cicids2017_clean_full.npz"),
        os.path.join(processed_dir, "cicids2017_clean_full.npz")
    ]
    for path in candidates:
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            return data["X"], data["y_binary"], data["y_multiclass"], data["y_raw"]

    raise FileNotFoundError(f"Processed dataset not found in '{processed_dir}'. Run preprocessing/prepare_dataset.py first.")


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
    
    clean_name = subset_name_or_size.lower().strip()
    if clean_name.startswith("subset_"):
        clean_name = clean_name.replace("subset_", "")
    if clean_name.endswith(".npz"):
        clean_name = clean_name.replace(".npz", "")
        
    filename = f"subset_{clean_name}.npz"
    folder_name = clean_name

    candidates = [
        os.path.join(processed_dir, "subsets", folder_name, filename),
        os.path.join(processed_dir, "subsets", filename),
        os.path.join(processed_dir, filename)
    ]

    for path in candidates:
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            return data["X"], data["y_binary"], data["y_multiclass"], data["y_raw"], data["indices"]

    raise FileNotFoundError(
        f"Subset '{subset_name_or_size}' not found in '{os.path.join(processed_dir, 'subsets')}'. Checked: {candidates}"
    )
