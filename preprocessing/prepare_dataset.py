"""
CICIDS2017 Dataset Preparation and Preprocessing Pipeline
Milestone M1: Dataset Preparation

This script implements a reproducible end-to-end dataset preparation workflow:
1. Ingestion of raw CICIDS2017 CSV files from data/raw/
2. Header sanitation and deduplication
3. Cleansing of invalid records (NaN, +Inf, -Inf, duplicate flows)
4. Domain-driven feature selection (removal of identifiers, redundant duplicates, zero-variance columns)
5. Separation of ground-truth labels (raw strings, multi-class codes, binary anomaly flags)
6. Z-score standard scaling (StandardScaler) fitted across reference dataset
7. Generation of reproducible, stratified experimental subsets (10K, 25K, 100K, 250K, 500K, 1M, Full)
8. Export to high-efficiency binary arrays (.npz) and JSON metadata manifest.
"""

import os
import glob
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Any

# Excluded columns and domain rationale
EXCLUDED_COLUMNS_RATIONALE = {
    "Label": "Ground-truth target label; must be separated from input features to prevent data leakage.",
    "Destination Port": "Network port identifier; treating discrete port IDs as Euclidean metrics creates artificial distance biases.",
    "Fwd Header Length.1": "Duplicate column of 'Fwd Header Length' produced by CICFlowMeter generation bug.",
    "Bwd PSH Flags": "Zero-variance constant column (all values 0.0 across all samples).",
    "Bwd URG Flags": "Zero-variance constant column (all values 0.0 across all samples).",
    "Fwd Avg Bytes/Bulk": "Zero-variance constant column (all values 0.0 across all samples).",
    "Fwd Avg Packets/Bulk": "Zero-variance constant column (all values 0.0 across all samples).",
    "Fwd Avg Bulk Rate": "Zero-variance constant column (all values 0.0 across all samples).",
    "Bwd Avg Bytes/Bulk": "Zero-variance constant column (all values 0.0 across all samples).",
    "Bwd Avg Packets/Bulk": "Zero-variance constant column (all values 0.0 across all samples).",
    "Bwd Avg Bulk Rate": "Zero-variance constant column (all values 0.0 across all samples)."
}

DEFAULT_SUBSET_SIZES = [10_000, 25_000, 100_000, 250_000, 500_000, 1_000_000]
RANDOM_SEED = 42


def sanitize_label_string(s: str) -> str:
    """Sanitize label string by stripping whitespace and replacing corrupted characters."""
    cleaned = (
        str(s)
        .strip()
        .encode("ascii", "replace")
        .decode("ascii")
        .replace("?", "-")
    )
    # Standardize spaces around hyphens
    if "Web Attack" in cleaned:
        parts = [p.strip() for p in cleaned.split("-")]
        return " - ".join(parts)
    return cleaned


def load_and_clean_raw_csvs(raw_dir: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load all raw CSV files, standardize headers, and remove invalid rows and duplicates."""
    csv_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    print(f"[*] Found {len(csv_files)} raw CSV files in '{raw_dir}'")
    dfs = []
    file_stats = []

    for f in csv_files:
        filename = os.path.basename(f)
        print(f"    - Loading: {filename} ...")
        df_part = pd.read_csv(f, low_memory=False)
        df_part.columns = [c.strip() for c in df_part.columns]
        file_stats.append({
            "filename": filename,
            "raw_rows": len(df_part),
            "raw_cols": len(df_part.columns)
        })
        dfs.append(df_part)

    df_full = pd.concat(dfs, ignore_index=True)
    raw_total_rows, raw_total_cols = df_full.shape
    print(f"[*] Total raw concatenated shape: {raw_total_rows:,} rows x {raw_total_cols} columns")

    # Sanitize label column
    label_col = "Label"
    if label_col not in df_full.columns:
        raise KeyError(f"Label column '{label_col}' not found in dataset columns.")
    df_full[label_col] = df_full[label_col].apply(sanitize_label_string)

    # Detect invalid (NaN / Inf) rows
    numeric_cols = df_full.select_dtypes(include=[np.number]).columns
    nan_mask = df_full.isna().any(axis=1)
    inf_mask = np.isinf(df_full[numeric_cols]).any(axis=1)
    invalid_mask = nan_mask | inf_mask

    nan_count = int(nan_mask.sum())
    inf_count = int(inf_mask.sum())
    invalid_count = int(invalid_mask.sum())
    print(f"[*] Identified {nan_count:,} rows with NaNs and {inf_count:,} rows with Infs (Total invalid: {invalid_count:,})")

    # Filter out invalid rows
    df_valid = df_full[~invalid_mask].copy()

    # Deduplication
    dup_count = int(df_valid.duplicated().sum())
    df_clean = df_valid.drop_duplicates().reset_index(drop=True)
    clean_rows = len(df_clean)
    print(f"[*] Removed {dup_count:,} duplicate rows. Cleaned dataset size: {clean_rows:,} rows")

    cleaning_metadata = {
        "raw_total_rows": raw_total_rows,
        "raw_total_cols": raw_total_cols,
        "nan_rows_removed": nan_count,
        "inf_rows_removed": inf_count,
        "total_invalid_rows_removed": invalid_count,
        "duplicate_rows_removed": dup_count,
        "cleaned_rows": clean_rows,
        "file_stats": file_stats
    }

    return df_clean, cleaning_metadata


def process_features_and_labels(
    df_clean: pd.DataFrame
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray, np.ndarray, Dict[str, Any], StandardScaler]:
    """Extract features, fit StandardScaler, and generate label arrays and mappings."""
    excluded_cols = list(EXCLUDED_COLUMNS_RATIONALE.keys())
    feature_names = [c for c in df_clean.columns if c not in excluded_cols]

    print(f"[*] Selected {len(feature_names)} numerical features (Excluded {len(excluded_cols)} columns)")

    # Extract numerical matrix as float32
    X_raw = df_clean[feature_names].values.astype(np.float32)

    # Standard Scaling
    print("[*] Fitting StandardScaler on cleaned dataset...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

    # Labels processing
    raw_labels = df_clean["Label"].values.astype(str)
    unique_labels = sorted(list(set(raw_labels)))
    # Ensure BENIGN is index 0
    if "BENIGN" in unique_labels:
        unique_labels.remove("BENIGN")
        unique_labels = ["BENIGN"] + unique_labels

    label_to_id = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    id_to_label = {idx: lbl for idx, lbl in enumerate(unique_labels)}

    y_multiclass = np.array([label_to_id[lbl] for lbl in raw_labels], dtype=np.int16)
    y_binary = np.array([0 if lbl == "BENIGN" else 1 for lbl in raw_labels], dtype=np.int8)

    # Class distribution statistics
    label_distribution = {}
    for lbl in unique_labels:
        count = int(np.sum(raw_labels == lbl))
        pct = float((count / len(raw_labels)) * 100)
        label_distribution[lbl] = {
            "class_id": label_to_id[lbl],
            "is_anomaly": 0 if lbl == "BENIGN" else 1,
            "count": count,
            "percentage": round(pct, 4)
        }

    label_metadata = {
        "unique_classes": unique_labels,
        "class_to_id": label_to_id,
        "id_to_class": id_to_label,
        "distribution": label_distribution,
        "total_benign": int(np.sum(y_binary == 0)),
        "total_anomalies": int(np.sum(y_binary == 1)),
        "anomaly_ratio": float(np.sum(y_binary == 1) / len(y_binary))
    }

    return X_scaled, feature_names, raw_labels, y_multiclass, y_binary, label_metadata, scaler


def create_stratified_subsets(
    X: np.ndarray,
    y_multiclass: np.ndarray,
    y_binary: np.ndarray,
    raw_labels: np.ndarray,
    subset_sizes: List[int],
    output_dir: str,
    random_seed: int = RANDOM_SEED
) -> Dict[str, Any]:
    """Generate reproducible stratified subsets for controlled scaling experiments."""
    total_samples = len(y_multiclass)
    subsets_dir = os.path.join(output_dir, "subsets")
    os.makedirs(subsets_dir, exist_ok=True)

    rng = np.random.default_rng(random_seed)
    subsets_metadata = {}

    for size in subset_sizes:
        if size > total_samples:
            print(f"[-] Requested subset size {size:,} exceeds total samples {total_samples:,}; skipping.")
            continue

        print(f"[*] Generating reproducible stratified subset of size: {size:,} (seed={random_seed})...")

        # Stratified sampling proportional to multiclass distribution
        classes, class_counts = np.unique(y_multiclass, return_counts=True)
        selected_indices = []

        for cls, count in zip(classes, class_counts):
            cls_indices = np.where(y_multiclass == cls)[0]
            # Proportionate allocation with minimum 1 sample if available
            cls_target = max(1, int(round((count / total_samples) * size)))
            if cls_target > count:
                cls_target = count
            sampled = rng.choice(cls_indices, size=cls_target, replace=False)
            selected_indices.extend(sampled)

        selected_indices = np.array(selected_indices)

        # Adjust size if rounding caused slight mismatch
        if len(selected_indices) > size:
            selected_indices = rng.choice(selected_indices, size=size, replace=False)
        elif len(selected_indices) < size:
            diff = size - len(selected_indices)
            remaining = np.setdiff1d(np.arange(total_samples), selected_indices)
            extra = rng.choice(remaining, size=diff, replace=False)
            selected_indices = np.concatenate([selected_indices, extra])

        # Sort indices for deterministic ordering
        selected_indices = np.sort(selected_indices)

        # Subset arrays
        X_sub = X[selected_indices]
        y_sub_multi = y_multiclass[selected_indices]
        y_sub_bin = y_binary[selected_indices]
        y_sub_raw = raw_labels[selected_indices]

        # Save subset
        subset_filename = f"subset_{size // 1000}k.npz" if size < 1_000_000 else f"subset_{size // 1_000_000}m.npz"
        subset_path = os.path.join(subsets_dir, subset_filename)
        np.savez_compressed(
            subset_path,
            X=X_sub,
            y_binary=y_sub_bin,
            y_multiclass=y_sub_multi,
            y_raw=y_sub_raw,
            indices=selected_indices
        )

        sub_benign = int(np.sum(y_sub_bin == 0))
        sub_anom = int(np.sum(y_sub_bin == 1))
        subsets_metadata[subset_filename] = {
            "size": size,
            "filename": subset_filename,
            "filepath": os.path.relpath(subset_path, start="."),
            "benign_count": sub_benign,
            "anomaly_count": sub_anom,
            "anomaly_ratio": round(sub_anom / size, 4),
            "file_size_mb": round(os.path.getsize(subset_path) / (1024 * 1024), 2)
        }
        print(f"    Saved: {subset_path} ({subsets_metadata[subset_filename]['file_size_mb']} MB)")

    return subsets_metadata


def run_pipeline(
    raw_dir: str = "data/raw",
    output_dir: str = "data/processed",
    subset_sizes: List[int] = DEFAULT_SUBSET_SIZES,
    random_seed: int = RANDOM_SEED
) -> Dict[str, Any]:
    """Execute complete dataset preparation pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 70)
    print("CICIDS2017 DATASET PREPARATION PIPELINE (MILESTONE M1)")
    print("=" * 70)

    # 1. Ingestion & Cleaning
    df_clean, cleaning_meta = load_and_clean_raw_csvs(raw_dir)

    # 2. Feature Selection, Scaling & Label Extraction
    X_scaled, feature_names, raw_labels, y_multiclass, y_binary, label_meta, scaler = process_features_and_labels(df_clean)

    # 3. Save Full Cleaned Dataset
    full_output_path = os.path.join(output_dir, "cicids2017_clean_full.npz")
    print(f"[*] Saving full processed dataset to '{full_output_path}'...")
    np.savez_compressed(
        full_output_path,
        X=X_scaled,
        y_binary=y_binary,
        y_multiclass=y_multiclass,
        y_raw=raw_labels
    )
    full_file_size_mb = round(os.path.getsize(full_output_path) / (1024 * 1024), 2)
    print(f"[*] Full dataset compressed size: {full_file_size_mb} MB")

    # 4. Generate Stratified Subsets
    subsets_meta = create_stratified_subsets(
        X_scaled, y_multiclass, y_binary, raw_labels, subset_sizes, output_dir, random_seed
    )

    # 5. Export Comprehensive Metadata Manifest
    manifest = {
        "dataset_name": "CICIDS2017",
        "milestone": "M1",
        "random_seed": random_seed,
        "cleaning_summary": cleaning_meta,
        "feature_summary": {
            "total_raw_columns": cleaning_meta["raw_total_cols"],
            "total_selected_features": len(feature_names),
            "total_excluded_columns": len(EXCLUDED_COLUMNS_RATIONALE),
            "feature_names": feature_names,
            "excluded_columns": EXCLUDED_COLUMNS_RATIONALE,
            "data_type": "float32",
            "scaler_type": "StandardScaler (z-score normalization)",
            "scaler_mean": [float(m) for m in scaler.mean_],
            "scaler_std": [float(s) for s in scaler.scale_]
        },
        "label_summary": label_meta,
        "full_dataset_file": {
            "filename": "cicids2017_clean_full.npz",
            "filepath": os.path.relpath(full_output_path, start="."),
            "total_samples": len(X_scaled),
            "feature_dim": X_scaled.shape[1],
            "file_size_mb": full_file_size_mb
        },
        "subsets": subsets_meta
    }

    manifest_path = os.path.join(output_dir, "dataset_metadata.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[*] Preprocessing metadata written to '{manifest_path}'")

    print("\n" + "=" * 70)
    print("DATASET PREPARATION COMPLETED SUCCESSFULLY")
    print(f"  - Total Cleaned Samples: {len(X_scaled):,}")
    print(f"  - Numerical Features: {X_scaled.shape[1]}")
    print(f"  - Benign Samples: {label_meta['total_benign']:,} ({label_meta['distribution']['BENIGN']['percentage']}%)")
    print(f"  - Anomaly Samples: {label_meta['total_anomalies']:,} ({round(label_meta['anomaly_ratio']*100, 2)}%)")
    print(f"  - Subsets Created: {list(subsets_meta.keys())}")
    print("=" * 70 + "\n")

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and preprocess CICIDS2017 dataset.")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw CSV files")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Path for processed outputs")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for subset generation")
    parser.add_argument(
        "--subsets",
        type=int,
        nargs="+",
        default=DEFAULT_SUBSET_SIZES,
        help="List of subset sample counts to generate"
    )
    args = parser.parse_args()

    run_pipeline(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        subset_sizes=args.subsets,
        random_seed=args.seed
    )
