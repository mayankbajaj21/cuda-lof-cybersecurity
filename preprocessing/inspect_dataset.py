"""
CICIDS2017 Dataset Inspection Script
Milestone M1: Dataset Preparation

Inspects all raw CSV files in data/raw/ to determine:
- File metadata, row/column counts
- Column name hygiene (whitespace, duplicates)
- Data types, missing (NaN) values, and infinite values
- Class / label distribution
- Potential identifier or zero-variance columns
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List


def inspect_csv_file(filepath: str) -> Dict[str, Any]:
    filename = os.path.basename(filepath)
    print(f"\n{'=' * 60}")
    print(f"Inspecting: {filename}")
    print(f"{'=' * 60}")

    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)

    # Read CSV
    # CICIDS2017 files may have leading/trailing whitespace in headers and mixed types
    df = pd.read_csv(filepath, low_memory=False)
    num_rows, num_cols = df.shape

    raw_columns = list(df.columns)
    stripped_columns = [c.strip() for c in raw_columns]

    # Find label column
    label_col = None
    for c in df.columns:
        if c.strip().lower() == "label":
            label_col = c
            break

    # Check label distribution
    label_counts = {}
    if label_col is not None:
        # Standardize strings, clean corrupted non-ASCII characters (e.g., \ufffd -> -)
        labels_clean = (
            df[label_col]
            .astype(str)
            .str.strip()
            .str.encode("ascii", "replace")
            .str.decode("ascii")
            .str.replace("?", "-", regex=False)
        )
        label_counts = labels_clean.value_counts().to_dict()

    # Numerical inspection
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Count NaNs
    nan_counts = df.isna().sum()
    total_nans = int(nan_counts.sum())
    cols_with_nans = nan_counts[nan_counts > 0].to_dict()

    # Count Infs in numeric columns
    inf_cols = {}
    for col in numeric_cols:
        inf_count = int(np.isinf(df[col]).sum())
        if inf_count > 0:
            inf_cols[col.strip()] = inf_count

    # Check duplicates
    duplicate_rows = int(df.duplicated().sum())

    print(f"File Size: {file_size_mb:.2f} MB")
    print(f"Rows: {num_rows:,} | Columns: {num_cols}")
    print(f"Numeric Columns: {len(numeric_cols)} | Non-Numeric: {len(non_numeric_cols)}")
    print(f"Total NaN values: {total_nans:,} across {len(cols_with_nans)} columns")
    print(f"Total Inf values: {sum(inf_cols.values()):,} across {len(inf_cols)} columns")
    print(f"Duplicate Rows: {duplicate_rows:,}")
    print(f"Labels: {label_counts}")

    return {
        "filename": filename,
        "size_mb": round(file_size_mb, 2),
        "num_rows": num_rows,
        "num_cols": num_cols,
        "raw_columns": raw_columns,
        "stripped_columns": stripped_columns,
        "label_column": label_col,
        "label_counts": label_counts,
        "numeric_cols_count": len(numeric_cols),
        "non_numeric_cols": non_numeric_cols,
        "total_nans": total_nans,
        "cols_with_nans": {k.strip(): int(v) for k, v in cols_with_nans.items()},
        "total_infs": sum(inf_cols.values()),
        "cols_with_infs": inf_cols,
        "duplicate_rows": duplicate_rows,
    }


def run_inspection(data_dir: str = "data/raw", output_report: str = "results/m1_inspection_report.json"):
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        print(f"No CSV files found in {data_dir}!")
        return

    os.makedirs(os.path.dirname(output_report), exist_ok=True)

    print(f"Found {len(csv_files)} CSV files in {data_dir}.")
    all_reports = []
    total_dataset_rows = 0
    global_label_counts: Dict[str, int] = {}
    all_column_sets: List[set] = []

    for f in csv_files:
        rep = inspect_csv_file(f)
        all_reports.append(rep)
        total_dataset_rows += rep["num_rows"]
        all_column_sets.append(set(rep["stripped_columns"]))
        for k, v in rep["label_counts"].items():
            global_label_counts[k] = global_label_counts.get(k, 0) + v

    # Check column consistency
    common_columns = set.intersection(*all_column_sets) if all_column_sets else set()
    union_columns = set.union(*all_column_sets) if all_column_sets else set()
    columns_differ = len(common_columns) != len(union_columns)

    summary = {
        "num_files": len(csv_files),
        "total_rows_across_all_files": total_dataset_rows,
        "common_column_count": len(common_columns),
        "total_unique_column_names": len(union_columns),
        "columns_differ_across_files": columns_differ,
        "global_label_counts": global_label_counts,
        "files": all_reports,
    }

    with open(output_report, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("DATASET AGGREGATE SUMMARY")
    print("=" * 60)
    print(f"Total Raw Files: {len(csv_files)}")
    print(f"Total Raw Rows: {total_dataset_rows:,}")
    print(f"Common Columns: {len(common_columns)}")
    print(f"All Columns Match Across Files: {not columns_differ}")
    print("\nGlobal Label Distribution:")
    for label, count in sorted(global_label_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_dataset_rows) * 100
        print(f"  {label:<35} : {count:>10,} ({pct:6.2f}%)")
    print(f"\nFull inspection report written to: {output_report}")


if __name__ == "__main__":
    run_inspection()
