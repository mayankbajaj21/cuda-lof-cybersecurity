"""
Naive CPU Baseline Implementation of Local Outlier Factor (LOF)
Milestone M2: Naive CPU LOF Baseline

This module provides a transparent, first-principles CPU reference implementation
of the Local Outlier Factor (Breunig et al., 2000) algorithm for unsupervised
anomaly detection on cybersecurity network traffic.

Mathematical Formulation:
1. Pairwise Euclidean Distance:
   d(p, q) = ||p - q||_2 = sqrt(sum_{f=1}^D (p_f - q_f)^2)

2. k-Nearest Neighbors N_k(p) & k-distance(p):
   N_k(p) = k closest distinct points to p (excluding self)
   k-distance(p) = distance from p to its k-th nearest neighbor

3. Reachability Distance:
   reach-dist_k(p, o) = max(k-distance(o), d(p, o))

4. Local Reachability Density (LRD):
   lrd_k(p) = |N_k(p)| / sum_{o in N_k(p)} reach-dist_k(p, o)
            = 1.0 / ( (1/k) * sum_{o in N_k(p)} reach-dist_k(p, o) )

5. Local Outlier Factor (LOF):
   LOF_k(p) = ( sum_{o in N_k(p)} (lrd_k(o) / lrd_k(p)) ) / |N_k(p)|
            = ( sum_{o in N_k(p)} lrd_k(o) ) / ( k * lrd_k(p) )
"""

import os
import sys
import time
import json
import platform
import ctypes
import subprocess
import argparse
import numpy as np
import scipy
import pandas as pd
import sklearn
from typing import Tuple, Dict, Any, Optional

# Ensure dataset loader is accessible
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from src.python.data.dataset import load_subset, load_full_dataset
except ImportError:
    from dataset import load_subset, load_full_dataset


def get_system_environment() -> Dict[str, Any]:
    """Capture host CPU, memory, OS, Python, and library environment metadata."""
    env = {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "architecture": platform.machine(),
        },
        "python": {
            "version": platform.python_version(),
            "compiler": platform.python_compiler(),
            "implementation": platform.python_implementation(),
        },
        "libraries": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "cpu": {
            "model": platform.processor(),
            "physical_cores": os.cpu_count() or 1,
            "logical_processors": os.cpu_count() or 1,
        },
        "memory": {
            "total_ram_gb": "N/A"
        }
    }

    # Query detailed CPU info on Windows
    try:
        cpu_cmd = "wmic cpu get Name, NumberOfCores, NumberOfLogicalProcessors /value"
        output = subprocess.check_output(cpu_cmd, shell=True, text=True).strip()
        cpu_data = {}
        for line in output.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                cpu_data[k.strip()] = v.strip()
        if "Name" in cpu_data and cpu_data["Name"]:
            env["cpu"]["model"] = cpu_data["Name"]
        if "NumberOfCores" in cpu_data:
            env["cpu"]["physical_cores"] = int(cpu_data["NumberOfCores"])
        if "NumberOfLogicalProcessors" in cpu_data:
            env["cpu"]["logical_processors"] = int(cpu_data["NumberOfLogicalProcessors"])
    except Exception:
        pass

    # Query total physical RAM on Windows
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        total_ram_gb = round(stat.ullTotalPhys / (1024**3), 2)
        env["memory"] = {
            "total_ram_bytes": stat.ullTotalPhys,
            "total_ram_gb": total_ram_gb
        }
    except Exception:
        pass

    return env


class CPULocalOutlierFactor:
    """
    Transparent, first-principles CPU implementation of Local Outlier Factor (LOF).
    
    Attributes:
        k (int): Number of nearest neighbors (default: 20).
        eps (float): Small numerical epsilon to prevent zero division.
        contamination (float): Expected proportion of outliers in the dataset.
        
        # Computed attributes
        n_samples_ (int): Number of samples in fitted dataset.
        n_features_ (int): Dimensionality of feature vectors.
        dist_matrix_ (np.ndarray): Pairwise Euclidean distance matrix (N, N).
        knn_indices_ (np.ndarray): Indices of k-nearest neighbors for each sample (N, k).
        knn_distances_ (np.ndarray): Distances to k-nearest neighbors (N, k).
        k_distances_ (np.ndarray): Distance to the k-th nearest neighbor for each sample (N,).
        reach_dists_ (np.ndarray): Reachability distances to k neighbors for each sample (N, k).
        lrd_ (np.ndarray): Local reachability density for each sample (N,).
        lof_scores_ (np.ndarray): Local Outlier Factor scores (N,).
        threshold_ (float): LOF decision threshold based on contamination.
        predictions_ (np.ndarray): Binary anomaly predictions (1: anomaly, 0: inlier).
        timings_ (Dict[str, float]): Execution time breakdown in seconds.
    """

    def __init__(self, k: int = 20, eps: float = 1e-10, contamination: float = 0.1):
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        if not (0.0 < contamination < 1.0):
            raise ValueError(f"contamination must be between 0 and 1, got {contamination}")
        self.k = k
        self.eps = eps
        self.contamination = contamination

        # State placeholders
        self.n_samples_: int = 0
        self.n_features_: int = 0
        self.dist_matrix_: Optional[np.ndarray] = None
        self.knn_indices_: Optional[np.ndarray] = None
        self.knn_distances_: Optional[np.ndarray] = None
        self.k_distances_: Optional[np.ndarray] = None
        self.reach_dists_: Optional[np.ndarray] = None
        self.lrd_: Optional[np.ndarray] = None
        self.lof_scores_: Optional[np.ndarray] = None
        self.threshold_: Optional[float] = None
        self.predictions_: Optional[np.ndarray] = None
        self.timings_: Dict[str, float] = {}

    def compute_pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute full pairwise Euclidean distance matrix between all points.
        d(p, q) = sqrt( sum_{f=1}^D (p_f - q_f)^2 )
        
        Args:
            X: Input feature matrix of shape (N, D), dtype float32/float64.
            
        Returns:
            Pairwise distance matrix of shape (N, N).
        """
        # Naive first-principles Euclidean distance using quadratic expansion:
        # ||p - q||^2 = ||p||^2 + ||q||^2 - 2 * <p, q>
        # Ensured non-negative inside sqrt for numerical stability.
        dot_product = np.dot(X, X.T)
        sq_norms = np.diag(dot_product)
        # Broadcast: ||p||^2 - 2<p, q> + ||q||^2
        sq_dists = sq_norms[:, np.newaxis] - 2.0 * dot_product + sq_norms[np.newaxis, :]
        # Numerical floor at 0.0 to prevent negative sqrt
        sq_dists = np.maximum(sq_dists, 0.0)
        # Ensure self-distance is exactly zero
        np.fill_diagonal(sq_dists, 0.0)
        dist_matrix = np.sqrt(sq_dists)
        return dist_matrix

    def compute_k_neighbors(self, dist_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Identify the k-nearest neighbors for each point, excluding self.
        
        Args:
            dist_matrix: Pairwise distance matrix of shape (N, N).
            
        Returns:
            knn_indices: (N, k) array of neighbor indices.
            knn_distances: (N, k) array of neighbor distances.
            k_distances: (N,) array of distance to the k-th neighbor.
        """
        N = dist_matrix.shape[0]
        if self.k >= N:
            raise ValueError(f"k ({self.k}) must be strictly less than number of samples N ({N})")

        # Sort distances along each row
        # Column 0 is the point itself (dist = 0), columns 1 to k are the k nearest neighbors
        sorted_indices = np.argsort(dist_matrix, axis=1)
        knn_indices = sorted_indices[:, 1 : self.k + 1]

        # Extract corresponding sorted distances
        row_indices = np.arange(N)[:, np.newaxis]
        knn_distances = dist_matrix[row_indices, knn_indices]

        # The k-distance is the distance to the k-th nearest neighbor (column k-1)
        k_distances = knn_distances[:, -1].copy()

        return knn_indices, knn_distances, k_distances

    def compute_reachability_distances(
        self, dist_matrix: np.ndarray, knn_indices: np.ndarray, k_distances: np.ndarray
    ) -> np.ndarray:
        """
        Compute reachability distance from each point p to all its k neighbors o in N_k(p):
        reach-dist_k(p, o) = max(k-distance(o), d(p, o))
        
        Args:
            dist_matrix: Pairwise distance matrix of shape (N, N).
            knn_indices: Indices of k-nearest neighbors of shape (N, k).
            k_distances: k-distances of all points of shape (N,).
            
        Returns:
            reach_dists: Reachability distances of shape (N, k).
        """
        N, k = knn_indices.shape
        row_indices = np.arange(N)[:, np.newaxis]
        
        # Direct Euclidean distance d(p, o) to each neighbor
        direct_dists = dist_matrix[row_indices, knn_indices]
        
        # k-distance of each neighbor o: k_distances[knn_indices]
        neighbor_k_dists = k_distances[knn_indices]
        
        # reach-dist_k(p, o) = max(k-distance(o), d(p, o))
        reach_dists = np.maximum(neighbor_k_dists, direct_dists)
        return reach_dists

    def compute_lrd(self, reach_dists: np.ndarray) -> np.ndarray:
        """
        Compute Local Reachability Density (LRD) for each point:
        lrd_k(p) = |N_k(p)| / sum_{o in N_k(p)} reach-dist_k(p, o)
                 = 1.0 / ( mean_{o in N_k(p)} reach-dist_k(p, o) )
        
        Args:
            reach_dists: Reachability distances to k neighbors of shape (N, k).
            
        Returns:
            lrd: Local reachability density array of shape (N,).
        """
        # Average reachability distance across k neighbors
        avg_reach_dist = np.mean(reach_dists, axis=1)
        # Prevent division by zero with epsilon guard
        lrd = 1.0 / (avg_reach_dist + self.eps)
        return lrd

    def compute_lof_scores(self, lrd: np.ndarray, knn_indices: np.ndarray) -> np.ndarray:
        """
        Compute Local Outlier Factor (LOF) score for each point:
        LOF_k(p) = ( sum_{o in N_k(p)} (lrd(o) / lrd(p)) ) / |N_k(p)|
                 = mean_{o in N_k(p)} (lrd(o)) / lrd(p)
                 
        Args:
            lrd: Local reachability density array of shape (N,).
            knn_indices: Neighbor indices of shape (N, k).
            
        Returns:
            lof_scores: LOF score for each point of shape (N,).
        """
        # Neighbor LRD values: shape (N, k)
        neighbor_lrds = lrd[knn_indices]
        # Average LRD of neighbors for each point: shape (N,)
        avg_neighbor_lrd = np.mean(neighbor_lrds, axis=1)
        # LOF score = avg(neighbor_lrd) / lrd(p)
        lof_scores = avg_neighbor_lrd / (lrd + self.eps)
        return lof_scores

    def fit(self, X: np.ndarray) -> "CPULocalOutlierFactor":
        """
        Fit the LOF model on feature matrix X and compute all intermediate states and LOF scores.
        
        Args:
            X: Input numerical array of shape (N, D).
            
        Returns:
            self: The fitted model instance.
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X.shape}")
        
        self.n_samples_, self.n_features_ = X.shape
        total_start = time.perf_counter()

        # Step 1: Pairwise Distance Matrix
        t0 = time.perf_counter()
        self.dist_matrix_ = self.compute_pairwise_distances(X)
        t_dist = time.perf_counter() - t0

        # Step 2: k-Nearest Neighbors & k-distance
        t0 = time.perf_counter()
        self.knn_indices_, self.knn_distances_, self.k_distances_ = self.compute_k_neighbors(self.dist_matrix_)
        t_knn = time.perf_counter() - t0

        # Step 3: Reachability Distances
        t0 = time.perf_counter()
        self.reach_dists_ = self.compute_reachability_distances(
            self.dist_matrix_, self.knn_indices_, self.k_distances_
        )
        t_reach = time.perf_counter() - t0

        # Step 4: Local Reachability Density (LRD)
        t0 = time.perf_counter()
        self.lrd_ = self.compute_lrd(self.reach_dists_)
        t_lrd = time.perf_counter() - t0

        # Step 5: LOF Scores
        t0 = time.perf_counter()
        self.lof_scores_ = self.compute_lof_scores(self.lrd_, self.knn_indices_)
        t_lof = time.perf_counter() - t0

        total_compute = time.perf_counter() - total_start

        # Step 6: Decision Threshold based on contamination
        self.threshold_ = float(np.percentile(self.lof_scores_, 100.0 * (1.0 - self.contamination)))
        self.predictions_ = np.where(self.lof_scores_ >= self.threshold_, 1, 0).astype(np.int8)

        # Store fine-grained timings
        self.timings_ = {
            "distance_matrix_seconds": round(t_dist, 5),
            "knn_search_seconds": round(t_knn, 5),
            "reachability_distance_seconds": round(t_reach, 5),
            "lrd_computation_seconds": round(t_lrd, 5),
            "lof_scoring_seconds": round(t_lof, 5),
            "total_lof_compute_seconds": round(total_compute, 5),
        }

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit model and return binary predictions (1: anomaly, 0: inlier).
        """
        self.fit(X)
        return self.predictions_

    def get_summary(self) -> Dict[str, Any]:
        """
        Return structured dictionary summarizing the execution and score statistics.
        """
        if self.lof_scores_ is None:
            raise RuntimeError("Model has not been fitted yet.")

        scores = self.lof_scores_
        return {
            "n_samples": self.n_samples_,
            "n_features": self.n_features_,
            "k": self.k,
            "contamination": self.contamination,
            "threshold": round(float(self.threshold_), 5),
            "score_statistics": {
                "min": round(float(np.min(scores)), 5),
                "max": round(float(np.max(scores)), 5),
                "mean": round(float(np.mean(scores)), 5),
                "std": round(float(np.std(scores)), 5),
                "median": round(float(np.median(scores)), 5),
                "p25": round(float(np.percentile(scores, 25)), 5),
                "p75": round(float(np.percentile(scores, 75)), 5),
                "p90": round(float(np.percentile(scores, 90)), 5),
                "p95": round(float(np.percentile(scores, 95)), 5),
                "p99": round(float(np.percentile(scores, 99)), 5),
            },
            "predicted_anomalies": int(np.sum(self.predictions_ == 1)),
            "predicted_inliers": int(np.sum(self.predictions_ == 0)),
            "timings": self.timings_,
        }


def run_cpu_lof_experiment(
    subset_name: str = "10k",
    k: int = 20,
    contamination: float = 0.1,
    output_dir: str = "results/m2_cpu_baseline",
    save_scores: bool = True
) -> Dict[str, Any]:
    """
    Run an end-to-end CPU LOF experiment on a designated dataset subset and save results.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print(f"RUNNING NAIVE CPU LOF BASELINE EXPERIMENT (Subset: {subset_name}, k={k})")
    print("=" * 70)

    # 1. Load Data
    t_load_start = time.perf_counter()
    print(f"[*] Loading processed dataset subset '{subset_name}'...")
    X, y_bin, y_multi, y_raw, indices = load_subset(subset_name)
    t_load = time.perf_counter() - t_load_start
    print(f"    Loaded {X.shape[0]:,} samples with {X.shape[1]} features in {t_load:.4f}s")

    # 2. Fit CPU LOF
    print(f"[*] Executing CPU LOF computation (k={k}, contamination={contamination})...")
    lof = CPULocalOutlierFactor(k=k, contamination=contamination)
    lof.fit(X)

    # 3. Overall execution timings
    timings = dict(lof.timings_)
    timings["data_loading_seconds"] = round(t_load, 5)
    timings["total_execution_seconds"] = round(t_load + timings["total_lof_compute_seconds"], 5)

    summary = lof.get_summary()
    summary["subset_name"] = subset_name
    summary["timings"] = timings
    summary["environment"] = get_system_environment()

    # Ground truth correlation / comparison metrics
    ground_truth_anomalies = int(np.sum(y_bin == 1))
    ground_truth_benign = int(np.sum(y_bin == 0))
    summary["ground_truth"] = {
        "benign_count": ground_truth_benign,
        "anomaly_count": ground_truth_anomalies,
        "anomaly_ratio": round(ground_truth_anomalies / len(y_bin), 4)
    }

    # Print summary table
    print("\n" + "-" * 70)
    print("CPU LOF BASELINE EXECUTION SUMMARY")
    print("-" * 70)
    print(f"  Dataset Subset            : {subset_name} (N = {X.shape[0]:,}, D = {X.shape[1]})")
    print(f"  Neighborhood Parameter (k): {k}")
    print(f"  LOF Score Range           : [{summary['score_statistics']['min']:.4f}, {summary['score_statistics']['max']:.4f}]")
    print(f"  LOF Score Mean +/- Std    : {summary['score_statistics']['mean']:.4f} +/- {summary['score_statistics']['std']:.4f}")
    print(f"  LOF Decision Threshold    : {summary['threshold']:.4f} (at contamination={contamination})")
    print("\n  Execution Timings Breakdown:")
    print(f"    - Data Loading          : {timings['data_loading_seconds']:8.4f} s")
    print(f"    - Pairwise Distance Mat : {timings['distance_matrix_seconds']:8.4f} s")
    print(f"    - k-NN Search & Sort    : {timings['knn_search_seconds']:8.4f} s")
    print(f"    - Reachability Distance : {timings['reachability_distance_seconds']:8.4f} s")
    print(f"    - LRD Computation       : {timings['lrd_computation_seconds']:8.4f} s")
    print(f"    - LOF Scoring           : {timings['lof_scoring_seconds']:8.4f} s")
    print(f"    ---------------------------------------")
    print(f"    - Total LOF Compute     : {timings['total_lof_compute_seconds']:8.4f} s")
    print(f"    - Total End-to-End Time : {timings['total_execution_seconds']:8.4f} s")
    print("=" * 70 + "\n")

    # Save JSON summary
    out_json = os.path.join(output_dir, f"cpu_lof_{subset_name}_k{k}.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[*] Results manifest saved to: '{out_json}'")

    # Save LOF score array if requested
    if save_scores:
        out_scores = os.path.join(output_dir, f"lof_scores_{subset_name}_k{k}.npy")
        np.save(out_scores, lof.lof_scores_)
        print(f"[*] LOF scores array saved to: '{out_scores}'")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Naive CPU Local Outlier Factor baseline.")
    parser.add_argument("--subset", type=str, default="10k", help="Subset size/name to run (e.g., 10k, 25k)")
    parser.add_argument("--k", type=int, default=20, help="Number of nearest neighbors")
    parser.add_argument("--contamination", type=float, default=0.1, help="Contamination factor")
    parser.add_argument("--output-dir", type=str, default="results/m2_cpu_baseline", help="Directory to save results")
    args = parser.parse_args()

    run_cpu_lof_experiment(
        subset_name=args.subset,
        k=args.k,
        contamination=args.contamination,
        output_dir=args.output_dir
    )
