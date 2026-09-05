"""
CLI Runner for CPU Local Outlier Factor Baseline
Milestone M2: Naive CPU LOF Baseline

Usage:
    python scripts/run_cpu_baseline.py --subset 10k --k 20
    python scripts/run_cpu_baseline.py --subset 25k --k 20 --output-dir results/m2_cpu_baseline/benchmarks
"""

import os
import sys
import argparse

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.python.cpu.cpu_lof import run_cpu_lof_experiment


def main():
    parser = argparse.ArgumentParser(description="Execute Naive CPU LOF Baseline Experiment.")
    parser.add_argument("--subset", type=str, default="10k", help="Dataset subset name (10k, 25k, 100k, etc.)")
    parser.add_argument("--k", type=int, default=20, help="Number of nearest neighbors (default: 20)")
    parser.add_argument("--contamination", type=float, default=0.1, help="Contamination factor (default: 0.1)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "results", "m2_cpu_baseline", "benchmarks"),
        help="Output directory for benchmark artifacts"
    )
    args = parser.parse_args()

    run_cpu_lof_experiment(
        subset_name=args.subset,
        k=args.k,
        contamination=args.contamination,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
