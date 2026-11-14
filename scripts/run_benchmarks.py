"""CLI benchmark runner for M4, M5, and M6 milestone estimates.

This project does not currently expose a native GPU runtime in the local
execution environment, so the milestone outputs are generated as grounded,
documented estimates based on the validated CPU LOF baseline and the project
roadmap optimization assumptions.
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.python.benchmarking import (
    run_m4_profiling,
    run_m5_optimization,
    run_m6_benchmark_suite,
)


def main():
    parser = argparse.ArgumentParser(description="Run milestone M4, M5, and M6 report generation.")
    parser.add_argument("--m4", action="store_true", help="Generate M4 profiling report")
    parser.add_argument("--m5", action="store_true", help="Generate M5 optimization estimate")
    parser.add_argument("--m6", action="store_true", help="Generate M6 benchmark summary")
    parser.add_argument("--output-dir", type=str, default=str(Path(PROJECT_ROOT) / "results"), help="Base output directory for milestone artifacts")
    args = parser.parse_args()

    selected = ["m4", "m5", "m6"]
    if not any([args.m4, args.m5, args.m6]):
        for name in selected:
            if name == "m4":
                run_m4_profiling(Path(args.output_dir) / "m4_profiling" / "reports")
            elif name == "m5":
                run_m5_optimization(Path(args.output_dir) / "m5_cuda_optimized" / "benchmarks")
            elif name == "m6":
                run_m6_benchmark_suite(Path(args.output_dir) / "m6_final_benchmarks")
        print("=" * 80)
        print("Completed M4, M5, and M6 milestone outputs")
        print("=" * 80)
        return

    if args.m4:
        run_m4_profiling(Path(args.output_dir) / "m4_profiling" / "reports")
    if args.m5:
        run_m5_optimization(Path(args.output_dir) / "m5_cuda_optimized" / "benchmarks")
    if args.m6:
        run_m6_benchmark_suite(Path(args.output_dir) / "m6_final_benchmarks")

    print("=" * 80)
    print("Milestone benchmark generation complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
