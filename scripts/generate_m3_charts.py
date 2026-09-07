"""
generate_m3_charts.py — CLI Runner to generate official Milestone 3 charts and tables
Usage:
    python scripts/generate_m3_charts.py
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.python.visualization.m3_visualizations import generate_all_charts


def main():
    benchmarks_dir = os.path.join(PROJECT_ROOT, "results", "m3_cuda_baseline", "benchmarks")
    charts_dir = os.path.join(PROJECT_ROOT, "results", "m3_cuda_baseline", "charts")
    tables_dir = os.path.join(PROJECT_ROOT, "results", "m3_cuda_baseline", "tables")

    generate_all_charts(
        benchmarks_dir=benchmarks_dir,
        charts_dir=charts_dir,
        tables_dir=tables_dir
    )


if __name__ == "__main__":
    main()
