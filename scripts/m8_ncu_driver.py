"""Runs one LOF pipeline variant once on the 10k subset; launched under Nsight Compute
by scripts/run_m8_advanced_optimizations.py. Usage: python scripts/m8_ncu_driver.py <variant>"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.python.cuda.lof_variants import compute_lof_variant  # noqa: E402
from src.python.data.dataset import load_subset  # noqa: E402

if __name__ == "__main__":
    variant = int(sys.argv[1])
    X = load_subset("10k")[0]
    compute_lof_variant(X, 20, variant)
