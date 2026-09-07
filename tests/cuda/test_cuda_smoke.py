"""
test_cuda_smoke.py - CUDA Smoke Test Validation (Python wrapper)

Milestone: M3 - CUDA LOF Baseline (Step 1)

Validates that the CUDA smoke test executable:
1. Exists and was compiled successfully
2. Runs and exits with code 0
3. Reports the expected GPU and correct computation

This test is SKIPPED automatically if:
  - The smoke_test.exe binary does not exist (CUDA not compiled)
  - No CUDA-capable GPU is available

This ensures the existing M1/M2 Python test suite is never broken
by CUDA availability requirements.
"""

import os
import sys
import subprocess
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SMOKE_TEST_EXE = os.path.join(PROJECT_ROOT, "bin", "smoke_test.exe")


def cuda_available():
    """Check if nvidia-smi is available (proxy for CUDA GPU presence)."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# Skip the entire module if no CUDA binary or no GPU
pytestmark = pytest.mark.skipif(
    not os.path.isfile(SMOKE_TEST_EXE) or not cuda_available(),
    reason="CUDA smoke test binary not found or no CUDA GPU available"
)


class TestCudaSmokeTest:
    """Validate the compiled CUDA smoke test program."""

    def test_smoke_test_runs_successfully(self):
        """The smoke test executable should run and exit with code 0."""
        result = subprocess.run(
            [SMOKE_TEST_EXE],
            capture_output=True, text=True, timeout=30
        )
        print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}", file=sys.stderr)

        assert result.returncode == 0, (
            f"Smoke test exited with code {result.returncode}.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    def test_gpu_detected(self):
        """The output should mention an NVIDIA GPU."""
        result = subprocess.run(
            [SMOKE_TEST_EXE],
            capture_output=True, text=True, timeout=30
        )
        output = result.stdout.lower()
        assert "nvidia" in output or "geforce" in output or "rtx" in output, (
            f"GPU name not found in smoke test output:\n{result.stdout}"
        )

    def test_computation_correct(self):
        """The output should report all elements correct."""
        result = subprocess.run(
            [SMOKE_TEST_EXE],
            capture_output=True, text=True, timeout=30
        )
        assert "SMOKE TEST PASSED" in result.stdout, (
            f"Smoke test did not report PASSED:\n{result.stdout}"
        )

    def test_kernel_launched(self):
        """The output should confirm kernel launch."""
        result = subprocess.run(
            [SMOKE_TEST_EXE],
            capture_output=True, text=True, timeout=30
        )
        assert "Kernel launched successfully" in result.stdout, (
            f"Kernel launch not confirmed:\n{result.stdout}"
        )
