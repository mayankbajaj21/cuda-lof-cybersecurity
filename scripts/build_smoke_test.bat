@echo off
REM ============================================================
REM  build_smoke_test.bat — Build the CUDA M3 smoke test
REM
REM  Project: CUDA LOF Cybersecurity
REM  Milestone: M3 - CUDA LOF Baseline (Step 1)
REM
REM  Target GPU: NVIDIA GeForce RTX 3050 (sm_86, Compute Capability 8.6)
REM  Compiler:   CUDA Toolkit 12.8 (nvcc) + MSVC v143 (VS 2022 v14.38-17.8)
REM
REM  Prerequisites:
REM    Run from an x64 Visual Studio Developer Command Prompt configured
REM    for the v143 toolset, e.g.:
REM      "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.38
REM
REM  Usage:
REM    scripts\build_smoke_test.bat          (build only)
REM    scripts\build_smoke_test.bat run      (build and run)
REM ============================================================

echo ============================================================
echo  Building CUDA Smoke Test (sm_86 / RTX 3050)
echo ============================================================

REM Navigate to project root
cd /d "%~dp0.."

REM Create output directory
if not exist "bin" mkdir bin

REM Compile with nvcc targeting sm_86
echo.
echo [BUILD] nvcc -arch=sm_86 -O2 -o bin\smoke_test.exe src\cuda\common\smoke_test.cu
echo.

nvcc -arch=sm_86 -O2 -o bin\smoke_test.exe src\cuda\common\smoke_test.cu

if %ERRORLEVEL% neq 0 (
    echo.
    echo [FAIL] Compilation failed with error code %ERRORLEVEL%
    echo        Ensure MSVC v143 is in PATH:
    echo        call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.38
    exit /b %ERRORLEVEL%
)

echo.
echo [PASS] Compilation succeeded: bin\smoke_test.exe
echo.

REM Optionally run the smoke test
if /i "%1"=="run" (
    echo ============================================================
    echo  Running CUDA Smoke Test
    echo ============================================================
    echo.
    bin\smoke_test.exe
    echo.
    echo Exit code: %ERRORLEVEL%
)
