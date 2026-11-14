@echo off
REM ============================================================
REM  build_cuda_lof_optimized.bat — Build Optimized CUDA LOF Pipeline
REM
REM  Project: CUDA LOF Cybersecurity
REM  Milestone: M5 - Profiling-Driven CUDA Optimization
REM
REM  Target GPU: set via ARCH env var (default sm_86); override
REM              per-machine, e.g. "set ARCH=sm_89 && scripts\build_cuda_lof_optimized.bat"
REM  Compiler:   CUDA Toolkit (nvcc) + matching MSVC host compiler
REM
REM  Keeps the naive baseline (bin\cuda_lof.dll) completely untouched -
REM  this produces a SEPARATE binary for before/after comparison.
REM
REM  Outputs:
REM    - bin\cuda_lof_optimized.dll
REM
REM  Usage:
REM    scripts\build_cuda_lof_optimized.bat
REM ============================================================

if "%ARCH%"=="" set "ARCH=sm_86"

echo ============================================================
echo  Building Optimized CUDA LOF Pipeline (%ARCH%)
echo ============================================================

cd /d "%~dp0.."

if not exist "bin" mkdir bin

set "SOURCES=src\cuda\kernels_optimized\distance_kernel_optimized.cu src\cuda\kernels_optimized\knn_kernel_optimized.cu src\cuda\kernels\reachability_kernel.cu src\cuda\kernels\lrd_kernel.cu src\cuda\kernels\lof_kernel.cu src\cuda\lof\cuda_lof_pipeline_optimized.cu"

echo.
echo [BUILD] Compiling bin\cuda_lof_optimized.dll...
echo.
nvcc -arch=%ARCH% -O2 --shared -o bin\cuda_lof_optimized.dll %SOURCES%
if %ERRORLEVEL% neq 0 (
    echo [FAIL] Compilation of cuda_lof_optimized.dll failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
echo [PASS] Compiled bin\cuda_lof_optimized.dll

echo.
echo [SUCCESS] Optimized CUDA LOF build completed.
echo.
