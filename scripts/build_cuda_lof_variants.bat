@echo off
REM ============================================================
REM  build_cuda_lof_variants.bat - Build the M8 V0..V5 LOF pipeline
REM
REM  Project: CUDA LOF Cybersecurity
REM  Milestone: M8 - Advanced Optimizations
REM
REM  Target GPU: set via ARCH env var (default sm_86); override
REM              per-machine, e.g. "set ARCH=sm_89 && scripts\build_cuda_lof_variants.bat"
REM              cp.async (V3, V5) needs sm_80+; older targets compile a
REM              synchronous-load fallback.
REM
REM  Produces a SEPARATE binary; bin\cuda_lof.dll (M3) and
REM  bin\cuda_lof_optimized.dll (M5) are not touched.
REM
REM  Outputs:
REM    - bin\cuda_lof_variants.dll
REM ============================================================

if "%ARCH%"=="" set "ARCH=sm_86"

echo ============================================================
echo  Building CUDA LOF Variants V0..V5 (%ARCH%)
echo ============================================================

cd /d "%~dp0.."

if not exist "bin" mkdir bin

set "SOURCES=src\cuda\kernels\distance_kernel.cu src\cuda\kernels\knn_kernel.cu src\cuda\kernels\reachability_kernel.cu src\cuda\kernels\lrd_kernel.cu src\cuda\kernels\lof_kernel.cu src\cuda\kernels_optimized\distance_kernel_optimized.cu src\cuda\kernels_optimized\knn_kernel_optimized.cu src\cuda\kernels_variants\variant_kernels.cu src\cuda\lof\cuda_lof_pipeline_variants.cu"

echo.
echo [BUILD] Compiling bin\cuda_lof_variants.dll...
echo.
nvcc -arch=%ARCH% -O2 -lineinfo --shared -o bin\cuda_lof_variants.dll %SOURCES%
if %ERRORLEVEL% neq 0 (
    echo [FAIL] Compilation of cuda_lof_variants.dll failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
echo [PASS] Compiled bin\cuda_lof_variants.dll

echo.
echo [SUCCESS] CUDA LOF variants build completed.
echo.
