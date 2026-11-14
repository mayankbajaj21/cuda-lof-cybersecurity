@echo off
REM ============================================================
REM  build_cuda_lof.bat — Build Complete CUDA LOF Baseline
REM
REM  Project: CUDA LOF Cybersecurity
REM  Milestone: M3 - CUDA LOF Baseline (Step 4)
REM
REM  Target GPU: set via ARCH env var (default sm_86 / RTX 3050); override
REM              per-machine, e.g. "set ARCH=sm_89 && scripts\build_cuda_lof.bat"
REM  Compiler:   CUDA Toolkit (nvcc) + matching MSVC host compiler
REM
REM  Outputs:
REM    - bin\cuda_lof.dll     (Shared library for Python ctypes)
REM    - bin\test_cuda_lof.exe(Standalone C++ verification executable)
REM
REM  Usage:
REM    scripts\build_cuda_lof.bat          (build only)
REM    scripts\build_cuda_lof.bat run      (build and run C++ verification)
REM ============================================================

if "%ARCH%"=="" set "ARCH=sm_86"

echo ============================================================
echo  Building Complete CUDA LOF Baseline (%ARCH%)
echo ============================================================

REM Navigate to project root
cd /d "%~dp0.."

REM Create output directory
if not exist "bin" mkdir bin

REM Source files
set "SOURCES=src\cuda\kernels\distance_kernel.cu src\cuda\kernels\knn_kernel.cu src\cuda\kernels\reachability_kernel.cu src\cuda\kernels\lrd_kernel.cu src\cuda\kernels\lof_kernel.cu src\cuda\lof\cuda_lof_pipeline.cu"

REM Compile DLL
echo.
echo [BUILD] Compiling bin\cuda_lof.dll...
echo.
nvcc -arch=%ARCH% -O2 --shared -o bin\cuda_lof.dll %SOURCES%
if %ERRORLEVEL% neq 0 (
    echo [FAIL] Compilation of cuda_lof.dll failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
echo [PASS] Compiled bin\cuda_lof.dll

REM Compile Standalone Test Executable
echo.
echo [BUILD] Compiling bin\test_cuda_lof.exe...
echo.
nvcc -arch=%ARCH% -O2 -o bin\test_cuda_lof.exe src\cuda\lof\test_cuda_lof.cu %SOURCES%
if %ERRORLEVEL% neq 0 (
    echo [FAIL] Compilation of test_cuda_lof.exe failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
echo [PASS] Compiled bin\test_cuda_lof.exe

echo.
echo [SUCCESS] Complete CUDA LOF build completed.
echo.

if /i "%1"=="run" (
    echo ============================================================
    echo  Running Standalone C++ CUDA LOF Verification
    echo ============================================================
    echo.
    bin\test_cuda_lof.exe
    echo.
    echo Exit code: %ERRORLEVEL%
)
