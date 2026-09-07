@echo off
REM ============================================================
REM  build_knn_kernel.bat — Build the CUDA M3 k-NN Kernel
REM
REM  Project: CUDA LOF Cybersecurity
REM  Milestone: M3 - CUDA LOF Baseline (Step 3)
REM
REM  Target GPU: NVIDIA GeForce RTX 3050 (sm_86, Compute Capability 8.6)
REM  Compiler:   CUDA Toolkit 12.8 (nvcc) + MSVC v143 (VS 2022 v14.38-17.8)
REM
REM  Outputs:
REM    - bin\knn_kernel.dll (Shared library for Python ctypes)
REM    - bin\test_knn.exe   (Standalone C++ verification executable)
REM
REM  Usage:
REM    scripts\build_knn_kernel.bat          (build only)
REM    scripts\build_knn_kernel.bat run      (build and run C++ verification)
REM ============================================================

echo ============================================================
echo  Building CUDA k-NN Kernel (sm_86 / RTX 3050)
echo ============================================================

REM Navigate to project root
cd /d "%~dp0.."

REM Create output directory
if not exist "bin" mkdir bin

REM Compile DLL
echo.
echo [BUILD] nvcc -arch=sm_86 -O2 --shared -o bin\knn_kernel.dll src\cuda\kernels\knn_kernel.cu
echo.
nvcc -arch=sm_86 -O2 --shared -o bin\knn_kernel.dll src\cuda\kernels\knn_kernel.cu
if %ERRORLEVEL% neq 0 (
    echo [FAIL] Compilation of knn_kernel.dll failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
echo [PASS] Compiled bin\knn_kernel.dll

REM Compile Standalone Test Executable
echo.
echo [BUILD] nvcc -arch=sm_86 -O2 -o bin\test_knn.exe src\cuda\kernels\test_knn.cu src\cuda\kernels\knn_kernel.cu src\cuda\kernels\distance_kernel.cu
echo.
nvcc -arch=sm_86 -O2 -o bin\test_knn.exe src\cuda\kernels\test_knn.cu src\cuda\kernels\knn_kernel.cu src\cuda\kernels\distance_kernel.cu
if %ERRORLEVEL% neq 0 (
    echo [FAIL] Compilation of test_knn.exe failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
echo [PASS] Compiled bin\test_knn.exe

echo.
echo [SUCCESS] k-NN kernel build completed.
echo.

if /i "%1"=="run" (
    echo ============================================================
    echo  Running Standalone C++ k-NN Kernel Verification
    echo ============================================================
    echo.
    bin\test_knn.exe
    echo.
    echo Exit code: %ERRORLEVEL%
)
