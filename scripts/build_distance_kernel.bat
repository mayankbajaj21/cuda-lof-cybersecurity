@echo off
REM ============================================================
REM  build_distance_kernel.bat — Build the CUDA M3 Distance Kernel
REM
REM  Project: CUDA LOF Cybersecurity
REM  Milestone: M3 - CUDA LOF Baseline (Step 2)
REM
REM  Target GPU: NVIDIA GeForce RTX 3050 (sm_86, Compute Capability 8.6)
REM  Compiler:   CUDA Toolkit 12.8 (nvcc) + MSVC v143 (VS 2022 v14.38-17.8)
REM
REM  Outputs:
REM    - bin\distance_kernel.dll (Shared library for Python ctypes)
REM    - bin\test_distance.exe   (Standalone C++ verification executable)
REM
REM  Usage:
REM    scripts\build_distance_kernel.bat          (build only)
REM    scripts\build_distance_kernel.bat run      (build and run C++ verification)
REM ============================================================

echo ============================================================
if "%ARCH%"=="" set "ARCH=sm_86"
echo  Building CUDA Pairwise Distance Kernel (%ARCH%)
echo ============================================================

REM Navigate to project root
cd /d "%~dp0.."

REM Create output directory
if not exist "bin" mkdir bin

REM Compile DLL
echo.
echo [BUILD] nvcc -arch=%ARCH% -O2 --shared -o bin\distance_kernel.dll src\cuda\kernels\distance_kernel.cu
echo.
nvcc -arch=%ARCH% -O2 --shared -o bin\distance_kernel.dll src\cuda\kernels\distance_kernel.cu
if %ERRORLEVEL% neq 0 (
    echo [FAIL] Compilation of distance_kernel.dll failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
echo [PASS] Compiled bin\distance_kernel.dll

REM Compile Standalone Test Executable
echo.
echo [BUILD] nvcc -arch=%ARCH% -O2 -o bin\test_distance.exe src\cuda\kernels\test_distance.cu src\cuda\kernels\distance_kernel.cu
echo.
nvcc -arch=%ARCH% -O2 -o bin\test_distance.exe src\cuda\kernels\test_distance.cu src\cuda\kernels\distance_kernel.cu
if %ERRORLEVEL% neq 0 (
    echo [FAIL] Compilation of test_distance.exe failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
echo [PASS] Compiled bin\test_distance.exe

echo.
echo [SUCCESS] Distance kernel build completed.
echo.

if /i "%1"=="run" (
    echo ============================================================
    echo  Running Standalone C++ Distance Kernel Verification
    echo ============================================================
    echo.
    bin\test_distance.exe
    echo.
    echo Exit code: %ERRORLEVEL%
)
