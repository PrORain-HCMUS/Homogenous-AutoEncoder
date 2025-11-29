@echo off
echo ============================================
echo    Homogenous AutoEncoder - Build All
echo ============================================

REM Setup Visual Studio environment for NVCC
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" 2>nul
if %errorlevel% neq 0 (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" 2>nul
)

echo.
echo === Phase 1: CPU Baseline ===
g++ -O3 -std=c++17 -fopenmp -Iinclude -o cpu_train.exe src/main.cpp src/dataset.cpp src/layers_cpu.cpp src/autoencoder.cpp
if %errorlevel% equ 0 (echo [OK] cpu_train.exe) else (echo [FAILED] cpu_train.exe)

echo.
echo === Phase 2: GPU Naive ===
nvcc -O2 -std=c++17 -arch=sm_75 --expt-relaxed-constexpr -Iinclude -o gpu_train.exe src/main_gpu.cu src/layers_gpu.cu src/gpu_autoencoder.cu src/dataset.cpp
if %errorlevel% equ 0 (echo [OK] gpu_train.exe) else (echo [FAILED] gpu_train.exe)

echo.
echo === Phase 3: GPU Optimized ===
nvcc -O2 -std=c++17 -arch=sm_75 --expt-relaxed-constexpr -Iinclude -DUSE_OPTIMIZED_KERNELS -o gpu_train_opt.exe src/main_gpu.cu src/layers_gpu.cu src/gpu_autoencoder.cu src/layers_gpu_opt.cu src/dataset.cpp
if %errorlevel% equ 0 (echo [OK] gpu_train_opt.exe) else (echo [FAILED] gpu_train_opt.exe)

echo.
echo ============================================
echo Build complete!
echo ============================================
pause
