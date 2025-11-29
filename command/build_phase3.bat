@echo off
echo === Building Phase 3: GPU Optimized ===

REM Setup Visual Studio environment for NVCC
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" 2>nul
if %errorlevel% neq 0 (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" 2>nul
)

nvcc -O2 -std=c++17 -arch=sm_75 --expt-relaxed-constexpr -Iinclude -DUSE_OPTIMIZED_KERNELS -o gpu_train_opt.exe src/main_gpu.cu src/layers_gpu.cu src/gpu_autoencoder.cu src/layers_gpu_opt.cu src/dataset.cpp
if %errorlevel% equ 0 (
    echo Build successful!
    echo Running Phase 3...
    gpu_train_opt.exe
) else (
    echo Build failed!
)
pause
