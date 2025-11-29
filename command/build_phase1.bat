@echo off
echo === Building Phase 1: CPU Baseline ===
g++ -O3 -std=c++17 -fopenmp -Iinclude -o cpu_train.exe src/main.cpp src/dataset.cpp src/layers_cpu.cpp src/autoencoder.cpp
if %errorlevel% equ 0 (
    echo Build successful!
    echo Running Phase 1...
    cpu_train.exe
) else (
    echo Build failed!
)
pause
