# CSC14120 - Parallel Programming: Unsupervised Feature Learning with Autoencoder

**Đại học Khoa học Tự nhiên, ĐHQG-HCM**  
**Khoa Công nghệ Thông tin**

Convolutional Autoencoder for unsupervised feature learning on CIFAR-10, accelerated with CUDA.

## Quick Start

```bash
# 1. Download CIFAR-10 binary dataset
cd data && curl -O https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -xzf cifar-10-binary.tar.gz && mv cifar-10-batches-bin/* . && cd ..

# 2. Build CPU baseline (Phase 1)
make cpu_train

# 3. Run CPU training
./cpu_train data 5 32 0.001 cpu_log.csv 1000

# 4. Build GPU version (Phase 2) - requires CUDA
make gpu_train

# 5. Run GPU training
./gpu_train --data data --epochs 20 --batch 64
```

## Project Structure

```
Homogenous-AutoEncoder/
├── include/
│   ├── autoencoder.h      # CPU autoencoder class
│   ├── gpu_autoencoder.h  # GPU autoencoder class (Phase 2+)
│   ├── layer.h            # CPU layer definitions (Conv2D, ReLU, MaxPool, etc.)
│   ├── gpu_layer.h        # GPU layer definitions with CUDA kernels
│   ├── dataset.h          # CIFAR-10 data loading
│   ├── cuda_utils.h       # CUDA error checking macros
│   └── svm_wrapper.h      # LIBSVM integration interface
├── src/
│   ├── main.cpp           # CPU training entry point
│   ├── main_gpu.cu        # GPU training entry point
│   ├── autoencoder.cpp    # CPU autoencoder implementation
│   ├── gpu_autoencoder.cu # GPU autoencoder implementation
│   ├── layers_cpu.cpp     # CPU layer implementations
│   ├── layers_gpu.cu      # Naive GPU kernels (Phase 2)
│   ├── layers_gpu_opt.cu  # Optimized GPU kernels (Phase 3)
│   ├── dataset.cpp        # Data loading implementation
│   └── svm_wrapper.cpp    # SVM classification (Phase 4)
├── data/                  # CIFAR-10 binary files
├── Makefile              # Build configuration
└── README.md
```

## Build Targets

| Target | Description | Command |
|--------|-------------|---------|
| `cpu_train` | CPU baseline (Phase 1) | `make cpu_train` |
| `cpu_train_omp` | CPU with OpenMP | `make cpu_train_omp` |
| `gpu_train` | Naive GPU (Phase 2) | `make gpu_train` |
| `gpu_train_opt` | Optimized GPU (Phase 3) | `make gpu_train_opt` |
| `full_pipeline` | GPU + SVM (Phase 4) | `make full_pipeline` |

## Network Architecture

```
INPUT: (N, 3, 32, 32)
  ↓
ENCODER:
  Conv2D(3→256, 3×3, pad=1) + ReLU → (N, 256, 32, 32)
  MaxPool(2×2)                     → (N, 256, 16, 16)
  Conv2D(256→128, 3×3, pad=1) + ReLU → (N, 128, 16, 16)
  MaxPool(2×2)                     → (N, 128, 8, 8) = 8,192 features
  ↓
LATENT: (N, 128, 8, 8)
  ↓
DECODER:
  Conv2D(128→128, 3×3, pad=1) + ReLU → (N, 128, 8, 8)
  UpSample(2×)                      → (N, 128, 16, 16)
  Conv2D(128→256, 3×3, pad=1) + ReLU → (N, 256, 16, 16)
  UpSample(2×)                      → (N, 256, 32, 32)
  Conv2D(256→3, 3×3, pad=1)         → (N, 3, 32, 32)
  ↓
OUTPUT: (N, 3, 32, 32)

Total Parameters: 751,875
```

## GPU Optimizations (Phase 3)

1. **Shared Memory Tiling**: Reduces global memory access in convolution
2. **Kernel Fusion**: Combined Conv+ReLU+Bias in single kernel
3. **Vectorized Access**: float4 loads for element-wise operations
4. **2D Thread Blocks**: Optimal spatial parallelization for pooling/upsampling

## Performance Targets

| Metric | Target |
|--------|--------|
| Training time (GPU) | < 10 minutes |
| Feature extraction | < 20 sec for 60K images |
| GPU speedup vs CPU | > 20× |
| Test accuracy | 60-65% |

## Command Line Options

### CPU Training (`cpu_train`)
```
./cpu_train <data_dir> <epochs> <batch_size> <lr> <log_csv> <max_images> <use_openmp>
```

### GPU Training (`gpu_train`)
```
./gpu_train [options]
  --data <dir>         CIFAR-10 data directory (default: data)
  --epochs <n>         Number of training epochs (default: 20)
  --batch <n>          Batch size (default: 64)
  --lr <f>             Learning rate (default: 0.001)
  --log <file>         CSV log file path
  --max-images <n>     Max training images (0=all)
  --load-weights <f>   Load weights from file
  --save-weights <f>   Save weights to file
```

## Requirements

### Cross-Platform Support (macOS, Linux, Windows)

This project supports building on multiple platforms:

**Linux (Ubuntu/Debian)**
```bash
# Install dependencies
sudo apt-get install build-essential

# For OpenMP support (usually included with GCC)
make cpu_train_omp

# For GPU support
sudo apt-get install nvidia-cuda-toolkit
make gpu_train
```

**macOS**
```bash
# Install Xcode command line tools
xcode-select --install

# For OpenMP support (choose one):
# Option 1: Install libomp via Homebrew
brew install libomp

# Option 2: Install GCC via Homebrew (includes OpenMP)
brew install gcc

# Build
make cpu_train      # Without OpenMP
make cpu_train_omp  # With OpenMP (if available)
```

**Windows (MinGW-w64)**
```powershell
# Install MSYS2 and MinGW-w64
# Then in MSYS2 terminal:
pacman -S mingw-w64-x86_64-gcc make

# Build
make cpu_train
make cpu_train_omp
```

### Build Requirements

- **CPU Build**: C++17 compiler (g++ 9+, clang++ 10+, or MSVC 2019+)
- **OpenMP (Optional)**: For parallel CPU execution
  - Linux: GCC with OpenMP (usually included)
  - macOS: `brew install libomp` or `brew install gcc`
  - Windows: MinGW-w64 with OpenMP
- **GPU Build**: CUDA Toolkit 11.0+, NVIDIA GPU (compute capability 6.0+)
- **SVM (Optional)**: LIBSVM library

### Check Your Configuration

Run `make info` to see detected platform and compiler settings:
```bash
make info
# Output example:
# === Build Configuration ===
# OS: Darwin
# Compiler: clang++
# CXXFLAGS: -O3 -std=c++17 -Wall -Wextra -march=native
# OpenMP available: 1
# ===========================
```

## References

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
- Hinton & Salakhutdinov (2006). "Reducing the Dimensionality of Data with Neural Networks"
