# CSC14120 - Parallel Programming: Autoencoder Feature Learning

**HCMUS - Faculty of Information Technology**

Convolutional Autoencoder for unsupervised feature learning on CIFAR-10, accelerated with CUDA.

## Project Overview

This project implements a 4-phase autoencoder training pipeline:

| Phase | Description | Build Target |
|-------|-------------|--------------|
| 1 | CPU Baseline | `cpu_train`, `cpu_train_omp` |
| 2 | Naive GPU | `gpu_train` |
| 3 | Optimized GPU | `gpu_train_opt` |
| 4 | Full Pipeline (GPU + SVM) | `full_pipeline` |

## Quick Start

```bash
# 1. Download CIFAR-10 dataset
cd data
curl -O https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -xzf cifar-10-binary.tar.gz
mv cifar-10-batches-bin/* .
cd ..

# 2. Build and run Phase 1 (CPU)
make cpu_train_omp
./cpu_train_omp data 5 32 0.001 cpu_training.log 1000 1

# 3. Build and run Phase 2 (GPU Naive)
make gpu_train
./gpu_train --data data --epochs 20 --batch 64

# 4. Build and run Phase 3 (GPU Optimized)
make gpu_train_opt
./gpu_train_opt --data data --epochs 20 --batch 64

# 5. Build and run Phase 4 (Full Pipeline with SVM)
make full_pipeline
./full_pipeline --data data --epochs 20 --batch 64
```

## Phase Details

### Phase 1: CPU Baseline

**Build:**
```bash
make cpu_train      # Without OpenMP
make cpu_train_omp  # With OpenMP parallelization
```

**Features:**
- OpenMP parallelization for Conv2D forward/backward
- Loop unrolling for 3x3 kernel convolutions
- SIMD vectorization for ReLU and MSE operations
- Thread-local gradient accumulators

**Usage:**
```bash
./cpu_train <data_dir> <epochs> <batch_size> <lr> <log_file> <max_images> <use_openmp> [csv_file]
# Example:
./cpu_train_omp data 5 32 0.001 cpu_training.log 5000 1 cpu_training.csv
```

### Phase 2: Naive GPU Implementation

**Build:**
```bash
make gpu_train
```

**Features:**
- Basic CUDA kernels for all layers
- 2D thread blocks (16×16) for spatial operations
- Global memory access patterns

**Usage:**
```bash
./gpu_train [options]
  --data <dir>         Data directory (default: data)
  --epochs <n>         Training epochs (default: 20)
  --batch <n>          Batch size (default: 64)
  --lr <f>             Learning rate (default: 0.001)
  --log <file>         CSV log file
  --log-txt <file>     TXT log file
  --max-images <n>     Limit training images (0=all)
  --load-weights <f>   Load pretrained weights
  --save-weights <f>   Save weights after training
```

### Phase 3: Optimized GPU Implementation

**Build:**
```bash
make gpu_train_opt
```

**Optimizations Applied:**
1. **Shared Memory Tiling**: Reduces global memory access in convolution
2. **Vectorized Memory Access**: float4 loads for ReLU operations
3. **Warp Shuffle Reduction**: Fast MSE loss computation
4. **Loop Unrolling**: Optimized 3×3 kernel and 2×2 pooling
5. **Kernel Fusion Ready**: Conv+ReLU fusion kernels available

### Phase 4: Full Pipeline (GPU + SVM)

**Build:**
```bash
# First, setup LIBSVM
git submodule add https://github.com/cjlin1/libsvm external/libsvm
make full_pipeline
```

**Features:**
- Feature extraction using trained encoder
- SVM classification with RBF kernel
- Confusion matrix output

## Network Architecture

```
INPUT: (N, 3, 32, 32)
  ↓
ENCODER:
  Conv2D(3→256, 3×3, pad=1) + ReLU → (N, 256, 32, 32)
  MaxPool(2×2)                     → (N, 256, 16, 16)
  Conv2D(256→128, 3×3, pad=1) + ReLU → (N, 128, 16, 16)
  MaxPool(2×2)                     → (N, 128, 8, 8)
  ↓
LATENT: (N, 128, 8, 8) = 8,192 features
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

## Performance Targets

| Metric | Target |
|--------|--------|
| Training time (GPU) | < 10 minutes |
| Feature extraction | < 20 sec for 60K images |
| GPU speedup vs CPU | > 20× |
| Test accuracy | 60-65% |

## Verification

Verify GPU implementation correctness:
```bash
make verify_gpu
./verify_gpu data
```

## Project Structure

```
Homogenous-AutoEncoder/
├── include/
│   ├── autoencoder.h      # CPU autoencoder class
│   ├── gpu_autoencoder.h  # GPU autoencoder class
│   ├── layer.h            # CPU layer definitions
│   ├── gpu_layer.h        # GPU layer definitions
│   ├── dataset.h          # CIFAR-10 data loading
│   ├── cuda_utils.h       # CUDA error checking
│   └── svm_wrapper.h      # LIBSVM interface
├── src/
│   ├── main.cpp           # CPU training (Phase 1)
│   ├── main_gpu.cu        # GPU training (Phase 2-4)
│   ├── autoencoder.cpp    # CPU autoencoder
│   ├── gpu_autoencoder.cu # GPU autoencoder
│   ├── layers_cpu.cpp     # CPU layer implementations
│   ├── layers_gpu.cu      # Naive GPU kernels (Phase 2)
│   ├── layers_gpu_opt.cu  # Optimized GPU kernels (Phase 3)
│   ├── dataset.cpp        # Data loading
│   ├── svm_wrapper.cpp    # SVM classification (Phase 4)
│   └── verify_gpu.cu      # GPU verification tool
├── data/                  # CIFAR-10 binary files
├── Makefile
└── README.md
```

## Build Configuration

Check your build configuration:
```bash
make info
```

### Requirements

**CPU Build:**
- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- OpenMP (optional, for parallel CPU execution)

**GPU Build:**
- CUDA Toolkit 11.0+
- NVIDIA GPU (compute capability 6.0+)

**SVM (Phase 4):**
- LIBSVM library

### Platform-Specific Setup

**Linux:**
```bash
sudo apt-get install build-essential
sudo apt-get install nvidia-cuda-toolkit  # For GPU
```

**macOS:**
```bash
xcode-select --install
brew install libomp  # For OpenMP
# Note: CUDA not supported on recent macOS
```

**Windows (MSYS2):**
```bash
pacman -S mingw-w64-x86_64-gcc make
```

## References

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
- Hinton & Salakhutdinov (2006). "Reducing the Dimensionality of Data with Neural Networks"
