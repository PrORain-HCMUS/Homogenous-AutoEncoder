# CSC14120 - Parallel Programming: Autoencoder Feature Learning

<p align="center">
  <strong>Vietnam National University, Ho Chi Minh City</strong><br>
  University of Science - Faculty of Information Technology
</p>

<p align="center">
  <img src="https://img.shields.io/badge/CUDA-12.x-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA">
  <img src="https://img.shields.io/badge/cuDNN-8.x-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="cuDNN">
  <img src="https://img.shields.io/badge/cuML-RAPIDS-7400B8?style=for-the-badge&logo=rapids&logoColor=white" alt="cuML">
  <img src="https://img.shields.io/badge/ThunderSVM-GPU-FF6B6B?style=for-the-badge" alt="ThunderSVM">
  <img src="https://img.shields.io/badge/C++-17-00599C?style=for-the-badge&logo=cplusplus&logoColor=white" alt="C++17">
  <img src="https://img.shields.io/badge/OpenMP-Parallel-0078D4?style=for-the-badge" alt="OpenMP">
</p>

---

## Project Overview

This project implements a **GPU-accelerated Convolutional Autoencoder** for unsupervised feature learning on CIFAR-10, achieving **~505× speedup** over CPU baseline and **70% classification accuracy** using extracted features with SVM.

### Group Members

| Member | Name | Student ID |
|:------:|:-----|:-----------|
| 1 | Le Dai Hoa | 22120108 |
| 2 | Nguyen Tuong Bach Hy | 22120455 |
| 3 | Le Hoang Vu | 22120461 |

### Key Achievements

| Metric | Target | Achieved | Status |
|:-------|:-------|:---------|:------:|
| GPU Speedup | >50× | **~505×** | Exceeded |
| Classification Accuracy | >50% | **70%** | Exceeded |
| Training Time (50K images) | <15 min | **~10.3 min** | Met |

---

## Implementation Phases

| Phase | Description | Key Optimization | Speedup |
|:-----:|:------------|:-----------------|:-------:|
| **1** | CPU Baseline | OpenMP parallelization | 1× |
| **2** | GPU Naive | Basic CUDA kernels | ~51× |
| **3.1** | GPU Tiled | Shared memory tiling | ~505× |
| **3.2** | GPU cuDNN | cuDNN library integration | ~482× |
| **3.3** | GPU BCE | BCE loss + Sigmoid activation | ~292× |
| **4** | SVM Classification | cuML GPU-accelerated SVM | - |

### Phase 3 Comparison

| Version | Training Time | Final Loss | SVM Accuracy | Best For |
|:--------|:--------------|:-----------|:-------------|:---------|
| Tiled | ~10.3 min | 0.0114 | ~67% | Learning CUDA optimization |
| cuDNN | ~10.8 min | 0.0114 | ~67% | Production deployment |
| **BCE** | ~17.8 min | 0.55 | **~70%** | Best classification accuracy |

> [!TIP]
> The **BCE loss version** produces better features for classification despite slower training, because BCE gradients are more stable for pixel-wise reconstruction.

---

## Network Architecture

```
INPUT: (N, 3, 32, 32) - CIFAR-10 RGB images
  ↓
ENCODER:
  Conv2D(3→256, 3×3, pad=1) + ReLU  → (N, 256, 32, 32)
  MaxPool(2×2)                       → (N, 256, 16, 16)
  Conv2D(256→128, 3×3, pad=1) + ReLU → (N, 128, 16, 16)
  MaxPool(2×2)                       → (N, 128, 8, 8)
  ↓
LATENT: (N, 128, 8, 8) = 8,192 features
  ↓
DECODER:
  Conv2D(128→128, 3×3, pad=1) + ReLU → (N, 128, 8, 8)
  UpSample(2×)                        → (N, 128, 16, 16)
  Conv2D(128→256, 3×3, pad=1) + ReLU  → (N, 256, 16, 16)
  UpSample(2×)                        → (N, 256, 32, 32)
  Conv2D(256→3, 3×3, pad=1) + Sigmoid → (N, 3, 32, 32)
  ↓
OUTPUT: (N, 3, 32, 32) - Reconstructed images

Total Parameters: 751,875
```

---

## Results Summary

### Training Performance

| Phase | Training Time (50K) | Speedup vs CPU | Final Loss |
|:------|:--------------------|:---------------|:-----------|
| CPU Baseline | ~86.7 hours | 1.0× | 0.2646 |
| GPU Naive | ~102.1 min | ~51× | 0.0224 |
| GPU Tiled | ~10.3 min | **~505×** | 0.0114 |
| GPU cuDNN | ~10.8 min | ~482× | 0.0114 |
| GPU BCE | ~17.8 min | ~292× | 0.4648 |

### Classification Accuracy (SVM on Extracted Features)

| Weights Source | Test Accuracy | Best Class | Worst Class |
|:---------------|:--------------|:-----------|:------------|
| Phase 1 CPU | 68.0% | ship | cat |
| Phase 2 GPU Naive | 67.0% | ship | cat |
| Phase 3 Tiled | 66.6% | ship | cat |
| Phase 3 cuDNN | 66.6% | ship | cat |
| **Phase 3 BCE** | **70.5%** | ship (~80%) | cat (~56%) |

---

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/PrORain-HCMUS/Homogenous-AutoEncoder.git
cd Homogenous-AutoEncoder

# 2. Download CIFAR-10 dataset
cd data
curl -O https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -xzf cifar-10-binary.tar.gz
mv cifar-10-batches-bin/* .
cd ..

# 3. Build and run (choose one)
make gpu_train_opt           # Phase 3 optimized
./gpu_train_opt --data data --epochs 20 --batch 64

make full_pipeline           # Phase 4 with SVM
./full_pipeline --data data --epochs 20 --batch 64
```

---

## Project Structure

```
Homogenous-AutoEncoder/
├── include/
│   ├── autoencoder.h       # CPU autoencoder class
│   ├── gpu_autoencoder.h   # GPU autoencoder class
│   ├── layer.h             # CPU layer definitions
│   ├── gpu_layer.h         # GPU layer definitions
│   ├── dataset.h           # CIFAR-10 data loading
│   └── cuda_utils.h        # CUDA error checking
├── src/
│   ├── main.cpp            # CPU training (Phase 1)
│   ├── main_gpu.cu         # GPU training (Phase 2-3)
│   ├── main_phase4.cu      # Full pipeline with SVM (Phase 4)
│   ├── autoencoder.cpp     # CPU autoencoder
│   ├── gpu_autoencoder.cu  # GPU autoencoder
│   ├── layers_cpu.cpp      # CPU layer implementations
│   ├── layers_gpu.cu       # Naive GPU kernels (Phase 2)
│   ├── layers_gpu_opt.cu   # Optimized GPU kernels (Phase 3)
│   └── dataset.cpp         # Data loading
├── notebooks/              # Jupyter notebooks (see below)
├── docs/                   # PDF reports
├── results/                # Training logs and weights
├── data/                   # CIFAR-10 binary files
├── Makefile
└── README.md
```

> [!IMPORTANT]
> For the full project report with detailed analysis, see [`docs/report.pdf`](docs/report.pdf) or run the [`report.ipynb`](https://github.com/PrORain-HCMUS/Homogenous-AutoEncoder/blob/report/notebooks/report.ipynb) notebook.

---

## Notebooks

All notebooks are available in the [`feat/enhancement`](https://github.com/PrORain-HCMUS/Homogenous-AutoEncoder/tree/feat/enhancement/notebooks) branch:

### Phase 2: GPU Naive
- [`Phase2_Colab.ipynb`](https://github.com/PrORain-HCMUS/Homogenous-AutoEncoder/blob/feat/enhancement/notebooks/Phase2_Colab.ipynb)
- [`Phase2_Kaggle.ipynb`](https://github.com/PrORain-HCMUS/Homogenous-AutoEncoder/blob/feat/enhancement/notebooks/Phase2_Kaggle.ipynb)

### Phase 3: GPU Optimized
- [`Phase3_Colab.ipynb`](https://github.com/PrORain-HCMUS/Homogenous-AutoEncoder/blob/feat/enhancement/notebooks/Phase3_Colab.ipynb) - Tiled convolution
- [`Phase3_Kaggle.ipynb`](https://github.com/PrORain-HCMUS/Homogenous-AutoEncoder/blob/feat/enhancement/notebooks/Phase3_Kaggle.ipynb)
- [`Phase3_Colab_BCE.ipynb`](https://github.com/PrORain-HCMUS/Homogenous-AutoEncoder/blob/feat/enhancement/notebooks/Phase3_Colab_BCE.ipynb) - BCE loss version
- [`Phase3_Kaggle_BCE.ipynb`](https://github.com/PrORain-HCMUS/Homogenous-AutoEncoder/blob/feat/enhancement/notebooks/Phase3_Kaggle_BCE.ipynb)
- [`enhancement/Phase3_Colab_Tiled.ipynb`](https://github.com/PrORain-HCMUS/Homogenous-AutoEncoder/blob/feat/enhancement/notebooks/enhancement/Phase3_Colab_Tiled.ipynb)

### Phase 4: SVM Classification
- [`Phase4_Colab.ipynb`](https://github.com/PrORain-HCMUS/Homogenous-AutoEncoder/blob/feat/enhancement/notebooks/Phase4_Colab.ipynb)
- [`Phase4_Kaggle.ipynb`](https://github.com/PrORain-HCMUS/Homogenous-AutoEncoder/blob/feat/enhancement/notebooks/Phase4_Kaggle.ipynb)
- [`Phase4_Kaggle_PCA.ipynb`](https://github.com/PrORain-HCMUS/Homogenous-AutoEncoder/blob/feat/enhancement/notebooks/Phase4_Kaggle_PCA.ipynb) - With PCA
- [`phase3-4-kaggle-bce-cuml.ipynb`](https://github.com/PrORain-HCMUS/Homogenous-AutoEncoder/blob/feat/enhancement/notebooks/phase3-4-kaggle-bce-cuml.ipynb) - BCE + cuML SVM

### Full Report
- [`report.ipynb`](https://github.com/PrORain-HCMUS/Homogenous-AutoEncoder/blob/report/notebooks/report.ipynb) - Complete project report with all phases

---

## Build Targets

| Target | Description | Command |
|:-------|:------------|:--------|
| `cpu_train` | CPU baseline (no OpenMP) | `make cpu_train` |
| `cpu_train_omp` | CPU with OpenMP | `make cpu_train_omp` |
| `gpu_train` | GPU naive (Phase 2) | `make gpu_train` |
| `gpu_train_opt` | GPU optimized (Phase 3) | `make gpu_train_opt` |
| `full_pipeline` | Full pipeline with SVM (Phase 4) | `make full_pipeline` |
| `feature_extractor` | Extract features only | `make feature_extractor` |

---

## Requirements

**GPU Build:**
- CUDA Toolkit 11.0+
- NVIDIA GPU (compute capability 6.0+)
- cuDNN (optional, for cuDNN version)

**CPU Build:**
- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- OpenMP (optional)

**SVM Classification (Phase 4):**
- cuML (recommended for GPU)
- ThunderSVM or sklearn (alternatives)

---

## References

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/)
- [cuML SVM](https://docs.rapids.ai/api/cuml/stable/)
- Hinton & Salakhutdinov (2006). "Reducing the Dimensionality of Data with Neural Networks"

---

## License

This project is for educational purposes as part of CSC14120 - Parallel Programming course.
