# CSC14120 - PARALLEL PROGRAMMING: FINAL PROJECT

## 1\. Introduction

### 1.1 Problem Statement

Feature engineering is a fundamental challenge in machine learning: how do we automatically discover good representations of data that capture its underlying structure? In this project, you will implement an Autoencoder-based unsupervised feature learning system for image classification on the CIFAR-10 dataset.

Traditional supervised learning trains a model end-to-end using labelled examples. However, autoencoders take a different approach: they learn meaningful representations by attempting to reconstruct the input data itself, without requiring any labels during the feature learning phase. This unsupervised pre-training can discover features that are often more robust and generalizable than those learned through direct supervision alone.

Your task is to build and optimize a complete two-stage pipeline:

**Stage 1 - Unsupervised Feature Learning:**

  * Train a convolutional autoencoder to reconstruct CIFAR-10 images.
  * The autoencoder learns to encode $32 \times 32 \times 3$ images into an 8,192-dimensional feature representation that captures meaningful visual patterns.
  * No labels are used during this training phase.
  * The network learns features that capture important visual patterns (edges, textures, shapes).

**Stage 2 - Supervised Classification:**

  * Extract features from the trained encoder for all images.
  * Train an SVM classifier on these learned features with class labels.
  * Evaluate classification performance on the test set.

The primary focus is on implementing and optimizing the autoencoder training and inference in CUDA, while using existing libraries (LIBSVM) for the SVM classifier. Through systematic optimization, you will accelerate the autoencoder from taking hours on CPU to completing in seconds on GPU.

### 1.2 CIFAR-10 Dataset

CIFAR-10 is one of the most widely used benchmark datasets in computer vision and machine learning research. It provides a challenging yet computationally manageable image classification task.

**Dataset Specifications:**

  * **Image size:** $32 \times 32$ pixels (RGB)
  * **10 classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
  * **Training set:** 50,000 images (5,000 per class)
  * **Test set:** 10,000 images (1,000 per class)
  * **Total images:** 60,000
  * **Format:** Binary files with uint8 pixel values
  * **Dataset Link:** [https://www.cs.toronto.edu/\~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

**Dataset Organization for This Project:**

  * **Autoencoder training:** All 50,000 training images (labels ignored)
  * **SVM training:** Same 50,000 images with labels (using extracted features)
  * **Evaluation:** 10,000 test images with labels

### 1.3 Expected Performance Targets

| Metric | Target |
| :--- | :--- |
| Autoencoder training time | \< 10 minutes |
| Feature extraction time | \< 20 seconds for all 60K images |
| Test classification accuracy | 60-65% |
| GPU speedup over CPU | \> 20x |

-----

## 2\. Background Knowledge

### 2.1 Autoencoders for Unsupervised Feature Learning

**What is an Autoencoder?**
An autoencoder is a neural network trained to reconstruct its input, forcing the network to learn a compressed, meaningful representation in the process. It consists of two parts:

  * **Encoder:** Compresses the input into a low-dimensional latent representation (feature vector).
  * **Decoder:** Reconstructs the original input from the latent representation.

**Training Objective:**
$$Loss = MSE(Input, Reconstructed\_Output)$$
$$= (1 / N) * \Sigma (x - decoder(encoder(x)))^2$$

The network learns to minimize reconstruction error, forcing the encoder to capture essential information about the input.

**Key Concepts:**

1.  **Bottleneck Layer (Latent Space):** The smallest layer in the middle. Forces compression and feature extraction. In this project: $(6,6,128) \rightarrow 4,608$ dimensions (Note: Text later says 8,192, see architecture section for clarification).
2.  **Symmetric Architecture:** Decoder mirrors the encoder. Upsampling operations reverse the downsampling. Helps in reconstruction quality.
3.  **Feature Extraction:** After training, we discard the decoder. Use only the encoder to extract features. These features feed into the SVM classifier.

**Recommended Reading Materials:**

  * **Foundational Papers:** Hinton, G. E., & Salakhutdinov, R. R. (2006). "Reducing the Dimensionality of Data with Neural Networks." [Link](https://www.google.com/search?q=https://www.cs.toronto.edu/~hinton/science.pdf)

  * **Book Chapters:** Goodfellow, I., Bengio, Y., & Courville, A. "Deep Learning" (2016) - Chapter 14: Autoencoders. [Link](https://www.deeplearningbook.org/contents/autoencoders.html)

  * **Video guide:** Autoencoders and Representation Learning. [YouTube Link](https://www.youtube.com/watch?v=R3DNKE3zKFk)

  * **Reference source code:**

      * [https://github.com/turkdogan/autoencoder](https://github.com/turkdogan/autoencoder)
      * [https://github.com/tbennun/cudnn-training](https://github.com/tbennun/cudnn-training)

  * **CNN references:**

      * Video: “How Convolutional Neural Networks work”
      * Stanford cheatsheet on CNNs
      * “Dive into deep learning” book

### 2.2 Support Vector Machine (SVM) for Classification

**Your Role with SVM:** You do NOT need to implement SVM from scratch. Use existing optimized libraries.

**Recommended SVM Libraries:**

1.  **LIBSVM (Primary Recommendation):**
      * Most popular SVM library, Easy to use C/C++ interface, Supports multi-class classification, RBF kernel built-in.
      * [Website](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) | [GitHub](https://github.com/cjlin1/libsvm)
2.  **LIBLINEAR (Alternative for Linear SVM):**
      * Faster for linear kernels, good for experimentation.
      * [Website](https://www.csie.ntu.edu.tw/~cjlin/liblinear/)
3.  **ThunderSVM (GPU-Accelerated):**
      * GPU implementation of SVM (Optional for extra credit).
      * [GitHub](https://github.com/Xtra-Computing/thundersvm)

-----

## 3\. Project Pipeline

The project follows a clear 5-step pipeline:

1.  **Step 1: Load CIFAR-10 Images**

      * 50,000 training images (for autoencoder)
      * 10,000 test images
      * Preprocess: normalize to [0,1]

2.  **Step 2: Train Autoencoder (Your CUDA Code)**

      * Use all 50k images (unsupervised training)
      * Ignore labels during autoencoder training
      * Train to minimize reconstruction loss
      * Save encoder weights

3.  **Step 3: Extract Features (Your CUDA Code)**

      * Load trained encoder weights
      * Run encoder forward pass (no decoder)
      * `train_features`: (50000, 8192)
      * `test_features`: (10000, 8192)

4.  **Step 4: Train SVM (Library)**

      * Input: `train_features` + labels
      * Kernel: RBF (Radial Basis Function)
      * Hyperparameters: C=10, gamma=auto
      * Output: trained SVM model

5.  **Step 5: Evaluate**

      * Predict on `test_features` using SVM
      * Calculate accuracy, confusion matrix
      * Expected accuracy: 60-65%

**Key Points:**

  * **Steps 2 and 3:** Focus on autoencoder implementation in CUDA.
  * **Step 4:** Use libraries for SVM training.
  * **Step 5:** Measure success.

-----

## 4\. Network Architecture

### 4.1 Architecture Overview

INPUT: (32, 32, 3) $\Rightarrow$ ENCODER (compress) $\Rightarrow$ LATENT: (8, 8, 128) = 8,192 features $\Rightarrow$ DECODER (reconstruct) $\Rightarrow$ OUTPUT: (32, 32, 3)

### 4.2 Detailed Architecture Specification

**ENCODER (Downsampling Path)**

  * **INPUT:** $(32, 32, 3)$
  * **Conv2D:** (256 filters, $3\times3$ kernel, padding=1, stride=1) + ReLU $\rightarrow (32,32,256)$
  * **MaxPool2D:** ($2\times2$, stride=2) $\rightarrow (16,16,256)$
  * **Conv2D:** (128 filters, $3\times3$ kernel, padding=1, stride=1) + ReLU $\rightarrow (16,16,128)$
  * **MaxPool2D:** ($2\times2$, stride=2) $\rightarrow (8,8,128)$
  * **LATENT REPRESENTATION:** $(8,8,128) = 8192$ dimensions

**DECODER (Upsampling Path - Mirror of Encoder)**

  * **LATENT:** $(8,8,128)$
  * **Conv2D:** (128 filters, $3\times3$ kernel, padding=1, stride=1) + ReLU $\rightarrow (8,8,128)$
  * **Upsample2D:** ($2\times2$) [Nearest neighbor or bilinear] $\rightarrow (16,16,128)$
  * **Conv2D:** (256 filters, $3\times3$ kernel, padding=1, stride=1) + ReLU $\rightarrow (16,16,256)$
  * **Upsample2D:** ($2\times2$) [Nearest neighbor or bilinear] $\rightarrow (32,32,256)$
  * **Conv2D:** (3 filters, $3\times3$ kernel, padding=1, stride=1) [No activation] $\rightarrow (32,32,3)$
  * **OUTPUT:** $(32,32,3)$

**Keras Reference Setup:**

```python
input_size = (32, 32, 3)
input_image = Input(shape=input_size)

# Encoder
x = Conv2D(256, (3, 3), activation='relu', padding='same')(input_image)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same', name='encoded_layer')(x)

# Decoder
x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), padding='same')(x)
```

**Parameter Summary:**

| Layer (type) | Output Shape | Param \# |
| :--- | :--- | :--- |
| input\_1 (InputLayer) | (None, 32, 32, 3) | 0 |
| conv2d\_1 (Conv2D) | (None, 32, 32, 256) | 7,168 |
| max\_pooling2d\_1 | (None, 16, 16, 256) | 0 |
| conv2d\_2 (Conv2D) | (None, 16, 16, 128) | 295,040 |
| encoded\_layer (MaxPool) | (None, 8, 8, 128) | 0 |
| conv2d\_3 (Conv2D) | (None, 8, 8, 128) | 147,584 |
| up\_sampling2d\_1 | (None, 16, 16, 128) | 0 |
| conv2d\_4 (Conv2D) | (None, 16, 16, 256) | 295,168 |
| up\_sampling2d\_2 | (None, 32, 32, 256) | 0 |
| conv2d\_5 (Conv2D) | (None, 32, 32, 3) | 6,915 |
| **Total params** | | **751,875** |

-----

## 5\. Implementation Guideline

You will implement this project in at least 4 progressive phases, focusing on CUDA optimization for the autoencoder training and inference.

### Phase 1: CPU Baseline & Data Pipeline

**Objective:** Set up the project infrastructure and create a working CPU baseline.

**What to Implement:**

1.  **Data Loading and Preprocessing:**

      * Create a CIFAR10 Dataset class.
      * Read binary files, parse format (1 byte label + 3072 bytes image).
      * Convert uint8 [0, 255] to float [0, 1].
      * Implement batch generation and shuffling.
      * Organize 50,000 train images and 10,000 test images.

2.  **CPU Neural Network Layers:**

      * Implement CPU versions: Conv2D, ReLU, MaxPool ($2\times2$), Upsampling (Nearest Neighbor), MSE Loss.

3.  **Autoencoder Class:**

      * Allocate memory for weights/biases.
      * Implement weight initialization.
      * Implement forward pass (Encoder -\> Decoder).
      * Implement backward pass.
      * Implement feature extraction method (Encoder only).
      * Add weight saving/loading.

4.  **Training Loop:**

      * Batch size (32), epochs (20), learning rate (0.001).
      * Track training loss and time per epoch.

**Deliverables:** Working data pipeline, Complete CPU implementation, Baseline measurements.

### Phase 2: Naive GPU Implementation

**Objective:** Port all operations to GPU with basic parallelization.

**What to Implement:**

1.  **GPU Memory Management:**

      * `GPUAutoencoder` class.
      * Allocate device memory for weights, activations, gradients.
      * Implement host-to-device copy functions and `cudaFree` cleanup.

2.  **Naive GPU Kernels:**

      * **Convolution:** Thread per pixel, nested loops for kernel/channels. Global memory.
      * **ReLU:** Simple in-place `max(0, x)`.
      * **MaxPooling:** Thread per output, max in $2\times2$ window.
      * **Upsampling:** Map output coords to input, nearest neighbor copy.
      * **MSE Loss:** Parallel reduction to sum squared differences (atomicAdd).

3.  **GPU Forward/Backward Pass & Training:**

      * Launch kernels sequentially.
      * Implement gradients for each layer type.
      * Implement SGD weight update.
      * Increase batch size to 64.

**Deliverables:** All layers on GPU, Working GPU training loop, Correctness verification against CPU.

### Phase 3: Advanced Optimization

**Objective:** Optimize memory access patterns and apply advanced techniques.

**Optimization Ideas (Suggestions):**

**Category 1: Memory Optimization**

1.  **Shared Memory Tiling for Convolution:** Load input tiles to shared memory to reduce global access.
2.  **Convert to Matrix Multiplication:** Transform convolution to GEMM.
3.  **Memory Coalescing:** Ensure consecutive memory access by warps.
4.  **Constant Memory:** Store small weights/biases in constant memory.
5.  **Pinned Memory:** Use `cudaMallocHost` for faster transfers.
6.  **Unified Memory:** Simplify management with auto-migration.
7.  **Memory Pool:** Reuse buffers to avoid allocation overhead.

**Category 2: Kernel-Level Optimization**
8\.  **Kernel Fusion:** Combine Conv + ReLU + Bias.
9\.  **Block-Level Fusion:** Fuse entire encoder/decoder paths.
10\. **Loop Unrolling:** Optimize loops in conv/pooling.
11\. **Vectorized Memory Access:** Use `float4`.
12\. **Optimized Block Dimensions:** Tune block sizes (e.g., $16\times16$).
13\. **Mixed Precision:** Use FP16 for forward pass.

**Category 3: Parallelism & Concurrency**
14\. **Gradient Checkpointing:** Recompute activations to save memory.
15\. **Multi-Stream Pipeline:** Overlap data transfer and computation.
16\. **Batched Operations:** Process multiple images in parallel.

### Phase 4: SVM Integration and Analysis

**Objective:** Complete the pipeline and thoroughly evaluate performance.

-----

## 6\. Project Report

**Submission Format:**

  * Project report must be submitted as a **Notebook (.ipynb)** that can run on Google Colab.

**Report Structure:**

**Section 1: Problem Description**

  * **1.1 Problem Statement:** Define task and motivation for GPU acceleration.
  * **1.2 CIFAR-10 Dataset Overview:** Specs, samples, preprocessing.
  * **1.3 Autoencoder Architecture:** Diagrams, dimensions, latent representation explanation.
  * **1.4 Project Objectives:** Performance goals and success criteria.

**Section 2: Implementation Phases**

  * **Phase 2.1: CPU Baseline:** Objectives, Data Pipeline details, Layer implementations, Training loop structure, Results (time, loss, images).
  * **Phase 2.2: GPU Basic:** Porting strategy, Kernel designs (Conv, Pool), Memory management, Results (Speedup vs CPU, verification).
  * **Phase 2.3: GPU Optimized (Version 1):** Specific technique applied (e.g., Shared Memory), Explanation, Code snippets, Analysis of why it worked/failed.
  * **Phase 2.4: GPU Optimized (Version 2):** Additional optimizations (e.g., Fusion), Results.
  * **Phase 2.5: SVM Integration:** Feature extraction details, LIBSVM parameters, Accuracy results, Confusion matrix, Analysis.

**Section 3: Comprehensive Performance Analysis**

  * **3.1 Performance Comparison:**

      * Table Format:

    | Phase | Training Time | Speedup (vs CPU) | Incremental Speedup | Memory Usage | Key Optimization |
    | :--- | :--- | :--- | :--- | :--- | :--- |
    | CPU Baseline | 1800s | 1.0x | - | - | - |
    | GPU Basic | 180s | 10.0x | 10.0x | 2.1 GB | Parallelization |
    | GPU Opt v1 | 45s | 40.0x | 4.0x | 2.3 GB | Shared memory |
    | GPU Opt v2 | 25s | 72.0x | 1.8x | 2.5 GB | Fusion + Streams |

  * **Visualization:** Bar chart of training times, Line graph of speedup.

**Section 4: Lessons Learned and Challenges**

  * **4.1 Key Technical Insights:** CUDA, Deep Learning, Optimization.
  * **4.2 Major Challenges:** Format: "Problem -\> Solution -\> Lesson".

**Section 5: Conclusion and Future Work**

  * **5.1 Project Summary:** Metrics summary.
  * **5.2 Key Achievements:** Max speedup, best accuracy.
  * **5.3 Limitations:** Bottlenecks, constraints.
  * **5.4 Future Improvements.**

-----

## 7\. Project Deliverable

**Submission Requirements:**

1.  **Team Plan and Work Distribution:** Document detailing responsibilities, timeline, and contribution %.
2.  **Project Report:** Jupyter Notebook (.ipynb), executable on Colab, exported to PDF.
3.  **Source Code Package:**
      * All .cpp, .cu, .h files.
      * **README.md** containing: Setup instructions, compilation commands, execution instructions, hardware requirements, expected outputs.
      * Trained model weights (or link).
4.  **Presentation Video (15-20 minutes):**
      * Upload to YouTube (Unlisted).
      * **Content:** Problem overview (2-3 min), Live Code Demo (5-7 min), Results/Analysis (5-7 min), Optimizations/Lessons (3-5 min).