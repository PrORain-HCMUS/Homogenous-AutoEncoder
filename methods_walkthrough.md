# 🎯 GPU Autoencoder Methods Walkthrough

> **Mục đích:** Tài liệu tổng hợp các methods theo từng phase, focus vào Phase 3 (3 versions) và Phase 4.
> **Đối tượng:** Speakers chuẩn bị thuyết trình về project.

---

## 📋 Tổng Quan Project Flow

```
Phase 1: CPU Baseline → Phase 2: Naive GPU → Phase 3: Optimized GPU → Phase 4: SVM Integration
```

| Phase | Mục tiêu | Key Methods |
|-------|----------|-------------|
| 1 | CPU implementation cơ bản | Conv2D, BatchNorm, ReLU naive |
| 2 | Port sang GPU (naive kernels) | CUDA kernels cơ bản |
| **3** | **Tối ưu GPU (3 versions)** | **cuDNN, Tiled Conv, BCE Loss** |
| **4** | **Feature extraction + SVM** | **cuML, PCA, ThunderSVM** |

---

# 🔷 PHASE 1 & 2: Core Training Infrastructure

## 1.1 BatchNorm Fix

### Method: Proper Training vs Inference Mode

**File:** `src/layers_gpu.cu` (lines 162-210)

```cuda
// Training mode: tính batch mean/var
__global__ void batchnorm_compute_mean_kernel(const float* input, float* mean, int N, int C, int H, int W) {
    int c = blockIdx.x;
    extern __shared__ float sdata[];
    // ... parallel reduction để tính mean
    if (tid == 0) mean[c] = sdata[0] / total;
}

// Update running stats với momentum
__global__ void batchnorm_update_running_kernel(float* running, const float* batch, float momentum, int C) {
    running[c] = momentum * running[c] + (1.0f - momentum) * batch[c];
}
```

**Tại sao dùng:**
- **Before:** Luôn dùng running_mean/var → model không học được distribution của batch hiện tại
- **After:** Training dùng batch stats, inference dùng running stats
- **Impact:** +5-10% accuracy

> **🎤 Speaker Notes:**
> - Giải thích sự khác biệt giữa training và inference mode
> - BatchNorm cần tính mean/var từ batch hiện tại khi training để normalize
> - Running stats được update với momentum để dùng khi inference
> - Đây là bug phổ biến khi implement BatchNorm từ đầu

---

## 1.2 AdamW Optimizer

### Method: Adaptive Learning Rate với Weight Decay

**File:** `src/layers_gpu.cu` (lines 244-255)

```cuda
__global__ void adamw_update_kernel(float* params, const float* grads, float* m, float* v, 
    float lr, float beta1, float beta2, float eps, float weight_decay, 
    float bias_correction1, float bias_correction2, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = grads[idx];
        // Momentum: m = β1*m + (1-β1)*g
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        // Velocity: v = β2*v + (1-β2)*g²
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        // Bias correction
        float m_hat = m[idx] / bias_correction1;
        float v_hat = v[idx] / bias_correction2;
        // AdamW update: weight decay applied to params directly
        params[idx] = params[idx] * (1.0f - lr * weight_decay) - lr * m_hat / (sqrtf(v_hat) + eps);
    }
}
```

**Config:** `β1=0.9, β2=0.999, ε=1e-8, weight_decay=1e-4`

**File:** `include/gpu_layer.h` (lines 24-30)
```cpp
struct OptimizerConfig {
    float momentum = 0.9f, weight_decay = 1e-4f;
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    bool use_adamw = true;
    int step = 0;  // For bias correction
};
```

**Tại sao dùng AdamW thay SGD:**
- **Adaptive LR:** Mỗi parameter có learning rate riêng dựa trên gradient history
- **Weight Decay Decoupled:** Không bị ảnh hưởng bởi adaptive scaling
- **Faster Convergence:** +2-5% accuracy, converge nhanh hơn

> **🎤 Speaker Notes:**
> - Adam = Adaptive Moment Estimation, kết hợp Momentum + RMSprop
> - AdamW khác Adam ở chỗ weight decay được apply trực tiếp vào params, không qua gradient
> - Bias correction quan trọng ở early steps vì m, v khởi tạo = 0
> - Tradeoff: Cần thêm memory cho m, v buffers (2x params)

---

## 1.3 Learning Rate Schedule: Warmup + Cosine Annealing

### Method: Gradual Warmup → Smooth Decay

**File:** `src/main_gpu.cu` (lines 20-33)

```cpp
float get_scheduled_lr(float base_lr, int epoch, int total_epochs) {
    const int warmup_epochs = 5;
    if (epoch < warmup_epochs) {
        // Linear warmup: 20% → 40% → 60% → 80% → 100%
        return base_lr * static_cast<float>(epoch + 1) / static_cast<float>(warmup_epochs);
    }
    // Cosine annealing after warmup
    const float min_lr = base_lr * 0.001f;  // 0.1% of base
    int adjusted_epoch = epoch - warmup_epochs;
    int adjusted_total = total_epochs - warmup_epochs;
    float cos_val = cosf(3.14159265f * static_cast<float>(adjusted_epoch) / static_cast<float>(adjusted_total));
    return min_lr + 0.5f * (base_lr - min_lr) * (1.0f + cos_val);
}
```

**Schedule visualization:**
```
Epoch 1-5:  Linear warmup (20% → 100%)
Epoch 6+:   Cosine decay (100% → 0.1%)
```

**Tại sao dùng:**
- **Warmup:** Tránh gradient explosion ở early training khi weights còn random
- **Cosine Annealing:** Smooth decay, không có sudden drops như step decay
- **Impact:** Stable training, better final loss

> **🎤 Speaker Notes:**
> - Warmup đặc biệt quan trọng với large batch sizes
> - Cosine annealing cho phép model "explore" ở đầu, "exploit" ở cuối
> - So sánh với step decay: cosine không có "jumps" gây instability

---

## 1.4 PReLU (Parametric ReLU)

### Method: Learnable Negative Slope

**File:** `src/layers_gpu.cu` (lines 446-506)

```cuda
// Forward: f(x) = max(0,x) + α[c] * min(0,x)
__global__ void prelu_forward_kernel(const float* input, float* output,
    const float* alpha, int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int c = (idx / (H * W)) % C;  // Channel index
    float x = input[idx];
    output[idx] = (x > 0.0f) ? x : alpha[c] * x;  // α is per-channel
}

// Backward: gradient flows through negative values
__global__ void prelu_backward_kernel(const float* input, const float* grad_output,
    float* grad_input, const float* alpha, float* grad_alpha, int N, int C, int H, int W) {
    int c = (idx / (H * W)) % C;
    float x = input[idx], go = grad_output[idx];
    grad_input[idx] = (x > 0.0f) ? go : alpha[c] * go;
    // Gradient w.r.t. alpha (atomic add vì nhiều elements contribute)
    if (x <= 0.0f) {
        atomicAdd(&grad_alpha[c], go * x);
    }
}
```

**Initialization:** `α = 0.25` (PReLU paper)

**So sánh với LeakyReLU:**
| | LeakyReLU | PReLU |
|--|-----------|-------|
| Slope | Fixed (0.01) | Learnable per channel |
| Params | 0 | num_channels |
| Flexibility | Low | High |

> **🎤 Speaker Notes:**
> - PReLU cho phép model tự học slope tối ưu cho mỗi channel
> - atomicAdd cần thiết vì nhiều spatial positions cùng contribute vào grad_alpha[c]
> - Tradeoff: Thêm params nhưng không đáng kể (chỉ = num_channels)

---

## 1.5 Data Augmentation

### Method: Random Transforms để Prevent Overfitting

**File:** `src/dataset.cpp` (lines 155-205)

```cpp
void CIFAR10Dataset::augment_image(float* image, const AugmentConfig& config, std::mt19937& rng) {
    // 1. Random Horizontal Flip (50%)
    if (config.horizontal_flip) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        if (dist(rng) < 0.5f) {
            horizontal_flip_image(image);
        }
    }
    
    // 2. Random Crop with Padding (32 → 40 → crop 32)
    if (config.random_crop && config.crop_padding > 0) {
        random_crop_image(image, config.crop_padding, rng);
    }
    
    // 3. Cutout (Random Erasing) - 50% probability
    if (config.cutout && config.cutout_size > 0) {
        if (prob_dist(rng) < 0.5f) {
            // Zero out 8x8 region at random position
            for (int c = 0; c < 3; ++c)
                for (int h = h_start; h < h_end; ++h)
                    for (int w = w_start; w < w_end; ++w)
                        image[c * H * W + h * W + w] = 0.0f;
        }
    }
}
```

**File:** `include/dataset.h` (lines 9-22)
```cpp
struct AugmentConfig {
    bool horizontal_flip = true;      // 50% chance
    bool random_crop = true;          // Padding=4
    int crop_padding = 4;             
    bool cutout = true;               // 8x8 random erasing
    int cutout_size = 8;              
};
```

**Tại sao dùng:**
- **Horizontal Flip:** Tăng data diversity, safe cho CIFAR-10 classes
- **Random Crop:** Translation invariance
- **Cutout:** Forces model to use global features, not rely on single region
- **Impact:** +2-5% accuracy, prevents overfitting

> **🎤 Speaker Notes:**
> - Augmentation là "free data" - tăng effective dataset size
> - Cutout đặc biệt hiệu quả cho small images như CIFAR-10
> - Augmentation chỉ apply khi training, không apply khi inference

---

## 1.6 Gradient Clipping

### Method: Prevent Gradient Explosion

**File:** `src/layers_gpu.cu` (lines 649-660)

```cuda
__global__ void gradient_clip_kernel(float* grad, float max_norm, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Clamp gradient to [-max_norm, max_norm]
        grad[idx] = fminf(fmaxf(grad[idx], -max_norm), max_norm);
    }
}

// Applied in loss functions
float gpu_mse_loss_with_grad(...) {
    mse_grad_kernel<<<...>>>(output, target, grad_output, 2.0f / n, n);
    gpu_clip_gradients(grad_output, 1.0f);  // max_norm = 1.0
    return loss;
}
```

**Tại sao dùng:**
- **BCE Loss:** Có thể tạo very large gradients khi prediction gần 0 hoặc 1
- **Stability:** Prevents NaN/Inf trong training
- **Tradeoff:** Có thể slow down learning nếu clip quá aggressive

> **🎤 Speaker Notes:**
> - Gradient clipping là safety net, không phải solution cho bad architecture
> - max_norm = 1.0 là conservative choice, có thể tune
> - Per-element clipping (dùng ở đây) vs global norm clipping (PyTorch default)

---

# 🔷 PHASE 3: GPU Optimization (3 Versions)

## ⚡ Overview: 3 Convolution Implementations

| Version | File | Method | Speedup | Use Case |
|---------|------|--------|---------|----------|
| **cuDNN** | `Phase3_Colab.ipynb` | NVIDIA cuDNN library | 10-50x | Production, best performance |
| **Tiled** | `Phase3_Colab_Tiled.ipynb` | Shared memory tiling | 3-5x | Educational, no external deps |
| **BCE** | `phase3-4-kaggle-bce-cuml.ipynb` | BCE loss + cuML | - | Better reconstruction |

---

## 3.1 Version 1: cuDNN Convolution

### Method: NVIDIA's Optimized Convolution Library

**File:** `src/layers_gpu_opt.cu` (lines 21-77)

```cuda
void gpu_conv2d_forward_cudnn(const GPUTensor4D& input, const float* d_weights, 
    const float* d_bias, GPUTensor4D& output, int in_c, int out_c, int k, int stride, int padding) {
    init_cudnn();
    
    // 1. Create descriptors
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W);
    cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_c, in_c, k, k);
    cudnnSetConvolution2dDescriptor(conv_desc, padding, padding, stride, stride, 1, 1, 
                                     CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    
    // 2. Auto-select best algorithm
    cudnnConvolutionFwdAlgoPerf_t algo_perf;
    cudnnGetConvolutionForwardAlgorithm_v7(cudnn_handle, input_desc, filter_desc, 
                                            conv_desc, output_desc, 1, &returned_algo_count, &algo_perf);
    
    // 3. Allocate workspace
    size_t workspace_size;
    cudnnGetConvolutionForwardWorkspaceSize(..., algo_perf.algo, &workspace_size);
    cudaMalloc(&d_workspace, workspace_size);
    
    // 4. Execute convolution
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(cudnn_handle, &alpha, input_desc, input.d_data, 
                            filter_desc, d_weights, conv_desc, algo_perf.algo, 
                            d_workspace, workspace_size, &beta, output_desc, output.d_data);
    
    // 5. Add bias
    cudnnAddTensor(cudnn_handle, &alpha, bias_desc, d_bias, &beta, output_desc, output.d_data);
}
```

**Backward pass:** `src/layers_gpu_opt.cu` (lines 46-77)
```cuda
void gpu_conv2d_backward_cudnn(...) {
    // Backward Data: dL/dX
    cudnnConvolutionBackwardData(cudnn_handle, &alpha, filter_desc, d_weights, 
                                  grad_output_desc, grad_output.d_data, conv_desc, 
                                  data_algo, d_workspace, workspace_size, 
                                  &beta, grad_input_desc, grad_input.d_data);
    
    // Backward Filter: dL/dW
    cudnnConvolutionBackwardFilter(cudnn_handle, &alpha, input_desc, input.d_data,
                                    grad_output_desc, grad_output.d_data, conv_desc,
                                    filter_algo, d_workspace, workspace_size,
                                    &beta, filter_desc, d_grad_weights);
    
    // Backward Bias: dL/db
    cudnnConvolutionBackwardBias(cudnn_handle, &alpha, grad_output_desc, 
                                  grad_output.d_data, &beta, bias_desc, d_grad_bias);
}
```

**Build command (Colab):**
```bash
nvcc -O3 -std=c++17 -arch=sm_75 --use_fast_math \
    -DUSE_OPTIMIZED_KERNELS -lcublas -lcudnn \
    -o gpu_train_opt src/main_gpu.cu src/layers_gpu.cu ...
```

**Tại sao dùng cuDNN:**
- **Auto Algorithm Selection:** cuDNN tự chọn algorithm tối ưu (Winograd, FFT, GEMM-based)
- **Highly Optimized:** NVIDIA engineers đã optimize cho từng GPU architecture
- **Speedup:** 10-50x so với naive implementation

**Bottlenecks & Tradeoffs:**
- **Workspace Memory:** Một số algorithms cần extra memory
- **Initialization Overhead:** cudnnCreate() tốn thời gian, nên reuse handle
- **Dependency:** Cần NVIDIA GPU + cuDNN library

> **🎤 Speaker Notes:**
> - cuDNN là "black box" - ta không biết chính xác algorithm nào được chọn
> - Algorithm selection có thể khác nhau giữa các GPU architectures
> - Winograd: tốt cho 3x3 kernels, FFT: tốt cho large kernels
> - Demo: So sánh time/epoch giữa naive và cuDNN

---

## 3.2 Version 2: Tiled Convolution (Shared Memory)

### Method: Manual Optimization với Shared Memory

**File:** `src/layers_gpu_opt.cu` (lines 126-156)

```cuda
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

__global__ void conv2d_forward_tiled_kernel(
    const float* input, const float* weights, const float* bias, float* output,
    int batch_size, int in_c, int in_h, int in_w, int out_c, int out_h, int out_w, 
    int k, int stride, int padding) {
    
    extern __shared__ float shared_mem[];  // Dynamic shared memory
    
    int ow = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int oh = blockIdx.y * TILE_HEIGHT + threadIdx.y;
    int oc = blockIdx.z % out_c, n = blockIdx.z / out_c;
    
    if (ow >= out_w || oh >= out_h || n >= batch_size) return;
    
    float sum = bias[oc];
    
    // Tile dimensions for input
    const int tile_h = TILE_HEIGHT * stride + k - stride;
    const int tile_w = TILE_WIDTH * stride + k - stride;
    float* s_input = shared_mem;
    
    for (int ic = 0; ic < in_c; ++ic) {
        // 1. Cooperative loading: All threads load input tile to shared memory
        const int in_start_h = blockIdx.y * TILE_HEIGHT * stride - padding;
        const int in_start_w = blockIdx.x * TILE_WIDTH * stride - padding;
        
        for (int load = 0; load < num_loads; ++load) {
            int linear_idx = load * threads_per_block + linear_tid;
            if (linear_idx < tile_size) {
                int sh = linear_idx / tile_w, sw = linear_idx % tile_w;
                int ih = in_start_h + sh, iw = in_start_w + sw;
                // Boundary check + load
                s_input[linear_idx] = (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) 
                                      ? input[...] : 0.0f;
            }
        }
        __syncthreads();  // Wait for all threads to finish loading
        
        // 2. Compute convolution using shared memory
        const int local_h = threadIdx.y * stride;
        const int local_w = threadIdx.x * stride;
        for (int kh = 0; kh < k; ++kh) 
            for (int kw = 0; kw < k; ++kw) 
                sum += s_input[(local_h + kh) * tile_w + local_w + kw] * weights[...];
        
        __syncthreads();  // Sync before next channel
    }
    
    output[...] = sum;
}
```

**Launch configuration:**
```cuda
void gpu_conv2d_forward_tiled(...) {
    dim3 block(TILE_WIDTH, TILE_HEIGHT);  // 16x16 = 256 threads
    dim3 grid((out_w + TILE_WIDTH - 1) / TILE_WIDTH, 
              (out_h + TILE_HEIGHT - 1) / TILE_HEIGHT, 
              batch_size * out_c);
    
    int tile_h = TILE_HEIGHT * stride + k - stride;
    int tile_w = TILE_WIDTH * stride + k - stride;
    size_t shared_mem_size = tile_h * tile_w * sizeof(float);
    
    conv2d_forward_tiled_kernel<<<grid, block, shared_mem_size>>>(...);
}
```

**Build command (Colab):**
```bash
nvcc -O3 -std=c++17 -arch=sm_75 --use_fast_math \
    -DUSE_OPTIMIZED_KERNELS -DUSE_TILED_CONV \
    -lcublas -lcurand \
    -o gpu_train_tiled src/main_gpu.cu ...
```

**Tại sao Tiled nhanh hơn Naive:**

| Aspect | Naive | Tiled |
|--------|-------|-------|
| Memory Access | Global only (~400-800 cycles) | Shared (~5 cycles) |
| Data Reuse | None | Each input pixel loaded once, used k×k times |
| Coalescing | Random access | Coalesced loading to shared mem |

**Memory Hierarchy:**
```
Registers (fastest) → Shared Memory → L1/L2 Cache → Global Memory (slowest)
```

**Bottlenecks & Tradeoffs:**
- **Shared Memory Limited:** 48KB per SM (Turing), limits tile size
- **Bank Conflicts:** Nếu access pattern không tốt, performance giảm
- **Complexity:** Code phức tạp hơn nhiều so với naive

> **🎤 Speaker Notes:**
> - Shared memory là key optimization cho GPU
> - `__syncthreads()` rất quan trọng - đảm bảo tất cả threads đã load xong
> - Tile size 16x16 là common choice, balance giữa occupancy và reuse
> - Demo: Vẽ diagram memory access pattern

---

## 3.3 Version 3: BCE Loss + cuML

### Method: Binary Cross-Entropy cho Reconstruction

**File:** `src/layers_gpu.cu` (lines 701-756)

```cuda
// BCE Loss: -[y*log(p) + (1-y)*log(1-p)]
__global__ void bce_loss_kernel(const float* output, const float* target, 
    float* partial_sums, size_t n) {
    const float eps = 1e-7f;  // Numerical stability
    float val = 0.0f;
    if (idx < n) {
        float y = target[idx];
        float p = fmaxf(fminf(output[idx], 1.0f - eps), eps);  // Clamp to [eps, 1-eps]
        val = -(y * logf(p) + (1.0f - y) * logf(1.0f - p));
    }
    // Parallel reduction...
}

// BCE Gradient: (p - y) / (p * (1-p))
__global__ void bce_grad_kernel(const float* output, const float* target, 
    float* grad_output, float scale, size_t n) {
    const float eps = 1e-7f;
    float y = target[idx];
    float p = fmaxf(fminf(output[idx], 1.0f - eps), eps);
    grad_output[idx] = scale * (p - y) / (p * (1.0f - p) + eps);
}
```

**Sigmoid activation (required for BCE):**
```cuda
// Numerically stable sigmoid
__global__ void sigmoid_forward_kernel(const float* input, float* output, size_t n) {
    float x = input[idx];
    // Avoid overflow: use different formula for positive/negative
    output[idx] = (x >= 0) ? (1.0f / (1.0f + expf(-x))) 
                           : (expf(x) / (1.0f + expf(x)));
}
```

**So sánh MSE vs BCE:**

| | MSE | BCE |
|--|-----|-----|
| Formula | (y - p)² | -[y·log(p) + (1-y)·log(1-p)] |
| Output range | Any | [0, 1] (cần Sigmoid) |
| Gradient behavior | Linear | Stronger near 0 and 1 |
| Use case | General regression | Pixel-wise reconstruction |

**Tại sao dùng BCE cho Autoencoder:**
- **Pixel values [0,1]:** CIFAR-10 normalized, BCE phù hợp
- **Sharper gradients:** BCE penalize wrong predictions mạnh hơn
- **Better reconstruction:** Đặc biệt cho edges và details

**Bottleneck:**
- **Gradient explosion:** BCE gradient có thể rất lớn → cần gradient clipping
- **Numerical stability:** Cần epsilon để tránh log(0)

> **🎤 Speaker Notes:**
> - BCE thường dùng cho classification, nhưng cũng tốt cho reconstruction
> - Sigmoid + BCE = stable combination
> - Demo: So sánh reconstructed images giữa MSE và BCE

---

## 3.4 Additional Phase 3 Optimizations

### 3.4.1 Vectorized Operations (float4)

**File:** `src/layers_gpu_opt.cu` (lines 171-193)

```cuda
// Process 4 floats at once
__global__ void relu_forward_vectorized_kernel(const float4* input, float4* output, size_t n4) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) { 
        float4 in = input[idx]; 
        output[idx] = make_float4(
            (in.x > 0.0f) ? in.x : 0.01f * in.x,
            (in.y > 0.0f) ? in.y : 0.01f * in.y,
            (in.z > 0.0f) ? in.z : 0.01f * in.z,
            (in.w > 0.0f) ? in.w : 0.01f * in.w);
    }
}

// Usage
void gpu_relu_forward_opt(const GPUTensor4D& input, GPUTensor4D& output) {
    size_t n = input.size(), n4 = n / 4;
    if (n % 4 == 0) {
        relu_forward_vectorized_kernel<<<(n4 + 255) / 256, 256>>>(
            reinterpret_cast<const float4*>(input.d_data), 
            reinterpret_cast<float4*>(output.d_data), n4);
    }
}
```

**Tại sao dùng float4:**
- **Memory bandwidth:** Load/store 4 floats trong 1 transaction
- **Instruction throughput:** 4x fewer instructions
- **Speedup:** ~2-4x cho memory-bound operations

### 3.4.2 Pinned Memory + Double Buffering

**File:** `src/main_gpu.cu` (lines 158-196)

```cpp
// Allocate pinned (page-locked) memory
float* h_batch[2];
CUDA_CHECK(cudaMallocHost(&h_batch[0], batch_size * 3 * 32 * 32 * sizeof(float)));
CUDA_CHECK(cudaMallocHost(&h_batch[1], batch_size * 3 * 32 * 32 * sizeof(float)));

// Create streams for overlap
cudaStream_t streams[2];
CUDA_CHECK(cudaStreamCreate(&streams[0]));
CUDA_CHECK(cudaStreamCreate(&streams[1]));

// Double buffering: overlap transfer and compute
for (int batch = 0; batch < num_batches; ++batch) {
    int curr_buf = batch % 2, next_buf = (batch + 1) % 2;
    
    // Async transfer current batch
    cudaMemcpyAsync(gpu_batch[curr_buf].d_data, h_batch[curr_buf], 
                    batch_size * sizeof(float), cudaMemcpyHostToDevice, streams[curr_buf]);
    
    // Prepare next batch on CPU while GPU is busy
    if (batch + 1 < num_batches) {
        // Copy next batch data to h_batch[next_buf]
        // Apply augmentation
    }
    
    // Wait and compute
    cudaStreamSynchronize(streams[curr_buf]);
    float loss = autoencoder.train_step_momentum(gpu_batch[curr_buf], ...);
}
```

**Tại sao dùng:**
- **Pinned Memory:** DMA transfer, không cần CPU intervention
- **Double Buffering:** Overlap CPU prep và GPU compute
- **Speedup:** ~1.5-2x cho data loading

> **🎤 Speaker Notes:**
> - Pinned memory "pins" pages in RAM, không bị swap
> - Double buffering là classic technique trong graphics/compute
> - Streams cho phép concurrent operations trên GPU

---

# 🔷 PHASE 4: Feature Extraction + SVM Classification

## 4.1 Feature Extraction từ Encoder

### Method: Sử dụng Latent Space làm Features

**File:** `src/gpu_autoencoder.cu` (lines 53-71)

```cpp
void GPUAutoencoder::encode(const GPUTensor4D& input, GPUTensor4D& latent) {
    // Encoder path only (không qua decoder)
    gpu_conv2d_forward_cudnn_wrapper(input, conv1_, x1_);  // 3→256
    bn1_.forward(x1_, x2_, false);
    prelu1_.forward(x2_, x3_);
    pool1_.forward(x3_, x4_);                              // 32→16
    
    gpu_conv2d_forward_cudnn_wrapper(x4_, conv2_, x5_);    // 256→128
    bn2_.forward(x5_, x6_, false);
    prelu2_.forward(x6_, x7_);
    pool2_.forward(x7_, latent);                           // 16→8
    
    // Output: latent = [batch, 128, 8, 8] = 8192 features
}
```

**File:** `src/main_phase4.cu` (lines 204-223)

```cpp
// Feature extraction loop
const int feature_dim = 128 * 8 * 8;  // 8192
std::vector<float> train_features(num_train * feature_dim);

GPUTensor4D single_image(1, 3, 32, 32);
GPUTensor4D latent;
std::vector<float> h_latent(feature_dim);

for (int i = 0; i < num_train; ++i) {
    // Copy image to GPU
    single_image.copy_from_host(train.images.data() + i * 3 * 32 * 32);
    
    // Extract features
    autoencoder.encode(single_image, latent);
    
    // Copy back to CPU
    latent.copy_to_host(h_latent.data());
    std::copy(h_latent.begin(), h_latent.end(), 
              train_features.begin() + i * feature_dim);
}
```

**Tại sao dùng Latent Space:**
- **Dimensionality Reduction:** 3×32×32 = 3072 → 128×8×8 = 8192 (nhưng semantic richer)
- **Learned Features:** Autoencoder học features hữu ích cho reconstruction
- **Transfer Learning:** Features này có thể dùng cho classification

> **🎤 Speaker Notes:**
> - Latent space chứa "compressed representation" của image
> - Features này capture high-level patterns, không phải raw pixels
> - Có thể visualize latent space với t-SNE/UMAP

---

## 4.2 L2 Normalization

### Method: Unit Vector Normalization

**File:** `src/main_phase4.cu` (lines 55-67)

```cpp
void l2_normalize(float* features, int num_samples, int dim) {
    for (int i = 0; i < num_samples; ++i) {
        float* vec = features + i * dim;
        
        // Compute L2 norm
        float norm = 0.0f;
        for (int j = 0; j < dim; ++j) {
            norm += vec[j] * vec[j];
        }
        norm = std::sqrt(norm) + 1e-8f;  // Epsilon for stability
        
        // Normalize
        for (int j = 0; j < dim; ++j) {
            vec[j] /= norm;
        }
    }
}
```

**Tại sao dùng L2 Normalization:**
- **Scale Invariance:** SVM với RBF kernel sensitive với feature scale
- **Better Kernel:** Cosine similarity sau L2 norm = dot product
- **Numerical Stability:** Tránh overflow trong kernel computation

> **🎤 Speaker Notes:**
> - L2 norm đưa tất cả vectors lên unit sphere
> - RBF kernel: K(x,y) = exp(-γ||x-y||²), cần features cùng scale
> - Alternative: StandardScaler (zero mean, unit variance)

---

## 4.3 SVM Classification

### Method: RBF Kernel SVM

**File:** `src/main_phase4.cu` (lines 330-342)

```cpp
// SVM Training
SVMWrapper svm;
svm.set_C(10.0);                    // Regularization
svm.set_gamma(1.0 / feature_dim);   // RBF gamma = 1/8192

svm.train(train_features.data(), train_labels.data(), num_train, feature_dim);

// Evaluation
float accuracy = svm.evaluate(test_features.data(), test_labels.data(), num_test, feature_dim);
```

**SVM Parameters:**
- **C = 10.0:** Regularization, higher = less regularization
- **gamma = 1/dim:** RBF kernel width, auto-scaled
- **Kernel:** RBF (Radial Basis Function)

**RBF Kernel:**
```
K(x, y) = exp(-γ ||x - y||²)
```

> **🎤 Speaker Notes:**
> - SVM tìm hyperplane tối ưu để separate classes
> - RBF kernel cho phép non-linear decision boundary
> - C và gamma là 2 hyperparameters quan trọng nhất

---

## 4.4 cuML Acceleration (Notebook Version)

### Method: GPU-accelerated ML với RAPIDS cuML

**File:** `notebooks/phase3-4-kaggle-bce-cuml.ipynb`

```python
# cuML thay thế sklearn
from cuml.preprocessing import StandardScaler
from cuml.decomposition import PCA
from cuml.svm import SVC

# GPU-accelerated preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# GPU PCA
pca = PCA(n_components=1024)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# GPU SVM
svm = SVC(C=10.0, kernel='rbf', gamma='scale')
svm.fit(X_train_pca, y_train)
predictions = svm.predict(X_test_pca)
```

**So sánh sklearn vs cuML:**

| Operation | sklearn (CPU) | cuML (GPU) | Speedup |
|-----------|---------------|------------|---------|
| StandardScaler | ~1s | ~0.05s | 20x |
| PCA (8192→1024) | ~30s | ~2s | 15x |
| SVM Training | ~300s | ~10s | 30x |
| **Total** | **~5 min** | **~15 sec** | **~20x** |

**Tại sao dùng PCA:**
- **Dimensionality Reduction:** 8192 → 1024 features
- **Noise Removal:** PCA giữ lại principal components
- **SVM Speed:** SVM training O(n²) với features, giảm features = nhanh hơn

**Bottlenecks & Tradeoffs:**
- **GPU Memory:** cuML cần load toàn bộ data vào GPU memory
- **Accuracy vs Speed:** PCA có thể mất information
- **Dependency:** Cần RAPIDS ecosystem (conda install)

> **🎤 Speaker Notes:**
> - cuML là drop-in replacement cho sklearn
> - RAPIDS ecosystem: cuDF, cuML, cuGraph - all GPU-accelerated
> - Demo: So sánh wall time sklearn vs cuML

---

# 📊 Performance Summary

## Training Time Comparison

| Phase | Method | Time/Epoch | Total (20 epochs) |
|-------|--------|------------|-------------------|
| 2 | Naive GPU | ~120-300s | ~40-100 min |
| 3 (cuDNN) | cuDNN | ~20-30s | ~7-10 min |
| 3 (Tiled) | Shared Memory | ~30-60s | ~10-20 min |

## Accuracy Progression

| Optimization | Expected Impact |
|--------------|-----------------|
| BatchNorm fix | +5-10% |
| AdamW | +2-5% |
| PReLU | +1-2% |
| Data Augmentation | +2-5% |
| BCE Loss | Better reconstruction |
| **SVM on Features** | **~50-60% classification** |

---

# 🎤 Speaker Notes Summary

## Key Talking Points

### Phase 1-2: Foundation
1. **BatchNorm:** Training vs Inference mode - common bug
2. **AdamW:** Adaptive LR + decoupled weight decay
3. **LR Schedule:** Warmup prevents early instability

### Phase 3: Optimization
1. **cuDNN:** Black-box optimization, auto algorithm selection
2. **Tiled Conv:** Shared memory hierarchy, data reuse
3. **BCE Loss:** Better for pixel-wise reconstruction

### Phase 4: Application
1. **Feature Extraction:** Latent space as learned representation
2. **cuML:** GPU-accelerated ML pipeline
3. **SVM:** Classic classifier on deep features

## Demo Suggestions
- [ ] So sánh training curves: naive vs optimized
- [ ] Visualize reconstructed images: MSE vs BCE
- [ ] Show nvprof/nsight output for kernel analysis
- [ ] t-SNE visualization of latent space

## Q&A Preparation
- "Tại sao không dùng PyTorch/TensorFlow?" → Educational purpose, understand low-level
- "cuDNN vs Tiled, khi nào dùng cái nào?" → cuDNN for production, Tiled for learning
- "Accuracy 50-60% có thấp không?" → CIFAR-10 baseline, focus là optimization không phải SOTA

---

*Generated for presentation preparation. Last updated: December 2024*
