# GPU Autoencoder - Giải thích kỹ thuật
> **Mục đích:** Tài liệu giải thích chuyên môn chi tiết, tổ chức theo các Phase trong notebook report_final.ipynb
> **Đối tượng:** Người đọc muốn hiểu sâu về kỹ thuật implementation

---

# Phase 1: CPU Baseline Implementation

## Giải thích chuyên môn

### 1.1 Kiến trúc Autoencoder

#### CPU Architecture (Phase 1)
<p align="center">
  <img src="../assets/img/cpu_architecture.jpg" alt="CPU Architecture" width="100%">
</p>

**Thiết kế đơn giản cho CPU Baseline:**
- Sử dụng kiến trúc **Conv + ReLU** cơ bản, không có BatchNorm
- Mục đích: Tạo baseline để verify correctness và đo performance
- Kiến trúc đơn giản giúp debug dễ dàng và so sánh output trực tiếp giữa CPU và GPU

**Cấu trúc mạng CPU:**
```
INPUT (3×32×32) → ENCODER → LATENT (128×8×8) → DECODER → OUTPUT (3×32×32)
```

**Encoder (Feature Extraction):**
| Layer | Input → Output | Parameters |
|-------|----------------|------------|
| Conv2D + ReLU | 3×32×32 → 256×32×32 | 3×3 kernel, pad=1 |
| MaxPool2D | 256×32×32 → 256×16×16 | 2×2, stride=2 |
| Conv2D + ReLU | 256×16×16 → 128×16×16 | 3×3 kernel, pad=1 |
| MaxPool2D | 128×16×16 → 128×8×8 | 2×2, stride=2 |

**Decoder (Reconstruction):**
| Layer | Input → Output | Parameters |
|-------|----------------|------------|
| Conv2D + ReLU | 128×8×8 → 128×8×8 | 3×3 kernel, pad=1 |
| Upsample | 128×8×8 → 128×16×16 | 2× nearest neighbor |
| Conv2D + ReLU | 128×16×16 → 256×16×16 | 3×3 kernel, pad=1 |
| Upsample | 256×16×16 → 256×32×32 | 2× nearest neighbor |
| Conv2D | 256×32×32 → 3×32×32 | 3×3 kernel, pad=1 |

**Latent Space:** 128×8×8 = **8,192 features** (bottleneck representation)

---

#### GPU Architecture (Phase 2-4)
<p align="center">
  <img src="../assets/img/gpu_architecture.jpg" alt="GPU Architecture" width="100%">
</p>

**Cải tiến so với CPU:**
- Thêm **BatchNorm** và **PReLU** layers
- Với sức mạnh GPU, có thể thêm các component phức tạp mà không ảnh hưởng training time

**Các thành phần nâng cao:**

1. **BatchNorm:** 
   - Training mode: Dùng batch mean/variance, update running stats
   - Inference mode: Dùng accumulated running stats
   - Impact: +5-10% accuracy và stabilize training
   - **Bug phổ biến:** Không phân biệt training/inference mode

2. **PReLU (Parametric ReLU):**
   - Công thức: `f(x) = max(0,x) + α[c] * min(0,x)`
   - α learnable per-channel (khởi tạo 0.25)
   - Khác LeakyReLU (fixed slope 0.01)
   - Model tự học optimal slopes cho mỗi channel

3. **Sigmoid (Optional):**
   - Chỉ áp dụng khi train với BCE loss
   - MSE loss output trực tiếp từ decoder (không cần [0,1] range)

**Cấu trúc mạng GPU:**
```
INPUT (3×32×32) → ENCODER → LATENT (128×8×8) → DECODER → OUTPUT (3×32×32) [→ Sigmoid]
```

**Encoder (Feature Extraction):**
| Layer | Input → Output | Parameters |
|-------|----------------|------------|
| Conv2D | 3×32×32 → 256×32×32 | 3×3 kernel, pad=1 |
| BatchNorm + PReLU | 256×32×32 → 256×32×32 | γ, β, α learnable |
| MaxPool2D | 256×32×32 → 256×16×16 | 2×2, stride=2 |
| Conv2D | 256×16×16 → 128×16×16 | 3×3 kernel, pad=1 |
| BatchNorm + PReLU | 128×16×16 → 128×16×16 | γ, β, α learnable |
| MaxPool2D | 128×16×16 → 128×8×8 | 2×2, stride=2 |

**Decoder (Reconstruction):**
| Layer | Input → Output | Parameters |
|-------|----------------|------------|
| Conv2D | 128×8×8 → 128×8×8 | 3×3 kernel, pad=1 |
| BatchNorm + PReLU | 128×8×8 → 128×8×8 | γ, β, α learnable |
| Upsample | 128×8×8 → 128×16×16 | 2× nearest neighbor |
| Conv2D | 128×16×16 → 256×16×16 | 3×3 kernel, pad=1 |
| BatchNorm + PReLU | 256×16×16 → 256×16×16 | γ, β, α learnable |
| Upsample | 256×16×16 → 256×32×32 | 2× nearest neighbor |
| Conv2D | 256×32×32 → 3×32×32 | 3×3 kernel, pad=1 |
| Sigmoid (BCE only) | 3×32×32 → 3×32×32 | Output ∈ [0,1] |

**Latent Space:** 128×8×8 = **8,192 features** (giống CPU, để so sánh công bằng)

### 1.2 Convolution Implementation (CPU)

**Complexity:** O(N · C_out · C_in · H · W · K²)

```cpp
void conv2d_forward_cpu(float* input, float* weights, float* output,
                        int batch, int in_c, int out_c, int h, int w, int k) {
    #pragma omp parallel for collapse(4)
    for (int b = 0; b < batch; b++)           // Batch
        for (int oc = 0; oc < out_c; oc++)    // Output channels
            for (int oh = 0; oh < h; oh++)    // Output height
                for (int ow = 0; ow < w; ow++) {  // Output width
                    float sum = 0.0f;
                    for (int ic = 0; ic < in_c; ic++)     // Input channels
                        for (int kh = 0; kh < k; kh++)    // Kernel height
                            for (int kw = 0; kw < k; kw++) // Kernel width
                                sum += input[...] * weights[...];
                    output[...] = sum;
                }
}
```

**Bottlenecks của CPU:**
- **Sequential memory access:** Cache misses trên large tensors
- **Limited parallelism:** OpenMP chỉ parallelize được outer loops
- **No SIMD optimization:** Compiler auto-vectorization bị giới hạn
- **Memory bandwidth:** CPU ~50 GB/s vs GPU ~900 GB/s

### 1.3 Training Configuration

| Parameter | Value | Lý do |
|-----------|-------|-------|
| Epochs | 20 | Đủ để converge với small dataset |
| Batch size | 32 | Cân bằng memory và gradient stability |
| Learning rate | 0.001 | Standard cho SGD |
| Loss function | MSE | Baseline cho reconstruction |
| Optimizer | SGD | Baseline, sau đó upgrade lên AdamW |

### 1.4 Kết quả Phase 1

- **Training images:** 100 (scaled to 50,000)
- **Avg Time/Epoch (scaled 50K):** ~260 minutes = 4.3 hours
- **Total Training (scaled 50K):** ~87 hours
- **Final Loss:** ~0.26 (MSE)

**Phân tích:**
- Loss oscillates (0.25-0.29) do dataset quá nhỏ (100 images)
- Không có clear convergence trend
- Convolution chiếm ~90% compute time
- **Kết luận:** CPU baseline không viable cho production, nhưng cần thiết để verify correctness

---

# Phase 2: GPU Basic Implementation

## Giải thích chuyên môn

### 2.1 Parallelization Strategy

**Thread Mapping:**
- 1 CUDA thread = 1 output element
- Thread blocks: `dim3(16, 16)` = 256 threads per block
- Grid size: calculated based on output dimensions

**Kernel Design:**
```cpp
__global__ void conv2d_forward_kernel(float* input, float* weights, float* output,
                                       int batch, int in_c, int out_c, int h, int w, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = idx / (out_c * h * w);
    int oc = (idx / (h * w)) % out_c;
    int oh = (idx / w) % h;
    int ow = idx % w;
    
    float sum = 0.0f;
    for (int ic = 0; ic < in_c; ic++)
        for (int kh = 0; kh < k; kh++)
            for (int kw = 0; kw < k; kw++)
                sum += input[...] * weights[...];
    output[idx] = sum;
}
```

### 2.2 Các Kernel Implementations

| Kernel | Thread Mapping | Description |
|--------|----------------|-------------|
| `conv2d_forward` | 1 thread → 1 output pixel | Each thread computes full convolution sum |
| `maxpool_forward` | 1 thread → 1 pooled output | Thread finds max in 2×2 window |
| `relu_forward` | 1 thread → 1 element | Simple max(0, x) |
| `upsample_forward` | 1 thread → 1 output pixel | Copy from nearest input |

### 2.3 BatchNorm Fix (Critical Bug)

**Vấn đề:** Implementation ban đầu luôn dùng running_mean/var cho cả training lẫn inference.

**Cơ chế BatchNorm:**
```
y = γ * (x - μ) / √(σ² + ε) + β
```

**Training mode:**
```cuda
__global__ void batchnorm_compute_mean_kernel(const float* input, float* mean, int N, int C, int H, int W) {
    int c = blockIdx.x;
    extern __shared__ float sdata[];
    // Parallel reduction để tính mean
    if (tid == 0) mean[c] = sdata[0] / total;
}

__global__ void batchnorm_update_running_kernel(float* running, const float* batch, float momentum, int C) {
    running[c] = momentum * running[c] + (1.0f - momentum) * batch[c];
}
```

**Fix:**
- **Training:** Tính mean/var từ batch hiện tại, update running stats với momentum 0.1
- **Inference:** Sử dụng running stats đã tích lũy
- **Impact:** +5-10% accuracy

### 2.4 AdamW Optimizer

**Công thức:**
```
m = β1*m + (1-β1)*g           // Momentum (first moment)
v = β2*v + (1-β2)*g²          // Velocity (second moment)
m_hat = m / (1 - β1^t)        // Bias correction
v_hat = v / (1 - β2^t)
params = params*(1 - lr*λ) - lr*m_hat / (√v_hat + ε)  // AdamW update
```

**Implementation:**
```cuda
__global__ void adamw_update_kernel(float* params, const float* grads, float* m, float* v, 
    float lr, float beta1, float beta2, float eps, float weight_decay, 
    float bias_correction1, float bias_correction2, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = grads[idx];
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        float m_hat = m[idx] / bias_correction1;
        float v_hat = v[idx] / bias_correction2;
        params[idx] = params[idx] * (1.0f - lr * weight_decay) - lr * m_hat / (sqrtf(v_hat) + eps);
    }
}
```

**Config:** `β1=0.9, β2=0.999, ε=1e-8, weight_decay=1e-4`

**So sánh với SGD:**
| Aspect | SGD | AdamW |
|--------|-----|-------|
| Learning rate | Fixed cho tất cả params | Adaptive per-param |
| Weight decay | Coupled với gradient | Decoupled |
| Memory | 1x params | 3x params (m, v) |
| Convergence | Chậm | Nhanh 2-5x |

### 2.5 Learning Rate Schedule

**Warmup + Cosine Annealing:**
```cpp
float get_scheduled_lr(float base_lr, int epoch, int total_epochs) {
    const int warmup_epochs = 5;
    if (epoch < warmup_epochs) {
        // Linear warmup: 20% → 100%
        return base_lr * static_cast<float>(epoch + 1) / static_cast<float>(warmup_epochs);
    }
    // Cosine annealing: 100% → 0.1%
    const float min_lr = base_lr * 0.001f;
    int adjusted_epoch = epoch - warmup_epochs;
    int adjusted_total = total_epochs - warmup_epochs;
    float cos_val = cosf(3.14159265f * static_cast<float>(adjusted_epoch) / static_cast<float>(adjusted_total));
    return min_lr + 0.5f * (base_lr - min_lr) * (1.0f + cos_val);
}
```

**Schedule:**
```
Epoch 1-5:  Linear warmup (20% → 100%)
Epoch 6+:   Cosine decay (100% → 0.1%)
```

### 2.6 PReLU (Parametric ReLU)

**Công thức:** `f(x) = max(0,x) + α[c] * min(0,x)`

```cuda
__global__ void prelu_forward_kernel(const float* input, float* output,
    const float* alpha, int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int c = (idx / (H * W)) % C;
    float x = input[idx];
    output[idx] = (x > 0.0f) ? x : alpha[c] * x;
}

__global__ void prelu_backward_kernel(const float* input, const float* grad_output,
    float* grad_input, const float* alpha, float* grad_alpha, int N, int C, int H, int W) {
    int c = (idx / (H * W)) % C;
    float x = input[idx], go = grad_output[idx];
    grad_input[idx] = (x > 0.0f) ? go : alpha[c] * go;
    if (x <= 0.0f) {
        atomicAdd(&grad_alpha[c], go * x);  // Atomic vì nhiều elements contribute
    }
}
```

**So sánh:**
| | ReLU | LeakyReLU | PReLU |
|--|------|-----------|-------|
| Negative slope | 0 | 0.01 (fixed) | Learnable per channel |
| Dying neurons | Yes | No | No |
| Extra params | 0 | 0 | num_channels |

### 2.7 Data Augmentation

```cpp
void CIFAR10Dataset::augment_image(float* image, const AugmentConfig& config, std::mt19937& rng) {
    // 1. Random Horizontal Flip (50%)
    if (config.horizontal_flip && dist(rng) < 0.5f) {
        horizontal_flip_image(image);
    }
    
    // 2. Random Crop (padding=4, 32→40→32)
    if (config.random_crop && config.crop_padding > 0) {
        random_crop_image(image, config.crop_padding, rng);
    }
    
    // 3. Cutout (8x8 random erasing, 50%)
    if (config.cutout && config.cutout_size > 0 && prob_dist(rng) < 0.5f) {
        for (int c = 0; c < 3; ++c)
            for (int h = h_start; h < h_end; ++h)
                for (int w = w_start; w < w_end; ++w)
                    image[c * H * W + h * W + w] = 0.0f;
    }
}
```

**Config:**
```cpp
struct AugmentConfig {
    bool horizontal_flip = true;   // 50% chance
    bool random_crop = true;       // Padding=4
    int crop_padding = 4;             
    bool cutout = true;            // 8x8 random erasing
    int cutout_size = 8;              
};
```

### 2.8 Gradient Clipping

```cuda
__global__ void gradient_clip_kernel(float* grad, float max_norm, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad[idx] = fminf(fmaxf(grad[idx], -max_norm), max_norm);
    }
}
```

**Áp dụng:** `max_norm = 1.0`, đặc biệt quan trọng với BCE Loss.

### 2.9 Kết quả Phase 2

- **Training images:** 5,000 (scaled to 50,000)
- **Avg Time/Epoch (scaled 50K):** ~5.1 minutes
- **Total Training (scaled 50K):** ~102 minutes
- **Final Loss:** 0.022 (MSE)
- **Speedup vs CPU:** ~51×

**Phân tích:**
- Rapid convergence từ ~0.08 xuống ~0.022 trong 5 epochs đầu
- Stable plateau sau epoch 10
- GPU naive đã đạt 51× speedup nhờ massive parallelism

**Remaining Bottlenecks:**
- Redundant global memory reads (mỗi input pixel đọc 9× cho 3×3 kernel)
- Uncoalesced memory access
- Low occupancy

---

# Phase 3: GPU Optimization

## Giải thích chuyên môn

### 3.1 cuDNN Convolution

**Workflow:**
```cuda
void gpu_conv2d_forward_cudnn(...) {
    init_cudnn();
    
    // 1. Create descriptors
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W);
    cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_c, in_c, k, k);
    cudnnSetConvolution2dDescriptor(conv_desc, padding, padding, stride, stride, 1, 1, 
                                     CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    
    // 2. Auto-select best algorithm
    cudnnGetConvolutionForwardAlgorithm_v7(cudnn_handle, input_desc, filter_desc, 
                                            conv_desc, output_desc, 1, &returned_algo_count, &algo_perf);
    
    // 3. Allocate workspace
    cudnnGetConvolutionForwardWorkspaceSize(..., algo_perf.algo, &workspace_size);
    cudaMalloc(&d_workspace, workspace_size);
    
    // 4. Execute convolution
    cudnnConvolutionForward(cudnn_handle, &alpha, input_desc, input.d_data, 
                            filter_desc, d_weights, conv_desc, algo_perf.algo, 
                            d_workspace, workspace_size, &beta, output_desc, output.d_data);
    
    // 5. Add bias
    cudnnAddTensor(cudnn_handle, &alpha, bias_desc, d_bias, &beta, output_desc, output.d_data);
}
```

**Backward pass:**
```cuda
void gpu_conv2d_backward_cudnn(...) {
    // Backward Data: dL/dX
    cudnnConvolutionBackwardData(...);
    
    // Backward Filter: dL/dW
    cudnnConvolutionBackwardFilter(...);
    
    // Backward Bias: dL/db
    cudnnConvolutionBackwardBias(...);
}
```

**Algorithms được cuDNN hỗ trợ:**
| Algorithm | Mô tả | Best for |
|-----------|-------|----------|
| GEMM-based | im2col + matrix multiply | General |
| Winograd | Transform domain, giảm phép nhân | 3×3 kernels |
| FFT-based | Frequency domain convolution | Large kernels (7×7+) |

**Build command:**
```bash
nvcc -O3 -std=c++17 -arch=sm_75 --use_fast_math \
    -DUSE_OPTIMIZED_KERNELS -lcublas -lcudnn \
    -o gpu_train_opt src/main_gpu.cu src/layers_gpu.cu ...
```

**Performance:** 10-50× speedup vs naive, ~20-30s/epoch

### 3.2 Tiled Convolution (Shared Memory)

**GPU Memory Hierarchy:**
```
Registers (~1 cycle) → Shared Memory (~5 cycles) → L1/L2 Cache (~50 cycles) → Global Memory (~400-800 cycles)
```

**Implementation:**
```cuda
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

__global__ void conv2d_forward_tiled_kernel(
    const float* input, const float* weights, const float* bias, float* output,
    int batch_size, int in_c, int in_h, int in_w, int out_c, int out_h, int out_w, 
    int k, int stride, int padding) {
    
    extern __shared__ float shared_mem[];
    
    int ow = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int oh = blockIdx.y * TILE_HEIGHT + threadIdx.y;
    int oc = blockIdx.z % out_c, n = blockIdx.z / out_c;
    
    if (ow >= out_w || oh >= out_h || n >= batch_size) return;
    
    float sum = bias[oc];
    
    const int tile_h = TILE_HEIGHT * stride + k - stride;
    const int tile_w = TILE_WIDTH * stride + k - stride;
    float* s_input = shared_mem;
    
    for (int ic = 0; ic < in_c; ++ic) {
        // 1. Cooperative loading to shared memory
        const int in_start_h = blockIdx.y * TILE_HEIGHT * stride - padding;
        const int in_start_w = blockIdx.x * TILE_WIDTH * stride - padding;
        
        for (int load = 0; load < num_loads; ++load) {
            int linear_idx = load * threads_per_block + linear_tid;
            if (linear_idx < tile_size) {
                int sh = linear_idx / tile_w, sw = linear_idx % tile_w;
                int ih = in_start_h + sh, iw = in_start_w + sw;
                s_input[linear_idx] = (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) 
                                      ? input[...] : 0.0f;
            }
        }
        __syncthreads();  // CRITICAL: Wait for all threads
        
        // 2. Compute using shared memory
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
dim3 block(TILE_WIDTH, TILE_HEIGHT);  // 16×16 = 256 threads
dim3 grid((out_w + TILE_WIDTH - 1) / TILE_WIDTH, 
          (out_h + TILE_HEIGHT - 1) / TILE_HEIGHT, 
          batch_size * out_c);

int tile_h = TILE_HEIGHT * stride + k - stride;
int tile_w = TILE_WIDTH * stride + k - stride;
size_t shared_mem_size = tile_h * tile_w * sizeof(float);

conv2d_forward_tiled_kernel<<<grid, block, shared_mem_size>>>(...);
```

**So sánh Naive vs Tiled:**
| Aspect | Naive | Tiled |
|--------|-------|-------|
| Memory Access | Global only (~400-800 cycles) | Shared (~5 cycles) |
| Data Reuse | None | Each input pixel loaded once, used k×k times |
| Coalescing | Random access | Coalesced loading to shared mem |

**Performance:** 3-5× speedup vs naive, ~30-60s/epoch

### 3.3 BCE Loss

**Công thức:**
```
Loss = -[y·log(p) + (1-y)·log(1-p)]
Gradient = (p - y) / (p * (1-p))
```

**Implementation:**
```cuda
__global__ void bce_loss_kernel(const float* output, const float* target, 
    float* partial_sums, size_t n) {
    const float eps = 1e-7f;
    float val = 0.0f;
    if (idx < n) {
        float y = target[idx];
        float p = fmaxf(fminf(output[idx], 1.0f - eps), eps);  // Clamp to [eps, 1-eps]
        val = -(y * logf(p) + (1.0f - y) * logf(1.0f - p));
    }
    // Parallel reduction...
}

__global__ void bce_grad_kernel(const float* output, const float* target, 
    float* grad_output, float scale, size_t n) {
    const float eps = 1e-7f;
    float y = target[idx];
    float p = fmaxf(fminf(output[idx], 1.0f - eps), eps);
    grad_output[idx] = scale * (p - y) / (p * (1.0f - p) + eps);
}
```

**Numerically stable sigmoid:**
```cuda
__global__ void sigmoid_forward_kernel(const float* input, float* output, size_t n) {
    float x = input[idx];
    output[idx] = (x >= 0) ? (1.0f / (1.0f + expf(-x))) 
                           : (expf(x) / (1.0f + expf(x)));
}
```

**So sánh MSE vs BCE:**
| Aspect | MSE | BCE |
|--------|-----|-----|
| Formula | (y - p)² | -[y·log(p) + (1-y)·log(1-p)] |
| Output range | Any | [0, 1] (cần Sigmoid) |
| Gradient behavior | Linear | Stronger near 0 and 1 |
| Reconstruction quality | Blurry | Sharper edges |

### 3.4 Vectorized Operations (float4)

```cuda
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

void gpu_relu_forward_opt(const GPUTensor4D& input, GPUTensor4D& output) {
    size_t n = input.size(), n4 = n / 4;
    if (n % 4 == 0) {
        relu_forward_vectorized_kernel<<<(n4 + 255) / 256, 256>>>(
            reinterpret_cast<const float4*>(input.d_data), 
            reinterpret_cast<float4*>(output.d_data), n4);
    }
}
```

**Benefits:**
- Memory bandwidth: Load/store 4 floats trong 1 transaction
- Instruction throughput: 4× fewer instructions
- Speedup: ~2-4× cho memory-bound operations

### 3.5 Pinned Memory + Double Buffering

```cpp
// Allocate pinned memory
float* h_batch[2];
CUDA_CHECK(cudaMallocHost(&h_batch[0], batch_size * 3 * 32 * 32 * sizeof(float)));
CUDA_CHECK(cudaMallocHost(&h_batch[1], batch_size * 3 * 32 * 32 * sizeof(float)));

// Create streams
cudaStream_t streams[2];
CUDA_CHECK(cudaStreamCreate(&streams[0]));
CUDA_CHECK(cudaStreamCreate(&streams[1]));

// Double buffering loop
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

**Timeline overlap:**
```
Time:     |----1----|----2----|----3----|----4----|
Buffer A: [Transfer] [Train  ] [Transfer] [Train  ]
Buffer B: [Prepare ] [Transfer] [Train  ] [Transfer]
```

**Benefits:**
- Pinned Memory: DMA transfer, không cần CPU intervention
- Double Buffering: Overlap CPU prep và GPU compute
- Speedup: ~1.5-2× cho data loading

### 3.6 Kết quả Phase 3

| Version | Method | Time/Epoch | Speedup vs Naive |
|---------|--------|------------|------------------|
| cuDNN | NVIDIA cuDNN library | ~30s | 50× |
| Tiled | Shared memory tiling | ~30s | 50× |
| BCE | BCE loss + cuML | - | Better reconstruction |

---

# Phase 4: Feature Extraction + SVM Classification

## Giải thích chuyên môn

### 4.1 Feature Extraction từ Encoder

**Encoder-only forward pass:**
```cpp
void GPUAutoencoder::encode(const GPUTensor4D& input, GPUTensor4D& latent) {
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

**Feature extraction loop:**
```cpp
const int feature_dim = 128 * 8 * 8;  // 8192
std::vector<float> train_features(num_train * feature_dim);

GPUTensor4D single_image(1, 3, 32, 32);
GPUTensor4D latent;
std::vector<float> h_latent(feature_dim);

for (int i = 0; i < num_train; ++i) {
    single_image.copy_from_host(train.images.data() + i * 3 * 32 * 32);
    autoencoder.encode(single_image, latent);
    latent.copy_to_host(h_latent.data());
    std::copy(h_latent.begin(), h_latent.end(), 
              train_features.begin() + i * feature_dim);
}
```

**Latent Space Properties:**
- **Dimension:** 128 × 8 × 8 = 8,192 features
- **Semantic richness:** Learned features vs raw pixels
- **Transfer learning:** Pre-trained encoder → downstream classification

### 4.2 L2 Normalization

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

**Tại sao L2 Normalization:**
- **Scale Invariance:** SVM với RBF kernel sensitive với feature scale
- **Unit sphere:** Tất cả vectors có ||x|| = 1
- **Cosine similarity = dot product** sau L2 norm

### 4.3 SVM Classification

```cpp
SVMWrapper svm;
svm.set_C(10.0);                    // Regularization
svm.set_gamma(1.0 / feature_dim);   // RBF gamma = 1/8192

svm.train(train_features.data(), train_labels.data(), num_train, feature_dim);
float accuracy = svm.evaluate(test_features.data(), test_labels.data(), num_test, feature_dim);
```

**RBF Kernel:**
```
K(x, y) = exp(-γ ||x - y||²)
```

**Hyperparameters:**
| Parameter | Value | Ý nghĩa |
|-----------|-------|---------|
| C | 10.0 | Regularization (higher = less regularization) |
| gamma | 1/dim | RBF kernel width (auto-scaled) |

### 4.4 cuML Acceleration

```python
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
---

*Tài liệu giải thích chuyên môn. Cập nhật: 27/12/2025*
