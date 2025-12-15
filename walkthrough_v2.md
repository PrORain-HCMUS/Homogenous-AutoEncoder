# Kernel Fusion Optimization Walkthrough

## Tổng kết kết quả

| Version | Thời gian | Cải thiện |
|---------|-----------|-----------|
| **Baseline** (trước fusion) | 59.1s | - |
| + Forward BN+PReLU fusion | 59.1s | 0% |
| + Backward PReLU+BN fusion | 52.2s | -11.7% |
| + Shared memory reduction | 43.0s | **-27.2%** |
| + Fused mean+var kernel | 43.0s | ~0.5% (measurement variance) |

---

## Chi tiết các thay đổi

### 1. Forward Pass: BatchNorm + PReLU Fusion

**File:** [layers_gpu_opt.cu](file:///c:/Users/PC/Desktop/mylove/encoder/src/layers_gpu_opt.cu)

**Kernel mới (Lines 543-567):**
```cuda
__global__ void batchnorm_prelu_fused_forward_kernel(
    const float* input, float* output,
    const float* bn_gamma, const float* bn_beta,
    const float* bn_mean, const float* bn_var,
    const float* prelu_alpha,
    int N, int C, int H, int W, float eps
) {
    // Fuses: BN normalize + PReLU activation
    float bn_result = bn_gamma[c] * (x - bn_mean[c]) * inv_std + bn_beta[c];
    output[idx] = (bn_result > 0.0f) ? bn_result : prelu_alpha[c] * bn_result;
}
```

**Wrapper (Lines 570-610):**
- [gpu_batchnorm_prelu_fused_forward()](file:///c:/Users/PC/Desktop/mylove/encoder/include/gpu_layer.h#173-175) - Được gọi từ [forward()](file:///c:/Users/PC/Desktop/mylove/encoder/include/gpu_layer.h#148-149) và `encode()`

---

### 2. Backward Pass: PReLU + BatchNorm Fusion (Main Optimization)

**File:** [layers_gpu_opt.cu](file:///c:/Users/PC/Desktop/mylove/encoder/src/layers_gpu_opt.cu)

**Kernel 1 - Compute Sums (Lines 618-688):**
```cuda
__global__ void prelu_batchnorm_compute_sums_fused_kernel(...) {
    // Fuses: PReLU gradient + BN sum_dy/sum_dy_xhat computation
    // Eliminates intermediate gradient tensor
}
```

**Kernel 2 - Backward with Shared Memory Reduction (Lines 690-757):**
```cuda
__global__ void prelu_batchnorm_backward_fused_kernel(...) {
    extern __shared__ float smem[];  // 2 * C floats
    float* s_grad_gamma = smem;
    float* s_grad_beta = smem + C;
    
    // Per-thread accumulation
    atomicAdd(&s_grad_gamma[c], prelu_grad * x_hat);  // Shared memory atomic
    atomicAdd(&s_grad_beta[c], prelu_grad);
    
    __syncthreads();
    
    // Final reduction to global memory (1 atomic per channel per block)
    atomicAdd(&grad_gamma[i], s_grad_gamma[i]);
}
```

**Impact:** Giảm atomicAdd từ N×H×W per channel → ~(total/256) per channel

---

### 3. Fused Mean + Variance Computation

**File:** [layers_gpu.cu](file:///c:/Users/PC/Desktop/mylove/encoder/src/layers_gpu.cu)

**Kernel (Lines 391-437):**
```cuda
__global__ void batchnorm_compute_mean_var_fused_kernel_v2(...) {
    // Single-pass: compute sum and sum_of_squares simultaneously
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;
    
    for (int i = tid; i < total; i += blockDim.x) {
        float val = input[idx];
        local_sum += val;
        local_sq_sum += val * val;  // For variance
    }
    
    // Reduction + final computation
    mean[c] = s_sum[0] / total;
    var[c] = s_sq_sum[0] / total - mean[c] * mean[c];
}
```

**Impact:** Giảm 50% memory reads trong training forward pass

---

### 4. Persistent Memory Buffers

**File:** [layers_gpu_opt.cu](file:///c:/Users/PC/Desktop/mylove/encoder/src/layers_gpu_opt.cu)

**Buffers (Lines 711-732):**
```cuda
static float* d_persistent_sum_dy = nullptr;
static float* d_persistent_sum_dy_xhat = nullptr;

void ensure_bn_backward_buffers(size_t required_size) {
    // Allocate once, reuse across iterations
}
```

**Impact:** Loại bỏ cudaMalloc/cudaFree overhead mỗi backward call

---

### 5. Header Updates

**File:** [gpu_layer.h](file:///c:/Users/PC/Desktop/mylove/encoder/include/gpu_layer.h)

**Getters cho GPUBatchNorm2D (Lines 73-82):**
```cpp
float* get_gamma() const { return d_gamma_; }
float* get_beta() const { return d_beta_; }
float* get_cache_mean() const { return d_cache_mean_; }
float* get_cache_var() const { return d_cache_var_; }
float* get_grad_gamma() const { return d_grad_gamma_; }
float* get_grad_beta() const { return d_grad_beta_; }
```

**Declarations (Lines 173-184):**
```cpp
void gpu_batchnorm_prelu_fused_forward(...);
void gpu_prelu_batchnorm_fused_backward(...);
```

---

### 6. Integration into Autoencoder

**File:** [gpu_autoencoder.cu](file:///c:/Users/PC/Desktop/mylove/encoder/src/gpu_autoencoder.cu)

**Forward (Lines 21-71):** Dùng [gpu_batchnorm_prelu_fused_forward()](file:///c:/Users/PC/Desktop/mylove/encoder/include/gpu_layer.h#173-175) cho inference

**Train Step Backward (Lines 152-199):**
```cpp
#ifdef USE_OPTIMIZED_KERNELS
    gpu_prelu_batchnorm_fused_backward(x13_, x14_, g15_, g13_, bn4_, prelu4_, learning_rate);
#endif
```

---

## Profiling So sánh

### Trước tối ưu:
| Kernel | Time % |
|--------|--------|
| batchnorm_backward_kernel_v2 | 38.62% |
| prelu_backward_kernel | 15.63% |
| batchnorm_compute_mean | 1.05% |
| batchnorm_compute_var | 1.02% |

### Sau tối ưu:
| Kernel | Time % |
|--------|--------|
| prelu_batchnorm_backward_fused | 29.37% |
| batchnorm_compute_mean_var_fused | 1.68% |

---

## Build & Test

```bash
make gpu_train_opt
nvprof ./gpu_train_bce --lr 0.001 --bce-loss --data data --epochs 1 --batch 64
```
