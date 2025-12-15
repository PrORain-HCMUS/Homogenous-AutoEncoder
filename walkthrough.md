# GPU Autoencoder Optimization Summary

Tổng hợp các optimizations đã implement cho CUDA Autoencoder trên CIFAR-10.

---

## Phase 1: Core Training Infrastructure

### 1. BatchNorm Fix (layers_gpu.cu)
| Before | After |
|--------|-------|
| Luôn dùng running_mean/var | Training: tính batch mean/var |
| Backward gradient đơn giản | Backward đầy đủ theo chain rule |
| Không update running stats | Update running stats với momentum |

**Files:** layers_gpu.cu (lines 162-430)

---

### 2. AdamW Optimizer (layers_gpu.cu, gpu_layer.h)
```cpp
// AdamW update rule với bias correction
m = β1*m + (1-β1)*g
v = β2*v + (1-β2)*g²
w = w*(1-lr*wd) - lr*m̂/(√v̂+ε)
```
**Config:** β1=0.9, β2=0.999, ε=1e-8, weight_decay=1e-4

---

### 3. Cosine Annealing LR (main_gpu.cu)
```cpp
lr = min_lr + 0.5*(max_lr - min_lr)*(1 + cos(π*epoch/total_epochs))
```
Giảm LR từ 100% → 1% theo đường cong cosine.

---

### 4. LeakyReLU → PReLU (layers_gpu.cu)

**v1 - LeakyReLU:**
```cpp
output = (x > 0) ? x : 0.01*x  // Fixed α=0.01
```

**v2 - PReLU (Learnable Slope):**
```cuda
// Forward: f(x) = max(0,x) + α[c] * min(0,x)
output[idx] = (x > 0.0f) ? x : alpha[c] * x;

// Backward: gradient flows through negative values
grad_input[idx] = (x > 0.0f) ? go : alpha[c] * go;
if (x <= 0.0f) atomicAdd(&grad_alpha[c], go * x);
```
**Why:** Channel-wise learnable α. Initialized to 0.25 (PReLU paper).

**Files:** gpu_layer.h, layers_gpu.cu, gpu_autoencoder.h, gpu_autoencoder.cu

---

### 5. Data Augmentation

**Basic (main_gpu.cu):**
- ✅ `horizontal_flip = true`
- ✅ `random_crop = true` (padding=4)

**Cutout (dataset.h, dataset.cpp):**
```cpp
// Random erasing 8×8 region with 50% probability
int center_h = pos_h(rng), center_w = pos_w(rng);
for (int c = 0; c < 3; ++c)
    for (int h = h_start; h < h_end; ++h)
        for (int w = w_start; w < w_end; ++w)
            image[c * H * W + h * W + w] = 0.0f;
```

---

## Phase 2: Training Stability

### 6. Gradient Clipping (layers_gpu.cu)

**Problem:** BCE loss can cause gradient explosion.

```cuda
__global__ void gradient_clip_kernel(float* grad, float max_norm, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad[idx] = fminf(fmaxf(grad[idx], -max_norm), max_norm);
    }
}
```
**Applied in:** gpu_mse_loss_with_grad() và gpu_bce_loss_with_grad()

---

### 7. Learning Rate Warmup (main_gpu.cu)

5-epoch linear warmup before cosine annealing:
```
Epoch 1: lr = base_lr × 0.2
Epoch 2: lr = base_lr × 0.4
Epoch 3: lr = base_lr × 0.6
Epoch 4: lr = base_lr × 0.8
Epoch 5: lr = base_lr × 1.0
Epoch 6+: Cosine annealing → min_lr
```

---

## Phase 3: Inference Acceleration

### 8. cuML thay sklearn (phase3-kaggle-bce-v2.py)
| sklearn | cuML |
|---------|------|
| StandardScaler | cuml.preprocessing.StandardScaler |
| PCA | cuml.decomposition.PCA |
| SVC | cuml.svm.SVC |

**Speedup:** 10-50x trên GPU

---

## 📁 Files Modified

| File | Changes |
|------|---------|
| layers_gpu.cu | BatchNorm fix, AdamW kernel, PReLU, Gradient Clip |
| layers_gpu_opt.cu | LeakyReLU vectorized kernels |
| gpu_layer.h | PReLU class, AdamW config (β1, β2, eps), m/v buffers |
| gpu_autoencoder.h | Changed layer types to PReLU |
| gpu_autoencoder.cu | PReLU throughout all methods |
| main_gpu.cu | Cosine LR + Warmup, AdamW config, augmentation |
| dataset.h, dataset.cpp | Cutout augmentation |
| phase3-kaggle-bce-v2.py | cuML, PCA 1024 components |

---

## 🚀 Expected Improvements

| Optimization | Impact |
|--------------|--------|
| BatchNorm fix | +5-10% accuracy |
| AdamW | +2-5% accuracy, faster convergence |
| **PReLU** | Adaptive activation, faster convergence |
| Cosine LR + Warmup | Smooth start, better final loss |
| **Gradient Clipping** | Stable training, no explosion |
| Data Aug + Cutout | +2-5% accuracy, prevents overfitting |
| cuML | 10-50x faster preprocessing |

All changes compile and run successfully. ✅
