#ifndef MIXED_PRECISION_CUH
#define MIXED_PRECISION_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>

struct LossScaler {
    float scale = 65536.0f;
    float growth_factor = 2.0f;
    float backoff_factor = 0.5f;
    int growth_interval = 2000;
    int growth_step = 0;
    bool overflow_detected = false;
    
    void update() {
        if (overflow_detected) {
            scale *= backoff_factor;
            growth_step = 0;
            overflow_detected = false;
        } else {
            growth_step++;
            if (growth_step >= growth_interval) {
                scale *= growth_factor;
                growth_step = 0;
            }
        }
    }
};

__global__ void fp32_to_fp16_kernel(const float* __restrict__ input, __half* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = __float2half(input[idx]);
}

__global__ void fp16_to_fp32_kernel(const __half* __restrict__ input, float* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) output[idx] = __half2float(input[idx]);
}

__global__ void fp32_to_fp16_vectorized_kernel(const float2* __restrict__ input, __half2* __restrict__ output, size_t n2) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n2) {
        float2 in = input[idx];
        output[idx] = __floats2half2_rn(in.x, in.y);
    }
}

__global__ void scale_gradients_kernel(float* grads, float scale, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) grads[idx] *= scale;
}

__global__ void check_overflow_kernel(const float* grads, int* overflow_flag, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float g = grads[idx];
        if (isinf(g) || isnan(g)) atomicExch(overflow_flag, 1);
    }
}

inline void convert_fp32_to_fp16(const float* d_fp32, __half* d_fp16, size_t n) {
    if (n % 2 == 0) {
        fp32_to_fp16_vectorized_kernel<<<(n/2 + 255) / 256, 256>>>(
            reinterpret_cast<const float2*>(d_fp32),
            reinterpret_cast<__half2*>(d_fp16), n / 2);
    } else {
        fp32_to_fp16_kernel<<<(n + 255) / 256, 256>>>(d_fp32, d_fp16, n);
    }
}

inline void convert_fp16_to_fp32(const __half* d_fp16, float* d_fp32, size_t n) {
    fp16_to_fp32_kernel<<<(n + 255) / 256, 256>>>(d_fp16, d_fp32, n);
}

inline void unscale_gradients(float* d_grads, float scale, size_t n) {
    scale_gradients_kernel<<<(n + 255) / 256, 256>>>(d_grads, 1.0f / scale, n);
}

inline bool check_gradients_overflow(const float* d_grads, size_t n, int* d_overflow_flag) {
    cudaMemset(d_overflow_flag, 0, sizeof(int));
    check_overflow_kernel<<<(n + 255) / 256, 256>>>(d_grads, d_overflow_flag, n);
    int h_overflow = 0;
    cudaMemcpy(&h_overflow, d_overflow_flag, sizeof(int), cudaMemcpyDeviceToHost);
    return h_overflow != 0;
}

#endif
