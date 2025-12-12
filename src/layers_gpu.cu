#include "gpu_layer.h"
#include "cuda_utils.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include <cstdio>
#include <cmath>
#include <cfloat>
#include <vector>
#include <random>

__global__ void fill_zero_kernel(float* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = 0.0f;
    }
}

// Kernel for He initialization (for ReLU networks)
__global__ void he_init_kernel(float* data, size_t n, float std_dev, unsigned long long seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        data[idx] = curand_normal(&state) * std_dev;
    }
}

GPUTensor4D::GPUTensor4D(int n_, int c_, int h_, int w_)
    : n(n_), c(c_), h(h_), w(w_), d_data(nullptr) {
    allocate(n_, c_, h_, w_);
}

GPUTensor4D::~GPUTensor4D() {
    free();
}

GPUTensor4D::GPUTensor4D(GPUTensor4D&& other) noexcept
    : n(other.n), c(other.c), h(other.h), w(other.w), d_data(other.d_data) {
    other.d_data = nullptr;
    other.n = other.c = other.h = other.w = 0;
}

GPUTensor4D& GPUTensor4D::operator=(GPUTensor4D&& other) noexcept {
    if (this != &other) {
        free();
        n = other.n;
        c = other.c;
        h = other.h;
        w = other.w;
        d_data = other.d_data;
        other.d_data = nullptr;
        other.n = other.c = other.h = other.w = 0;
    }
    return *this;
}

void GPUTensor4D::allocate(int n_, int c_, int h_, int w_) {
    free();
    n = n_; c = c_; h = h_; w = w_;
    if (size() > 0) {
        CUDA_CHECK(cudaMalloc(&d_data, bytes()));
        CUDA_CHECK(cudaMemset(d_data, 0, bytes()));
    }
}

void GPUTensor4D::free() {
    if (d_data) {
        CUDA_CHECK(cudaFree(d_data));
        d_data = nullptr;
    }
    n = c = h = w = 0;
}

void GPUTensor4D::copy_from_host(const float* h_data) {
    if (d_data && size() > 0) {
        CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes(), cudaMemcpyHostToDevice));
    }
}

void GPUTensor4D::copy_to_host(float* h_data) const {
    if (d_data && size() > 0) {
        CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes(), cudaMemcpyDeviceToHost));
    }
}

__global__ void conv2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int k, int stride, int padding
) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int oc_n = blockIdx.z;
    int oc = oc_n % out_c;
    int n = oc_n / out_c;
    
    if (ow >= out_w || oh >= out_h || n >= batch_size) return;
    
    float sum = bias[oc];
    
    const size_t in_n_offset = static_cast<size_t>(n) * in_c * in_h * in_w;
    const size_t w_oc_offset = static_cast<size_t>(oc) * in_c * k * k;
    const int ih_base = oh * stride - padding;
    const int iw_base = ow * stride - padding;
    
    if (k == 3) {
        #pragma unroll
        for (int ic = 0; ic < in_c; ++ic) {
            const size_t in_ic_offset = in_n_offset + static_cast<size_t>(ic) * in_h * in_w;
            const size_t w_ic_offset = w_oc_offset + static_cast<size_t>(ic) * 9;
            
            #pragma unroll
            for (int kh = 0; kh < 3; ++kh) {
                int ih = ih_base + kh;
                if (ih >= 0 && ih < in_h) {
                    const size_t in_row = in_ic_offset + ih * in_w;
                    const size_t w_kh = w_ic_offset + kh * 3;
                    
                    #pragma unroll
                    for (int kw = 0; kw < 3; ++kw) {
                        int iw = iw_base + kw;
                        if (iw >= 0 && iw < in_w) {
                            sum += input[in_row + iw] * weights[w_kh + kw];
                        }
                    }
                }
            }
        }
    } else {
        for (int ic = 0; ic < in_c; ++ic) {
            for (int kh = 0; kh < k; ++kh) {
                for (int kw = 0; kw < k; ++kw) {
                    int ih = ih_base + kh;
                    int iw = iw_base + kw;
                    
                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                        size_t in_idx = in_n_offset + (static_cast<size_t>(ic) * in_h + ih) * in_w + iw;
                        size_t w_idx = w_oc_offset + (static_cast<size_t>(ic) * k + kh) * k + kw;
                        sum += input[in_idx] * weights[w_idx];
                    }
                }
            }
        }
    }
    
    size_t out_idx = ((static_cast<size_t>(n) * out_c + oc) * out_h + oh) * out_w + ow;
    output[out_idx] = sum;
}

__global__ void conv2d_backward_data_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ weights,
    float* __restrict__ grad_input,
    int batch_size, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int k, int stride, int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_inputs = batch_size * in_c * in_h * in_w;
    
    if (idx >= total_inputs) return;
    
    int iw = idx % in_w;
    int temp = idx / in_w;
    int ih = temp % in_h;
    temp = temp / in_h;
    int ic = temp % in_c;
    int n = temp / in_c;
    
    float sum = 0.0f;
    
    for (int oc = 0; oc < out_c; ++oc) {
        for (int kh = 0; kh < k; ++kh) {
            for (int kw = 0; kw < k; ++kw) {
                int oh_check = ih + padding - kh;
                int ow_check = iw + padding - kw;
                
                if (oh_check % stride == 0 && ow_check % stride == 0) {
                    int oh = oh_check / stride;
                    int ow = ow_check / stride;
                    
                    if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                        size_t go_idx = ((static_cast<size_t>(n) * out_c + oc) * out_h + oh) * out_w + ow;
                        size_t w_idx = ((static_cast<size_t>(oc) * in_c + ic) * k + kh) * k + kw;
                        sum += grad_output[go_idx] * weights[w_idx];
                    }
                }
            }
        }
    }
    
    grad_input[idx] = sum;
}

__global__ void conv2d_backward_weights_kernel(
    const float* __restrict__ input,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_weights,
    float* __restrict__ grad_bias,
    int batch_size, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int k, int stride, int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_weights = out_c * in_c * k * k;
    
    if (idx >= total_weights) return;
    
    int kw = idx % k;
    int temp = idx / k;
    int kh = temp % k;
    temp = temp / k;
    int ic = temp % in_c;
    int oc = temp / in_c;
    
    float sum = 0.0f;
    
    for (int n = 0; n < batch_size; ++n) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                int ih = oh * stride + kh - padding;
                int iw = ow * stride + kw - padding;
                
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    size_t in_idx = ((static_cast<size_t>(n) * in_c + ic) * in_h + ih) * in_w + iw;
                    size_t go_idx = ((static_cast<size_t>(n) * out_c + oc) * out_h + oh) * out_w + ow;
                    sum += input[in_idx] * grad_output[go_idx];
                }
            }
        }
    }
    
    grad_weights[idx] = sum;
}

__global__ void conv2d_backward_bias_kernel(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_bias,
    int batch_size, int out_c, int out_h, int out_w
) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= out_c) return;
    
    float sum = 0.0f;
    for (int n = 0; n < batch_size; ++n) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                size_t idx = ((static_cast<size_t>(n) * out_c + oc) * out_h + oh) * out_w + ow;
                sum += grad_output[idx];
            }
        }
    }
    grad_bias[oc] = sum;
}

__global__ void sgd_update_kernel(float* params, const float* grads, float lr, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        params[idx] -= lr * grads[idx];
    }
}

// ============================================================
// MOMENTUM SGD WITH WEIGHT DECAY
// ============================================================
// v = momentum * v + grad + weight_decay * params
// params = params - lr * v
// ============================================================
__global__ void sgd_momentum_update_kernel(
    float* params,           // Parameters to update
    const float* grads,      // Gradients
    float* velocity,         // Momentum buffer
    float lr,                // Learning rate
    float momentum,          // Momentum coefficient (typically 0.9)
    float weight_decay,      // L2 regularization (typically 1e-4)
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Compute gradient with weight decay (L2 regularization)
        float grad_with_decay = grads[idx] + weight_decay * params[idx];
        
        // Update velocity: v = momentum * v + grad_with_decay
        float v = momentum * velocity[idx] + grad_with_decay;
        velocity[idx] = v;
        
        // Update parameters: params = params - lr * v
        params[idx] -= lr * v;
    }
}

GPUConv2DLayer::GPUConv2DLayer(int in_channels, int out_channels, int kernel_size,
                               int stride, int padding)
    : in_c_(in_channels), out_c_(out_channels), k_(kernel_size),
      stride_(stride), padding_(padding) {
    
    weights_size_ = static_cast<size_t>(out_c_) * in_c_ * k_ * k_;
    
    CUDA_CHECK(cudaMalloc(&d_weights_, weights_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias_, out_c_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_weights_, weights_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_bias_, out_c_ * sizeof(float)));
    
    // Allocate momentum buffers (velocity) - initialized to zero
    CUDA_CHECK(cudaMalloc(&d_velocity_weights_, weights_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_velocity_bias_, out_c_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_velocity_weights_, 0, weights_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_velocity_bias_, 0, out_c_ * sizeof(float)));
    
    // He initialization for weights: std = sqrt(2 / fan_in)
    // fan_in = in_channels * kernel_size * kernel_size
    int fan_in = in_c_ * k_ * k_;
    float std_dev = sqrtf(2.0f / static_cast<float>(fan_in));
    
    int block_size = 256;
    int grid_size = (weights_size_ + block_size - 1) / block_size;
    
    // Use random seed based on layer parameters for reproducibility
    unsigned long long seed = static_cast<unsigned long long>(in_c_ * 1000 + out_c_ * 100 + k_);
    he_init_kernel<<<grid_size, block_size>>>(d_weights_, weights_size_, std_dev, seed);
    CUDA_CHECK(cudaGetLastError());
    
    // Initialize bias to zero (standard practice)
    CUDA_CHECK(cudaMemset(d_bias_, 0, out_c_ * sizeof(float)));
}

GPUConv2DLayer::~GPUConv2DLayer() {
    if (d_weights_) cudaFree(d_weights_);
    if (d_bias_) cudaFree(d_bias_);
    if (d_grad_weights_) cudaFree(d_grad_weights_);
    if (d_grad_bias_) cudaFree(d_grad_bias_);
    if (d_velocity_weights_) cudaFree(d_velocity_weights_);
    if (d_velocity_bias_) cudaFree(d_velocity_bias_);
}

void GPUConv2DLayer::copy_weights_from_host(const float* h_weights, const float* h_bias) {
    CUDA_CHECK(cudaMemcpy(d_weights_, h_weights, weights_size_ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias_, h_bias, out_c_ * sizeof(float), cudaMemcpyHostToDevice));
}

void GPUConv2DLayer::copy_weights_to_host(float* h_weights, float* h_bias) const {
    CUDA_CHECK(cudaMemcpy(h_weights, d_weights_, weights_size_ * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bias, d_bias_, out_c_ * sizeof(float), cudaMemcpyDeviceToHost));
}

void GPUConv2DLayer::forward(const GPUTensor4D& input, GPUTensor4D& output) const {
    int out_h = get_output_h(input.h);
    int out_w = get_output_w(input.w);
    
    if (output.n != input.n || output.c != out_c_ || 
        output.h != out_h || output.w != out_w) {
        output.allocate(input.n, out_c_, out_h, out_w);
    }
    
    dim3 block(16, 16);
    dim3 grid(
        (out_w + block.x - 1) / block.x,
        (out_h + block.y - 1) / block.y,
        input.n * out_c_
    );
    
    conv2d_forward_kernel<<<grid, block>>>(
        input.d_data, d_weights_, d_bias_, output.d_data,
        input.n, in_c_, input.h, input.w,
        out_c_, out_h, out_w,
        k_, stride_, padding_
    );
    CUDA_CHECK(cudaGetLastError());
}

void GPUConv2DLayer::backward(const GPUTensor4D& input, const GPUTensor4D& grad_output,
                              GPUTensor4D& grad_input, float learning_rate) {
    int out_h = grad_output.h;
    int out_w = grad_output.w;
    
    if (grad_input.n != input.n || grad_input.c != in_c_ ||
        grad_input.h != input.h || grad_input.w != input.w) {
        grad_input.allocate(input.n, in_c_, input.h, input.w);
    }
    
    int block_size = 256;
    
    int total_inputs = input.n * in_c_ * input.h * input.w;
    int grid_inputs = (total_inputs + block_size - 1) / block_size;
    conv2d_backward_data_kernel<<<grid_inputs, block_size>>>(
        grad_output.d_data, d_weights_, grad_input.d_data,
        input.n, in_c_, input.h, input.w,
        out_c_, out_h, out_w,
        k_, stride_, padding_
    );
    CUDA_CHECK(cudaGetLastError());
    
    int total_weights = static_cast<int>(weights_size_);
    int grid_weights = (total_weights + block_size - 1) / block_size;
    conv2d_backward_weights_kernel<<<grid_weights, block_size>>>(
        input.d_data, grad_output.d_data, d_grad_weights_, d_grad_bias_,
        input.n, in_c_, input.h, input.w,
        out_c_, out_h, out_w,
        k_, stride_, padding_
    );
    CUDA_CHECK(cudaGetLastError());
    
    int grid_bias = (out_c_ + block_size - 1) / block_size;
    conv2d_backward_bias_kernel<<<grid_bias, block_size>>>(
        grad_output.d_data, d_grad_bias_,
        input.n, out_c_, out_h, out_w
    );
    CUDA_CHECK(cudaGetLastError());
    
    int grid_w_update = (total_weights + block_size - 1) / block_size;
    sgd_update_kernel<<<grid_w_update, block_size>>>(d_weights_, d_grad_weights_, learning_rate, weights_size_);
    CUDA_CHECK(cudaGetLastError());
    
    int grid_b_update = (out_c_ + block_size - 1) / block_size;
    sgd_update_kernel<<<grid_b_update, block_size>>>(d_bias_, d_grad_bias_, learning_rate, out_c_);
    CUDA_CHECK(cudaGetLastError());
}

void GPUConv2DLayer::backward_momentum(const GPUTensor4D& input, const GPUTensor4D& grad_output,
                                        GPUTensor4D& grad_input, float learning_rate,
                                        const OptimizerConfig& opt_config) {
    int out_h = grad_output.h;
    int out_w = grad_output.w;
    
    if (grad_input.n != input.n || grad_input.c != in_c_ ||
        grad_input.h != input.h || grad_input.w != input.w) {
        grad_input.allocate(input.n, in_c_, input.h, input.w);
    }
    
    int block_size = 256;
    
    // Backward data (same as before)
    int total_inputs = input.n * in_c_ * input.h * input.w;
    int grid_inputs = (total_inputs + block_size - 1) / block_size;
    conv2d_backward_data_kernel<<<grid_inputs, block_size>>>(
        grad_output.d_data, d_weights_, grad_input.d_data,
        input.n, in_c_, input.h, input.w,
        out_c_, out_h, out_w,
        k_, stride_, padding_
    );
    CUDA_CHECK(cudaGetLastError());
    
    // Backward weights
    int total_weights = static_cast<int>(weights_size_);
    int grid_weights = (total_weights + block_size - 1) / block_size;
    conv2d_backward_weights_kernel<<<grid_weights, block_size>>>(
        input.d_data, grad_output.d_data, d_grad_weights_, d_grad_bias_,
        input.n, in_c_, input.h, input.w,
        out_c_, out_h, out_w,
        k_, stride_, padding_
    );
    CUDA_CHECK(cudaGetLastError());
    
    // Backward bias
    int grid_bias = (out_c_ + block_size - 1) / block_size;
    conv2d_backward_bias_kernel<<<grid_bias, block_size>>>(
        grad_output.d_data, d_grad_bias_,
        input.n, out_c_, out_h, out_w
    );
    CUDA_CHECK(cudaGetLastError());
    
    // Update weights with Momentum SGD + Weight Decay
    int grid_w_update = (total_weights + block_size - 1) / block_size;
    sgd_momentum_update_kernel<<<grid_w_update, block_size>>>(
        d_weights_, d_grad_weights_, d_velocity_weights_,
        learning_rate, opt_config.momentum, opt_config.weight_decay, weights_size_
    );
    CUDA_CHECK(cudaGetLastError());
    
    // Update bias with Momentum SGD (typically no weight decay on bias)
    int grid_b_update = (out_c_ + block_size - 1) / block_size;
    sgd_momentum_update_kernel<<<grid_b_update, block_size>>>(
        d_bias_, d_grad_bias_, d_velocity_bias_,
        learning_rate, opt_config.momentum, 0.0f, out_c_  // No weight decay on bias
    );
    CUDA_CHECK(cudaGetLastError());
}

__global__ void relu_forward_kernel(const float* input, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

__global__ void relu_backward_kernel(const float* input, const float* grad_output,
                                     float* grad_input, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}

void GPUReLULayer::forward(const GPUTensor4D& input, GPUTensor4D& output) const {
    if (output.n != input.n || output.c != input.c ||
        output.h != input.h || output.w != input.w) {
        output.allocate(input.n, input.c, input.h, input.w);
    }
    
#ifdef USE_OPTIMIZED_KERNELS
    gpu_relu_forward_opt(input, output);
#else
    size_t total = input.size();
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    
    relu_forward_kernel<<<grid_size, block_size>>>(input.d_data, output.d_data, total);
    CUDA_CHECK(cudaGetLastError());
#endif
}

void GPUReLULayer::backward(const GPUTensor4D& input, const GPUTensor4D& grad_output,
                            GPUTensor4D& grad_input) const {
    if (grad_input.n != input.n || grad_input.c != input.c ||
        grad_input.h != input.h || grad_input.w != input.w) {
        grad_input.allocate(input.n, input.c, input.h, input.w);
    }
    
#ifdef USE_OPTIMIZED_KERNELS
    gpu_relu_backward_opt(input, grad_output, grad_input);
#else
    size_t total = input.size();
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    
    relu_backward_kernel<<<grid_size, block_size>>>(input.d_data, grad_output.d_data, 
                                                     grad_input.d_data, total);
    CUDA_CHECK(cudaGetLastError());
#endif
}

__global__ void maxpool2d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels, int in_h, int in_w,
    int out_h, int out_w, int k, int stride
) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c_n = blockIdx.z;
    int c = c_n % channels;
    int n = c_n / channels;
    
    if (ow >= out_w || oh >= out_h || n >= batch_size) return;
    
    const size_t in_base = ((static_cast<size_t>(n) * channels + c) * in_h) * in_w;
    const int ih_base = oh * stride;
    const int iw_base = ow * stride;
    
    float max_val = -FLT_MAX;
    
    if (k == 2 && stride == 2) {
        const size_t idx00 = in_base + ih_base * in_w + iw_base;
        const size_t idx01 = idx00 + 1;
        const size_t idx10 = idx00 + in_w;
        const size_t idx11 = idx10 + 1;
        
        max_val = input[idx00];
        max_val = fmaxf(max_val, input[idx01]);
        max_val = fmaxf(max_val, input[idx10]);
        max_val = fmaxf(max_val, input[idx11]);
    } else {
        #pragma unroll 4
        for (int kh = 0; kh < k; ++kh) {
            #pragma unroll 4
            for (int kw = 0; kw < k; ++kw) {
                int ih = ih_base + kh;
                int iw = iw_base + kw;
                
                if (ih < in_h && iw < in_w) {
                    size_t in_idx = in_base + ih * in_w + iw;
                    max_val = fmaxf(max_val, input[in_idx]);
                }
            }
        }
    }
    
    size_t out_idx = ((static_cast<size_t>(n) * channels + c) * out_h + oh) * out_w + ow;
    output[out_idx] = max_val;
}

__global__ void maxpool2d_backward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int batch_size, int channels, int in_h, int in_w,
    int out_h, int out_w, int k, int stride
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * out_h * out_w;
    
    if (idx >= total) return;
    
    int ow = idx % out_w;
    int temp = idx / out_w;
    int oh = temp % out_h;
    temp = temp / out_h;
    int c = temp % channels;
    int n = temp / channels;
    
    float max_val = -FLT_MAX;
    int max_ih = -1, max_iw = -1;
    
    for (int kh = 0; kh < k; ++kh) {
        for (int kw = 0; kw < k; ++kw) {
            int ih = oh * stride + kh;
            int iw = ow * stride + kw;
            
            if (ih < in_h && iw < in_w) {
                size_t in_idx = ((static_cast<size_t>(n) * channels + c) * in_h + ih) * in_w + iw;
                if (input[in_idx] > max_val) {
                    max_val = input[in_idx];
                    max_ih = ih;
                    max_iw = iw;
                }
            }
        }
    }
    
    if (max_ih >= 0 && max_iw >= 0) {
        size_t max_idx = ((static_cast<size_t>(n) * channels + c) * in_h + max_ih) * in_w + max_iw;
        atomicAdd(&grad_input[max_idx], grad_output[idx]);
    }
}

GPUMaxPool2DLayer::GPUMaxPool2DLayer(int kernel_size, int stride)
    : k_(kernel_size), stride_(stride) {}

void GPUMaxPool2DLayer::forward(const GPUTensor4D& input, GPUTensor4D& output) const {
    int out_h = get_output_h(input.h);
    int out_w = get_output_w(input.w);
    
    if (output.n != input.n || output.c != input.c ||
        output.h != out_h || output.w != out_w) {
        output.allocate(input.n, input.c, out_h, out_w);
    }
    
#ifdef USE_OPTIMIZED_KERNELS
    gpu_maxpool2d_forward_opt(input, output, k_, stride_);
#else
    dim3 block(16, 16);
    dim3 grid(
        (out_w + block.x - 1) / block.x,
        (out_h + block.y - 1) / block.y,
        input.n * input.c
    );
    
    maxpool2d_forward_kernel<<<grid, block>>>(
        input.d_data, output.d_data,
        input.n, input.c, input.h, input.w,
        out_h, out_w, k_, stride_
    );
    CUDA_CHECK(cudaGetLastError());
#endif
}

void GPUMaxPool2DLayer::backward(const GPUTensor4D& input, const GPUTensor4D& grad_output,
                                 GPUTensor4D& grad_input) const {
    if (grad_input.n != input.n || grad_input.c != input.c ||
        grad_input.h != input.h || grad_input.w != input.w) {
        grad_input.allocate(input.n, input.c, input.h, input.w);
    }
    
    CUDA_CHECK(cudaMemset(grad_input.d_data, 0, grad_input.bytes()));
    
    int total = input.n * input.c * grad_output.h * grad_output.w;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    
    maxpool2d_backward_kernel<<<grid_size, block_size>>>(
        input.d_data, grad_output.d_data, grad_input.d_data,
        input.n, input.c, input.h, input.w,
        grad_output.h, grad_output.w, k_, stride_
    );
    CUDA_CHECK(cudaGetLastError());
}

__global__ void upsample2d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels, int in_h, int in_w,
    int out_h, int out_w, int scale
) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c_n = blockIdx.z;
    int c = c_n % channels;
    int n = c_n / channels;
    
    if (ow >= out_w || oh >= out_h || n >= batch_size) return;
    
    int ih = oh / scale;
    int iw = ow / scale;
    
    size_t in_idx = ((static_cast<size_t>(n) * channels + c) * in_h + ih) * in_w + iw;
    size_t out_idx = ((static_cast<size_t>(n) * channels + c) * out_h + oh) * out_w + ow;
    output[out_idx] = input[in_idx];
}

__global__ void upsample2d_backward_kernel(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int batch_size, int channels, int in_h, int in_w,
    int out_h, int out_w, int scale
) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c_n = blockIdx.z;
    int c = c_n % channels;
    int n = c_n / channels;
    
    if (ow >= out_w || oh >= out_h || n >= batch_size) return;
    
    int ih = oh / scale;
    int iw = ow / scale;
    
    size_t in_idx = ((static_cast<size_t>(n) * channels + c) * in_h + ih) * in_w + iw;
    size_t out_idx = ((static_cast<size_t>(n) * channels + c) * out_h + oh) * out_w + ow;
    atomicAdd(&grad_input[in_idx], grad_output[out_idx]);
}

GPUUpSample2DLayer::GPUUpSample2DLayer(int scale) : scale_(scale) {}

void GPUUpSample2DLayer::forward(const GPUTensor4D& input, GPUTensor4D& output) const {
    int out_h = get_output_h(input.h);
    int out_w = get_output_w(input.w);
    
    if (output.n != input.n || output.c != input.c ||
        output.h != out_h || output.w != out_w) {
        output.allocate(input.n, input.c, out_h, out_w);
    }
    
#ifdef USE_OPTIMIZED_KERNELS
    gpu_upsample2d_forward_opt(input, output, scale_);
#else
    dim3 block(16, 16);
    dim3 grid(
        (out_w + block.x - 1) / block.x,
        (out_h + block.y - 1) / block.y,
        input.n * input.c
    );
    
    upsample2d_forward_kernel<<<grid, block>>>(
        input.d_data, output.d_data,
        input.n, input.c, input.h, input.w,
        out_h, out_w, scale_
    );
    CUDA_CHECK(cudaGetLastError());
#endif
}

void GPUUpSample2DLayer::backward(const GPUTensor4D& input, const GPUTensor4D& grad_output,
                                  GPUTensor4D& grad_input) const {
    if (grad_input.n != input.n || grad_input.c != input.c ||
        grad_input.h != input.h || grad_input.w != input.w) {
        grad_input.allocate(input.n, input.c, input.h, input.w);
    }
    
    CUDA_CHECK(cudaMemset(grad_input.d_data, 0, grad_input.bytes()));
    
    int out_h = grad_output.h;
    int out_w = grad_output.w;
    
    dim3 block(16, 16);
    dim3 grid(
        (out_w + block.x - 1) / block.x,
        (out_h + block.y - 1) / block.y,
        input.n * input.c
    );
    
    upsample2d_backward_kernel<<<grid, block>>>(
        grad_output.d_data, grad_input.d_data,
        input.n, input.c, input.h, input.w,
        out_h, out_w, scale_
    );
    CUDA_CHECK(cudaGetLastError());
}

__global__ void mse_loss_kernel(
    const float* __restrict__ output,
    const float* __restrict__ target,
    float* __restrict__ partial_sums,
    size_t n
) {
    extern __shared__ float sdata[];
    
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    float val = 0.0f;
    if (idx < n) {
        float diff = output[idx] - target[idx];
        val = diff * diff;
    }
    if (idx + blockDim.x < n) {
        float diff = output[idx + blockDim.x] - target[idx + blockDim.x];
        val += diff * diff;
    }
    
    sdata[tid] = val;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid < 32) {
        volatile float* vsmem = sdata;
        if (blockDim.x >= 64) vsmem[tid] += vsmem[tid + 32];
        float myVal = vsmem[tid];
        
        for (int offset = 16; offset > 0; offset /= 2) {
            myVal += __shfl_down_sync(0xffffffff, myVal, offset);
        }
        
        if (tid == 0) {
            partial_sums[blockIdx.x] = myVal;
        }
    }
}

__global__ void mse_grad_kernel(
    const float* __restrict__ output,
    const float* __restrict__ target,
    float* __restrict__ grad_output,
    float scale,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grad_output[idx] = scale * (output[idx] - target[idx]);
    }
}

static float* d_persistent_partial_sums = nullptr;
static size_t persistent_partial_sums_size = 0;

float gpu_mse_loss(const GPUTensor4D& output, const GPUTensor4D& target) {
    size_t n = output.size();
    int block_size = 256;
    int grid_size = (n + block_size * 2 - 1) / (block_size * 2);
    
    if (d_persistent_partial_sums == nullptr || persistent_partial_sums_size < static_cast<size_t>(grid_size)) {
        if (d_persistent_partial_sums) {
            cudaFree(d_persistent_partial_sums);
        }
        persistent_partial_sums_size = static_cast<size_t>(grid_size) * 2;
        CUDA_CHECK(cudaMalloc(&d_persistent_partial_sums, persistent_partial_sums_size * sizeof(float)));
    }
    
    size_t shared_mem = block_size * sizeof(float);
    mse_loss_kernel<<<grid_size, block_size, shared_mem>>>(
        output.d_data, target.d_data, d_persistent_partial_sums, n
    );
    CUDA_CHECK(cudaGetLastError());
    
    std::vector<float> h_partial_sums(grid_size);
    CUDA_CHECK(cudaMemcpy(h_partial_sums.data(), d_persistent_partial_sums, 
                          grid_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    float sum = 0.0f;
    for (int i = 0; i < grid_size; ++i) {
        sum += h_partial_sums[i];
    }
    
    return sum / static_cast<float>(n);
}

float gpu_mse_loss_with_grad(const GPUTensor4D& output, const GPUTensor4D& target,
                              GPUTensor4D& grad_output) {
    size_t n = output.size();
    
    if (grad_output.n != output.n || grad_output.c != output.c ||
        grad_output.h != output.h || grad_output.w != output.w) {
        grad_output.allocate(output.n, output.c, output.h, output.w);
    }
    
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    float scale = 2.0f / static_cast<float>(n);
    mse_grad_kernel<<<grid_size, block_size>>>(
        output.d_data, target.d_data, grad_output.d_data, scale, n
    );
    CUDA_CHECK(cudaGetLastError());
    
    return gpu_mse_loss(output, target);
}

// ============================================================
// SIGMOID ACTIVATION LAYER
// ============================================================
// sigmoid(x) = 1 / (1 + exp(-x))
// ============================================================

__global__ void sigmoid_forward_kernel(const float* __restrict__ input,
                                        float* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Numerically stable sigmoid
        float x = input[idx];
        if (x >= 0) {
            output[idx] = 1.0f / (1.0f + expf(-x));
        } else {
            float exp_x = expf(x);
            output[idx] = exp_x / (1.0f + exp_x);
        }
    }
}

__global__ void sigmoid_backward_kernel(const float* __restrict__ output,
                                         const float* __restrict__ grad_output,
                                         float* __restrict__ grad_input, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // d_sigmoid/d_x = sigmoid(x) * (1 - sigmoid(x)) = output * (1 - output)
        float s = output[idx];
        grad_input[idx] = grad_output[idx] * s * (1.0f - s);
    }
}

void GPUSigmoidLayer::forward(const GPUTensor4D& input, GPUTensor4D& output) const {
    if (output.n != input.n || output.c != input.c ||
        output.h != input.h || output.w != input.w) {
        output.allocate(input.n, input.c, input.h, input.w);
    }
    
    size_t n = input.size();
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    sigmoid_forward_kernel<<<grid_size, block_size>>>(input.d_data, output.d_data, n);
    CUDA_CHECK(cudaGetLastError());
}

void GPUSigmoidLayer::backward(const GPUTensor4D& output, const GPUTensor4D& grad_output,
                                GPUTensor4D& grad_input) const {
    if (grad_input.n != output.n || grad_input.c != output.c ||
        grad_input.h != output.h || grad_input.w != output.w) {
        grad_input.allocate(output.n, output.c, output.h, output.w);
    }
    
    size_t n = output.size();
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    sigmoid_backward_kernel<<<grid_size, block_size>>>(
        output.d_data, grad_output.d_data, grad_input.d_data, n);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================
// BINARY CROSS-ENTROPY LOSS
// ============================================================
// BCE = -[y*log(ŷ + eps) + (1-y)*log(1-ŷ + eps)] / N
// grad = (ŷ - y) / (ŷ * (1-ŷ) + eps) / N
// ============================================================

__global__ void bce_loss_kernel(
    const float* __restrict__ output,
    const float* __restrict__ target,
    float* __restrict__ partial_sums,
    size_t n
) {
    extern __shared__ float sdata[];
    
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    const float eps = 1e-7f;
    float val = 0.0f;
    
    if (idx < n) {
        float y = target[idx];
        float p = fmaxf(fminf(output[idx], 1.0f - eps), eps);  // Clamp to [eps, 1-eps]
        val = -(y * logf(p) + (1.0f - y) * logf(1.0f - p));
    }
    if (idx + blockDim.x < n) {
        float y = target[idx + blockDim.x];
        float p = fmaxf(fminf(output[idx + blockDim.x], 1.0f - eps), eps);
        val += -(y * logf(p) + (1.0f - y) * logf(1.0f - p));
    }
    
    sdata[tid] = val;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid < 32) {
        volatile float* vsmem = sdata;
        if (blockDim.x >= 64) vsmem[tid] += vsmem[tid + 32];
        float myVal = vsmem[tid];
        
        for (int offset = 16; offset > 0; offset /= 2) {
            myVal += __shfl_down_sync(0xffffffff, myVal, offset);
        }
        
        if (tid == 0) {
            partial_sums[blockIdx.x] = myVal;
        }
    }
}

__global__ void bce_grad_kernel(
    const float* __restrict__ output,
    const float* __restrict__ target,
    float* __restrict__ grad_output,
    float scale,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const float eps = 1e-7f;
        float y = target[idx];
        float p = fmaxf(fminf(output[idx], 1.0f - eps), eps);
        // d_BCE/d_p = -y/p + (1-y)/(1-p) = (p - y) / (p * (1-p))
        // Simplified: grad = (p - y) / (p * (1-p) + eps)
        grad_output[idx] = scale * (p - y) / (p * (1.0f - p) + eps);
    }
}

static float* d_bce_partial_sums = nullptr;
static size_t bce_partial_sums_size = 0;

float gpu_bce_loss(const GPUTensor4D& output, const GPUTensor4D& target) {
    size_t n = output.size();
    int block_size = 256;
    int grid_size = (n + block_size * 2 - 1) / (block_size * 2);
    
    if (d_bce_partial_sums == nullptr || bce_partial_sums_size < static_cast<size_t>(grid_size)) {
        if (d_bce_partial_sums) {
            cudaFree(d_bce_partial_sums);
        }
        bce_partial_sums_size = static_cast<size_t>(grid_size) * 2;
        CUDA_CHECK(cudaMalloc(&d_bce_partial_sums, bce_partial_sums_size * sizeof(float)));
    }
    
    size_t shared_mem = block_size * sizeof(float);
    bce_loss_kernel<<<grid_size, block_size, shared_mem>>>(
        output.d_data, target.d_data, d_bce_partial_sums, n
    );
    CUDA_CHECK(cudaGetLastError());
    
    std::vector<float> h_partial_sums(grid_size);
    CUDA_CHECK(cudaMemcpy(h_partial_sums.data(), d_bce_partial_sums, 
                          grid_size * sizeof(float), cudaMemcpyDeviceToHost));
    
    float sum = 0.0f;
    for (int i = 0; i < grid_size; ++i) {
        sum += h_partial_sums[i];
    }
    
    return sum / static_cast<float>(n);
}

float gpu_bce_loss_with_grad(const GPUTensor4D& output, const GPUTensor4D& target,
                              GPUTensor4D& grad_output) {
    size_t n = output.size();
    
    if (grad_output.n != output.n || grad_output.c != output.c ||
        grad_output.h != output.h || grad_output.w != output.w) {
        grad_output.allocate(output.n, output.c, output.h, output.w);
    }
    
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    
    float scale = 1.0f / static_cast<float>(n);
    bce_grad_kernel<<<grid_size, block_size>>>(
        output.d_data, target.d_data, grad_output.d_data, scale, n
    );
    CUDA_CHECK(cudaGetLastError());
    
    return gpu_bce_loss(output, target);
}
