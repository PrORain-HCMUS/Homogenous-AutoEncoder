#include "cuda_utils.h"
#include "gpu_layer.h"
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
    if (idx < n) data[idx] = 0.0f;
}

__global__ void he_init_kernel(float* data, size_t n, float std_dev, unsigned long long seed) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        data[idx] = curand_normal(&state) * std_dev;
    }
}

GPUTensor4D::GPUTensor4D(int n_, int c_, int h_, int w_) : n(n_), c(c_), h(h_), w(w_), d_data(nullptr) { allocate(n_, c_, h_, w_); }
GPUTensor4D::~GPUTensor4D() { free(); }

GPUTensor4D::GPUTensor4D(GPUTensor4D&& other) noexcept : n(other.n), c(other.c), h(other.h), w(other.w), d_data(other.d_data) {
    other.d_data = nullptr; other.n = other.c = other.h = other.w = 0;
}

GPUTensor4D& GPUTensor4D::operator=(GPUTensor4D&& other) noexcept {
    if (this != &other) {
        free(); n = other.n; c = other.c; h = other.h; w = other.w; d_data = other.d_data;
        other.d_data = nullptr; other.n = other.c = other.h = other.w = 0;
    }
    return *this;
}

void GPUTensor4D::allocate(int n_, int c_, int h_, int w_) {
    free(); n = n_; c = c_; h = h_; w = w_;
    if (size() > 0) { CUDA_CHECK(cudaMalloc(&d_data, bytes())); CUDA_CHECK(cudaMemset(d_data, 0, bytes())); }
}

void GPUTensor4D::free() {
    if (d_data) { CUDA_CHECK(cudaFree(d_data)); d_data = nullptr; }
    n = c = h = w = 0;
}

void GPUTensor4D::copy_from_host(const float* h_data) {
    if (d_data && size() > 0) CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes(), cudaMemcpyHostToDevice));
}

void GPUTensor4D::copy_to_host(float* h_data) const {
    if (d_data && size() > 0) CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes(), cudaMemcpyDeviceToHost));
}

__global__ void conv2d_forward_kernel(const float* __restrict__ input, const float* __restrict__ weights, const float* __restrict__ bias, float* __restrict__ output,
    int batch_size, int in_c, int in_h, int in_w, int out_c, int out_h, int out_w, int k, int stride, int padding) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int oc_n = blockIdx.z, oc = oc_n % out_c, n = oc_n / out_c;
    if (ow >= out_w || oh >= out_h || n >= batch_size) return;
    float sum = bias[oc];
    const size_t in_n_offset = static_cast<size_t>(n) * in_c * in_h * in_w;
    const size_t w_oc_offset = static_cast<size_t>(oc) * in_c * k * k;
    const int ih_base = oh * stride - padding, iw_base = ow * stride - padding;
    if (k == 3) {
        #pragma unroll
        for (int ic = 0; ic < in_c; ++ic) {
            const size_t in_ic_offset = in_n_offset + static_cast<size_t>(ic) * in_h * in_w;
            const size_t w_ic_offset = w_oc_offset + static_cast<size_t>(ic) * 9;
            #pragma unroll
            for (int kh = 0; kh < 3; ++kh) {
                int ih = ih_base + kh;
                if (ih >= 0 && ih < in_h) {
                    const size_t in_row = in_ic_offset + ih * in_w, w_kh = w_ic_offset + kh * 3;
                    #pragma unroll
                    for (int kw = 0; kw < 3; ++kw) {
                        int iw = iw_base + kw;
                        if (iw >= 0 && iw < in_w) sum += input[in_row + iw] * weights[w_kh + kw];
                    }
                }
            }
        }
    } else {
        for (int ic = 0; ic < in_c; ++ic) {
            for (int kh = 0; kh < k; ++kh) {
                for (int kw = 0; kw < k; ++kw) {
                    int ih = ih_base + kh, iw = iw_base + kw;
                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                        size_t in_idx = in_n_offset + (static_cast<size_t>(ic) * in_h + ih) * in_w + iw;
                        size_t w_idx = w_oc_offset + (static_cast<size_t>(ic) * k + kh) * k + kw;
                        sum += input[in_idx] * weights[w_idx];
                    }
                }
            }
        }
    }
    output[((static_cast<size_t>(n) * out_c + oc) * out_h + oh) * out_w + ow] = sum;
}

__global__ void conv2d_backward_data_kernel(const float* __restrict__ grad_output, const float* __restrict__ weights, float* __restrict__ grad_input,
    int batch_size, int in_c, int in_h, int in_w, int out_c, int out_h, int out_w, int k, int stride, int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * in_c * in_h * in_w) return;
    int iw = idx % in_w, temp = idx / in_w, ih = temp % in_h; temp = temp / in_h; int ic = temp % in_c, n = temp / in_c;
    float sum = 0.0f;
    for (int oc = 0; oc < out_c; ++oc) {
        for (int kh = 0; kh < k; ++kh) {
            for (int kw = 0; kw < k; ++kw) {
                int oh_check = ih + padding - kh, ow_check = iw + padding - kw;
                if (oh_check % stride == 0 && ow_check % stride == 0) {
                    int oh = oh_check / stride, ow = ow_check / stride;
                    if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                        sum += grad_output[((static_cast<size_t>(n) * out_c + oc) * out_h + oh) * out_w + ow] *
                               weights[((static_cast<size_t>(oc) * in_c + ic) * k + kh) * k + kw];
                    }
                }
            }
        }
    }
    grad_input[idx] = sum;
}

__global__ void conv2d_backward_weights_kernel(const float* __restrict__ input, const float* __restrict__ grad_output, float* __restrict__ grad_weights, float* __restrict__ grad_bias,
    int batch_size, int in_c, int in_h, int in_w, int out_c, int out_h, int out_w, int k, int stride, int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= out_c * in_c * k * k) return;
    int kw = idx % k, temp = idx / k, kh = temp % k; temp = temp / k; int ic = temp % in_c, oc = temp / in_c;
    float sum = 0.0f;
    for (int n = 0; n < batch_size; ++n) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                int ih = oh * stride + kh - padding, iw = ow * stride + kw - padding;
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w)
                    sum += input[((static_cast<size_t>(n) * in_c + ic) * in_h + ih) * in_w + iw] *
                           grad_output[((static_cast<size_t>(n) * out_c + oc) * out_h + oh) * out_w + ow];
            }
        }
    }
    grad_weights[idx] = sum;
}

__global__ void conv2d_backward_bias_kernel(const float* __restrict__ grad_output, float* __restrict__ grad_bias, int batch_size, int out_c, int out_h, int out_w) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= out_c) return;
    float sum = 0.0f;
    for (int n = 0; n < batch_size; ++n)
        for (int oh = 0; oh < out_h; ++oh)
            for (int ow = 0; ow < out_w; ++ow)
                sum += grad_output[((static_cast<size_t>(n) * out_c + oc) * out_h + oh) * out_w + ow];
    grad_bias[oc] = sum;
}

__global__ void sgd_update_kernel(float* params, const float* grads, float lr, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) params[idx] -= lr * grads[idx];
}

__global__ void batchnorm_compute_mean_kernel(const float* __restrict__ input, float* __restrict__ mean, int N, int C, int H, int W) {
    int c = blockIdx.x;
    if (c >= C) return;
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int spatial = H * W;
    int total = N * spatial;
    float sum = 0.0f;
    for (int i = tid; i < total; i += blockDim.x) {
        int n = i / spatial, hw = i % spatial;
        sum += input[((static_cast<size_t>(n) * C + c) * H + hw / W) * W + hw % W];
    }
    sdata[tid] = sum; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) sdata[tid] += sdata[tid + s]; __syncthreads(); }
    if (tid == 0) mean[c] = sdata[0] / total;
}

__global__ void batchnorm_compute_var_kernel(const float* __restrict__ input, const float* __restrict__ mean, float* __restrict__ var, int N, int C, int H, int W) {
    int c = blockIdx.x;
    if (c >= C) return;
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int spatial = H * W;
    int total = N * spatial;
    float m = mean[c], sum = 0.0f;
    for (int i = tid; i < total; i += blockDim.x) {
        int n = i / spatial, hw = i % spatial;
        float diff = input[((static_cast<size_t>(n) * C + c) * H + hw / W) * W + hw % W] - m;
        sum += diff * diff;
    }
    sdata[tid] = sum; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) sdata[tid] += sdata[tid + s]; __syncthreads(); }
    if (tid == 0) var[c] = sdata[0] / total;
}

__global__ void batchnorm_normalize_kernel(const float* __restrict__ input, float* __restrict__ output, const float* __restrict__ gamma, const float* __restrict__ beta,
    const float* __restrict__ mean, const float* __restrict__ var, int N, int C, int H, int W, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * H * W) return;
    int c = (idx / (H * W)) % C;
    float inv_std = rsqrtf(var[c] + eps);
    output[idx] = gamma[c] * (input[idx] - mean[c]) * inv_std + beta[c];
}

__global__ void batchnorm_update_running_kernel(float* __restrict__ running, const float* __restrict__ batch, float momentum, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < C) running[c] = momentum * running[c] + (1.0f - momentum) * batch[c];
}

__global__ void batchnorm_backward_kernel_v2(const float* __restrict__ input, const float* __restrict__ grad_output, float* __restrict__ grad_input,
    float* __restrict__ grad_gamma, float* __restrict__ grad_beta, const float* __restrict__ gamma, const float* __restrict__ mean, const float* __restrict__ var,
    const float* __restrict__ sum_dy, const float* __restrict__ sum_dy_xhat, int N, int C, int H, int W, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * C * H * W) return;
    int c = (idx / (H * W)) % C;
    int M = N * H * W;
    float inv_std = rsqrtf(var[c] + eps);
    float x_hat = (input[idx] - mean[c]) * inv_std;
    grad_input[idx] = gamma[c] * inv_std * (grad_output[idx] - sum_dy[c] / M - x_hat * sum_dy_xhat[c] / M);
    atomicAdd(&grad_gamma[c], grad_output[idx] * x_hat);
    atomicAdd(&grad_beta[c], grad_output[idx]);
}

__global__ void batchnorm_compute_sums_kernel(const float* __restrict__ input, const float* __restrict__ grad_output,
    const float* __restrict__ mean, const float* __restrict__ var, float* __restrict__ sum_dy, float* __restrict__ sum_dy_xhat, int N, int C, int H, int W, float eps) {
    int c = blockIdx.x;
    if (c >= C) return;
    extern __shared__ float sdata[];
    float* s_dy = sdata, *s_dy_xhat = sdata + blockDim.x;
    int tid = threadIdx.x, spatial = H * W, total = N * spatial;
    float m = mean[c], inv_std = rsqrtf(var[c] + eps), local_dy = 0.0f, local_dy_xhat = 0.0f;
    for (int i = tid; i < total; i += blockDim.x) {
        int n = i / spatial, hw = i % spatial;
        size_t idx = ((static_cast<size_t>(n) * C + c) * H + hw / W) * W + hw % W;
        float dy = grad_output[idx], x_hat = (input[idx] - m) * inv_std;
        local_dy += dy; local_dy_xhat += dy * x_hat;
    }
    s_dy[tid] = local_dy; s_dy_xhat[tid] = local_dy_xhat; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) { s_dy[tid] += s_dy[tid + s]; s_dy_xhat[tid] += s_dy_xhat[tid + s]; } __syncthreads(); }
    if (tid == 0) { sum_dy[c] = s_dy[0]; sum_dy_xhat[c] = s_dy_xhat[0]; }
}

__global__ void adamw_update_kernel(float* params, const float* grads, float* m, float* v, 
    float lr, float beta1, float beta2, float eps, float weight_decay, float bias_correction1, float bias_correction2, size_t n) {
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

GPUConv2DLayer::GPUConv2DLayer(int in_channels, int out_channels, int kernel_size, int stride, int padding)
    : in_c_(in_channels), out_c_(out_channels), k_(kernel_size), stride_(stride), padding_(padding) {
    weights_size_ = static_cast<size_t>(out_c_) * in_c_ * k_ * k_;
    CUDA_CHECK(cudaMalloc(&d_weights_, weights_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias_, out_c_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_weights_, weights_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_bias_, out_c_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m_weights_, weights_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_weights_, weights_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_m_bias_, out_c_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v_bias_, out_c_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_m_weights_, 0, weights_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_weights_, 0, weights_size_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_m_bias_, 0, out_c_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_v_bias_, 0, out_c_ * sizeof(float)));
    float std_dev = sqrtf(2.0f / static_cast<float>(in_c_ * k_ * k_));
    int block_size = 256, grid_size = (weights_size_ + block_size - 1) / block_size;
    he_init_kernel<<<grid_size, block_size>>>(d_weights_, weights_size_, std_dev, static_cast<unsigned long long>(in_c_ * 1000 + out_c_ * 100 + k_));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemset(d_bias_, 0, out_c_ * sizeof(float)));
}

GPUConv2DLayer::~GPUConv2DLayer() {
    if (d_weights_) cudaFree(d_weights_); if (d_bias_) cudaFree(d_bias_);
    if (d_grad_weights_) cudaFree(d_grad_weights_); if (d_grad_bias_) cudaFree(d_grad_bias_);
    if (d_m_weights_) cudaFree(d_m_weights_); if (d_v_weights_) cudaFree(d_v_weights_);
    if (d_m_bias_) cudaFree(d_m_bias_); if (d_v_bias_) cudaFree(d_v_bias_);
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
    int out_h = get_output_h(input.h), out_w = get_output_w(input.w);
    if (output.n != input.n || output.c != out_c_ || output.h != out_h || output.w != out_w)
        output.allocate(input.n, out_c_, out_h, out_w);
    dim3 block(16, 16), grid((out_w + 15) / 16, (out_h + 15) / 16, input.n * out_c_);
    conv2d_forward_kernel<<<grid, block>>>(input.d_data, d_weights_, d_bias_, output.d_data, input.n, in_c_, input.h, input.w, out_c_, out_h, out_w, k_, stride_, padding_);
    CUDA_CHECK(cudaGetLastError());
}

void GPUConv2DLayer::backward(const GPUTensor4D& input, const GPUTensor4D& grad_output, GPUTensor4D& grad_input, float learning_rate) {
    int out_h = grad_output.h, out_w = grad_output.w;
    if (grad_input.n != input.n || grad_input.c != in_c_ || grad_input.h != input.h || grad_input.w != input.w)
        grad_input.allocate(input.n, in_c_, input.h, input.w);
    int total_inputs = input.n * in_c_ * input.h * input.w;
    conv2d_backward_data_kernel<<<(total_inputs + 255) / 256, 256>>>(grad_output.d_data, d_weights_, grad_input.d_data, input.n, in_c_, input.h, input.w, out_c_, out_h, out_w, k_, stride_, padding_);
    CUDA_CHECK(cudaGetLastError());
    int total_weights = static_cast<int>(weights_size_);
    conv2d_backward_weights_kernel<<<(total_weights + 255) / 256, 256>>>(input.d_data, grad_output.d_data, d_grad_weights_, d_grad_bias_, input.n, in_c_, input.h, input.w, out_c_, out_h, out_w, k_, stride_, padding_);
    CUDA_CHECK(cudaGetLastError());
    conv2d_backward_bias_kernel<<<(out_c_ + 255) / 256, 256>>>(grad_output.d_data, d_grad_bias_, input.n, out_c_, out_h, out_w);
    CUDA_CHECK(cudaGetLastError());
    sgd_update_kernel<<<(total_weights + 255) / 256, 256>>>(d_weights_, d_grad_weights_, learning_rate, weights_size_);
    CUDA_CHECK(cudaGetLastError());
    sgd_update_kernel<<<(out_c_ + 255) / 256, 256>>>(d_bias_, d_grad_bias_, learning_rate, out_c_);
    CUDA_CHECK(cudaGetLastError());
}

void GPUConv2DLayer::backward_momentum(const GPUTensor4D& input, const GPUTensor4D& grad_output, GPUTensor4D& grad_input, float learning_rate, const OptimizerConfig& opt_config) {
    int out_h = grad_output.h, out_w = grad_output.w;
    if (grad_input.n != input.n || grad_input.c != in_c_ || grad_input.h != input.h || grad_input.w != input.w)
        grad_input.allocate(input.n, in_c_, input.h, input.w);
    int total_inputs = input.n * in_c_ * input.h * input.w;
    conv2d_backward_data_kernel<<<(total_inputs + 255) / 256, 256>>>(grad_output.d_data, d_weights_, grad_input.d_data, input.n, in_c_, input.h, input.w, out_c_, out_h, out_w, k_, stride_, padding_);
    CUDA_CHECK(cudaGetLastError());
    int total_weights = static_cast<int>(weights_size_);
    conv2d_backward_weights_kernel<<<(total_weights + 255) / 256, 256>>>(input.d_data, grad_output.d_data, d_grad_weights_, d_grad_bias_, input.n, in_c_, input.h, input.w, out_c_, out_h, out_w, k_, stride_, padding_);
    CUDA_CHECK(cudaGetLastError());
    conv2d_backward_bias_kernel<<<(out_c_ + 255) / 256, 256>>>(grad_output.d_data, d_grad_bias_, input.n, out_c_, out_h, out_w);
    CUDA_CHECK(cudaGetLastError());
    float bc1 = 1.0f - powf(opt_config.beta1, opt_config.step + 1);
    float bc2 = 1.0f - powf(opt_config.beta2, opt_config.step + 1);
    adamw_update_kernel<<<(total_weights + 255) / 256, 256>>>(d_weights_, d_grad_weights_, d_m_weights_, d_v_weights_, learning_rate, opt_config.beta1, opt_config.beta2, opt_config.eps, opt_config.weight_decay, bc1, bc2, weights_size_);
    CUDA_CHECK(cudaGetLastError());
    adamw_update_kernel<<<(out_c_ + 255) / 256, 256>>>(d_bias_, d_grad_bias_, d_m_bias_, d_v_bias_, learning_rate, opt_config.beta1, opt_config.beta2, opt_config.eps, 0.0f, bc1, bc2, out_c_);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void relu_forward_kernel(const float* input, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = (x > 0.0f) ? x : 0.01f * x;
    }
}

__global__ void relu_backward_kernel(const float* input, const float* grad_output, float* grad_input, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.01f * grad_output[idx];
}

void GPUReLULayer::forward(const GPUTensor4D& input, GPUTensor4D& output) const {
    if (output.n != input.n || output.c != input.c || output.h != input.h || output.w != input.w)
        output.allocate(input.n, input.c, input.h, input.w);
#ifdef USE_OPTIMIZED_KERNELS
    gpu_relu_forward_opt(input, output);
#else
    size_t total = input.size();
    relu_forward_kernel<<<(total + 255) / 256, 256>>>(input.d_data, output.d_data, total);
    CUDA_CHECK(cudaGetLastError());
#endif
}

GPUBatchNorm2D::GPUBatchNorm2D(int num_features, float momentum, float eps) : num_features_(num_features), momentum_(momentum), eps_(eps) {
    CUDA_CHECK(cudaMalloc(&d_gamma_, num_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta_, num_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_running_mean_, num_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_running_var_, num_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_gamma_, num_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_beta_, num_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cache_mean_, num_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cache_var_, num_features * sizeof(float)));
    std::vector<float> ones(num_features, 1.0f), zeros(num_features, 0.0f);
    CUDA_CHECK(cudaMemcpy(d_gamma_, ones.data(), num_features * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta_, zeros.data(), num_features * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_running_mean_, zeros.data(), num_features * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_running_var_, ones.data(), num_features * sizeof(float), cudaMemcpyHostToDevice));
}

GPUBatchNorm2D::~GPUBatchNorm2D() {
    if (d_gamma_) cudaFree(d_gamma_); if (d_beta_) cudaFree(d_beta_);
    if (d_running_mean_) cudaFree(d_running_mean_); if (d_running_var_) cudaFree(d_running_var_);
    if (d_grad_gamma_) cudaFree(d_grad_gamma_); if (d_grad_beta_) cudaFree(d_grad_beta_);
    if (d_cache_mean_) cudaFree(d_cache_mean_); if (d_cache_var_) cudaFree(d_cache_var_);
}

void GPUBatchNorm2D::forward(const GPUTensor4D& input, GPUTensor4D& output, bool training) {
    if (output.n != input.n || output.c != input.c || output.h != input.h || output.w != input.w)
        output.allocate(input.n, input.c, input.h, input.w);
    int total = input.n * input.c * input.h * input.w;
    if (training) {
        batchnorm_compute_mean_kernel<<<num_features_, 256, 256 * sizeof(float)>>>(input.d_data, d_cache_mean_, input.n, input.c, input.h, input.w);
        CUDA_CHECK(cudaGetLastError());
        batchnorm_compute_var_kernel<<<num_features_, 256, 256 * sizeof(float)>>>(input.d_data, d_cache_mean_, d_cache_var_, input.n, input.c, input.h, input.w);
        CUDA_CHECK(cudaGetLastError());
        batchnorm_normalize_kernel<<<(total + 255) / 256, 256>>>(input.d_data, output.d_data, d_gamma_, d_beta_, d_cache_mean_, d_cache_var_, input.n, input.c, input.h, input.w, eps_);
        CUDA_CHECK(cudaGetLastError());
        batchnorm_update_running_kernel<<<(num_features_ + 255) / 256, 256>>>(d_running_mean_, d_cache_mean_, momentum_, num_features_);
        batchnorm_update_running_kernel<<<(num_features_ + 255) / 256, 256>>>(d_running_var_, d_cache_var_, momentum_, num_features_);
        CUDA_CHECK(cudaGetLastError());
    } else {
        batchnorm_normalize_kernel<<<(total + 255) / 256, 256>>>(input.d_data, output.d_data, d_gamma_, d_beta_, d_running_mean_, d_running_var_, input.n, input.c, input.h, input.w, eps_);
        CUDA_CHECK(cudaGetLastError());
    }
}

void GPUBatchNorm2D::backward(const GPUTensor4D& input, const GPUTensor4D& grad_output, GPUTensor4D& grad_input, float learning_rate) {
    if (grad_input.n != input.n || grad_input.c != input.c || grad_input.h != input.h || grad_input.w != input.w)
        grad_input.allocate(input.n, input.c, input.h, input.w);
    int total = input.n * input.c * input.h * input.w;
    CUDA_CHECK(cudaMemset(d_grad_gamma_, 0, num_features_ * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_grad_beta_, 0, num_features_ * sizeof(float)));
    float* d_sum_dy = nullptr, *d_sum_dy_xhat = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sum_dy, num_features_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sum_dy_xhat, num_features_ * sizeof(float)));
    batchnorm_compute_sums_kernel<<<num_features_, 256, 2 * 256 * sizeof(float)>>>(input.d_data, grad_output.d_data, d_cache_mean_, d_cache_var_, d_sum_dy, d_sum_dy_xhat, input.n, input.c, input.h, input.w, eps_);
    CUDA_CHECK(cudaGetLastError());
    batchnorm_backward_kernel_v2<<<(total + 255) / 256, 256>>>(input.d_data, grad_output.d_data, grad_input.d_data, d_grad_gamma_, d_grad_beta_, d_gamma_, d_cache_mean_, d_cache_var_, d_sum_dy, d_sum_dy_xhat, input.n, input.c, input.h, input.w, eps_);
    CUDA_CHECK(cudaGetLastError());
    cudaFree(d_sum_dy); cudaFree(d_sum_dy_xhat);
    sgd_update_kernel<<<(num_features_ + 255) / 256, 256>>>(d_gamma_, d_grad_gamma_, learning_rate, num_features_);
    sgd_update_kernel<<<(num_features_ + 255) / 256, 256>>>(d_beta_, d_grad_beta_, learning_rate, num_features_);
    CUDA_CHECK(cudaGetLastError());
}

void GPUReLULayer::backward(const GPUTensor4D& input, const GPUTensor4D& grad_output, GPUTensor4D& grad_input) const {
    if (grad_input.n != input.n || grad_input.c != input.c || grad_input.h != input.h || grad_input.w != input.w)
        grad_input.allocate(input.n, input.c, input.h, input.w);
#ifdef USE_OPTIMIZED_KERNELS
    gpu_relu_backward_opt(input, grad_output, grad_input);
#else
    size_t total = input.size();
    relu_backward_kernel<<<(total + 255) / 256, 256>>>(input.d_data, grad_output.d_data, grad_input.d_data, total);
    CUDA_CHECK(cudaGetLastError());
#endif
}

__global__ void maxpool2d_forward_kernel(const float* __restrict__ input, float* __restrict__ output, int batch_size, int channels, int in_h, int in_w, int out_h, int out_w, int k, int stride) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x, oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c_n = blockIdx.z, c = c_n % channels, n = c_n / channels;
    if (ow >= out_w || oh >= out_h || n >= batch_size) return;
    const size_t in_base = ((static_cast<size_t>(n) * channels + c) * in_h) * in_w;
    const int ih_base = oh * stride, iw_base = ow * stride;
    float max_val = -FLT_MAX;
    if (k == 2 && stride == 2) {
        size_t idx00 = in_base + ih_base * in_w + iw_base;
        max_val = fmaxf(fmaxf(input[idx00], input[idx00 + 1]), fmaxf(input[idx00 + in_w], input[idx00 + in_w + 1]));
    } else {
        for (int kh = 0; kh < k; ++kh) for (int kw = 0; kw < k; ++kw) {
            int ih = ih_base + kh, iw = iw_base + kw;
            if (ih < in_h && iw < in_w) max_val = fmaxf(max_val, input[in_base + ih * in_w + iw]);
        }
    }
    output[((static_cast<size_t>(n) * channels + c) * out_h + oh) * out_w + ow] = max_val;
}

__global__ void maxpool2d_backward_kernel(const float* __restrict__ input, const float* __restrict__ grad_output, float* __restrict__ grad_input,
    int batch_size, int channels, int in_h, int in_w, int out_h, int out_w, int k, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * channels * out_h * out_w) return;
    int ow = idx % out_w, temp = idx / out_w, oh = temp % out_h; temp = temp / out_h; int c = temp % channels, n = temp / channels;
    float max_val = -FLT_MAX; int max_ih = -1, max_iw = -1;
    for (int kh = 0; kh < k; ++kh) for (int kw = 0; kw < k; ++kw) {
        int ih = oh * stride + kh, iw = ow * stride + kw;
        if (ih < in_h && iw < in_w) {
            size_t in_idx = ((static_cast<size_t>(n) * channels + c) * in_h + ih) * in_w + iw;
            if (input[in_idx] > max_val) { max_val = input[in_idx]; max_ih = ih; max_iw = iw; }
        }
    }
    if (max_ih >= 0 && max_iw >= 0)
        atomicAdd(&grad_input[((static_cast<size_t>(n) * channels + c) * in_h + max_ih) * in_w + max_iw], grad_output[idx]);
}

GPUMaxPool2DLayer::GPUMaxPool2DLayer(int kernel_size, int stride) : k_(kernel_size), stride_(stride) {}

void GPUMaxPool2DLayer::forward(const GPUTensor4D& input, GPUTensor4D& output) const {
    int out_h = get_output_h(input.h), out_w = get_output_w(input.w);
    if (output.n != input.n || output.c != input.c || output.h != out_h || output.w != out_w)
        output.allocate(input.n, input.c, out_h, out_w);
#ifdef USE_OPTIMIZED_KERNELS
    gpu_maxpool2d_forward_opt(input, output, k_, stride_);
#else
    dim3 block(16, 16), grid((out_w + 15) / 16, (out_h + 15) / 16, input.n * input.c);
    maxpool2d_forward_kernel<<<grid, block>>>(input.d_data, output.d_data, input.n, input.c, input.h, input.w, out_h, out_w, k_, stride_);
    CUDA_CHECK(cudaGetLastError());
#endif
}

void GPUMaxPool2DLayer::backward(const GPUTensor4D& input, const GPUTensor4D& grad_output, GPUTensor4D& grad_input) const {
    if (grad_input.n != input.n || grad_input.c != input.c || grad_input.h != input.h || grad_input.w != input.w)
        grad_input.allocate(input.n, input.c, input.h, input.w);
    CUDA_CHECK(cudaMemset(grad_input.d_data, 0, grad_input.bytes()));
    int total = input.n * input.c * grad_output.h * grad_output.w;
    maxpool2d_backward_kernel<<<(total + 255) / 256, 256>>>(input.d_data, grad_output.d_data, grad_input.d_data, input.n, input.c, input.h, input.w, grad_output.h, grad_output.w, k_, stride_);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void upsample2d_forward_kernel(const float* __restrict__ input, float* __restrict__ output, int batch_size, int channels, int in_h, int in_w, int out_h, int out_w, int scale) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x, oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c_n = blockIdx.z, c = c_n % channels, n = c_n / channels;
    if (ow >= out_w || oh >= out_h || n >= batch_size) return;
    output[((static_cast<size_t>(n) * channels + c) * out_h + oh) * out_w + ow] = input[((static_cast<size_t>(n) * channels + c) * in_h + oh / scale) * in_w + ow / scale];
}

__global__ void upsample2d_backward_kernel(const float* __restrict__ grad_output, float* __restrict__ grad_input, int batch_size, int channels, int in_h, int in_w, int out_h, int out_w, int scale) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x, oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c_n = blockIdx.z, c = c_n % channels, n = c_n / channels;
    if (ow >= out_w || oh >= out_h || n >= batch_size) return;
    atomicAdd(&grad_input[((static_cast<size_t>(n) * channels + c) * in_h + oh / scale) * in_w + ow / scale], grad_output[((static_cast<size_t>(n) * channels + c) * out_h + oh) * out_w + ow]);
}

GPUUpSample2DLayer::GPUUpSample2DLayer(int scale) : scale_(scale) {}

void GPUUpSample2DLayer::forward(const GPUTensor4D& input, GPUTensor4D& output) const {
    int out_h = get_output_h(input.h), out_w = get_output_w(input.w);
    if (output.n != input.n || output.c != input.c || output.h != out_h || output.w != out_w)
        output.allocate(input.n, input.c, out_h, out_w);
#ifdef USE_OPTIMIZED_KERNELS
    gpu_upsample2d_forward_opt(input, output, scale_);
#else
    dim3 block(16, 16), grid((out_w + 15) / 16, (out_h + 15) / 16, input.n * input.c);
    upsample2d_forward_kernel<<<grid, block>>>(input.d_data, output.d_data, input.n, input.c, input.h, input.w, out_h, out_w, scale_);
    CUDA_CHECK(cudaGetLastError());
#endif
}

void GPUUpSample2DLayer::backward(const GPUTensor4D& input, const GPUTensor4D& grad_output, GPUTensor4D& grad_input) const {
    if (grad_input.n != input.n || grad_input.c != input.c || grad_input.h != input.h || grad_input.w != input.w)
        grad_input.allocate(input.n, input.c, input.h, input.w);
    CUDA_CHECK(cudaMemset(grad_input.d_data, 0, grad_input.bytes()));
    int out_h = grad_output.h, out_w = grad_output.w;
    dim3 block(16, 16), grid((out_w + 15) / 16, (out_h + 15) / 16, input.n * input.c);
    upsample2d_backward_kernel<<<grid, block>>>(grad_output.d_data, grad_input.d_data, input.n, input.c, input.h, input.w, out_h, out_w, scale_);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void mse_loss_kernel(const float* __restrict__ output, const float* __restrict__ target, float* __restrict__ partial_sums, size_t n) {
    extern __shared__ float sdata[];
    size_t tid = threadIdx.x, idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    float val = 0.0f;
    if (idx < n) { float diff = output[idx] - target[idx]; val = diff * diff; }
    if (idx + blockDim.x < n) { float diff = output[idx + blockDim.x] - target[idx + blockDim.x]; val += diff * diff; }
    sdata[tid] = val; __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) { if (tid < s) sdata[tid] += sdata[tid + s]; __syncthreads(); }
    if (tid < 32) {
        volatile float* vsmem = sdata;
        if (blockDim.x >= 64) vsmem[tid] += vsmem[tid + 32];
        float myVal = vsmem[tid];
        for (int offset = 16; offset > 0; offset /= 2) myVal += __shfl_down_sync(0xffffffff, myVal, offset);
        if (tid == 0) partial_sums[blockIdx.x] = myVal;
    }
}

__global__ void mse_grad_kernel(const float* __restrict__ output, const float* __restrict__ target, float* __restrict__ grad_output, float scale, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) grad_output[idx] = scale * (output[idx] - target[idx]);
}

static float* d_persistent_partial_sums = nullptr;
static size_t persistent_partial_sums_size = 0;

float gpu_mse_loss(const GPUTensor4D& output, const GPUTensor4D& target) {
    size_t n = output.size();
    int block_size = 256, grid_size = (n + block_size * 2 - 1) / (block_size * 2);
    if (d_persistent_partial_sums == nullptr || persistent_partial_sums_size < static_cast<size_t>(grid_size)) {
        if (d_persistent_partial_sums) cudaFree(d_persistent_partial_sums);
        persistent_partial_sums_size = static_cast<size_t>(grid_size) * 2;
        CUDA_CHECK(cudaMalloc(&d_persistent_partial_sums, persistent_partial_sums_size * sizeof(float)));
    }
    mse_loss_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(output.d_data, target.d_data, d_persistent_partial_sums, n);
    CUDA_CHECK(cudaGetLastError());
    std::vector<float> h_partial_sums(grid_size);
    CUDA_CHECK(cudaMemcpy(h_partial_sums.data(), d_persistent_partial_sums, grid_size * sizeof(float), cudaMemcpyDeviceToHost));
    float sum = 0.0f;
    for (int i = 0; i < grid_size; ++i) sum += h_partial_sums[i];
    return sum / static_cast<float>(n);
}

float gpu_mse_loss_with_grad(const GPUTensor4D& output, const GPUTensor4D& target, GPUTensor4D& grad_output) {
    size_t n = output.size();
    if (grad_output.n != output.n || grad_output.c != output.c || grad_output.h != output.h || grad_output.w != output.w)
        grad_output.allocate(output.n, output.c, output.h, output.w);
    mse_grad_kernel<<<(n + 255) / 256, 256>>>(output.d_data, target.d_data, grad_output.d_data, 2.0f / static_cast<float>(n), n);
    CUDA_CHECK(cudaGetLastError());
    return gpu_mse_loss(output, target);
}

__global__ void sigmoid_forward_kernel(const float* __restrict__ input, float* __restrict__ output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = (x >= 0) ? (1.0f / (1.0f + expf(-x))) : (expf(x) / (1.0f + expf(x)));
    }
}

__global__ void sigmoid_backward_kernel(const float* __restrict__ output, const float* __restrict__ grad_output, float* __restrict__ grad_input, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) { float s = output[idx]; grad_input[idx] = grad_output[idx] * s * (1.0f - s); }
}

void GPUSigmoidLayer::forward(const GPUTensor4D& input, GPUTensor4D& output) const {
    if (output.n != input.n || output.c != input.c || output.h != input.h || output.w != input.w)
        output.allocate(input.n, input.c, input.h, input.w);
    size_t n = input.size();
    sigmoid_forward_kernel<<<(n + 255) / 256, 256>>>(input.d_data, output.d_data, n);
    CUDA_CHECK(cudaGetLastError());
}

void GPUSigmoidLayer::backward(const GPUTensor4D& output, const GPUTensor4D& grad_output, GPUTensor4D& grad_input) const {
    if (grad_input.n != output.n || grad_input.c != output.c || grad_input.h != output.h || grad_input.w != output.w)
        grad_input.allocate(output.n, output.c, output.h, output.w);
    size_t n = output.size();
    sigmoid_backward_kernel<<<(n + 255) / 256, 256>>>(output.d_data, grad_output.d_data, grad_input.d_data, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void bce_loss_kernel(const float* __restrict__ output, const float* __restrict__ target, float* __restrict__ partial_sums, size_t n) {
    extern __shared__ float sdata[];
    size_t tid = threadIdx.x, idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    const float eps = 1e-7f;
    float val = 0.0f;
    if (idx < n) { float y = target[idx], p = fmaxf(fminf(output[idx], 1.0f - eps), eps); val = -(y * logf(p) + (1.0f - y) * logf(1.0f - p)); }
    if (idx + blockDim.x < n) { float y = target[idx + blockDim.x], p = fmaxf(fminf(output[idx + blockDim.x], 1.0f - eps), eps); val += -(y * logf(p) + (1.0f - y) * logf(1.0f - p)); }
    sdata[tid] = val; __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) { if (tid < s) sdata[tid] += sdata[tid + s]; __syncthreads(); }
    if (tid < 32) {
        volatile float* vsmem = sdata;
        if (blockDim.x >= 64) vsmem[tid] += vsmem[tid + 32];
        float myVal = vsmem[tid];
        for (int offset = 16; offset > 0; offset /= 2) myVal += __shfl_down_sync(0xffffffff, myVal, offset);
        if (tid == 0) partial_sums[blockIdx.x] = myVal;
    }
}

__global__ void bce_grad_kernel(const float* __restrict__ output, const float* __restrict__ target, float* __restrict__ grad_output, float scale, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const float eps = 1e-7f;
        float y = target[idx], p = fmaxf(fminf(output[idx], 1.0f - eps), eps);
        grad_output[idx] = scale * (p - y) / (p * (1.0f - p) + eps);
    }
}

static float* d_bce_partial_sums = nullptr;
static size_t bce_partial_sums_size = 0;

float gpu_bce_loss(const GPUTensor4D& output, const GPUTensor4D& target) {
    size_t n = output.size();
    int block_size = 256, grid_size = (n + block_size * 2 - 1) / (block_size * 2);
    if (d_bce_partial_sums == nullptr || bce_partial_sums_size < static_cast<size_t>(grid_size)) {
        if (d_bce_partial_sums) cudaFree(d_bce_partial_sums);
        bce_partial_sums_size = static_cast<size_t>(grid_size) * 2;
        CUDA_CHECK(cudaMalloc(&d_bce_partial_sums, bce_partial_sums_size * sizeof(float)));
    }
    bce_loss_kernel<<<grid_size, block_size, block_size * sizeof(float)>>>(output.d_data, target.d_data, d_bce_partial_sums, n);
    CUDA_CHECK(cudaGetLastError());
    std::vector<float> h_partial_sums(grid_size);
    CUDA_CHECK(cudaMemcpy(h_partial_sums.data(), d_bce_partial_sums, grid_size * sizeof(float), cudaMemcpyDeviceToHost));
    float sum = 0.0f;
    for (int i = 0; i < grid_size; ++i) sum += h_partial_sums[i];
    return sum / static_cast<float>(n);
}

float gpu_bce_loss_with_grad(const GPUTensor4D& output, const GPUTensor4D& target, GPUTensor4D& grad_output) {
    size_t n = output.size();
    if (grad_output.n != output.n || grad_output.c != output.c || grad_output.h != output.h || grad_output.w != output.w)
        grad_output.allocate(output.n, output.c, output.h, output.w);
    bce_grad_kernel<<<(n + 255) / 256, 256>>>(output.d_data, target.d_data, grad_output.d_data, 1.0f / static_cast<float>(n), n);
    CUDA_CHECK(cudaGetLastError());
    return gpu_bce_loss(output, target);
}
