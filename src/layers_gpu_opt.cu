#ifdef USE_OPTIMIZED_KERNELS

#include "gpu_layer.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <cstdio>

static cublasHandle_t cublas_handle = nullptr;
static bool cublas_initialized = false;
static cudnnHandle_t cudnn_handle = nullptr;
static bool cudnn_initialized = false;

void init_cublas() { if (!cublas_initialized) { cublasCreate(&cublas_handle); cublas_initialized = true; } }
void cleanup_cublas() { if (cublas_initialized && cublas_handle) { cublasDestroy(cublas_handle); cublas_handle = nullptr; cublas_initialized = false; } }
void init_cudnn() { if (!cudnn_initialized) { cudnnCreate(&cudnn_handle); cudnn_initialized = true; } }
void cleanup_cudnn() { if (cudnn_initialized && cudnn_handle) { cudnnDestroy(cudnn_handle); cudnn_handle = nullptr; cudnn_initialized = false; } }

void gpu_conv2d_forward_cudnn(const GPUTensor4D& input, const float* d_weights, const float* d_bias, GPUTensor4D& output, int in_c, int out_c, int k, int stride, int padding) {
    init_cudnn();
    int out_h = (input.h + 2 * padding - k) / stride + 1, out_w = (input.w + 2 * padding - k) / stride + 1;
    if (output.n != input.n || output.c != out_c || output.h != out_h || output.w != out_w) output.allocate(input.n, out_c, out_h, out_w);
    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc; cudnnFilterDescriptor_t filter_desc; cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateTensorDescriptor(&input_desc); cudnnCreateTensorDescriptor(&output_desc); cudnnCreateTensorDescriptor(&bias_desc);
    cudnnCreateFilterDescriptor(&filter_desc); cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input.n, in_c, input.h, input.w);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input.n, out_c, out_h, out_w);
    cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, out_c, 1, 1);
    cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_c, in_c, k, k);
    cudnnSetConvolution2dDescriptor(conv_desc, padding, padding, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    cudnnConvolutionFwdAlgoPerf_t algo_perf; int returned_algo_count;
    cudnnGetConvolutionForwardAlgorithm_v7(cudnn_handle, input_desc, filter_desc, conv_desc, output_desc, 1, &returned_algo_count, &algo_perf);
    size_t workspace_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle, input_desc, filter_desc, conv_desc, output_desc, algo_perf.algo, &workspace_size);
    void* d_workspace = nullptr; if (workspace_size > 0) cudaMalloc(&d_workspace, workspace_size);
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(cudnn_handle, &alpha, input_desc, input.d_data, filter_desc, d_weights, conv_desc, algo_perf.algo, d_workspace, workspace_size, &beta, output_desc, output.d_data);
    beta = 1.0f; cudnnAddTensor(cudnn_handle, &alpha, bias_desc, d_bias, &beta, output_desc, output.d_data);
    if (d_workspace) cudaFree(d_workspace);
    cudnnDestroyTensorDescriptor(input_desc); cudnnDestroyTensorDescriptor(output_desc); cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyFilterDescriptor(filter_desc); cudnnDestroyConvolutionDescriptor(conv_desc);
}

void gpu_conv2d_backward_cudnn(const GPUTensor4D& input, const GPUTensor4D& grad_output, const float* d_weights, float* d_grad_weights, float* d_grad_bias, GPUTensor4D& grad_input, int in_c, int out_c, int k, int stride, int padding, float learning_rate) {
    init_cudnn();
    int out_h = grad_output.h, out_w = grad_output.w;
    if (grad_input.n != input.n || grad_input.c != in_c || grad_input.h != input.h || grad_input.w != input.w) grad_input.allocate(input.n, in_c, input.h, input.w);
    cudnnTensorDescriptor_t input_desc, grad_output_desc, grad_input_desc, bias_desc; cudnnFilterDescriptor_t filter_desc; cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateTensorDescriptor(&input_desc); cudnnCreateTensorDescriptor(&grad_output_desc); cudnnCreateTensorDescriptor(&grad_input_desc);
    cudnnCreateTensorDescriptor(&bias_desc); cudnnCreateFilterDescriptor(&filter_desc); cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input.n, in_c, input.h, input.w);
    cudnnSetTensor4dDescriptor(grad_output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input.n, out_c, out_h, out_w);
    cudnnSetTensor4dDescriptor(grad_input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input.n, in_c, input.h, input.w);
    cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, out_c, 1, 1);
    cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, out_c, in_c, k, k);
    cudnnSetConvolution2dDescriptor(conv_desc, padding, padding, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionBwdDataAlgoPerf_t data_algo_perf; int returned_algo_count;
    cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnn_handle, filter_desc, grad_output_desc, conv_desc, grad_input_desc, 1, &returned_algo_count, &data_algo_perf);
    size_t data_workspace_size = 0;
    cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle, filter_desc, grad_output_desc, conv_desc, grad_input_desc, data_algo_perf.algo, &data_workspace_size);
    void* d_data_workspace = nullptr; if (data_workspace_size > 0) cudaMalloc(&d_data_workspace, data_workspace_size);
    cudnnConvolutionBackwardData(cudnn_handle, &alpha, filter_desc, d_weights, grad_output_desc, grad_output.d_data, conv_desc, data_algo_perf.algo, d_data_workspace, data_workspace_size, &beta, grad_input_desc, grad_input.d_data);
    if (d_data_workspace) cudaFree(d_data_workspace);
    cudnnConvolutionBwdFilterAlgoPerf_t filter_algo_perf;
    cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnn_handle, input_desc, grad_output_desc, conv_desc, filter_desc, 1, &returned_algo_count, &filter_algo_perf);
    size_t filter_workspace_size = 0;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle, input_desc, grad_output_desc, conv_desc, filter_desc, filter_algo_perf.algo, &filter_workspace_size);
    void* d_filter_workspace = nullptr; if (filter_workspace_size > 0) cudaMalloc(&d_filter_workspace, filter_workspace_size);
    cudnnConvolutionBackwardFilter(cudnn_handle, &alpha, input_desc, input.d_data, grad_output_desc, grad_output.d_data, conv_desc, filter_algo_perf.algo, d_filter_workspace, filter_workspace_size, &beta, filter_desc, d_grad_weights);
    if (d_filter_workspace) cudaFree(d_filter_workspace);
    cudnnConvolutionBackwardBias(cudnn_handle, &alpha, grad_output_desc, grad_output.d_data, &beta, bias_desc, d_grad_bias);
    cudnnDestroyTensorDescriptor(input_desc); cudnnDestroyTensorDescriptor(grad_output_desc); cudnnDestroyTensorDescriptor(grad_input_desc);
    cudnnDestroyTensorDescriptor(bias_desc); cudnnDestroyFilterDescriptor(filter_desc); cudnnDestroyConvolutionDescriptor(conv_desc);
}

void gpu_conv2d_forward_cudnn_wrapper(const GPUTensor4D& input, const GPUConv2DLayer& conv, GPUTensor4D& output) {
    gpu_conv2d_forward_cudnn(input, conv.get_weights(), conv.get_bias(), output, conv.get_in_channels(), conv.get_out_channels(), conv.get_kernel_size(), conv.get_stride(), conv.get_padding());
}

void gpu_conv2d_backward_cudnn_full(const GPUTensor4D& input, const GPUTensor4D& grad_output, GPUTensor4D& grad_input, const GPUConv2DLayer& conv, float learning_rate) {
    gpu_conv2d_backward_cudnn(input, grad_output, conv.get_weights(), conv.get_grad_weights(), conv.get_grad_bias(), grad_input, conv.get_in_channels(), conv.get_out_channels(), conv.get_kernel_size(), conv.get_stride(), conv.get_padding(), learning_rate);
    gpu_sgd_update_opt(conv.get_weights(), conv.get_grad_weights(), learning_rate, conv.get_weights_size());
    gpu_sgd_update_opt(conv.get_bias(), conv.get_grad_bias(), learning_rate, conv.get_out_channels());
}

// AdamW kernel declaration (defined in layers_gpu.cu)
extern __global__ void adamw_update_kernel(float* params, const float* grads, float* m, float* v, 
    float lr, float beta1, float beta2, float eps, float weight_decay, float bias_correction1, float bias_correction2, size_t n);

void gpu_conv2d_backward_cudnn_adamw(const GPUTensor4D& input, const GPUTensor4D& grad_output, GPUTensor4D& grad_input, 
                                     GPUConv2DLayer& conv, float learning_rate, const OptimizerConfig& opt_config) {
    // Compute gradients using cuDNN
    gpu_conv2d_backward_cudnn(input, grad_output, conv.get_weights(), conv.get_grad_weights(), conv.get_grad_bias(), 
                              grad_input, conv.get_in_channels(), conv.get_out_channels(), 
                              conv.get_kernel_size(), conv.get_stride(), conv.get_padding(), learning_rate);
    
    // Apply AdamW update instead of SGD
    float bc1 = 1.0f - powf(opt_config.beta1, opt_config.step + 1);
    float bc2 = 1.0f - powf(opt_config.beta2, opt_config.step + 1);
    size_t weights_size = conv.get_weights_size();
    int out_c = conv.get_out_channels();
    
    // Update weights with AdamW
    adamw_update_kernel<<<(weights_size + 255) / 256, 256>>>(
        conv.get_weights(), conv.get_grad_weights(), conv.get_m_weights(), conv.get_v_weights(),
        learning_rate, opt_config.beta1, opt_config.beta2, opt_config.eps, opt_config.weight_decay, bc1, bc2, weights_size);
    CUDA_CHECK(cudaGetLastError());
    
    // Update bias with AdamW (no weight decay for bias)
    adamw_update_kernel<<<(out_c + 255) / 256, 256>>>(
        conv.get_bias(), conv.get_grad_bias(), conv.get_m_bias(), conv.get_v_bias(),
        learning_rate, opt_config.beta1, opt_config.beta2, opt_config.eps, 0.0f, bc1, bc2, out_c);
    CUDA_CHECK(cudaGetLastError());
}

extern __global__ void relu_forward_kernel(const float* input, float* output, size_t n);
#define TILE_WIDTH 16
#define TILE_HEIGHT 16

__constant__ float c_weights_small[8192];
__constant__ float c_bias_small[256];

__global__ void conv2d_forward_tiled_kernel(const float* __restrict__ input, const float* __restrict__ weights, const float* __restrict__ bias, float* __restrict__ output,
    int batch_size, int in_c, int in_h, int in_w, int out_c, int out_h, int out_w, int k, int stride, int padding) {
    extern __shared__ float shared_mem[];
    int ow = blockIdx.x * TILE_WIDTH + threadIdx.x, oh = blockIdx.y * TILE_HEIGHT + threadIdx.y;
    int oc = blockIdx.z % out_c, n = blockIdx.z / out_c;
    if (ow >= out_w || oh >= out_h || n >= batch_size) return;
    float sum = bias[oc];
    const int tile_h = TILE_HEIGHT * stride + k - stride, tile_w = TILE_WIDTH * stride + k - stride;
    const int tile_size = tile_h * tile_w, threads_per_block = blockDim.x * blockDim.y;
    const int num_loads = (tile_size + threads_per_block - 1) / threads_per_block;
    const int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
    float* s_input = shared_mem;
    for (int ic = 0; ic < in_c; ++ic) {
        const int in_start_h = blockIdx.y * TILE_HEIGHT * stride - padding, in_start_w = blockIdx.x * TILE_WIDTH * stride - padding;
        const size_t in_channel_offset = (static_cast<size_t>(n) * in_c + ic) * in_h * in_w;
        for (int load = 0; load < num_loads; ++load) {
            int linear_idx = load * threads_per_block + linear_tid;
            if (linear_idx < tile_size) {
                int sh = linear_idx / tile_w, sw = linear_idx % tile_w;
                int ih = in_start_h + sh, iw = in_start_w + sw;
                s_input[linear_idx] = (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) ? input[in_channel_offset + ih * in_w + iw] : 0.0f;
            }
        }
        __syncthreads();
        const int local_h = threadIdx.y * stride, local_w = threadIdx.x * stride;
        const size_t w_ic_offset = (static_cast<size_t>(oc) * in_c + ic) * k * k;
        for (int kh = 0; kh < k; ++kh) for (int kw = 0; kw < k; ++kw) sum += s_input[(local_h + kh) * tile_w + local_w + kw] * weights[w_ic_offset + kh * k + kw];
        __syncthreads();
    }
    output[((static_cast<size_t>(n) * out_c + oc) * out_h + oh) * out_w + ow] = sum;
}

__global__ void conv2d_relu_forward_kernel(const float* __restrict__ input, const float* __restrict__ weights, const float* __restrict__ bias, float* __restrict__ output,
    int batch_size, int in_c, int in_h, int in_w, int out_c, int out_h, int out_w, int k, int stride, int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_c * out_h * out_w) return;
    int ow = idx % out_w, temp = idx / out_w, oh = temp % out_h; temp = temp / out_h; int oc = temp % out_c, n = temp / out_c;
    float sum = bias[oc];
    for (int ic = 0; ic < in_c; ++ic) for (int kh = 0; kh < k; ++kh) for (int kw = 0; kw < k; ++kw) {
        int ih = oh * stride + kh - padding, iw = ow * stride + kw - padding;
        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) sum += input[((static_cast<size_t>(n) * in_c + ic) * in_h + ih) * in_w + iw] * weights[((static_cast<size_t>(oc) * in_c + ic) * k + kh) * k + kw];
    }
    output[idx] = fmaxf(0.0f, sum);
}

__global__ void relu_forward_vectorized_kernel(const float4* input, float4* output, size_t n4) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) { 
        float4 in = input[idx]; 
        output[idx] = make_float4(
            (in.x > 0.0f) ? in.x : 0.01f * in.x,
            (in.y > 0.0f) ? in.y : 0.01f * in.y,
            (in.z > 0.0f) ? in.z : 0.01f * in.z,
            (in.w > 0.0f) ? in.w : 0.01f * in.w);  // LeakyReLU
    }
}

__global__ void relu_backward_vectorized_kernel(const float4* input, const float4* grad_output, float4* grad_input, size_t n4) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 in = input[idx], go = grad_output[idx];
        grad_input[idx] = make_float4(
            (in.x > 0.0f) ? go.x : 0.01f * go.x,
            (in.y > 0.0f) ? go.y : 0.01f * go.y,
            (in.z > 0.0f) ? go.z : 0.01f * go.z,
            (in.w > 0.0f) ? go.w : 0.01f * go.w);
    }
}

__global__ void maxpool2d_forward_opt_kernel(const float* __restrict__ input, float* __restrict__ output, int batch_size, int channels, int in_h, int in_w, int out_h, int out_w, int k, int stride) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x, oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels, n = blockIdx.z / channels;
    if (ow >= out_w || oh >= out_h || n >= batch_size) return;
    float max_val = -1e30f;
    for (int kh = 0; kh < k; ++kh) for (int kw = 0; kw < k; ++kw) {
        int ih = oh * stride + kh, iw = ow * stride + kw;
        if (ih < in_h && iw < in_w) max_val = fmaxf(max_val, input[((static_cast<size_t>(n) * channels + c) * in_h + ih) * in_w + iw]);
    }
    output[((static_cast<size_t>(n) * channels + c) * out_h + oh) * out_w + ow] = max_val;
}

__global__ void upsample2d_forward_opt_kernel(const float* __restrict__ input, float* __restrict__ output, int batch_size, int channels, int in_h, int in_w, int out_h, int out_w, int scale) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x, oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels, n = blockIdx.z / channels;
    if (ow >= out_w || oh >= out_h || n >= batch_size) return;
    output[((static_cast<size_t>(n) * channels + c) * out_h + oh) * out_w + ow] = input[((static_cast<size_t>(n) * channels + c) * in_h + oh / scale) * in_w + ow / scale];
}

__global__ void mse_loss_grad_fused_kernel(const float* __restrict__ output, const float* __restrict__ target, float* __restrict__ grad_output, float* __restrict__ partial_loss, float scale, size_t n) {
    extern __shared__ float sdata[];
    size_t tid = threadIdx.x, idx = blockIdx.x * blockDim.x + threadIdx.x;
    float local_loss = 0.0f;
    if (idx < n) { float diff = output[idx] - target[idx]; local_loss = diff * diff; grad_output[idx] = scale * diff; }
    sdata[tid] = local_loss; __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) sdata[tid] += sdata[tid + s]; __syncthreads(); }
    if (tid == 0) partial_loss[blockIdx.x] = sdata[0];
}

void gpu_relu_forward_opt(const GPUTensor4D& input, GPUTensor4D& output) {
    size_t n = input.size(), n4 = n / 4;
    if (output.n != input.n || output.c != input.c || output.h != input.h || output.w != input.w) output.allocate(input.n, input.c, input.h, input.w);
    if (n % 4 == 0) { relu_forward_vectorized_kernel<<<(n4 + 255) / 256, 256>>>(reinterpret_cast<const float4*>(input.d_data), reinterpret_cast<float4*>(output.d_data), n4); }
    else { relu_forward_kernel<<<(n + 255) / 256, 256>>>(input.d_data, output.d_data, n); }
    CUDA_CHECK(cudaGetLastError());
}

void gpu_relu_backward_opt(const GPUTensor4D& input, const GPUTensor4D& grad_output, GPUTensor4D& grad_input) {
    size_t n = input.size(), n4 = n / 4;
    if (grad_input.n != input.n || grad_input.c != input.c || grad_input.h != input.h || grad_input.w != input.w) grad_input.allocate(input.n, input.c, input.h, input.w);
    if (n % 4 == 0) { relu_backward_vectorized_kernel<<<(n4 + 255) / 256, 256>>>(reinterpret_cast<const float4*>(input.d_data), reinterpret_cast<const float4*>(grad_output.d_data), reinterpret_cast<float4*>(grad_input.d_data), n4); }
    CUDA_CHECK(cudaGetLastError());
}

void gpu_maxpool2d_forward_opt(const GPUTensor4D& input, GPUTensor4D& output, int k, int stride) {
    int out_h = (input.h - k) / stride + 1, out_w = (input.w - k) / stride + 1;
    if (output.n != input.n || output.c != input.c || output.h != out_h || output.w != out_w) output.allocate(input.n, input.c, out_h, out_w);
    dim3 block(16, 16), grid((out_w + 15) / 16, (out_h + 15) / 16, input.n * input.c);
    maxpool2d_forward_opt_kernel<<<grid, block>>>(input.d_data, output.d_data, input.n, input.c, input.h, input.w, out_h, out_w, k, stride);
    CUDA_CHECK(cudaGetLastError());
}

void gpu_upsample2d_forward_opt(const GPUTensor4D& input, GPUTensor4D& output, int scale) {
    int out_h = input.h * scale, out_w = input.w * scale;
    if (output.n != input.n || output.c != input.c || output.h != out_h || output.w != out_w) output.allocate(input.n, input.c, out_h, out_w);
    dim3 block(16, 16), grid((out_w + 15) / 16, (out_h + 15) / 16, input.n * input.c);
    upsample2d_forward_opt_kernel<<<grid, block>>>(input.d_data, output.d_data, input.n, input.c, input.h, input.w, out_h, out_w, scale);
    CUDA_CHECK(cudaGetLastError());
}

void gpu_conv2d_relu_forward_opt(const GPUTensor4D& input, const float* d_weights, const float* d_bias, GPUTensor4D& output, int in_c, int out_c, int k, int stride, int padding) {
    int out_h = (input.h + 2 * padding - k) / stride + 1, out_w = (input.w + 2 * padding - k) / stride + 1;
    if (output.n != input.n || output.c != out_c || output.h != out_h || output.w != out_w) output.allocate(input.n, out_c, out_h, out_w);
    int total = input.n * out_c * out_h * out_w;
    conv2d_relu_forward_kernel<<<(total + 255) / 256, 256>>>(input.d_data, d_weights, d_bias, output.d_data, input.n, in_c, input.h, input.w, out_c, out_h, out_w, k, stride, padding);
    CUDA_CHECK(cudaGetLastError());
}

void gpu_conv2d_forward_tiled(const GPUTensor4D& input, const float* d_weights, const float* d_bias, GPUTensor4D& output, int in_c, int out_c, int k, int stride, int padding) {
    int out_h = (input.h + 2 * padding - k) / stride + 1, out_w = (input.w + 2 * padding - k) / stride + 1;
    if (output.n != input.n || output.c != out_c || output.h != out_h || output.w != out_w) output.allocate(input.n, out_c, out_h, out_w);
    dim3 block(TILE_WIDTH, TILE_HEIGHT), grid((out_w + TILE_WIDTH - 1) / TILE_WIDTH, (out_h + TILE_HEIGHT - 1) / TILE_HEIGHT, input.n * out_c);
    int tile_h = TILE_HEIGHT * stride + k - stride, tile_w = TILE_WIDTH * stride + k - stride;
    conv2d_forward_tiled_kernel<<<grid, block, tile_h * tile_w * sizeof(float)>>>(input.d_data, d_weights, d_bias, output.d_data, input.n, in_c, input.h, input.w, out_c, out_h, out_w, k, stride, padding);
    CUDA_CHECK(cudaGetLastError());
}

void gpu_conv2d_forward_opt(const GPUTensor4D& input, const GPUConv2DLayer& conv, GPUTensor4D& output) {
    gpu_conv2d_forward_tiled(input, conv.get_weights(), conv.get_bias(), output, conv.get_in_channels(), conv.get_out_channels(), conv.get_kernel_size(), conv.get_stride(), conv.get_padding());
}

void gpu_conv2d_relu_fused_forward(const GPUTensor4D& input, const GPUConv2DLayer& conv, GPUTensor4D& output) {
    gpu_conv2d_relu_forward_opt(input, conv.get_weights(), conv.get_bias(), output, conv.get_in_channels(), conv.get_out_channels(), conv.get_kernel_size(), conv.get_stride(), conv.get_padding());
}

__global__ void col2im_backward_kernel(const float* __restrict__ grad_output, const float* __restrict__ weights, float* __restrict__ grad_input,
    int batch_size, int in_c, int in_h, int in_w, int out_c, int out_h, int out_w, int k, int stride, int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_c * out_h * out_w) return;
    int ow = idx % out_w, tmp = idx / out_w, oh = tmp % out_h; tmp = tmp / out_h; int oc = tmp % out_c, n = tmp / out_c;
    float grad_val = grad_output[idx];
    for (int ic = 0; ic < in_c; ++ic) {
        size_t w_base = (static_cast<size_t>(oc) * in_c + ic) * k * k;
        for (int kh = 0; kh < k; ++kh) { int ih = oh * stride + kh - padding; if (ih < 0 || ih >= in_h) continue;
            for (int kw = 0; kw < k; ++kw) { int iw = ow * stride + kw - padding; if (iw < 0 || iw >= in_w) continue;
                atomicAdd(&grad_input[((static_cast<size_t>(n) * in_c + ic) * in_h + ih) * in_w + iw], grad_val * weights[w_base + kh * k + kw]);
            }
        }
    }
}

__global__ void conv2d_backward_data_opt_kernel(const float* __restrict__ grad_output, const float* __restrict__ weights, float* __restrict__ grad_input,
    int batch_size, int in_c, int in_h, int in_w, int out_c, int out_h, int out_w, int k, int stride, int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * in_c * in_h * in_w) return;
    int iw = idx % in_w, tmp = idx / in_w, ih = tmp % in_h; tmp = tmp / in_h; int ic = tmp % in_c, n = tmp / in_c;
    float sum = 0.0f;
    int oh_start = max(0, ih + padding - k + 1), oh_end = min(out_h, ih + padding + 1);
    int ow_start = max(0, iw + padding - k + 1), ow_end = min(out_w, iw + padding + 1);
    for (int oc = 0; oc < out_c; ++oc) {
        size_t w_ic_base = (static_cast<size_t>(oc) * in_c + ic) * k * k;
        for (int oh = oh_start; oh < oh_end; ++oh) {
            int kh = ih + padding - oh;
            for (int ow = ow_start; ow < ow_end; ++ow) {
                int kw = iw + padding - ow;
                sum += grad_output[((static_cast<size_t>(n) * out_c + oc) * out_h + oh) * out_w + ow] * weights[w_ic_base + kh * k + kw];
            }
        }
    }
    grad_input[idx] = sum;
}

__global__ void conv2d_backward_weights_opt_kernel(const float* __restrict__ input, const float* __restrict__ grad_output, float* __restrict__ grad_weights,
    int batch_size, int in_c, int in_h, int in_w, int out_c, int out_h, int out_w, int k, int stride, int padding) {
    extern __shared__ float sdata[];
    int weight_idx = blockIdx.x;
    if (weight_idx >= out_c * in_c * k * k) return;
    int kw = weight_idx % k, temp = weight_idx / k, kh = temp % k; temp = temp / k; int ic = temp % in_c, oc = temp / in_c;
    int tid = threadIdx.x, total_outputs = batch_size * out_h * out_w;
    float local_sum = 0.0f;
    for (int i = tid; i < total_outputs; i += blockDim.x) {
        int ow = i % out_w, temp2 = i / out_w, oh = temp2 % out_h, n = temp2 / out_h;
        int ih = oh * stride + kh - padding, iw_ = ow * stride + kw - padding;
        if (ih >= 0 && ih < in_h && iw_ >= 0 && iw_ < in_w)
            local_sum += input[((static_cast<size_t>(n) * in_c + ic) * in_h + ih) * in_w + iw_] * grad_output[((static_cast<size_t>(n) * out_c + oc) * out_h + oh) * out_w + ow];
    }
    sdata[tid] = local_sum; __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) sdata[tid] += sdata[tid + s]; __syncthreads(); }
    if (tid == 0) grad_weights[weight_idx] = sdata[0];
}

__global__ void conv2d_backward_bias_opt_kernel(const float* __restrict__ grad_output, float* __restrict__ grad_bias, int batch_size, int out_c, int out_h, int out_w) {
    extern __shared__ float sdata[];
    int oc = blockIdx.x;
    if (oc >= out_c) return;
    int tid = threadIdx.x, total = batch_size * out_h * out_w;
    float local_sum = 0.0f;
    for (int i = tid; i < total; i += blockDim.x) {
        int ow = i % out_w, temp = i / out_w, oh = temp % out_h, n = temp / out_h;
        local_sum += grad_output[((static_cast<size_t>(n) * out_c + oc) * out_h + oh) * out_w + ow];
    }
    sdata[tid] = local_sum; __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) { if (tid < s) sdata[tid] += sdata[tid + s]; __syncthreads(); }
    if (tid == 0) grad_bias[oc] = sdata[0];
}

__global__ void sgd_update_vectorized_kernel(float4* params, const float4* grads, float lr, size_t n4) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) { float4 p = params[idx], g = grads[idx]; params[idx] = make_float4(p.x - lr * g.x, p.y - lr * g.y, p.z - lr * g.z, p.w - lr * g.w); }
}

__global__ void sgd_update_kernel_opt(float* params, const float* grads, float lr, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) params[idx] -= lr * grads[idx];
}

void gpu_conv2d_backward_opt(const GPUTensor4D& input, const GPUTensor4D& grad_output, const float* d_weights, float* d_grad_weights, float* d_grad_bias, GPUTensor4D& grad_input, int in_c, int out_c, int k, int stride, int padding) {
    int out_h = grad_output.h, out_w = grad_output.w;
    if (grad_input.n != input.n || grad_input.c != in_c || grad_input.h != input.h || grad_input.w != input.w) grad_input.allocate(input.n, in_c, input.h, input.w);
    int total_inputs = input.n * in_c * input.h * input.w;
    conv2d_backward_data_opt_kernel<<<(total_inputs + 255) / 256, 256>>>(grad_output.d_data, d_weights, grad_input.d_data, input.n, in_c, input.h, input.w, out_c, out_h, out_w, k, stride, padding);
    CUDA_CHECK(cudaGetLastError());
    int total_weights = out_c * in_c * k * k;
    conv2d_backward_weights_opt_kernel<<<total_weights, 128, 128 * sizeof(float)>>>(input.d_data, grad_output.d_data, d_grad_weights, input.n, in_c, input.h, input.w, out_c, out_h, out_w, k, stride, padding);
    CUDA_CHECK(cudaGetLastError());
    conv2d_backward_bias_opt_kernel<<<out_c, 256, 256 * sizeof(float)>>>(grad_output.d_data, d_grad_bias, input.n, out_c, out_h, out_w);
    CUDA_CHECK(cudaGetLastError());
}

void gpu_sgd_update_opt(float* params, const float* grads, float lr, size_t n) {
    if (n % 4 == 0) { sgd_update_vectorized_kernel<<<((n / 4) + 255) / 256, 256>>>(reinterpret_cast<float4*>(params), reinterpret_cast<const float4*>(grads), lr, n / 4); }
    else { sgd_update_kernel_opt<<<(n + 255) / 256, 256>>>(params, grads, lr, n); }
    CUDA_CHECK(cudaGetLastError());
}

void gpu_conv2d_backward_full_opt(const GPUTensor4D& input, const GPUTensor4D& grad_output, GPUTensor4D& grad_input, const GPUConv2DLayer& conv, float learning_rate) {
    gpu_conv2d_backward_opt(input, grad_output, conv.get_weights(), conv.get_grad_weights(), conv.get_grad_bias(), grad_input, conv.get_in_channels(), conv.get_out_channels(), conv.get_kernel_size(), conv.get_stride(), conv.get_padding());
    gpu_sgd_update_opt(conv.get_weights(), conv.get_grad_weights(), learning_rate, conv.get_weights_size());
    gpu_sgd_update_opt(conv.get_bias(), conv.get_grad_bias(), learning_rate, conv.get_out_channels());
}

__global__ void im2col_kernel(const float* __restrict__ input, float* __restrict__ col, int batch_size, int in_c, int in_h, int in_w, int out_h, int out_w, int k, int stride, int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_h * out_w) return;
    int ow = idx % out_w, temp = idx / out_w, oh = temp % out_h, n = temp / out_h;
    const int col_height = in_c * k * k;
    float* col_ptr = col + static_cast<size_t>(n) * out_h * out_w * col_height + static_cast<size_t>(oh * out_w + ow) * col_height;
    int col_idx = 0;
    for (int ic = 0; ic < in_c; ++ic) for (int kh = 0; kh < k; ++kh) for (int kw = 0; kw < k; ++kw) {
        int ih = oh * stride + kh - padding, iw = ow * stride + kw - padding;
        col_ptr[col_idx++] = (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) ? input[((static_cast<size_t>(n) * in_c + ic) * in_h + ih) * in_w + iw] : 0.0f;
    }
}

__global__ void gemm_conv_kernel(const float* __restrict__ weights, const float* __restrict__ col, const float* __restrict__ bias, float* __restrict__ output,
    int batch_size, int out_c, int out_h, int out_w, int col_height, bool apply_relu) {
    extern __shared__ float smem[];
    int oc = blockIdx.y, spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = spatial_idx / (out_h * out_w), spatial_pos = spatial_idx % (out_h * out_w);
    if (n >= batch_size || oc >= out_c) return;
    float* s_weights = smem;
    for (int i = threadIdx.x; i < col_height; i += blockDim.x) s_weights[i] = weights[static_cast<size_t>(oc) * col_height + i];
    __syncthreads();
    if (spatial_idx >= batch_size * out_h * out_w) return;
    const float* col_ptr = col + static_cast<size_t>(n) * out_h * out_w * col_height + static_cast<size_t>(spatial_pos) * col_height;
    float sum = bias[oc];
    for (int i = 0; i < col_height; ++i) sum += s_weights[i] * col_ptr[i];
    if (apply_relu) sum = fmaxf(0.0f, sum);
    output[((static_cast<size_t>(n) * out_c + oc) * out_h + spatial_pos / out_w) * out_w + spatial_pos % out_w] = sum;
}

static float* d_im2col_buffer = nullptr;
static size_t im2col_buffer_size = 0;

void ensure_im2col_buffer(size_t required_size) {
    if (d_im2col_buffer == nullptr || im2col_buffer_size < required_size) {
        if (d_im2col_buffer) cudaFree(d_im2col_buffer);
        im2col_buffer_size = required_size * 2;
        CUDA_CHECK(cudaMalloc(&d_im2col_buffer, im2col_buffer_size * sizeof(float)));
    }
}

__global__ void add_bias_kernel(float* __restrict__ output, const float* __restrict__ bias, int batch_size, int out_c, int out_h, int out_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * out_c * out_h * out_w) return;
    output[idx] += bias[(idx / (out_h * out_w)) % out_c];
}

void gpu_conv2d_forward_cublas(const GPUTensor4D& input, const float* d_weights, const float* d_bias, GPUTensor4D& output, int in_c, int out_c, int k, int stride, int padding) {
    init_cublas();
    int out_h = (input.h + 2 * padding - k) / stride + 1, out_w = (input.w + 2 * padding - k) / stride + 1;
    if (output.n != input.n || output.c != out_c || output.h != out_h || output.w != out_w) output.allocate(input.n, out_c, out_h, out_w);
    int col_height = in_c * k * k;
    ensure_im2col_buffer(static_cast<size_t>(input.n) * out_h * out_w * col_height);
    im2col_kernel<<<((input.n * out_h * out_w) + 255) / 256, 256>>>(input.d_data, d_im2col_buffer, input.n, in_c, input.h, input.w, out_h, out_w, k, stride, padding);
    CUDA_CHECK(cudaGetLastError());
    int M = input.n * out_h * out_w, N = out_c, K = col_height;
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, d_weights, K, d_im2col_buffer, K, &beta, output.d_data, N);
    add_bias_kernel<<<((input.n * out_c * out_h * out_w) + 255) / 256, 256>>>(output.d_data, d_bias, input.n, out_c, out_h, out_w);
    CUDA_CHECK(cudaGetLastError());
}

void gpu_conv2d_forward_cublas_wrapper(const GPUTensor4D& input, const GPUConv2DLayer& conv, GPUTensor4D& output) {
    gpu_conv2d_forward_cublas(input, conv.get_weights(), conv.get_bias(), output, conv.get_in_channels(), conv.get_out_channels(), conv.get_kernel_size(), conv.get_stride(), conv.get_padding());
}

__global__ void maxpool2d_backward_opt_kernel(const float* __restrict__ input, const float* __restrict__ grad_output, float* __restrict__ grad_input,
    int batch_size, int channels, int in_h, int in_w, int out_h, int out_w, int k, int stride) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x, oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels, n = blockIdx.z / channels;
    if (ow >= out_w || oh >= out_h || n >= batch_size) return;
    float max_val = -1e30f; int max_ih = -1, max_iw = -1;
    const size_t in_base = ((static_cast<size_t>(n) * channels + c) * in_h) * in_w;
    for (int kh = 0; kh < 2; ++kh) for (int kw = 0; kw < 2; ++kw) {
        int ih = oh * stride + kh, iw = ow * stride + kw;
        if (ih < in_h && iw < in_w) { float val = input[in_base + ih * in_w + iw]; if (val > max_val) { max_val = val; max_ih = ih; max_iw = iw; } }
    }
    if (max_ih >= 0 && max_iw >= 0) atomicAdd(&grad_input[in_base + max_ih * in_w + max_iw], grad_output[((static_cast<size_t>(n) * channels + c) * out_h + oh) * out_w + ow]);
}

__global__ void upsample2d_backward_opt_kernel(const float* __restrict__ grad_output, float* __restrict__ grad_input, int batch_size, int channels, int in_h, int in_w, int scale) {
    int iw = blockIdx.x * blockDim.x + threadIdx.x, ih = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels, n = blockIdx.z / channels;
    if (iw >= in_w || ih >= in_h || n >= batch_size) return;
    int out_h = in_h * scale, out_w = in_w * scale;
    float sum = 0.0f;
    for (int dh = 0; dh < 2; ++dh) for (int dw = 0; dw < 2; ++dw) sum += grad_output[((static_cast<size_t>(n) * channels + c) * out_h + ih * scale + dh) * out_w + iw * scale + dw];
    grad_input[((static_cast<size_t>(n) * channels + c) * in_h + ih) * in_w + iw] = sum;
}

void gpu_upsample2d_backward_opt(const GPUTensor4D& input, const GPUTensor4D& grad_output, GPUTensor4D& grad_input, int scale) {
    if (grad_input.n != input.n || grad_input.c != input.c || grad_input.h != input.h || grad_input.w != input.w) grad_input.allocate(input.n, input.c, input.h, input.w);
    dim3 block(16, 16), grid((input.w + 15) / 16, (input.h + 15) / 16, input.n * input.c);
    upsample2d_backward_opt_kernel<<<grid, block>>>(grad_output.d_data, grad_input.d_data, input.n, input.c, input.h, input.w, scale);
    CUDA_CHECK(cudaGetLastError());
}

void gpu_maxpool2d_backward_opt(const GPUTensor4D& input, const GPUTensor4D& grad_output, GPUTensor4D& grad_input, int k, int stride) {
    if (grad_input.n != input.n || grad_input.c != input.c || grad_input.h != input.h || grad_input.w != input.w) grad_input.allocate(input.n, input.c, input.h, input.w);
    CUDA_CHECK(cudaMemset(grad_input.d_data, 0, grad_input.bytes()));
    dim3 block(16, 16), grid((grad_output.w + 15) / 16, (grad_output.h + 15) / 16, input.n * input.c);
    maxpool2d_backward_opt_kernel<<<grid, block>>>(input.d_data, grad_output.d_data, grad_input.d_data, input.n, input.c, input.h, input.w, grad_output.h, grad_output.w, k, stride);
    CUDA_CHECK(cudaGetLastError());
}

void cleanup_gpu_opt_buffers() {
    if (d_im2col_buffer) { cudaFree(d_im2col_buffer); d_im2col_buffer = nullptr; im2col_buffer_size = 0; }
}

#endif