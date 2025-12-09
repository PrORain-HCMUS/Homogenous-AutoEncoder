#ifdef USE_OPTIMIZED_KERNELS

#include "gpu_layer.h"
#include "cuda_utils.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include <cstdio>

// Global cuBLAS handle for GEMM operations
static cublasHandle_t cublas_handle = nullptr;
static bool cublas_initialized = false;

// Global cuDNN handle and descriptors
static cudnnHandle_t cudnn_handle = nullptr;
static bool cudnn_initialized = false;

void init_cublas() {
    if (!cublas_initialized) {
        cublasCreate(&cublas_handle);
        cublas_initialized = true;
    }
}

void cleanup_cublas() {
    if (cublas_initialized && cublas_handle) {
        cublasDestroy(cublas_handle);
        cublas_handle = nullptr;
        cublas_initialized = false;
    }
}

// cuDNN initialization and cleanup
void init_cudnn() {
    if (!cudnn_initialized) {
        cudnnCreate(&cudnn_handle);
        cudnn_initialized = true;
    }
}

void cleanup_cudnn() {
    if (cudnn_initialized && cudnn_handle) {
        cudnnDestroy(cudnn_handle);
        cudnn_handle = nullptr;
        cudnn_initialized = false;
    }
}

// cuDNN-based convolution forward
void gpu_conv2d_forward_cudnn(
    const GPUTensor4D& input,
    const float* d_weights,
    const float* d_bias,
    GPUTensor4D& output,
    int in_c, int out_c, int k, int stride, int padding
) {
    init_cudnn();
    
    int out_h = (input.h + 2 * padding - k) / stride + 1;
    int out_w = (input.w + 2 * padding - k) / stride + 1;
    
    if (output.n != input.n || output.c != out_c ||
        output.h != out_h || output.w != out_w) {
        output.allocate(input.n, out_c, out_h, out_w);
    }
    
    // Create tensor descriptors
    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnCreateTensorDescriptor(&bias_desc);
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);
    
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                input.n, in_c, input.h, input.w);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                input.n, out_c, out_h, out_w);
    cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                1, out_c, 1, 1);
    cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                out_c, in_c, k, k);
    cudnnSetConvolution2dDescriptor(conv_desc, padding, padding, stride, stride, 1, 1,
                                     CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    
    // Find best algorithm
    cudnnConvolutionFwdAlgoPerf_t algo_perf;
    int returned_algo_count;
    cudnnGetConvolutionForwardAlgorithm_v7(cudnn_handle, input_desc, filter_desc,
                                            conv_desc, output_desc, 1, &returned_algo_count, &algo_perf);
    cudnnConvolutionFwdAlgo_t algo = algo_perf.algo;
    
    // Get workspace size
    size_t workspace_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle, input_desc, filter_desc,
                                             conv_desc, output_desc, algo, &workspace_size);
    
    // Allocate workspace
    void* d_workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&d_workspace, workspace_size);
    }
    
    // Perform convolution
    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(cudnn_handle, &alpha, input_desc, input.d_data,
                            filter_desc, d_weights, conv_desc, algo,
                            d_workspace, workspace_size, &beta, output_desc, output.d_data);
    
    // Add bias
    beta = 1.0f;
    cudnnAddTensor(cudnn_handle, &alpha, bias_desc, d_bias, &beta, output_desc, output.d_data);
    
    // Cleanup
    if (d_workspace) cudaFree(d_workspace);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
}

// cuDNN-based convolution backward (data + weights + bias)
void gpu_conv2d_backward_cudnn(
    const GPUTensor4D& input,
    const GPUTensor4D& grad_output,
    const float* d_weights,
    float* d_grad_weights,
    float* d_grad_bias,
    GPUTensor4D& grad_input,
    int in_c, int out_c, int k, int stride, int padding,
    float learning_rate
) {
    init_cudnn();
    
    int out_h = grad_output.h;
    int out_w = grad_output.w;
    
    if (grad_input.n != input.n || grad_input.c != in_c ||
        grad_input.h != input.h || grad_input.w != input.w) {
        grad_input.allocate(input.n, in_c, input.h, input.w);
    }
    
    // Create descriptors
    cudnnTensorDescriptor_t input_desc, grad_output_desc, grad_input_desc, bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    
    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&grad_output_desc);
    cudnnCreateTensorDescriptor(&grad_input_desc);
    cudnnCreateTensorDescriptor(&bias_desc);
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);
    
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                input.n, in_c, input.h, input.w);
    cudnnSetTensor4dDescriptor(grad_output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                input.n, out_c, out_h, out_w);
    cudnnSetTensor4dDescriptor(grad_input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                input.n, in_c, input.h, input.w);
    cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                1, out_c, 1, 1);
    cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                out_c, in_c, k, k);
    cudnnSetConvolution2dDescriptor(conv_desc, padding, padding, stride, stride, 1, 1,
                                     CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    
    float alpha = 1.0f, beta = 0.0f;
    
    // 1. Backward data (compute grad_input)
    cudnnConvolutionBwdDataAlgoPerf_t data_algo_perf;
    int returned_algo_count;
    cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnn_handle, filter_desc, grad_output_desc,
                                                 conv_desc, grad_input_desc, 1, &returned_algo_count, &data_algo_perf);
    
    size_t data_workspace_size = 0;
    cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle, filter_desc, grad_output_desc,
                                                  conv_desc, grad_input_desc, data_algo_perf.algo, &data_workspace_size);
    
    void* d_data_workspace = nullptr;
    if (data_workspace_size > 0) cudaMalloc(&d_data_workspace, data_workspace_size);
    
    cudnnConvolutionBackwardData(cudnn_handle, &alpha, filter_desc, d_weights,
                                  grad_output_desc, grad_output.d_data, conv_desc, data_algo_perf.algo,
                                  d_data_workspace, data_workspace_size, &beta, grad_input_desc, grad_input.d_data);
    
    if (d_data_workspace) cudaFree(d_data_workspace);
    
    // 2. Backward filter (compute grad_weights)
    cudnnConvolutionBwdFilterAlgoPerf_t filter_algo_perf;
    cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnn_handle, input_desc, grad_output_desc,
                                                   conv_desc, filter_desc, 1, &returned_algo_count, &filter_algo_perf);
    
    size_t filter_workspace_size = 0;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle, input_desc, grad_output_desc,
                                                    conv_desc, filter_desc, filter_algo_perf.algo, &filter_workspace_size);
    
    void* d_filter_workspace = nullptr;
    if (filter_workspace_size > 0) cudaMalloc(&d_filter_workspace, filter_workspace_size);
    
    cudnnConvolutionBackwardFilter(cudnn_handle, &alpha, input_desc, input.d_data,
                                    grad_output_desc, grad_output.d_data, conv_desc, filter_algo_perf.algo,
                                    d_filter_workspace, filter_workspace_size, &beta, filter_desc, d_grad_weights);
    
    if (d_filter_workspace) cudaFree(d_filter_workspace);
    
    // 3. Backward bias
    cudnnConvolutionBackwardBias(cudnn_handle, &alpha, grad_output_desc, grad_output.d_data,
                                  &beta, bias_desc, d_grad_bias);
    
    // Cleanup (weight update done in wrapper)
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(grad_output_desc);
    cudnnDestroyTensorDescriptor(grad_input_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
}

// Wrapper for cuDNN conv forward
void gpu_conv2d_forward_cudnn_wrapper(const GPUTensor4D& input,
                                       const GPUConv2DLayer& conv,
                                       GPUTensor4D& output) {
    gpu_conv2d_forward_cudnn(
        input,
        conv.get_weights(),
        conv.get_bias(),
        output,
        conv.get_in_channels(),
        conv.get_out_channels(),
        conv.get_kernel_size(),
        conv.get_stride(),
        conv.get_padding()
    );
}

// Full cuDNN backward for conv layer
void gpu_conv2d_backward_cudnn_full(
    const GPUTensor4D& input,
    const GPUTensor4D& grad_output,
    GPUTensor4D& grad_input,
    const GPUConv2DLayer& conv,
    float learning_rate
) {
    gpu_conv2d_backward_cudnn(
        input, grad_output,
        conv.get_weights(),
        conv.get_grad_weights(),
        conv.get_grad_bias(),
        grad_input,
        conv.get_in_channels(),
        conv.get_out_channels(),
        conv.get_kernel_size(),
        conv.get_stride(),
        conv.get_padding(),
        learning_rate
    );
    
    // Update weights
    size_t weights_size = conv.get_weights_size();
    gpu_sgd_update_opt(conv.get_weights(), conv.get_grad_weights(), learning_rate, weights_size);
    gpu_sgd_update_opt(conv.get_bias(), conv.get_grad_bias(), learning_rate, conv.get_out_channels());
}

// Declare external kernel from layers_gpu.cu
extern __global__ void relu_forward_kernel(const float* input, float* output, size_t n);

#define TILE_WIDTH 16
#define TILE_HEIGHT 16

#define WARP_SIZE 32

__constant__ float c_weights_small[8192];
__constant__ float c_bias_small[256];

__global__ void conv2d_forward_tiled_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int k, int stride, int padding
) {
    extern __shared__ float shared_mem[];
    
    int ow = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int oh = blockIdx.y * TILE_HEIGHT + threadIdx.y;
    int oc = blockIdx.z % out_c;
    int n = blockIdx.z / out_c;
    
    if (ow >= out_w || oh >= out_h || n >= batch_size) return;
    
    float sum = bias[oc];
    
    const int tile_h = TILE_HEIGHT * stride + k - stride;
    const int tile_w = TILE_WIDTH * stride + k - stride;
    const int tile_size = tile_h * tile_w;
    const int threads_per_block = blockDim.x * blockDim.y;
    const int num_loads = (tile_size + threads_per_block - 1) / threads_per_block;
    const int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    float* s_input = shared_mem;
    
    for (int ic = 0; ic < in_c; ++ic) {
        const int in_start_h = blockIdx.y * TILE_HEIGHT * stride - padding;
        const int in_start_w = blockIdx.x * TILE_WIDTH * stride - padding;
        const size_t in_channel_offset = (static_cast<size_t>(n) * in_c + ic) * in_h * in_w;
        
        #pragma unroll 4
        for (int load = 0; load < num_loads; ++load) {
            int linear_idx = load * threads_per_block + linear_tid;
            if (linear_idx < tile_size) {
                int sh = linear_idx / tile_w;
                int sw = linear_idx % tile_w;
                int ih = in_start_h + sh;
                int iw = in_start_w + sw;
                
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    s_input[linear_idx] = input[in_channel_offset + ih * in_w + iw];
                } else {
                    s_input[linear_idx] = 0.0f;
                }
            }
        }
        __syncthreads();
        
        const int local_h = threadIdx.y * stride;
        const int local_w = threadIdx.x * stride;
        const size_t w_ic_offset = (static_cast<size_t>(oc) * in_c + ic) * k * k;
        
        #pragma unroll
        for (int kh = 0; kh < k; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < k; ++kw) {
                sum += s_input[(local_h + kh) * tile_w + local_w + kw] * weights[w_ic_offset + kh * k + kw];
            }
        }
        __syncthreads();
    }
    
    size_t out_idx = ((static_cast<size_t>(n) * out_c + oc) * out_h + oh) * out_w + ow;
    output[out_idx] = sum;
}

__global__ void conv2d_relu_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int k, int stride, int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_c * out_h * out_w;
    
    if (idx >= total_outputs) return;
    
    int ow = idx % out_w;
    int temp = idx / out_w;
    int oh = temp % out_h;
    temp = temp / out_h;
    int oc = temp % out_c;
    int n = temp / out_c;
    
    float sum = bias[oc];
    
    #pragma unroll 4
    for (int ic = 0; ic < in_c; ++ic) {
        for (int kh = 0; kh < k; ++kh) {
            for (int kw = 0; kw < k; ++kw) {
                int ih = oh * stride + kh - padding;
                int iw = ow * stride + kw - padding;
                
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    size_t in_idx = ((static_cast<size_t>(n) * in_c + ic) * in_h + ih) * in_w + iw;
                    size_t w_idx = ((static_cast<size_t>(oc) * in_c + ic) * k + kh) * k + kw;
                    sum += input[in_idx] * weights[w_idx];
                }
            }
        }
    }
    
    output[idx] = fmaxf(0.0f, sum);
}

__global__ void relu_forward_vectorized_kernel(const float4* input, float4* output, size_t n4) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 in = input[idx];
        float4 out;
        out.x = fmaxf(0.0f, in.x);
        out.y = fmaxf(0.0f, in.y);
        out.z = fmaxf(0.0f, in.z);
        out.w = fmaxf(0.0f, in.w);
        output[idx] = out;
    }
}

__global__ void relu_backward_vectorized_kernel(
    const float4* input, const float4* grad_output,
    float4* grad_input, size_t n4
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 in = input[idx];
        float4 go = grad_output[idx];
        float4 gi;
        gi.x = (in.x > 0.0f) ? go.x : 0.0f;
        gi.y = (in.y > 0.0f) ? go.y : 0.0f;
        gi.z = (in.z > 0.0f) ? go.z : 0.0f;
        gi.w = (in.w > 0.0f) ? go.w : 0.0f;
        grad_input[idx] = gi;
    }
}

__global__ void maxpool2d_forward_opt_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels, int in_h, int in_w,
    int out_h, int out_w, int k, int stride
) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels;
    int n = blockIdx.z / channels;
    
    if (ow >= out_w || oh >= out_h || n >= batch_size) return;
    
    float max_val = -1e30f;
    
    #pragma unroll
    for (int kh = 0; kh < k; ++kh) {
        #pragma unroll
        for (int kw = 0; kw < k; ++kw) {
            int ih = oh * stride + kh;
            int iw = ow * stride + kw;
            
            if (ih < in_h && iw < in_w) {
                size_t in_idx = ((static_cast<size_t>(n) * channels + c) * in_h + ih) * in_w + iw;
                max_val = fmaxf(max_val, input[in_idx]);
            }
        }
    }
    
    size_t out_idx = ((static_cast<size_t>(n) * channels + c) * out_h + oh) * out_w + ow;
    output[out_idx] = max_val;
}

__global__ void upsample2d_forward_opt_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch_size, int channels, int in_h, int in_w,
    int out_h, int out_w, int scale
) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels;
    int n = blockIdx.z / channels;
    
    if (ow >= out_w || oh >= out_h || n >= batch_size) return;
    
    int ih = oh / scale;
    int iw = ow / scale;
    
    size_t in_idx = ((static_cast<size_t>(n) * channels + c) * in_h + ih) * in_w + iw;
    size_t out_idx = ((static_cast<size_t>(n) * channels + c) * out_h + oh) * out_w + ow;
    
    output[out_idx] = input[in_idx];
}

__global__ void mse_loss_grad_fused_kernel(
    const float* __restrict__ output,
    const float* __restrict__ target,
    float* __restrict__ grad_output,
    float* __restrict__ partial_loss,
    float scale,
    size_t n
) {
    extern __shared__ float sdata[];
    
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float local_loss = 0.0f;
    
    if (idx < n) {
        float diff = output[idx] - target[idx];
        local_loss = diff * diff;
        grad_output[idx] = scale * diff;
    }
    
    sdata[tid] = local_loss;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_loss[blockIdx.x] = sdata[0];
    }
}

void gpu_relu_forward_opt(const GPUTensor4D& input, GPUTensor4D& output) {
    size_t n = input.size();
    size_t n4 = n / 4;
    
    if (output.n != input.n || output.c != input.c ||
        output.h != input.h || output.w != input.w) {
        output.allocate(input.n, input.c, input.h, input.w);
    }
    
    if (n % 4 == 0) {
        int block_size = 256;
        int grid_size = (n4 + block_size - 1) / block_size;
        relu_forward_vectorized_kernel<<<grid_size, block_size>>>(
            reinterpret_cast<const float4*>(input.d_data),
            reinterpret_cast<float4*>(output.d_data),
            n4
        );
    } else {
        int block_size = 256;
        int grid_size = (n + block_size - 1) / block_size;
        relu_forward_kernel<<<grid_size, block_size>>>(input.d_data, output.d_data, n);
    }
    CUDA_CHECK(cudaGetLastError());
}

void gpu_relu_backward_opt(const GPUTensor4D& input, const GPUTensor4D& grad_output,
                            GPUTensor4D& grad_input) {
    size_t n = input.size();
    size_t n4 = n / 4;
    
    if (grad_input.n != input.n || grad_input.c != input.c ||
        grad_input.h != input.h || grad_input.w != input.w) {
        grad_input.allocate(input.n, input.c, input.h, input.w);
    }
    
    if (n % 4 == 0) {
        int block_size = 256;
        int grid_size = (n4 + block_size - 1) / block_size;
        relu_backward_vectorized_kernel<<<grid_size, block_size>>>(
            reinterpret_cast<const float4*>(input.d_data),
            reinterpret_cast<const float4*>(grad_output.d_data),
            reinterpret_cast<float4*>(grad_input.d_data),
            n4
        );
    }
    CUDA_CHECK(cudaGetLastError());
}

void gpu_maxpool2d_forward_opt(const GPUTensor4D& input, GPUTensor4D& output,
                                int k, int stride) {
    int out_h = (input.h - k) / stride + 1;
    int out_w = (input.w - k) / stride + 1;
    
    if (output.n != input.n || output.c != input.c ||
        output.h != out_h || output.w != out_w) {
        output.allocate(input.n, input.c, out_h, out_w);
    }
    
    dim3 block(16, 16);
    dim3 grid(
        (out_w + block.x - 1) / block.x,
        (out_h + block.y - 1) / block.y,
        input.n * input.c
    );
    
    maxpool2d_forward_opt_kernel<<<grid, block>>>(
        input.d_data, output.d_data,
        input.n, input.c, input.h, input.w,
        out_h, out_w, k, stride
    );
    CUDA_CHECK(cudaGetLastError());
}

void gpu_upsample2d_forward_opt(const GPUTensor4D& input, GPUTensor4D& output, int scale) {
    int out_h = input.h * scale;
    int out_w = input.w * scale;
    
    if (output.n != input.n || output.c != input.c ||
        output.h != out_h || output.w != out_w) {
        output.allocate(input.n, input.c, out_h, out_w);
    }
    
    dim3 block(16, 16);
    dim3 grid(
        (out_w + block.x - 1) / block.x,
        (out_h + block.y - 1) / block.y,
        input.n * input.c
    );
    
    upsample2d_forward_opt_kernel<<<grid, block>>>(
        input.d_data, output.d_data,
        input.n, input.c, input.h, input.w,
        out_h, out_w, scale
    );
    CUDA_CHECK(cudaGetLastError());
}

void gpu_conv2d_relu_forward_opt(
    const GPUTensor4D& input,
    const float* d_weights,
    const float* d_bias,
    GPUTensor4D& output,
    int in_c, int out_c, int k, int stride, int padding
) {
    int out_h = (input.h + 2 * padding - k) / stride + 1;
    int out_w = (input.w + 2 * padding - k) / stride + 1;
    
    if (output.n != input.n || output.c != out_c ||
        output.h != out_h || output.w != out_w) {
        output.allocate(input.n, out_c, out_h, out_w);
    }
    
    int total = input.n * out_c * out_h * out_w;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;
    
    conv2d_relu_forward_kernel<<<grid_size, block_size>>>(
        input.d_data, d_weights, d_bias, output.d_data,
        input.n, in_c, input.h, input.w,
        out_c, out_h, out_w,
        k, stride, padding
    );
    CUDA_CHECK(cudaGetLastError());
}

void gpu_conv2d_forward_tiled(
    const GPUTensor4D& input,
    const float* d_weights,
    const float* d_bias,
    GPUTensor4D& output,
    int in_c, int out_c, int k, int stride, int padding
) {
    int out_h = (input.h + 2 * padding - k) / stride + 1;
    int out_w = (input.w + 2 * padding - k) / stride + 1;
    
    if (output.n != input.n || output.c != out_c ||
        output.h != out_h || output.w != out_w) {
        output.allocate(input.n, out_c, out_h, out_w);
    }
    
    dim3 block(TILE_WIDTH, TILE_HEIGHT);
    dim3 grid(
        (out_w + TILE_WIDTH - 1) / TILE_WIDTH,
        (out_h + TILE_HEIGHT - 1) / TILE_HEIGHT,
        input.n * out_c
    );
    
    int tile_h = TILE_HEIGHT * stride + k - stride;
    int tile_w = TILE_WIDTH * stride + k - stride;
    size_t shared_size = tile_h * tile_w * sizeof(float);
    
    conv2d_forward_tiled_kernel<<<grid, block, shared_size>>>(
        input.d_data, d_weights, d_bias, output.d_data,
        input.n, in_c, input.h, input.w,
        out_c, out_h, out_w,
        k, stride, padding
    );
    CUDA_CHECK(cudaGetLastError());
}

// Wrapper function for optimized conv forward using layer object
void gpu_conv2d_forward_opt(const GPUTensor4D& input,
                            const GPUConv2DLayer& conv,
                            GPUTensor4D& output) {
    gpu_conv2d_forward_tiled(
        input,
        conv.get_weights(),
        conv.get_bias(),
        output,
        conv.get_in_channels(),
        conv.get_out_channels(),
        conv.get_kernel_size(),
        conv.get_stride(),
        conv.get_padding()
    );
}

// Fused Conv+ReLU forward - combines two operations into one kernel launch
void gpu_conv2d_relu_fused_forward(const GPUTensor4D& input, 
                                    const GPUConv2DLayer& conv,
                                    GPUTensor4D& output) {
    gpu_conv2d_relu_forward_opt(
        input,
        conv.get_weights(),
        conv.get_bias(),
        output,
        conv.get_in_channels(),
        conv.get_out_channels(),
        conv.get_kernel_size(),
        conv.get_stride(),
        conv.get_padding()
    );
}

// ============================================================================
// OPTIMIZED BACKWARD KERNELS
// Use col2im approach for backward data - much faster than naive
// ============================================================================

// Col2im kernel - scatter gradients from output to input
// This is the transpose of im2col, used for backward data pass
__global__ void col2im_backward_kernel(
    const float* __restrict__ grad_output,  // [batch, out_c, out_h, out_w]
    const float* __restrict__ weights,       // [out_c, in_c, k, k]
    float* __restrict__ grad_input,          // [batch, in_c, in_h, in_w]
    int batch_size, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int k, int stride, int padding
) {
    // Each thread handles one (n, oc, oh, ow) and scatters to grad_input
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_c * out_h * out_w;
    if (idx >= total) return;
    
    int ow = idx % out_w;
    int tmp = idx / out_w;
    int oh = tmp % out_h;
    tmp = tmp / out_h;
    int oc = tmp % out_c;
    int n = tmp / out_c;
    
    float grad_val = grad_output[idx];
    
    // Scatter this gradient to all input positions that contributed
    #pragma unroll
    for (int ic = 0; ic < in_c; ++ic) {
        size_t w_base = (static_cast<size_t>(oc) * in_c + ic) * k * k;
        
        #pragma unroll
        for (int kh = 0; kh < k; ++kh) {
            int ih = oh * stride + kh - padding;
            if (ih < 0 || ih >= in_h) continue;
            
            #pragma unroll
            for (int kw = 0; kw < k; ++kw) {
                int iw = ow * stride + kw - padding;
                if (iw < 0 || iw >= in_w) continue;
                
                size_t gi_idx = ((static_cast<size_t>(n) * in_c + ic) * in_h + ih) * in_w + iw;
                float w = weights[w_base + kh * k + kw];
                atomicAdd(&grad_input[gi_idx], grad_val * w);
            }
        }
    }
}

// Alternative: gather-based backward data (no atomics, but more work per thread)
// Each thread computes one grad_input element
__global__ void conv2d_backward_data_opt_kernel(
    const float* __restrict__ grad_output,
    const float* __restrict__ weights,
    float* __restrict__ grad_input,
    int batch_size, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int k, int stride, int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * in_c * in_h * in_w;
    if (idx >= total) return;
    
    int iw = idx % in_w;
    int tmp = idx / in_w;
    int ih = tmp % in_h;
    tmp = tmp / in_h;
    int ic = tmp % in_c;
    int n = tmp / in_c;
    
    float sum = 0.0f;
    
    // For stride=1, k=3, the valid oh range is [ih-2, ih] intersected with [0, out_h)
    int oh_start = max(0, ih + padding - k + 1);
    int oh_end = min(out_h, ih + padding + 1);
    int ow_start = max(0, iw + padding - k + 1);
    int ow_end = min(out_w, iw + padding + 1);
    
    for (int oc = 0; oc < out_c; ++oc) {
        size_t w_ic_base = (static_cast<size_t>(oc) * in_c + ic) * k * k;
        
        for (int oh = oh_start; oh < oh_end; ++oh) {
            int kh = ih + padding - oh;
            size_t go_row = (static_cast<size_t>(n) * out_c + oc) * out_h + oh;
            
            for (int ow = ow_start; ow < ow_end; ++ow) {
                int kw = iw + padding - ow;
                size_t go_idx = go_row * out_w + ow;
                size_t w_idx = w_ic_base + kh * k + kw;
                sum += grad_output[go_idx] * weights[w_idx];
            }
        }
    }
    
    grad_input[idx] = sum;
}

// Optimized backward weights kernel using parallel reduction
__global__ void conv2d_backward_weights_opt_kernel(
    const float* __restrict__ input,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_weights,
    int batch_size, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int k, int stride, int padding
) {
    extern __shared__ float sdata[];
    
    int weight_idx = blockIdx.x;
    int total_weights = out_c * in_c * k * k;
    if (weight_idx >= total_weights) return;
    
    // Decode weight index
    int kw = weight_idx % k;
    int temp = weight_idx / k;
    int kh = temp % k;
    temp = temp / k;
    int ic = temp % in_c;
    int oc = temp / in_c;
    
    // Each thread handles multiple (n, oh, ow) combinations
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int total_outputs = batch_size * out_h * out_w;
    
    float local_sum = 0.0f;
    
    for (int i = tid; i < total_outputs; i += num_threads) {
        int ow = i % out_w;
        int temp2 = i / out_w;
        int oh = temp2 % out_h;
        int n = temp2 / out_h;
        
        int ih = oh * stride + kh - padding;
        int iw = ow * stride + kw - padding;
        
        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
            size_t in_idx = ((static_cast<size_t>(n) * in_c + ic) * in_h + ih) * in_w + iw;
            size_t go_idx = ((static_cast<size_t>(n) * out_c + oc) * out_h + oh) * out_w + ow;
            local_sum += input[in_idx] * grad_output[go_idx];
        }
    }
    
    // Parallel reduction in shared memory
    sdata[tid] = local_sum;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        grad_weights[weight_idx] = sdata[0];
    }
}

// Optimized backward bias kernel using parallel reduction
__global__ void conv2d_backward_bias_opt_kernel(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_bias,
    int batch_size, int out_c, int out_h, int out_w
) {
    extern __shared__ float sdata[];
    
    int oc = blockIdx.x;
    if (oc >= out_c) return;
    
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int total = batch_size * out_h * out_w;
    
    float local_sum = 0.0f;
    
    for (int i = tid; i < total; i += num_threads) {
        int ow = i % out_w;
        int temp = i / out_w;
        int oh = temp % out_h;
        int n = temp / out_h;
        
        size_t idx = ((static_cast<size_t>(n) * out_c + oc) * out_h + oh) * out_w + ow;
        local_sum += grad_output[idx];
    }
    
    sdata[tid] = local_sum;
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        grad_bias[oc] = sdata[0];
    }
}

// Vectorized SGD update using float4
__global__ void sgd_update_vectorized_kernel(float4* params, const float4* grads, float lr, size_t n4) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 p = params[idx];
        float4 g = grads[idx];
        p.x -= lr * g.x;
        p.y -= lr * g.y;
        p.z -= lr * g.z;
        p.w -= lr * g.w;
        params[idx] = p;
    }
}

__global__ void sgd_update_kernel_opt(float* params, const float* grads, float lr, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        params[idx] -= lr * grads[idx];
    }
}

// Optimized conv backward function
void gpu_conv2d_backward_opt(
    const GPUTensor4D& input,
    const GPUTensor4D& grad_output,
    const float* d_weights,
    float* d_grad_weights,
    float* d_grad_bias,
    GPUTensor4D& grad_input,
    int in_c, int out_c, int k, int stride, int padding
) {
    int out_h = grad_output.h;
    int out_w = grad_output.w;
    
    // Allocate grad_input if needed
    if (grad_input.n != input.n || grad_input.c != in_c ||
        grad_input.h != input.h || grad_input.w != input.w) {
        grad_input.allocate(input.n, in_c, input.h, input.w);
    }
    
    // Use gather approach - no atomics, each thread computes one grad_input element
    int total_inputs = input.n * in_c * input.h * input.w;
    int block_size = 256;
    int grid_size = (total_inputs + block_size - 1) / block_size;
    
    conv2d_backward_data_opt_kernel<<<grid_size, block_size>>>(
        grad_output.d_data, d_weights, grad_input.d_data,
        input.n, in_c, input.h, input.w,
        out_c, out_h, out_w,
        k, stride, padding
    );
    CUDA_CHECK(cudaGetLastError());
    
    // Backward weights with parallel reduction
    int total_weights = out_c * in_c * k * k;
    int threads_per_weight = 128;  // Threads for reduction per weight
    size_t shared_size = threads_per_weight * sizeof(float);
    conv2d_backward_weights_opt_kernel<<<total_weights, threads_per_weight, shared_size>>>(
        input.d_data, grad_output.d_data, d_grad_weights,
        input.n, in_c, input.h, input.w,
        out_c, out_h, out_w,
        k, stride, padding
    );
    CUDA_CHECK(cudaGetLastError());
    
    // Backward bias with parallel reduction
    int threads_per_bias = 256;
    shared_size = threads_per_bias * sizeof(float);
    conv2d_backward_bias_opt_kernel<<<out_c, threads_per_bias, shared_size>>>(
        grad_output.d_data, d_grad_bias,
        input.n, out_c, out_h, out_w
    );
    CUDA_CHECK(cudaGetLastError());
}

// Optimized SGD update
void gpu_sgd_update_opt(float* params, const float* grads, float lr, size_t n) {
    int block_size = 256;
    
    if (n % 4 == 0) {
        size_t n4 = n / 4;
        int grid_size = (n4 + block_size - 1) / block_size;
        sgd_update_vectorized_kernel<<<grid_size, block_size>>>(
            reinterpret_cast<float4*>(params),
            reinterpret_cast<const float4*>(grads),
            lr, n4
        );
    } else {
        int grid_size = (n + block_size - 1) / block_size;
        sgd_update_kernel_opt<<<grid_size, block_size>>>(params, grads, lr, n);
    }
    CUDA_CHECK(cudaGetLastError());
}

// Full optimized backward for conv layer - uses pre-allocated gradient buffers
void gpu_conv2d_backward_full_opt(
    const GPUTensor4D& input,
    const GPUTensor4D& grad_output,
    GPUTensor4D& grad_input,
    const GPUConv2DLayer& conv,
    float learning_rate
) {
    int in_c = conv.get_in_channels();
    int out_c = conv.get_out_channels();
    int k = conv.get_kernel_size();
    int stride = conv.get_stride();
    int padding = conv.get_padding();
    size_t weights_size = conv.get_weights_size();
    
    // Get internal pointers - reuse pre-allocated buffers (no malloc overhead!)
    float* d_weights = conv.get_weights();
    float* d_bias = conv.get_bias();
    float* d_grad_weights = conv.get_grad_weights();
    float* d_grad_bias = conv.get_grad_bias();
    
    // Compute gradients using optimized kernels
    gpu_conv2d_backward_opt(
        input, grad_output,
        d_weights, d_grad_weights, d_grad_bias,
        grad_input,
        in_c, out_c, k, stride, padding
    );
    
    // Update weights using vectorized SGD
    gpu_sgd_update_opt(d_weights, d_grad_weights, learning_rate, weights_size);
    gpu_sgd_update_opt(d_bias, d_grad_bias, learning_rate, out_c);
}

// ============================================================================
// HIGHLY OPTIMIZED CONVOLUTION USING IM2COL + GEMM APPROACH
// This transforms convolution into matrix multiplication for better performance
// ============================================================================

// Im2col kernel - transforms input patches into columns for GEMM
__global__ void im2col_kernel(
    const float* __restrict__ input,
    float* __restrict__ col,
    int batch_size, int in_c, int in_h, int in_w,
    int out_h, int out_w,
    int k, int stride, int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_h * out_w;
    
    if (idx >= total) return;
    
    int ow = idx % out_w;
    int temp = idx / out_w;
    int oh = temp % out_h;
    int n = temp / out_h;
    
    // Each column contains all input values for one output position
    const int col_height = in_c * k * k;
    float* col_ptr = col + static_cast<size_t>(n) * out_h * out_w * col_height + 
                     static_cast<size_t>(oh * out_w + ow) * col_height;
    
    int col_idx = 0;
    for (int ic = 0; ic < in_c; ++ic) {
        for (int kh = 0; kh < k; ++kh) {
            for (int kw = 0; kw < k; ++kw) {
                int ih = oh * stride + kh - padding;
                int iw = ow * stride + kw - padding;
                
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    size_t in_idx = ((static_cast<size_t>(n) * in_c + ic) * in_h + ih) * in_w + iw;
                    col_ptr[col_idx] = input[in_idx];
                } else {
                    col_ptr[col_idx] = 0.0f;
                }
                col_idx++;
            }
        }
    }
}

// Optimized GEMM kernel for small matrices (convolution output)
__global__ void gemm_conv_kernel(
    const float* __restrict__ weights,  // [out_c, in_c * k * k]
    const float* __restrict__ col,      // [batch * out_h * out_w, in_c * k * k]
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int out_c, int out_h, int out_w,
    int col_height,  // in_c * k * k
    bool apply_relu
) {
    extern __shared__ float smem[];
    
    int oc = blockIdx.y;
    int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = spatial_idx / (out_h * out_w);
    int spatial_pos = spatial_idx % (out_h * out_w);
    
    if (n >= batch_size || oc >= out_c) return;
    
    // Load weights for this output channel into shared memory
    float* s_weights = smem;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    for (int i = tid; i < col_height; i += num_threads) {
        s_weights[i] = weights[static_cast<size_t>(oc) * col_height + i];
    }
    __syncthreads();
    
    if (spatial_idx >= batch_size * out_h * out_w) return;
    
    // Compute dot product
    const float* col_ptr = col + static_cast<size_t>(n) * out_h * out_w * col_height + 
                           static_cast<size_t>(spatial_pos) * col_height;
    
    float sum = bias[oc];
    
    #pragma unroll 4
    for (int i = 0; i < col_height; ++i) {
        sum += s_weights[i] * col_ptr[i];
    }
    
    if (apply_relu) {
        sum = fmaxf(0.0f, sum);
    }
    
    size_t out_idx = ((static_cast<size_t>(n) * out_c + oc) * out_h + spatial_pos / out_w) * out_w + 
                     spatial_pos % out_w;
    output[out_idx] = sum;
}

// Memory pool for im2col buffer to avoid repeated allocations
static float* d_im2col_buffer = nullptr;
static size_t im2col_buffer_size = 0;

void ensure_im2col_buffer(size_t required_size) {
    if (d_im2col_buffer == nullptr || im2col_buffer_size < required_size) {
        if (d_im2col_buffer) {
            cudaFree(d_im2col_buffer);
        }
        im2col_buffer_size = required_size * 2;  // Allocate 2x to reduce reallocations
        CUDA_CHECK(cudaMalloc(&d_im2col_buffer, im2col_buffer_size * sizeof(float)));
    }
}

// Add bias kernel - adds bias to each output channel
__global__ void add_bias_kernel(
    float* __restrict__ output,
    const float* __restrict__ bias,
    int batch_size, int out_c, int out_h, int out_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_c * out_h * out_w;
    if (idx >= total) return;
    
    int spatial = out_h * out_w;
    int oc = (idx / spatial) % out_c;
    output[idx] += bias[oc];
}

// ============================================================================
// CUBLAS-BASED CONVOLUTION - Uses im2col + cuBLAS SGEMM
// This is the fastest approach for convolution on GPU
// ============================================================================

void gpu_conv2d_forward_cublas(
    const GPUTensor4D& input,
    const float* d_weights,  // [out_c, in_c * k * k]
    const float* d_bias,
    GPUTensor4D& output,
    int in_c, int out_c, int k, int stride, int padding
) {
    init_cublas();
    
    int out_h = (input.h + 2 * padding - k) / stride + 1;
    int out_w = (input.w + 2 * padding - k) / stride + 1;
    
    if (output.n != input.n || output.c != out_c ||
        output.h != out_h || output.w != out_w) {
        output.allocate(input.n, out_c, out_h, out_w);
    }
    
    int col_height = in_c * k * k;
    size_t col_size = static_cast<size_t>(input.n) * out_h * out_w * col_height;
    ensure_im2col_buffer(col_size);
    
    // Step 1: Im2col - transform input to column matrix
    int total_positions = input.n * out_h * out_w;
    int block_size = 256;
    int grid_size = (total_positions + block_size - 1) / block_size;
    
    im2col_kernel<<<grid_size, block_size>>>(
        input.d_data, d_im2col_buffer,
        input.n, in_c, input.h, input.w,
        out_h, out_w, k, stride, padding
    );
    CUDA_CHECK(cudaGetLastError());
    
    // Step 2: GEMM using cuBLAS
    // output = weights * col^T + bias
    // weights: [out_c, col_height]
    // col: [batch * out_h * out_w, col_height] -> need transpose
    // output: [out_c, batch * out_h * out_w]
    
    // cuBLAS uses column-major, so we compute: output^T = col * weights^T
    // which gives us output in row-major format
    
    int M = input.n * out_h * out_w;  // rows of col
    int N = out_c;                     // cols of weights^T
    int K = col_height;                // shared dimension
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // C = alpha * A * B + beta * C
    // A = col [M x K], B = weights^T [K x N], C = output [M x N]
    // In column-major: C^T = B^T * A^T
    // cublasSgemm(handle, transB, transA, N, M, K, alpha, B, ldb, A, lda, beta, C, ldc)
    
    cublasStatus_t status = cublasSgemm(
        cublas_handle,
        CUBLAS_OP_T,    // transpose weights
        CUBLAS_OP_N,    // no transpose col
        N,              // rows of result (out_c)
        M,              // cols of result (batch * out_h * out_w)
        K,              // shared dim (col_height)
        &alpha,
        d_weights, K,   // weights [out_c x K] stored row-major = [K x out_c] col-major
        d_im2col_buffer, K,  // col [M x K] stored row-major = [K x M] col-major
        &beta,
        output.d_data, N  // output [M x N] stored row-major = [N x M] col-major
    );
    
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS SGEMM failed with status %d\n", status);
    }
    
    // Step 3: Add bias
    int total_output = input.n * out_c * out_h * out_w;
    grid_size = (total_output + block_size - 1) / block_size;
    add_bias_kernel<<<grid_size, block_size>>>(
        output.d_data, d_bias,
        input.n, out_c, out_h, out_w
    );
    CUDA_CHECK(cudaGetLastError());
}

// Wrapper for cuBLAS conv forward using layer object
void gpu_conv2d_forward_cublas_wrapper(const GPUTensor4D& input,
                                        const GPUConv2DLayer& conv,
                                        GPUTensor4D& output) {
    gpu_conv2d_forward_cublas(
        input,
        conv.get_weights(),
        conv.get_bias(),
        output,
        conv.get_in_channels(),
        conv.get_out_channels(),
        conv.get_kernel_size(),
        conv.get_stride(),
        conv.get_padding()
    );
}

// Optimized maxpool backward with better memory access
__global__ void maxpool2d_backward_opt_kernel(
    const float* __restrict__ input,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int batch_size, int channels, int in_h, int in_w,
    int out_h, int out_w, int k, int stride
) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels;
    int n = blockIdx.z / channels;
    
    if (ow >= out_w || oh >= out_h || n >= batch_size) return;
    
    // Find max position
    float max_val = -1e30f;
    int max_ih = -1, max_iw = -1;
    
    const size_t in_base = ((static_cast<size_t>(n) * channels + c) * in_h) * in_w;
    const int ih_base = oh * stride;
    const int iw_base = ow * stride;
    
    #pragma unroll
    for (int kh = 0; kh < 2; ++kh) {
        #pragma unroll
        for (int kw = 0; kw < 2; ++kw) {
            int ih = ih_base + kh;
            int iw = iw_base + kw;
            
            if (ih < in_h && iw < in_w) {
                float val = input[in_base + ih * in_w + iw];
                if (val > max_val) {
                    max_val = val;
                    max_ih = ih;
                    max_iw = iw;
                }
            }
        }
    }
    
    // Write gradient to max position
    if (max_ih >= 0 && max_iw >= 0) {
        size_t out_idx = ((static_cast<size_t>(n) * channels + c) * out_h + oh) * out_w + ow;
        size_t max_idx = in_base + max_ih * in_w + max_iw;
        atomicAdd(&grad_input[max_idx], grad_output[out_idx]);
    }
}

// Optimized upsample backward - accumulate gradients more efficiently
__global__ void upsample2d_backward_opt_kernel(
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    int batch_size, int channels, int in_h, int in_w,
    int scale
) {
    int iw = blockIdx.x * blockDim.x + threadIdx.x;
    int ih = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z % channels;
    int n = blockIdx.z / channels;
    
    if (iw >= in_w || ih >= in_h || n >= batch_size) return;
    
    int out_h = in_h * scale;
    int out_w = in_w * scale;
    
    // Sum all gradients from the upsampled region
    float sum = 0.0f;
    int oh_base = ih * scale;
    int ow_base = iw * scale;
    
    #pragma unroll
    for (int dh = 0; dh < 2; ++dh) {
        #pragma unroll
        for (int dw = 0; dw < 2; ++dw) {
            int oh = oh_base + dh;
            int ow = ow_base + dw;
            size_t out_idx = ((static_cast<size_t>(n) * channels + c) * out_h + oh) * out_w + ow;
            sum += grad_output[out_idx];
        }
    }
    
    size_t in_idx = ((static_cast<size_t>(n) * channels + c) * in_h + ih) * in_w + iw;
    grad_input[in_idx] = sum;
}

// Wrapper for optimized upsample backward
void gpu_upsample2d_backward_opt(const GPUTensor4D& input, const GPUTensor4D& grad_output,
                                  GPUTensor4D& grad_input, int scale) {
    if (grad_input.n != input.n || grad_input.c != input.c ||
        grad_input.h != input.h || grad_input.w != input.w) {
        grad_input.allocate(input.n, input.c, input.h, input.w);
    }
    
    dim3 block(16, 16);
    dim3 grid(
        (input.w + block.x - 1) / block.x,
        (input.h + block.y - 1) / block.y,
        input.n * input.c
    );
    
    upsample2d_backward_opt_kernel<<<grid, block>>>(
        grad_output.d_data, grad_input.d_data,
        input.n, input.c, input.h, input.w,
        scale
    );
    CUDA_CHECK(cudaGetLastError());
}

// Wrapper for optimized maxpool backward
void gpu_maxpool2d_backward_opt(const GPUTensor4D& input, const GPUTensor4D& grad_output,
                                 GPUTensor4D& grad_input, int k, int stride) {
    if (grad_input.n != input.n || grad_input.c != input.c ||
        grad_input.h != input.h || grad_input.w != input.w) {
        grad_input.allocate(input.n, input.c, input.h, input.w);
    }
    
    // Zero out grad_input first
    CUDA_CHECK(cudaMemset(grad_input.d_data, 0, grad_input.bytes()));
    
    int out_h = grad_output.h;
    int out_w = grad_output.w;
    
    dim3 block(16, 16);
    dim3 grid(
        (out_w + block.x - 1) / block.x,
        (out_h + block.y - 1) / block.y,
        input.n * input.c
    );
    
    maxpool2d_backward_opt_kernel<<<grid, block>>>(
        input.d_data, grad_output.d_data, grad_input.d_data,
        input.n, input.c, input.h, input.w,
        out_h, out_w, k, stride
    );
    CUDA_CHECK(cudaGetLastError());
}

// Cleanup function for memory pool
void cleanup_gpu_opt_buffers() {
    if (d_im2col_buffer) {
        cudaFree(d_im2col_buffer);
        d_im2col_buffer = nullptr;
        im2col_buffer_size = 0;
    }
}

#endif  // USE_OPTIMIZED_KERNELS