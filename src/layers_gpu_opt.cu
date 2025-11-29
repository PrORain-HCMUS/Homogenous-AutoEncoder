#ifdef USE_OPTIMIZED_KERNELS

#include "gpu_layer.h"
#include "cuda_utils.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdio>

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
        // (would call the original kernel here)
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

#endif  // USE_OPTIMIZED_KERNELS