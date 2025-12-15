#ifndef GPU_AUGMENTATION_CUH
#define GPU_AUGMENTATION_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void init_curand_states(curandState* states, unsigned long long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) curand_init(seed, idx, 0, &states[idx]);
}

__global__ void random_crop_flip_jitter_cutout_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    curandState* __restrict__ states,
    int N, int C, int H, int W,
    int padding, float flip_prob,
    bool color_jitter, float brightness_range,
    bool cutout, int cutout_size
) {
    int n = blockIdx.z;
    if (n >= N) return;
    
    curandState local_state = states[n];
    
    int crop_y = static_cast<int>(curand_uniform(&local_state) * (2 * padding + 1));
    int crop_x = static_cast<int>(curand_uniform(&local_state) * (2 * padding + 1));
    bool do_flip = curand_uniform(&local_state) < flip_prob;
    
    float brightness_delta = 0.0f;
    if (color_jitter) {
        brightness_delta = (curand_uniform(&local_state) * 2.0f - 1.0f) * brightness_range;
    }
    
    int cutout_center_h = 0, cutout_center_w = 0;
    bool apply_cutout = false;
    if (cutout && cutout_size > 0) {
        apply_cutout = curand_uniform(&local_state) < 0.5f;
        if (apply_cutout) {
            cutout_center_h = static_cast<int>(curand_uniform(&local_state) * H);
            cutout_center_w = static_cast<int>(curand_uniform(&local_state) * W);
        }
    }
    
    if (threadIdx.x == 0 && threadIdx.y == 0) states[n] = local_state;
    
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    if (oy >= H || ox >= W) return;
    
    int iy = oy + crop_y - padding;
    int ix = do_flip ? (W - 1 - ox) + crop_x - padding : ox + crop_x - padding;
    
    int half_cutout = cutout_size / 2;
    bool in_cutout = apply_cutout && 
        (oy >= cutout_center_h - half_cutout) && (oy < cutout_center_h + half_cutout) &&
        (ox >= cutout_center_w - half_cutout) && (ox < cutout_center_w + half_cutout);
    
    for (int c = 0; c < C; ++c) {
        float val = 0.0f;
        if (!in_cutout) {
            if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                val = input[((static_cast<size_t>(n) * C + c) * H + iy) * W + ix];
            }
            val = fminf(1.0f, fmaxf(0.0f, val + brightness_delta));
        }
        output[((static_cast<size_t>(n) * C + c) * H + oy) * W + ox] = val;
    }
}

class GPUAugmenter {
    curandState* d_states_ = nullptr;
    int max_batch_size_ = 0;
    
public:
    GPUAugmenter() = default;
    
    ~GPUAugmenter() {
        if (d_states_) cudaFree(d_states_);
    }
    
    void init(int batch_size, unsigned long long seed = 42) {
        if (batch_size > max_batch_size_) {
            if (d_states_) cudaFree(d_states_);
            cudaMalloc(&d_states_, batch_size * sizeof(curandState));
            max_batch_size_ = batch_size;
        }
        init_curand_states<<<(batch_size + 255) / 256, 256>>>(d_states_, seed, batch_size);
    }
    
    void augment(const float* d_input, float* d_output, int N, int C, int H, int W,
                 int padding = 4, float flip_prob = 0.5f,
                 bool color_jitter = true, float brightness_range = 0.2f,
                 bool cutout = true, int cutout_size = 8) {
        if (N > max_batch_size_) init(N);
        
        dim3 block(16, 16);
        dim3 grid((W + 15) / 16, (H + 15) / 16, N);
        random_crop_flip_jitter_cutout_kernel<<<grid, block>>>(
            d_input, d_output, d_states_, N, C, H, W,
            padding, flip_prob,
            color_jitter, brightness_range,
            cutout, cutout_size);
    }
};

#endif
