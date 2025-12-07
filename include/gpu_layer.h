#ifndef GPU_LAYER_H
#define GPU_LAYER_H

#include <cstddef>

struct GPUTensor4D {
    int n = 0;
    int c = 0;
    int h = 0;
    int w = 0;
    float* d_data = nullptr;

    GPUTensor4D() = default;
    GPUTensor4D(int n_, int c_, int h_, int w_);
    ~GPUTensor4D();

    GPUTensor4D(const GPUTensor4D&) = delete;
    GPUTensor4D& operator=(const GPUTensor4D&) = delete;

    GPUTensor4D(GPUTensor4D&& other) noexcept;
    GPUTensor4D& operator=(GPUTensor4D&& other) noexcept;

    void allocate(int n_, int c_, int h_, int w_);
    void free();
    size_t size() const { return static_cast<size_t>(n) * c * h * w; }
    size_t bytes() const { return size() * sizeof(float); }

    void copy_from_host(const float* h_data);
    void copy_to_host(float* h_data) const;
};

class GPUConv2DLayer {
public:
    GPUConv2DLayer(int in_channels, int out_channels, int kernel_size,
                   int stride = 1, int padding = 1);
    ~GPUConv2DLayer();

    void forward(const GPUTensor4D& input, GPUTensor4D& output) const;
    void backward(const GPUTensor4D& input, const GPUTensor4D& grad_output,
                  GPUTensor4D& grad_input, float learning_rate);

    // Expose weights for optimized kernels
    float* get_weights() const { return d_weights_; }
    float* get_bias() const { return d_bias_; }
    float* get_grad_weights() const { return d_grad_weights_; }
    float* get_grad_bias() const { return d_grad_bias_; }
    size_t get_weights_size() const { return weights_size_; }

    void copy_weights_from_host(const float* h_weights, const float* h_bias);
    void copy_weights_to_host(float* h_weights, float* h_bias) const;

    int get_output_h(int input_h) const { return (input_h + 2 * padding_ - k_) / stride_ + 1; }
    int get_output_w(int input_w) const { return (input_w + 2 * padding_ - k_) / stride_ + 1; }
    int get_out_channels() const { return out_c_; }
    int get_in_channels() const { return in_c_; }
    int get_kernel_size() const { return k_; }
    int get_stride() const { return stride_; }
    int get_padding() const { return padding_; }

private:
    int in_c_, out_c_, k_, stride_, padding_;
    float* d_weights_ = nullptr;
    float* d_bias_ = nullptr;
    float* d_grad_weights_ = nullptr;
    float* d_grad_bias_ = nullptr;
    size_t weights_size_;
};

class GPUReLULayer {
public:
    void forward(const GPUTensor4D& input, GPUTensor4D& output) const;
    void backward(const GPUTensor4D& input, const GPUTensor4D& grad_output,
                  GPUTensor4D& grad_input) const;
};

class GPUMaxPool2DLayer {
public:
    explicit GPUMaxPool2DLayer(int kernel_size = 2, int stride = 2);

    void forward(const GPUTensor4D& input, GPUTensor4D& output) const;
    void backward(const GPUTensor4D& input, const GPUTensor4D& grad_output,
                  GPUTensor4D& grad_input) const;

    int get_output_h(int input_h) const { return (input_h - k_) / stride_ + 1; }
    int get_output_w(int input_w) const { return (input_w - k_) / stride_ + 1; }

private:
    int k_, stride_;
};

class GPUUpSample2DLayer {
public:
    explicit GPUUpSample2DLayer(int scale = 2);

    void forward(const GPUTensor4D& input, GPUTensor4D& output) const;
    void backward(const GPUTensor4D& input, const GPUTensor4D& grad_output,
                  GPUTensor4D& grad_input) const;

    int get_output_h(int input_h) const { return input_h * scale_; }
    int get_output_w(int input_w) const { return input_w * scale_; }

private:
    int scale_;
};

float gpu_mse_loss(const GPUTensor4D& output, const GPUTensor4D& target);
float gpu_mse_loss_with_grad(const GPUTensor4D& output, const GPUTensor4D& target,
                              GPUTensor4D& grad_output);

#ifdef USE_OPTIMIZED_KERNELS
void gpu_relu_forward_opt(const GPUTensor4D& input, GPUTensor4D& output);
void gpu_relu_backward_opt(const GPUTensor4D& input, const GPUTensor4D& grad_output,
                           GPUTensor4D& grad_input);
void gpu_maxpool2d_forward_opt(const GPUTensor4D& input, GPUTensor4D& output,
                               int k, int stride);
void gpu_upsample2d_forward_opt(const GPUTensor4D& input, GPUTensor4D& output, int scale);
void gpu_conv2d_forward_tiled(const GPUTensor4D& input, const float* d_weights,
                              const float* d_bias, GPUTensor4D& output,
                              int in_c, int out_c, int k, int stride, int padding);
void gpu_conv2d_relu_forward_opt(const GPUTensor4D& input, const float* d_weights,
                                 const float* d_bias, GPUTensor4D& output,
                                 int in_c, int out_c, int k, int stride, int padding);

// Fused Conv+ReLU forward pass - skips ReLU layer entirely
void gpu_conv2d_relu_fused_forward(const GPUTensor4D& input, 
                                    const GPUConv2DLayer& conv,
                                    GPUTensor4D& output);

// Optimized conv forward using tiled shared memory
void gpu_conv2d_forward_opt(const GPUTensor4D& input,
                            const GPUConv2DLayer& conv,
                            GPUTensor4D& output);

// Optimized backward pass with parallel reduction
void gpu_conv2d_backward_opt(
    const GPUTensor4D& input,
    const GPUTensor4D& grad_output,
    const float* d_weights,
    float* d_grad_weights,
    float* d_grad_bias,
    GPUTensor4D& grad_input,
    int in_c, int out_c, int k, int stride, int padding
);

// Optimized SGD update with vectorized access
void gpu_sgd_update_opt(float* params, const float* grads, float lr, size_t n);

// Full optimized backward for conv layer (includes weight update)
void gpu_conv2d_backward_full_opt(
    const GPUTensor4D& input,
    const GPUTensor4D& grad_output,
    GPUTensor4D& grad_input,
    const GPUConv2DLayer& conv,
    float learning_rate
);
#endif

#endif  // GPU_LAYER_H
