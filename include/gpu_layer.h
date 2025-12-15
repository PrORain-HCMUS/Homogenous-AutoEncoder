#ifndef GPU_LAYER_H
#define GPU_LAYER_H

#include <cstddef>

struct GPUTensor4D {
    int n = 0, c = 0, h = 0, w = 0;
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

struct OptimizerConfig {
    float momentum = 0.9f, weight_decay = 1e-4f;
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    bool use_adamw = true;
    int step = 0;
    OptimizerConfig() = default;
};

class GPUConv2DLayer {
public:
    GPUConv2DLayer(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 1);
    ~GPUConv2DLayer();
    void forward(const GPUTensor4D& input, GPUTensor4D& output) const;
    void backward(const GPUTensor4D& input, const GPUTensor4D& grad_output, GPUTensor4D& grad_input, float learning_rate);
    void backward_momentum(const GPUTensor4D& input, const GPUTensor4D& grad_output, GPUTensor4D& grad_input, float learning_rate, const OptimizerConfig& opt_config);
    float* get_weights() const { return d_weights_; }
    float* get_bias() const { return d_bias_; }
    float* get_grad_weights() const { return d_grad_weights_; }
    float* get_grad_bias() const { return d_grad_bias_; }
    float* get_m_weights() const { return d_m_weights_; }
    float* get_v_weights() const { return d_v_weights_; }
    float* get_m_bias() const { return d_m_bias_; }
    float* get_v_bias() const { return d_v_bias_; }
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
    float *d_weights_ = nullptr, *d_bias_ = nullptr, *d_grad_weights_ = nullptr, *d_grad_bias_ = nullptr;
    float *d_m_weights_ = nullptr, *d_v_weights_ = nullptr, *d_m_bias_ = nullptr, *d_v_bias_ = nullptr;
    size_t weights_size_;
};

class GPUBatchNorm2D {
public:
    GPUBatchNorm2D(int num_features, float momentum = 0.9f, float eps = 1e-5f);
    ~GPUBatchNorm2D();
    void forward(const GPUTensor4D& input, GPUTensor4D& output, bool training = true);
    void backward(const GPUTensor4D& input, const GPUTensor4D& grad_output, GPUTensor4D& grad_input, float learning_rate);
    void copy_params_to_host(float* h_gamma, float* h_beta, float* h_mean, float* h_var) const;
    void copy_params_from_host(const float* h_gamma, const float* h_beta, const float* h_mean, const float* h_var);
    float* get_gamma() const { return d_gamma_; }
    float* get_beta() const { return d_beta_; }
    float* get_cache_mean() const { return d_cache_mean_; }
    float* get_cache_var() const { return d_cache_var_; }
    float* get_running_mean() const { return d_running_mean_; }
    float* get_running_var() const { return d_running_var_; }
    float get_eps() const { return eps_; }
    float get_momentum() const { return momentum_; }
    int get_num_features() const { return num_features_; }
    float* get_grad_gamma() const { return d_grad_gamma_; }
    float* get_grad_beta() const { return d_grad_beta_; }
private:
    int num_features_;
    float momentum_, eps_;
    float *d_gamma_, *d_beta_, *d_running_mean_, *d_running_var_;
    float *d_grad_gamma_, *d_grad_beta_, *d_cache_mean_, *d_cache_var_, *d_cache_normalized_;
};

class GPUReLULayer {
public:
    void forward(const GPUTensor4D& input, GPUTensor4D& output) const;
    void backward(const GPUTensor4D& input, const GPUTensor4D& grad_output, GPUTensor4D& grad_input) const;
};

// PReLU with channel-wise learnable slope: f(x) = max(0,x) + alpha * min(0,x)
class GPUPReLULayer {
public:
    explicit GPUPReLULayer(int num_channels);
    ~GPUPReLULayer();
    void forward(const GPUTensor4D& input, GPUTensor4D& output) const;
    void backward(const GPUTensor4D& input, const GPUTensor4D& grad_output, GPUTensor4D& grad_input, float learning_rate);
    float* get_alpha() const { return d_alpha_; }
    float* get_grad_alpha() const { return d_grad_alpha_; }
    int get_num_channels() const { return num_channels_; }
private:
    int num_channels_;
    float* d_alpha_ = nullptr;      // Learnable slope per channel
    float* d_grad_alpha_ = nullptr; // Gradient for alpha
};

class GPUMaxPool2DLayer {
public:
    explicit GPUMaxPool2DLayer(int kernel_size = 2, int stride = 2);
    void forward(const GPUTensor4D& input, GPUTensor4D& output) const;
    void backward(const GPUTensor4D& input, const GPUTensor4D& grad_output, GPUTensor4D& grad_input) const;
    int get_output_h(int input_h) const { return (input_h - k_) / stride_ + 1; }
    int get_output_w(int input_w) const { return (input_w - k_) / stride_ + 1; }
private:
    int k_, stride_;
};

class GPUUpSample2DLayer {
public:
    explicit GPUUpSample2DLayer(int scale = 2);
    void forward(const GPUTensor4D& input, GPUTensor4D& output) const;
    void backward(const GPUTensor4D& input, const GPUTensor4D& grad_output, GPUTensor4D& grad_input) const;
    int get_output_h(int input_h) const { return input_h * scale_; }
    int get_output_w(int input_w) const { return input_w * scale_; }
private:
    int scale_;
};

float gpu_mse_loss(const GPUTensor4D& output, const GPUTensor4D& target);
float gpu_mse_loss_with_grad(const GPUTensor4D& output, const GPUTensor4D& target, GPUTensor4D& grad_output);

class GPUSigmoidLayer {
public:
    void forward(const GPUTensor4D& input, GPUTensor4D& output) const;
    void backward(const GPUTensor4D& output, const GPUTensor4D& grad_output, GPUTensor4D& grad_input) const;
};

float gpu_bce_loss(const GPUTensor4D& output, const GPUTensor4D& target);
float gpu_bce_loss_with_grad(const GPUTensor4D& output, const GPUTensor4D& target, GPUTensor4D& grad_output);
void gpu_clip_gradients(GPUTensor4D& grad, float max_norm);

#ifdef USE_OPTIMIZED_KERNELS
void gpu_relu_forward_opt(const GPUTensor4D& input, GPUTensor4D& output);
void gpu_relu_backward_opt(const GPUTensor4D& input, const GPUTensor4D& grad_output, GPUTensor4D& grad_input);
void gpu_maxpool2d_forward_opt(const GPUTensor4D& input, GPUTensor4D& output, int k, int stride);
void gpu_upsample2d_forward_opt(const GPUTensor4D& input, GPUTensor4D& output, int scale);
void gpu_conv2d_forward_tiled(const GPUTensor4D& input, const float* d_weights, const float* d_bias, GPUTensor4D& output, int in_c, int out_c, int k, int stride, int padding);
void gpu_conv2d_relu_forward_opt(const GPUTensor4D& input, const float* d_weights, const float* d_bias, GPUTensor4D& output, int in_c, int out_c, int k, int stride, int padding);
void gpu_conv2d_relu_fused_forward(const GPUTensor4D& input, const GPUConv2DLayer& conv, GPUTensor4D& output);
void gpu_conv2d_forward_opt(const GPUTensor4D& input, const GPUConv2DLayer& conv, GPUTensor4D& output);
void gpu_conv2d_backward_opt(const GPUTensor4D& input, const GPUTensor4D& grad_output, const float* d_weights, float* d_grad_weights, float* d_grad_bias, GPUTensor4D& grad_input, int in_c, int out_c, int k, int stride, int padding);
void gpu_sgd_update_opt(float* params, const float* grads, float lr, size_t n);
void gpu_conv2d_backward_full_opt(const GPUTensor4D& input, const GPUTensor4D& grad_output, GPUTensor4D& grad_input, const GPUConv2DLayer& conv, float learning_rate);
void gpu_maxpool2d_backward_opt(const GPUTensor4D& input, const GPUTensor4D& grad_output, GPUTensor4D& grad_input, int k, int stride);
void gpu_upsample2d_backward_opt(const GPUTensor4D& input, const GPUTensor4D& grad_output, GPUTensor4D& grad_input, int scale);
void cleanup_gpu_opt_buffers();
void ensure_im2col_buffer(size_t required_size);
void init_cublas();
void cleanup_cublas();
void gpu_conv2d_forward_cublas(const GPUTensor4D& input, const float* d_weights, const float* d_bias, GPUTensor4D& output, int in_c, int out_c, int k, int stride, int padding);
void gpu_conv2d_forward_cublas_wrapper(const GPUTensor4D& input, const GPUConv2DLayer& conv, GPUTensor4D& output);
void init_cudnn();
void cleanup_cudnn();
void gpu_conv2d_forward_cudnn(const GPUTensor4D& input, const float* d_weights, const float* d_bias, GPUTensor4D& output, int in_c, int out_c, int k, int stride, int padding);
void gpu_conv2d_forward_cudnn_wrapper(const GPUTensor4D& input, const GPUConv2DLayer& conv, GPUTensor4D& output);
void gpu_conv2d_backward_cudnn_full(const GPUTensor4D& input, const GPUTensor4D& grad_output, GPUTensor4D& grad_input, const GPUConv2DLayer& conv, float learning_rate);
void gpu_conv2d_backward_cudnn_adamw(const GPUTensor4D& input, const GPUTensor4D& grad_output, GPUTensor4D& grad_input, GPUConv2DLayer& conv, float learning_rate, const OptimizerConfig& opt_config);
void gpu_batchnorm_prelu_fused_forward(const GPUTensor4D& input, GPUTensor4D& output,
    GPUBatchNorm2D& bn, const GPUPReLULayer& prelu, bool training);
void gpu_batchnorm_prelu_fused_forward_with_intermediate(
    const GPUTensor4D& input, GPUTensor4D& bn_output, GPUTensor4D& prelu_output,
    GPUBatchNorm2D& bn, const GPUPReLULayer& prelu, bool training);
void gpu_batchnorm_prelu_maxpool_fused_forward(const GPUTensor4D& input, GPUTensor4D& output,
    GPUBatchNorm2D& bn, const GPUPReLULayer& prelu, int pool_k, int pool_stride, bool training);
void gpu_prelu_batchnorm_fused_backward(
    const GPUTensor4D& bn_input,
    const GPUTensor4D& prelu_input,
    const GPUTensor4D& grad_output,
    GPUTensor4D& grad_input,
    GPUBatchNorm2D& bn, GPUPReLULayer& prelu, float learning_rate);
void gpu_maxpool_prelu_batchnorm_fused_backward(
    const GPUTensor4D& pool_input, const GPUTensor4D& bn_input, const GPUTensor4D& prelu_input,
    const GPUTensor4D& grad_output, GPUTensor4D& grad_input,
    GPUBatchNorm2D& bn, GPUPReLULayer& prelu, int pool_k, int pool_stride, float learning_rate);

void gpu_upsample_conv_fused_forward(const GPUTensor4D& input, const float* weights, const float* bias,
    GPUTensor4D& output, int in_c, int out_c, int k, int stride, int padding, int scale);
void gpu_upsample_conv_fused_forward_wrapper(const GPUTensor4D& input, const GPUConv2DLayer& conv, 
    GPUTensor4D& output, int scale);

#ifdef USE_FP16
void gpu_conv2d_forward_cudnn_fp16(const GPUTensor4D& input, const float* d_weights, const float* d_bias,
    GPUTensor4D& output, int in_c, int out_c, int k, int stride, int padding);
void gpu_conv2d_forward_cudnn_wrapper_fp16(const GPUTensor4D& input, const GPUConv2DLayer& conv, GPUTensor4D& output);
void cleanup_fp16_workspace();
#endif
#endif

#endif