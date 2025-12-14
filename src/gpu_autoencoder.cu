#include "gpu_autoencoder.h"
#include "autoencoder.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <iostream>

GPUAutoencoder::GPUAutoencoder(LossType loss_type)
    : loss_type_(loss_type), 
      conv1_(3, 256, 3, 1, 1), bn1_(256), pool1_(2, 2), 
      conv2_(256, 128, 3, 1, 1), bn2_(128), pool2_(2, 2),
      conv3_(128, 128, 3, 1, 1), bn3_(128), up1_(2), 
      conv4_(128, 256, 3, 1, 1), bn4_(256), up2_(2), 
      conv5_(256, 3, 3, 1, 1) {}

GPUAutoencoder::~GPUAutoencoder() {}
void GPUAutoencoder::synchronize() { CUDA_CHECK(cudaDeviceSynchronize()); }

void GPUAutoencoder::forward(const GPUTensor4D& input, GPUTensor4D& output) {
    conv1_.forward(input, x1_);
    bn1_.forward(x1_, x2_, false);
    relu1_.forward(x2_, x3_);
    pool1_.forward(x3_, x4_);
    
    conv2_.forward(x4_, x5_);
    bn2_.forward(x5_, x6_, false);
    relu2_.forward(x6_, x7_);
    pool2_.forward(x7_, x8_);
    
    conv3_.forward(x8_, x9_);
    bn3_.forward(x9_, x10_, false);
    relu3_.forward(x10_, x11_);
    up1_.forward(x11_, x12_);
    
    conv4_.forward(x12_, x13_);
    bn4_.forward(x13_, x14_, false);
    relu4_.forward(x14_, x15_);
    up2_.forward(x15_, x16_);
    
    conv5_.forward(x16_, x17_);
    
    if (loss_type_ == LossType::BCE) {
        sigmoid_.forward(x17_, output);
    } else {
        if (output.n != x17_.n || output.c != x17_.c || output.h != x17_.h || output.w != x17_.w)
            output.allocate(x17_.n, x17_.c, x17_.h, x17_.w);
        CUDA_CHECK(cudaMemcpy(output.d_data, x17_.d_data, x17_.bytes(), cudaMemcpyDeviceToDevice));
    }
}

void GPUAutoencoder::encode(const GPUTensor4D& input, GPUTensor4D& latent) {
#ifdef USE_OPTIMIZED_KERNELS
    gpu_conv2d_forward_cudnn_wrapper(input, conv1_, x1_);
#else
    conv1_.forward(input, x1_);
#endif
    bn1_.forward(x1_, x2_, false);
    relu1_.forward(x2_, x3_);
    pool1_.forward(x3_, x4_);
    
#ifdef USE_OPTIMIZED_KERNELS
    gpu_conv2d_forward_cudnn_wrapper(x4_, conv2_, x5_);
#else
    conv2_.forward(x4_, x5_);
#endif
    bn2_.forward(x5_, x6_, false);
    relu2_.forward(x6_, x7_);
    pool2_.forward(x7_, latent);
}

float GPUAutoencoder::train_step(const GPUTensor4D& input, const GPUTensor4D& target, float learning_rate) {
    // Forward pass
#ifdef USE_OPTIMIZED_KERNELS
    gpu_conv2d_forward_cudnn_wrapper(input, conv1_, x1_);
#else
    conv1_.forward(input, x1_);
#endif
    bn1_.forward(x1_, x2_, true);
    relu1_.forward(x2_, x3_);
    pool1_.forward(x3_, x4_);
    
#ifdef USE_OPTIMIZED_KERNELS
    gpu_conv2d_forward_cudnn_wrapper(x4_, conv2_, x5_);
#else
    conv2_.forward(x4_, x5_);
#endif
    bn2_.forward(x5_, x6_, true);
    relu2_.forward(x6_, x7_);
    pool2_.forward(x7_, x8_);
    
#ifdef USE_OPTIMIZED_KERNELS
    gpu_conv2d_forward_cudnn_wrapper(x8_, conv3_, x9_);
#else
    conv3_.forward(x8_, x9_);
#endif
    bn3_.forward(x9_, x10_, true);
    relu3_.forward(x10_, x11_);
    up1_.forward(x11_, x12_);
    
#ifdef USE_OPTIMIZED_KERNELS
    gpu_conv2d_forward_cudnn_wrapper(x12_, conv4_, x13_);
#else
    conv4_.forward(x12_, x13_);
#endif
    bn4_.forward(x13_, x14_, true);
    relu4_.forward(x14_, x15_);
    up2_.forward(x15_, x16_);
    
#ifdef USE_OPTIMIZED_KERNELS
    gpu_conv2d_forward_cudnn_wrapper(x16_, conv5_, x17_);
#else
    conv5_.forward(x16_, x17_);
#endif
    
    // Loss + gradient
    float loss;
    if (loss_type_ == LossType::BCE) {
        sigmoid_.forward(x17_, x18_);
        loss = gpu_bce_loss_with_grad(x18_, target, g18_);
        sigmoid_.backward(x18_, g18_, g17_);
    } else {
        loss = gpu_mse_loss_with_grad(x17_, target, g17_);
    }
    
    // Backward pass
#ifdef USE_OPTIMIZED_KERNELS
    gpu_conv2d_backward_cudnn_full(x16_, g17_, g16_, conv5_, learning_rate);
#else
    conv5_.backward(x16_, g17_, g16_, learning_rate);
#endif
    
    up2_.backward(x15_, g16_, g15_);
    relu4_.backward(x14_, g15_, g14_);
    bn4_.backward(x13_, g14_, g13_, learning_rate);
#ifdef USE_OPTIMIZED_KERNELS
    gpu_conv2d_backward_cudnn_full(x12_, g13_, g12_, conv4_, learning_rate);
#else
    conv4_.backward(x12_, g13_, g12_, learning_rate);
#endif
    
    up1_.backward(x11_, g12_, g11_);
    relu3_.backward(x10_, g11_, g10_);
    bn3_.backward(x9_, g10_, g9_, learning_rate);
#ifdef USE_OPTIMIZED_KERNELS
    gpu_conv2d_backward_cudnn_full(x8_, g9_, g8_, conv3_, learning_rate);
#else
    conv3_.backward(x8_, g9_, g8_, learning_rate);
#endif
    
    pool2_.backward(x7_, g8_, g7_);
    relu2_.backward(x6_, g7_, g6_);
    bn2_.backward(x5_, g6_, g5_, learning_rate);
#ifdef USE_OPTIMIZED_KERNELS
    gpu_conv2d_backward_cudnn_full(x4_, g5_, g4_, conv2_, learning_rate);
#else
    conv2_.backward(x4_, g5_, g4_, learning_rate);
#endif
    
    pool1_.backward(x3_, g4_, g3_);
    relu1_.backward(x2_, g3_, g2_);
    bn1_.backward(x1_, g2_, g1_, learning_rate);
#ifdef USE_OPTIMIZED_KERNELS
    gpu_conv2d_backward_cudnn_full(input, g1_, g0_, conv1_, learning_rate);
#else
    conv1_.backward(input, g1_, g0_, learning_rate);
#endif
    
    return loss;
}

float GPUAutoencoder::train_step_momentum(const GPUTensor4D& input, const GPUTensor4D& target, 
                                          float learning_rate, const OptimizerConfig& opt_config) {
    // Forward pass
#ifdef USE_OPTIMIZED_KERNELS
    gpu_conv2d_forward_cudnn_wrapper(input, conv1_, x1_);
#else
    conv1_.forward(input, x1_);
#endif
    bn1_.forward(x1_, x2_, true);
    relu1_.forward(x2_, x3_);
    pool1_.forward(x3_, x4_);
    
#ifdef USE_OPTIMIZED_KERNELS
    gpu_conv2d_forward_cudnn_wrapper(x4_, conv2_, x5_);
#else
    conv2_.forward(x4_, x5_);
#endif
    bn2_.forward(x5_, x6_, true);
    relu2_.forward(x6_, x7_);
    pool2_.forward(x7_, x8_);
    
#ifdef USE_OPTIMIZED_KERNELS
    gpu_conv2d_forward_cudnn_wrapper(x8_, conv3_, x9_);
#else
    conv3_.forward(x8_, x9_);
#endif
    bn3_.forward(x9_, x10_, true);
    relu3_.forward(x10_, x11_);
    up1_.forward(x11_, x12_);
    
#ifdef USE_OPTIMIZED_KERNELS
    gpu_conv2d_forward_cudnn_wrapper(x12_, conv4_, x13_);
#else
    conv4_.forward(x12_, x13_);
#endif
    bn4_.forward(x13_, x14_, true);
    relu4_.forward(x14_, x15_);
    up2_.forward(x15_, x16_);
    
#ifdef USE_OPTIMIZED_KERNELS
    gpu_conv2d_forward_cudnn_wrapper(x16_, conv5_, x17_);
#else
    conv5_.forward(x16_, x17_);
#endif
    
    // Loss + gradient
    float loss;
    if (loss_type_ == LossType::BCE) {
        sigmoid_.forward(x17_, x18_);
        loss = gpu_bce_loss_with_grad(x18_, target, g18_);
        sigmoid_.backward(x18_, g18_, g17_);
    } else {
        loss = gpu_mse_loss_with_grad(x17_, target, g17_);
    }
#ifdef USE_OPTIMIZED_KERNELS
    gpu_conv2d_backward_cudnn_adamw(x16_, g17_, g16_, conv5_, learning_rate, opt_config);
#else
    conv5_.backward_momentum(x16_, g17_, g16_, learning_rate, opt_config);
#endif
    
    up2_.backward(x15_, g16_, g15_);
    relu4_.backward(x14_, g15_, g14_);
    bn4_.backward(x13_, g14_, g13_, learning_rate);
#ifdef USE_OPTIMIZED_KERNELS
    gpu_conv2d_backward_cudnn_adamw(x12_, g13_, g12_, conv4_, learning_rate, opt_config);
#else
    conv4_.backward_momentum(x12_, g13_, g12_, learning_rate, opt_config);
#endif
    
    up1_.backward(x11_, g12_, g11_);
    relu3_.backward(x10_, g11_, g10_);
    bn3_.backward(x9_, g10_, g9_, learning_rate);
#ifdef USE_OPTIMIZED_KERNELS
    gpu_conv2d_backward_cudnn_adamw(x8_, g9_, g8_, conv3_, learning_rate, opt_config);
#else
    conv3_.backward_momentum(x8_, g9_, g8_, learning_rate, opt_config);
#endif
    
    pool2_.backward(x7_, g8_, g7_);
    relu2_.backward(x6_, g7_, g6_);
    bn2_.backward(x5_, g6_, g5_, learning_rate);
#ifdef USE_OPTIMIZED_KERNELS
    gpu_conv2d_backward_cudnn_adamw(x4_, g5_, g4_, conv2_, learning_rate, opt_config);
#else
    conv2_.backward_momentum(x4_, g5_, g4_, learning_rate, opt_config);
#endif
    
    pool1_.backward(x3_, g4_, g3_);
    relu1_.backward(x2_, g3_, g2_);
    bn1_.backward(x1_, g2_, g1_, learning_rate);
#ifdef USE_OPTIMIZED_KERNELS
    gpu_conv2d_backward_cudnn_adamw(input, g1_, g0_, conv1_, learning_rate, opt_config);
#else
    conv1_.backward_momentum(input, g1_, g0_, learning_rate, opt_config);
#endif
    
    return loss;
}

bool GPUAutoencoder::save_weights(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    const uint32_t MAGIC = 0x48414557, VERSION = 2, NUM_LAYERS = 5;
    out.write(reinterpret_cast<const char*>(&MAGIC), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&VERSION), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&NUM_LAYERS), sizeof(uint32_t));
    auto save_conv = [&out](const GPUConv2DLayer& layer) {
        int in_c = layer.get_in_channels(), out_c = layer.get_out_channels(), k = layer.get_kernel_size();
        out.write(reinterpret_cast<const char*>(&in_c), sizeof(int));
        out.write(reinterpret_cast<const char*>(&out_c), sizeof(int));
        out.write(reinterpret_cast<const char*>(&k), sizeof(int));
        size_t w_size = static_cast<size_t>(out_c) * in_c * k * k;
        int b_size = out_c;
        std::vector<float> h_weights(w_size), h_bias(b_size);
        layer.copy_weights_to_host(h_weights.data(), h_bias.data());
        int w_size_int = static_cast<int>(w_size);
        out.write(reinterpret_cast<const char*>(&w_size_int), sizeof(int));
        out.write(reinterpret_cast<const char*>(h_weights.data()), static_cast<std::streamsize>(w_size) * sizeof(float));
        out.write(reinterpret_cast<const char*>(&b_size), sizeof(int));
        out.write(reinterpret_cast<const char*>(h_bias.data()), static_cast<std::streamsize>(b_size) * sizeof(float));
    };
    save_conv(conv1_); save_conv(conv2_); save_conv(conv3_); save_conv(conv4_); save_conv(conv5_);
    return static_cast<bool>(out);
}

bool GPUAutoencoder::load_weights(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    const uint32_t EXPECTED_MAGIC = 0x48414557;
    uint32_t magic = 0, version = 0, num_layers = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&num_layers), sizeof(uint32_t));
    if (magic != EXPECTED_MAGIC) { std::cerr << "Invalid weight file format" << std::endl; return false; }
    if (version != 1 && version != 2) { std::cerr << "Unsupported version: " << version << std::endl; return false; }
    if (num_layers != 5) { std::cerr << "Layer count mismatch" << std::endl; return false; }
    auto load_conv = [&in](GPUConv2DLayer& layer) -> bool {
        int in_c = 0, out_c = 0, k = 0;
        in.read(reinterpret_cast<char*>(&in_c), sizeof(int));
        in.read(reinterpret_cast<char*>(&out_c), sizeof(int));
        in.read(reinterpret_cast<char*>(&k), sizeof(int));
        if (in_c != layer.get_in_channels() || out_c != layer.get_out_channels() || k != layer.get_kernel_size()) return false;
        size_t expected_w = static_cast<size_t>(out_c) * in_c * k * k;
        int w_size = 0, b_size = 0;
        in.read(reinterpret_cast<char*>(&w_size), sizeof(int));
        if (static_cast<size_t>(w_size) != expected_w) return false;
        std::vector<float> h_weights(w_size);
        in.read(reinterpret_cast<char*>(h_weights.data()), static_cast<std::streamsize>(w_size) * sizeof(float));
        in.read(reinterpret_cast<char*>(&b_size), sizeof(int));
        if (b_size != out_c) return false;
        std::vector<float> h_bias(b_size);
        in.read(reinterpret_cast<char*>(h_bias.data()), static_cast<std::streamsize>(b_size) * sizeof(float));
        layer.copy_weights_from_host(h_weights.data(), h_bias.data());
        return true;
    };
    try {
        if (!load_conv(conv1_) || !load_conv(conv2_) || !load_conv(conv3_) || !load_conv(conv4_) || !load_conv(conv5_)) return false;
    } catch (...) { return false; }
    return static_cast<bool>(in);
}

void GPUAutoencoder::load_weights_from_cpu(const Autoencoder&) {
    std::cerr << "Warning: load_weights_from_cpu not implemented." << std::endl;
}

void tensor_cpu_to_gpu(const Tensor4D& cpu_tensor, GPUTensor4D& gpu_tensor) {
    if (gpu_tensor.n != cpu_tensor.n || gpu_tensor.c != cpu_tensor.c || gpu_tensor.h != cpu_tensor.h || gpu_tensor.w != cpu_tensor.w)
        gpu_tensor.allocate(cpu_tensor.n, cpu_tensor.c, cpu_tensor.h, cpu_tensor.w);
    gpu_tensor.copy_from_host(cpu_tensor.data.data());
}

void tensor_gpu_to_cpu(const GPUTensor4D& gpu_tensor, Tensor4D& cpu_tensor) {
    cpu_tensor = Tensor4D(gpu_tensor.n, gpu_tensor.c, gpu_tensor.h, gpu_tensor.w);
    gpu_tensor.copy_to_host(cpu_tensor.data.data());
}

void batch_cpu_to_gpu(const float* cpu_data, int n, int c, int h, int w, GPUTensor4D& gpu_tensor) {
    if (gpu_tensor.n != n || gpu_tensor.c != c || gpu_tensor.h != h || gpu_tensor.w != w)
        gpu_tensor.allocate(n, c, h, w);
    gpu_tensor.copy_from_host(cpu_data);
}
