#include "gpu_autoencoder.h"
#include "autoencoder.h"
#include "cuda_utils.h"

#include <cuda_runtime.h>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <iostream>

GPUAutoencoder::GPUAutoencoder()
    : conv1_(3, 256, 3, 1, 1),
      pool1_(2, 2),
      conv2_(256, 128, 3, 1, 1),
      pool2_(2, 2),
      conv3_(128, 128, 3, 1, 1),
      up1_(2),
      conv4_(128, 256, 3, 1, 1),
      up2_(2),
      conv5_(256, 3, 3, 1, 1)
{
}

GPUAutoencoder::~GPUAutoencoder() {
}

void GPUAutoencoder::synchronize() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

void GPUAutoencoder::forward(const GPUTensor4D& input, GPUTensor4D& output) {
    conv1_.forward(input, x1_);
    relu1_.forward(x1_, x2_);
    pool1_.forward(x2_, x3_);

    conv2_.forward(x3_, x4_);
    relu2_.forward(x4_, x5_);
    pool2_.forward(x5_, x6_);

    conv3_.forward(x6_, x7_);
    relu3_.forward(x7_, x8_);
    up1_.forward(x8_, x9_);

    conv4_.forward(x9_, x10_);
    relu4_.forward(x10_, x11_);
    up2_.forward(x11_, x12_);

    conv5_.forward(x12_, output);
}

void GPUAutoencoder::encode(const GPUTensor4D& input, GPUTensor4D& latent) {
    conv1_.forward(input, x1_);
    relu1_.forward(x1_, x2_);
    pool1_.forward(x2_, x3_);

    conv2_.forward(x3_, x4_);
    relu2_.forward(x4_, x5_);
    pool2_.forward(x5_, latent);
}

float GPUAutoencoder::train_step(const GPUTensor4D& input, const GPUTensor4D& target, 
                                  float learning_rate) {
    conv1_.forward(input, x1_);
    relu1_.forward(x1_, x2_);
    pool1_.forward(x2_, x3_);

    conv2_.forward(x3_, x4_);
    relu2_.forward(x4_, x5_);
    pool2_.forward(x5_, x6_);

    conv3_.forward(x6_, x7_);
    relu3_.forward(x7_, x8_);
    up1_.forward(x8_, x9_);

    conv4_.forward(x9_, x10_);
    relu4_.forward(x10_, x11_);
    up2_.forward(x11_, x12_);

    conv5_.forward(x12_, x13_);

    float loss = gpu_mse_loss_with_grad(x13_, target, g13_);

    conv5_.backward(x12_, g13_, g12_, learning_rate);
    up2_.backward(x11_, g12_, g11_);
    relu4_.backward(x10_, g11_, g10_);
    conv4_.backward(x9_, g10_, g9_, learning_rate);
    up1_.backward(x8_, g9_, g8_);
    relu3_.backward(x7_, g8_, g7_);
    conv3_.backward(x6_, g7_, g6_, learning_rate);
    pool2_.backward(x5_, g6_, g5_);
    relu2_.backward(x4_, g5_, g4_);
    conv2_.backward(x3_, g4_, g3_, learning_rate);
    pool1_.backward(x2_, g3_, g2_);
    relu1_.backward(x1_, g2_, g1_);
    conv1_.backward(input, g1_, g0_, learning_rate);

    return loss;
}

bool GPUAutoencoder::save_weights(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;

    const uint32_t MAGIC = 0x48414557;
    const uint32_t VERSION = 1;
    const uint32_t NUM_LAYERS = 5;
    out.write(reinterpret_cast<const char*>(&MAGIC), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&VERSION), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&NUM_LAYERS), sizeof(uint32_t));

    auto save_conv = [&out](const GPUConv2DLayer& layer) {
        int in_c = layer.get_in_channels();
        int out_c = layer.get_out_channels();
        int k = layer.get_kernel_size();
        out.write(reinterpret_cast<const char*>(&in_c), sizeof(int));
        out.write(reinterpret_cast<const char*>(&out_c), sizeof(int));
        out.write(reinterpret_cast<const char*>(&k), sizeof(int));
        
        size_t w_size = static_cast<size_t>(out_c) * in_c * k * k;
        int b_size = out_c;

        std::vector<float> h_weights(w_size);
        std::vector<float> h_bias(b_size);
        layer.copy_weights_to_host(h_weights.data(), h_bias.data());

        int w_size_int = static_cast<int>(w_size);
        out.write(reinterpret_cast<const char*>(&w_size_int), sizeof(int));
        out.write(reinterpret_cast<const char*>(h_weights.data()), 
                  static_cast<std::streamsize>(w_size) * sizeof(float));
        out.write(reinterpret_cast<const char*>(&b_size), sizeof(int));
        out.write(reinterpret_cast<const char*>(h_bias.data()), 
                  static_cast<std::streamsize>(b_size) * sizeof(float));
    };

    save_conv(conv1_);
    save_conv(conv2_);
    save_conv(conv3_);
    save_conv(conv4_);
    save_conv(conv5_);

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
    
    if (magic != EXPECTED_MAGIC) {
        std::cerr << "Invalid weight file format (bad magic number)" << std::endl;
        return false;
    }
    if (version != 1) {
        std::cerr << "Unsupported weight file version: " << version << std::endl;
        return false;
    }
    if (num_layers != 5) {
        std::cerr << "Layer count mismatch: expected 5, got " << num_layers << std::endl;
        return false;
    }

    auto load_conv = [&in](GPUConv2DLayer& layer) -> bool {
        int in_c = 0, out_c = 0, k = 0;
        in.read(reinterpret_cast<char*>(&in_c), sizeof(int));
        in.read(reinterpret_cast<char*>(&out_c), sizeof(int));
        in.read(reinterpret_cast<char*>(&k), sizeof(int));
        
        if (in_c != layer.get_in_channels() || out_c != layer.get_out_channels() || 
            k != layer.get_kernel_size()) {
            std::cerr << "Layer shape mismatch: expected (" << layer.get_in_channels() 
                      << "," << layer.get_out_channels() << "," << layer.get_kernel_size()
                      << "), got (" << in_c << "," << out_c << "," << k << ")" << std::endl;
            return false;
        }
        
        size_t expected_w = static_cast<size_t>(out_c) * in_c * k * k;
        int expected_b = out_c;

        int w_size = 0, b_size = 0;
        in.read(reinterpret_cast<char*>(&w_size), sizeof(int));
        if (static_cast<size_t>(w_size) != expected_w) return false;

        std::vector<float> h_weights(w_size);
        in.read(reinterpret_cast<char*>(h_weights.data()), 
                static_cast<std::streamsize>(w_size) * sizeof(float));

        in.read(reinterpret_cast<char*>(&b_size), sizeof(int));
        if (b_size != expected_b) return false;

        std::vector<float> h_bias(b_size);
        in.read(reinterpret_cast<char*>(h_bias.data()), 
                static_cast<std::streamsize>(b_size) * sizeof(float));

        layer.copy_weights_from_host(h_weights.data(), h_bias.data());
        return true;
    };

    try {
        if (!load_conv(conv1_)) return false;
        if (!load_conv(conv2_)) return false;
        if (!load_conv(conv3_)) return false;
        if (!load_conv(conv4_)) return false;
        if (!load_conv(conv5_)) return false;
    } catch (...) {
        return false;
    }

    return static_cast<bool>(in);
}

void GPUAutoencoder::load_weights_from_cpu(const Autoencoder& /* cpu_ae */) {
    std::cerr << "Warning: load_weights_from_cpu not implemented. "
              << "Use save/load_weights instead." << std::endl;
}

void tensor_cpu_to_gpu(const Tensor4D& cpu_tensor, GPUTensor4D& gpu_tensor) {
    if (gpu_tensor.n != cpu_tensor.n || gpu_tensor.c != cpu_tensor.c ||
        gpu_tensor.h != cpu_tensor.h || gpu_tensor.w != cpu_tensor.w) {
        gpu_tensor.allocate(cpu_tensor.n, cpu_tensor.c, cpu_tensor.h, cpu_tensor.w);
    }
    gpu_tensor.copy_from_host(cpu_tensor.data.data());
}

void tensor_gpu_to_cpu(const GPUTensor4D& gpu_tensor, Tensor4D& cpu_tensor) {
    cpu_tensor = Tensor4D(gpu_tensor.n, gpu_tensor.c, gpu_tensor.h, gpu_tensor.w);
    gpu_tensor.copy_to_host(cpu_tensor.data.data());
}

void batch_cpu_to_gpu(const float* cpu_data, int n, int c, int h, int w, GPUTensor4D& gpu_tensor) {
    if (gpu_tensor.n != n || gpu_tensor.c != c ||
        gpu_tensor.h != h || gpu_tensor.w != w) {
        gpu_tensor.allocate(n, c, h, w);
    }
    gpu_tensor.copy_from_host(cpu_data);
}
