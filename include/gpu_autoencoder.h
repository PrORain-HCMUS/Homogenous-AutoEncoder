#ifndef GPU_AUTOENCODER_H
#define GPU_AUTOENCODER_H

#include "gpu_layer.h"
#include "layer.h"

#include <string>
#include <vector>

class GPUAutoencoder {
public:
    GPUAutoencoder();
    ~GPUAutoencoder();

    void load_weights_from_cpu(const class Autoencoder& cpu_ae);

    void forward(const GPUTensor4D& input, GPUTensor4D& output);

    float train_step(const GPUTensor4D& input, const GPUTensor4D& target, float learning_rate);

    void encode(const GPUTensor4D& input, GPUTensor4D& latent);

    bool save_weights(const std::string& path) const;
    bool load_weights(const std::string& path);

    void synchronize();

private:
    GPUConv2DLayer conv1_;
    GPUReLULayer relu1_;
    GPUMaxPool2DLayer pool1_;

    GPUConv2DLayer conv2_;
    GPUReLULayer relu2_;
    GPUMaxPool2DLayer pool2_;

    GPUConv2DLayer conv3_;
    GPUReLULayer relu3_;
    GPUUpSample2DLayer up1_;

    GPUConv2DLayer conv4_;
    GPUReLULayer relu4_;
    GPUUpSample2DLayer up2_;

    GPUConv2DLayer conv5_;

    GPUTensor4D x0_, x1_, x2_, x3_, x4_, x5_, x6_;
    GPUTensor4D x7_, x8_, x9_, x10_, x11_, x12_, x13_;
    
    GPUTensor4D g0_, g1_, g2_, g3_, g4_, g5_, g6_;
    GPUTensor4D g7_, g8_, g9_, g10_, g11_, g12_, g13_;

    void copy_input(const GPUTensor4D& input);
};

void tensor_cpu_to_gpu(const Tensor4D& cpu_tensor, GPUTensor4D& gpu_tensor);
void tensor_gpu_to_cpu(const GPUTensor4D& gpu_tensor, Tensor4D& cpu_tensor);

void batch_cpu_to_gpu(const float* cpu_data, int n, int c, int h, int w, GPUTensor4D& gpu_tensor);

#endif  // GPU_AUTOENCODER_H
