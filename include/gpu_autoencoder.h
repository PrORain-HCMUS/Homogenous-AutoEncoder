#ifndef GPU_AUTOENCODER_H
#define GPU_AUTOENCODER_H

#include "gpu_layer.h"
#include "layer.h"
#include <string>
#include <vector>

enum class LossType { MSE, BCE };

class GPUAutoencoder {
public:
    GPUAutoencoder(LossType loss_type = LossType::MSE);
    ~GPUAutoencoder();

    void load_weights_from_cpu(const class Autoencoder& cpu_ae);
    void forward(const GPUTensor4D& input, GPUTensor4D& output);
    float train_step(const GPUTensor4D& input, const GPUTensor4D& target, float learning_rate);
    float train_step_momentum(const GPUTensor4D& input, const GPUTensor4D& target, 
                              float learning_rate, const OptimizerConfig& opt_config);
    void encode(const GPUTensor4D& input, GPUTensor4D& latent);
    bool save_weights(const std::string& path) const;
    bool load_weights(const std::string& path);
    void synchronize();
    LossType get_loss_type() const { return loss_type_; }
    void set_loss_type(LossType lt) { loss_type_ = lt; }

private:
    LossType loss_type_;
    
    // Layers
    GPUConv2DLayer conv1_, conv2_, conv3_, conv4_, conv5_;
    GPUBatchNorm2D bn1_, bn2_, bn3_, bn4_;
    GPUPReLULayer prelu1_, prelu2_, prelu3_, prelu4_;
    GPUMaxPool2DLayer pool1_, pool2_;
    GPUUpSample2DLayer up1_, up2_;
    GPUSigmoidLayer sigmoid_;
    
    // Forward pass intermediate tensors
    GPUTensor4D x1_, x2_, x3_, x4_, x5_, x6_, x7_, x8_, x9_;
    GPUTensor4D x10_, x11_, x12_, x13_, x14_, x15_, x16_, x17_, x18_;
    
    // Backward pass gradient tensors
    GPUTensor4D g0_, g1_, g2_, g3_, g4_, g5_, g6_, g7_, g8_, g9_;
    GPUTensor4D g10_, g11_, g12_, g13_, g14_, g15_, g16_, g17_, g18_;
};

void tensor_cpu_to_gpu(const Tensor4D& cpu_tensor, GPUTensor4D& gpu_tensor);
void tensor_gpu_to_cpu(const GPUTensor4D& gpu_tensor, Tensor4D& cpu_tensor);
void batch_cpu_to_gpu(const float* cpu_data, int n, int c, int h, int w, GPUTensor4D& gpu_tensor);

#endif
