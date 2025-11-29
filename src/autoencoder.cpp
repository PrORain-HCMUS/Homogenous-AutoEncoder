#include "../include/autoencoder.h"
#include <fstream>

Autoencoder::Autoencoder() {
    // --- ENCODER [cite: 174-191] ---
    // Input: 32x32x3
    // Layer 1: Conv2D(3 -> 256) + ReLU
    layers.push_back(new Conv2D(3, 256, 3, 1, 1, {3, 32, 32}));
    layers.push_back(new ReLU({256, 32, 32}));
    // Layer 2: MaxPool
    layers.push_back(new MaxPool2D(2, 2, {256, 32, 32})); // Out: 16x16x256
    
    // Layer 3: Conv2D(256 -> 128) + ReLU
    layers.push_back(new Conv2D(256, 128, 3, 1, 1, {256, 16, 16}));
    layers.push_back(new ReLU({128, 16, 16}));
    // Layer 4: MaxPool (Latent)
    layers.push_back(new MaxPool2D(2, 2, {128, 16, 16})); // Out: 8x8x128 (Bottleneck)

    // --- DECODER [cite: 192-207] ---
    // Layer 5: Conv2D(128 -> 128) + ReLU
    layers.push_back(new Conv2D(128, 128, 3, 1, 1, {128, 8, 8}));
    layers.push_back(new ReLU({128, 8, 8}));
    
    // Layer 6: UpSample
    layers.push_back(new UpSample2D(2, {128, 8, 8})); // Out: 16x16x128
    
    // Layer 7: Conv2D(128 -> 256) + ReLU
    layers.push_back(new Conv2D(128, 256, 3, 1, 1, {128, 16, 16}));
    layers.push_back(new ReLU({256, 16, 16}));
    
    // Layer 8: UpSample
    layers.push_back(new UpSample2D(2, {256, 16, 16})); // Out: 32x32x256
    
    // Layer 9: Output Conv2D(256 -> 3) (No activation or Sigmoid/Tanh depending on norm)
    // Tài liệu không yêu cầu activation cuối cùng [cite: 204]
    layers.push_back(new Conv2D(256, 3, 3, 1, 1, {256, 32, 32})); 
}

Autoencoder::~Autoencoder() {
    for (Layer* layer : layers) {
        delete layer;
    }
}

void Autoencoder::forward(const std::vector<float>& input, std::vector<float>& output) {
    std::vector<float> current_in = input;
    std::vector<float> current_out;

    for (Layer* layer : layers) {
        current_out.resize(layer->output_shape.size());
        layer->forward(current_in, current_out);
        current_in = current_out; // Output lớp này là Input lớp sau
    }
    output = current_out;
}

void Autoencoder::backward(const std::vector<float>& loss_grad, float learning_rate) {
    std::vector<float> current_grad_out = loss_grad;
    std::vector<float> current_grad_in;

    // Duyệt ngược từ layer cuối về đầu
    for (int i = layers.size() - 1; i >= 0; --i) {
        current_grad_in.resize(layers[i]->input_shape.size());
        layers[i]->backward(current_grad_out, current_grad_in, learning_rate);
        current_grad_out = current_grad_in; // Gradient input lớp này là Gradient output lớp trước
    }
}

void Autoencoder::save_weights(const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open file to save: " << filepath << std::endl;
        return;
    }
    
    // Duyệt qua từng layer, nếu là Conv2D thì lưu weights + bias
    for (Layer* layer : layers) {
        // Kiểm tra xem layer có phải Conv2D không (bằng cách ép kiểu)
        Conv2D* conv = dynamic_cast<Conv2D*>(layer);
        if (conv) {
            // Lưu kích thước để kiểm tra
            int w_size = conv->weights.size();
            int b_size = conv->biases.size();
            file.write((char*)&w_size, sizeof(int));
            file.write((char*)conv->weights.data(), w_size * sizeof(float));
            file.write((char*)conv->biases.data(), b_size * sizeof(float));
        }
        // Tương tự nếu bạn muốn lưu parameters của các layer khác
    }
    file.close();
    std::cout << "Saved model to " << filepath << std::endl;
}