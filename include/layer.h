#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <algorithm>

// Cấu trúc Tensor 3 chiều đơn giản để quản lý kích thước
struct Shape {
    int channels;
    int height;
    int width;
    int size() const { return channels * height * width; }
};

// Lớp cha (Base Class)
class Layer {
public:
    Shape input_shape;
    Shape output_shape;
    std::string name;

    virtual ~Layer() {}

    // Forward pass: Input -> Output
    virtual void forward(const std::vector<float>& input, std::vector<float>& output) = 0;

    // Backward pass: Tính Gradient và cập nhật trọng số (nếu có)
    // grad_output: Gradient từ lớp phía sau truyền ngược lại
    // grad_input: Gradient tính toán để truyền về lớp phía trước
    virtual void backward(const std::vector<float>& grad_output, std::vector<float>& grad_input, float learning_rate) = 0;
};

// --- CONVOLUTION LAYER ---
class Conv2D : public Layer {
public:
    int k_size;      // Kernel size (e.g., 3 for 3x3)
    int stride;
    int padding;
    int in_channels;
    int out_channels;

    std::vector<float> weights;      // Filter weights: [out_c][in_c][k][k]
    std::vector<float> biases;       // Biases: [out_c]
    std::vector<float> input_cache;  // Lưu input để dùng cho backward

    Conv2D(int in_c, int out_c, int kernel_size, int s, int p, Shape in_shape);

    void forward(const std::vector<float>& input, std::vector<float>& output) override;
    void backward(const std::vector<float>& grad_output, std::vector<float>& grad_input, float learning_rate) override;
    
    // Khởi tạo trọng số (Xavier/He initialization)
    void init_weights();
};

// --- RELU LAYER ---
class ReLU : public Layer {
public:
    std::vector<float> input_cache; // Lưu input để tính đạo hàm (mask)

    ReLU(Shape in_shape);
    void forward(const std::vector<float>& input, std::vector<float>& output) override;
    void backward(const std::vector<float>& grad_output, std::vector<float>& grad_input, float learning_rate) override;
};

// --- MAX POOLING LAYER ---
class MaxPool2D : public Layer {
public:
    int pool_size;
    int stride;
    std::vector<int> mask; // Lưu chỉ số (index) của phần tử max để backprop

    MaxPool2D(int size, int s, Shape in_shape);
    void forward(const std::vector<float>& input, std::vector<float>& output) override;
    void backward(const std::vector<float>& grad_output, std::vector<float>& grad_input, float learning_rate) override;
};

// --- UPSAMPLING LAYER (NEAREST NEIGHBOR) ---
class UpSample2D : public Layer {
public:
    int scale;

    UpSample2D(int scale_factor, Shape in_shape);
    void forward(const std::vector<float>& input, std::vector<float>& output) override;
    void backward(const std::vector<float>& grad_output, std::vector<float>& grad_input, float learning_rate) override;
};

#endif