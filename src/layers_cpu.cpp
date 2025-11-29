#include "../include/layer.h"

// --- Helper Functions ---
// Truy cập mảng 1 chiều như mảng 3 chiều: index = c * H * W + h * W + w
inline int get_idx(int c, int h, int w, int height, int width) {
    return c * height * width + h * width + w;
}

// --- CONV2D IMPLEMENTATION ---
Conv2D::Conv2D(int in_c, int out_c, int kernel_size, int s, int p, Shape in_shape) {
    this->in_channels = in_c;
    this->out_channels = out_c;
    this->k_size = kernel_size;
    this->stride = s;
    this->padding = p;
    this->input_shape = in_shape;

    // Tính kích thước Output: (H - K + 2P)/S + 1
    this->output_shape.channels = out_c;
    this->output_shape.height = (in_shape.height - k_size + 2 * padding) / stride + 1;
    this->output_shape.width = (in_shape.width - k_size + 2 * padding) / stride + 1;
    this->name = "Conv2D";

    // Cấp phát bộ nhớ
    int weights_size = out_c * in_c * k_size * k_size;
    weights.resize(weights_size);
    biases.resize(out_c, 0.0f);
    
    init_weights();
}

void Conv2D::init_weights() {
    // Xavier Initialization
    std::default_random_engine generator;
    float limit = sqrt(6.0f / (in_channels * k_size * k_size + out_channels * k_size * k_size));
    std::uniform_real_distribution<float> distribution(-limit, limit);

    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = distribution(generator);
    }
}

void Conv2D::forward(const std::vector<float>& input, std::vector<float>& output) {
    this->input_cache = input; // Lưu input cho backward
    output.assign(output.size(), 0.0f); // Reset output

    int H_in = input_shape.height;
    int W_in = input_shape.width;
    int H_out = output_shape.height;
    int W_out = output_shape.width;

    // Convolution Logic (Naive implementation)
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int oh = 0; oh < H_out; ++oh) {
            for (int ow = 0; ow < W_out; ++ow) {
                float sum = 0.0f;
                
                // Duyệt qua input channels
                for (int ic = 0; ic < in_channels; ++ic) {
                    // Duyệt qua kernel
                    for (int kh = 0; kh < k_size; ++kh) {
                        for (int kw = 0; kw < k_size; ++kw) {
                            // Tính index trên input (có padding)
                            int ih = oh * stride - padding + kh;
                            int iw = ow * stride - padding + kw;

                            if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                                int input_idx = get_idx(ic, ih, iw, H_in, W_in);
                                // Weight index: oc -> ic -> kh -> kw
                                int weight_idx = oc * (in_channels * k_size * k_size) 
                                               + ic * (k_size * k_size) 
                                               + kh * k_size + kw;
                                sum += input[input_idx] * weights[weight_idx];
                            }
                        }
                    }
                }
                // Cộng bias
                output[get_idx(oc, oh, ow, H_out, W_out)] = sum + biases[oc];
            }
        }
    }
}

void Conv2D::backward(const std::vector<float>& grad_output, std::vector<float>& grad_input, float learning_rate) {
    // Reset grad_input
    grad_input.assign(grad_input.size(), 0.0f);

    int H_in = input_shape.height;
    int W_in = input_shape.width;
    int H_out = output_shape.height;
    int W_out = output_shape.width;

    // Gradient w.r.t Weights & Input
    for (int oc = 0; oc < out_channels; ++oc) {
        for (int oh = 0; oh < H_out; ++oh) {
            for (int ow = 0; ow < W_out; ++ow) {
                // Lấy gradient từ lớp sau
                float grad_val = grad_output[get_idx(oc, oh, ow, H_out, W_out)];
                
                // Gradient w.r.t Bias
                biases[oc] -= learning_rate * grad_val; 

                for (int ic = 0; ic < in_channels; ++ic) {
                    for (int kh = 0; kh < k_size; ++kh) {
                        for (int kw = 0; kw < k_size; ++kw) {
                            int ih = oh * stride - padding + kh;
                            int iw = ow * stride - padding + kw;

                            if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                                int input_idx = get_idx(ic, ih, iw, H_in, W_in);
                                int weight_idx = oc * (in_channels * k_size * k_size) 
                                               + ic * (k_size * k_size) 
                                               + kh * k_size + kw;

                                // 1. Tích lũy Gradient cho Input (để truyền về lớp trước)
                                // dL/dx += dL/dy * w
                                grad_input[input_idx] += grad_val * weights[weight_idx];

                                // 2. Cập nhật Weights (SGD)
                                // dL/dw = dL/dy * x
                                // w_new = w_old - lr * grad_weight
                                float grad_weight = grad_val * input_cache[input_idx];
                                weights[weight_idx] -= learning_rate * grad_weight;
                            }
                        }
                    }
                }
            }
        }
    }
}

// --- RELU IMPLEMENTATION ---
ReLU::ReLU(Shape in_shape) {
    this->input_shape = in_shape;
    this->output_shape = in_shape; // Shape không đổi
    this->name = "ReLU";
}

void ReLU::forward(const std::vector<float>& input, std::vector<float>& output) {
    this->input_cache = input;
    for (size_t i = 0; i < input.size(); ++i) {
        // max(0, x)
        output[i] = (input[i] > 0.0f) ? input[i] : 0.0f;
    }
}

void ReLU::backward(const std::vector<float>& grad_output, std::vector<float>& grad_input, float learning_rate) {
    for (size_t i = 0; i < grad_output.size(); ++i) {
        // Đạo hàm ReLU: 1 nếu x > 0, 0 nếu x <= 0
        grad_input[i] = (input_cache[i] > 0.0f) ? grad_output[i] : 0.0f;
    }
}

// --- MAX POOLING IMPLEMENTATION ---
MaxPool2D::MaxPool2D(int size, int s, Shape in_shape) {
    this->pool_size = size;
    this->stride = s;
    this->input_shape = in_shape;
    
    this->output_shape.channels = in_shape.channels;
    this->output_shape.height = (in_shape.height - pool_size) / stride + 1;
    this->output_shape.width = (in_shape.width - pool_size) / stride + 1;
    this->name = "MaxPool2D";
}

void MaxPool2D::forward(const std::vector<float>& input, std::vector<float>& output) {
    int H_out = output_shape.height;
    int W_out = output_shape.width;
    int H_in = input_shape.height;
    int W_in = input_shape.width;
    
    // Resize mask để lưu vị trí max
    mask.assign(output.size(), -1);

    for (int c = 0; c < input_shape.channels; ++c) {
        for (int oh = 0; oh < H_out; ++oh) {
            for (int ow = 0; ow < W_out; ++ow) {
                
                int start_h = oh * stride;
                int start_w = ow * stride;
                
                float max_val = -1e9;
                int max_idx = -1;

                // Quét trong vùng pool window
                for (int kh = 0; kh < pool_size; ++kh) {
                    for (int kw = 0; kw < pool_size; ++kw) {
                        int ih = start_h + kh;
                        int iw = start_w + kw;
                        
                        if (ih < H_in && iw < W_in) {
                            int idx = get_idx(c, ih, iw, H_in, W_in);
                            if (input[idx] > max_val) {
                                max_val = input[idx];
                                max_idx = idx;
                            }
                        }
                    }
                }
                
                int out_idx = get_idx(c, oh, ow, H_out, W_out);
                output[out_idx] = max_val;
                mask[out_idx] = max_idx; // Lưu lại index để backprop
            }
        }
    }
}

void MaxPool2D::backward(const std::vector<float>& grad_output, std::vector<float>& grad_input, float learning_rate) {
    // Reset grad input
    grad_input.assign(grad_input.size(), 0.0f);

    // Chỉ truyền gradient về đúng vị trí max (dựa vào mask)
    for (size_t i = 0; i < grad_output.size(); ++i) {
        if (mask[i] != -1) {
            grad_input[mask[i]] += grad_output[i];
        }
    }
}

// --- UPSAMPLING IMPLEMENTATION (NEAREST NEIGHBOR) ---
UpSample2D::UpSample2D(int scale_factor, Shape in_shape) {
    this->scale = scale_factor;
    this->input_shape = in_shape;
    
    this->output_shape.channels = in_shape.channels;
    this->output_shape.height = in_shape.height * scale;
    this->output_shape.width = in_shape.width * scale;
    this->name = "UpSample2D";
}

void UpSample2D::forward(const std::vector<float>& input, std::vector<float>& output) {
    int H_in = input_shape.height;
    int W_in = input_shape.width;
    int H_out = output_shape.height;
    int W_out = output_shape.width;

    for (int c = 0; c < output_shape.channels; ++c) {
        for (int oh = 0; oh < H_out; ++oh) {
            for (int ow = 0; ow < W_out; ++ow) {
                // Nearest Neighbor: chia tọa độ output cho scale để lấy tọa độ input
                int ih = oh / scale;
                int iw = ow / scale;
                
                output[get_idx(c, oh, ow, H_out, W_out)] = input[get_idx(c, ih, iw, H_in, W_in)];
            }
        }
    }
}

void UpSample2D::backward(const std::vector<float>& grad_output, std::vector<float>& grad_input, float learning_rate) {
    grad_input.assign(grad_input.size(), 0.0f);

    int H_out = output_shape.height; // Kích thước của lớp UpSample Output (lớn)
    int W_out = output_shape.width;
    int H_in = input_shape.height;   // Kích thước của lớp UpSample Input (nhỏ)
    int W_in = input_shape.width;

    // Backward của Upsample là cộng dồn gradient từ các pixel mở rộng về pixel gốc
    for (int c = 0; c < output_shape.channels; ++c) {
        for (int oh = 0; oh < H_out; ++oh) {
            for (int ow = 0; ow < W_out; ++ow) {
                int ih = oh / scale;
                int iw = ow / scale;
                
                int in_idx = get_idx(c, ih, iw, H_in, W_in);
                int out_idx = get_idx(c, oh, ow, H_out, W_out);
                
                grad_input[in_idx] += grad_output[out_idx];
            }
        }
    }
}