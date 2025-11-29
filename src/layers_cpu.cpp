#include "layer.h"

#include <cmath>
#include <istream>
#include <ostream>
#include <random>
#include <stdexcept>

Conv2DLayer::Conv2DLayer(int in_channels, int out_channels, int kernel_size,
                         int stride, int padding)
    : in_c_(in_channels),
      out_c_(out_channels),
      k_(kernel_size),
      stride_(stride),
      padding_(padding),
      weights_(static_cast<std::size_t>(out_channels) * in_channels * kernel_size * kernel_size),
      bias_(out_channels, 0.0f) {
    // Simple Gaussian initialization for weights
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.01f);
    for (auto &w : weights_) {
        w = dist(rng);
    }
}

Tensor4D Conv2DLayer::forward(const Tensor4D &input) const {
    int out_h = (input.h + 2 * padding_ - k_) / stride_ + 1;
    int out_w = (input.w + 2 * padding_ - k_) / stride_ + 1;

    Tensor4D output(input.n, out_c_, out_h, out_w);

#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int n = 0; n < input.n; ++n) {
        for (int oc = 0; oc < out_c_; ++oc) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float sum = bias_[oc];
                    for (int ic = 0; ic < in_c_; ++ic) {
                        for (int kh = 0; kh < k_; ++kh) {
                            for (int kw = 0; kw < k_; ++kw) {
                                int ih = oh * stride_ + kh - padding_;
                                int iw = ow * stride_ + kw - padding_;
                                if (ih < 0 || ih >= input.h || iw < 0 || iw >= input.w) {
                                    continue;
                                }
                                float val = input.at(n, ic, ih, iw);
                                std::size_t w_idx =
                                    (((static_cast<std::size_t>(oc) * in_c_ + ic) * k_ + kh) * k_ +
                                     kw);
                                sum += val * weights_[w_idx];
                            }
                        }
                    }
                    output.at(n, oc, oh, ow) = sum;
                }
            }
        }
    }

    return output;
}

void Conv2DLayer::save(std::ostream &os) const {
    int w_size = static_cast<int>(weights_.size());
    int b_size = static_cast<int>(bias_.size());
    os.write(reinterpret_cast<const char *>(&w_size), sizeof(int));
    os.write(reinterpret_cast<const char *>(weights_.data()),
             static_cast<std::streamsize>(w_size) * sizeof(float));
    os.write(reinterpret_cast<const char *>(&b_size), sizeof(int));
    os.write(reinterpret_cast<const char *>(bias_.data()),
             static_cast<std::streamsize>(b_size) * sizeof(float));
}

void Conv2DLayer::load(std::istream &is) {
    int w_size = 0;
    int b_size = 0;
    is.read(reinterpret_cast<char *>(&w_size), sizeof(int));
    if (w_size != static_cast<int>(weights_.size())) {
        throw std::runtime_error("Conv2DLayer::load: weight size mismatch");
    }
    is.read(reinterpret_cast<char *>(weights_.data()),
            static_cast<std::streamsize>(w_size) * sizeof(float));
    is.read(reinterpret_cast<char *>(&b_size), sizeof(int));
    if (b_size != static_cast<int>(bias_.size())) {
        throw std::runtime_error("Conv2DLayer::load: bias size mismatch");
    }
    is.read(reinterpret_cast<char *>(bias_.data()),
            static_cast<std::streamsize>(b_size) * sizeof(float));
}

Tensor4D MaxPool2DLayer::backward(const Tensor4D &input, const Tensor4D &grad_output) const {
    int out_h = (input.h - k_) / stride_ + 1;
    int out_w = (input.w - k_) / stride_ + 1;

    Tensor4D grad_input(input.n, input.c, input.h, input.w);

    for (int n = 0; n < input.n; ++n) {
        for (int c = 0; c < input.c; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    int max_h = -1;
                    int max_w = -1;
                    for (int kh = 0; kh < k_; ++kh) {
                        for (int kw = 0; kw < k_; ++kw) {
                            int ih = oh * stride_ + kh;
                            int iw = ow * stride_ + kw;
                            if (ih < 0 || ih >= input.h || iw < 0 || iw >= input.w) {
                                continue;
                            }
                            float val = input.at(n, c, ih, iw);
                            if (val > max_val) {
                                max_val = val;
                                max_h = ih;
                                max_w = iw;
                            }
                        }
                    }
                    float go = grad_output.at(n, c, oh, ow);
                    if (max_h >= 0 && max_w >= 0) {
                        grad_input.at(n, c, max_h, max_w) += go;
                    }
                }
            }
        }
    }

    return grad_input;
}

Tensor4D Conv2DLayer::backward(const Tensor4D &input, const Tensor4D &grad_output,
                               float learning_rate) {
    int out_h = grad_output.h;
    int out_w = grad_output.w;

    Tensor4D grad_input(input.n, in_c_, input.h, input.w);
    std::vector<float> grad_weights(weights_.size(), 0.0f);
    std::vector<float> grad_bias(out_c_, 0.0f);

    for (int n = 0; n < input.n; ++n) {
        for (int oc = 0; oc < out_c_; ++oc) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float go = grad_output.at(n, oc, oh, ow);
                    grad_bias[oc] += go;
                    for (int ic = 0; ic < in_c_; ++ic) {
                        for (int kh = 0; kh < k_; ++kh) {
                            for (int kw = 0; kw < k_; ++kw) {
                                int ih = oh * stride_ + kh - padding_;
                                int iw = ow * stride_ + kw - padding_;
                                if (ih < 0 || ih >= input.h || iw < 0 || iw >= input.w) {
                                    continue;
                                }
                                float val = input.at(n, ic, ih, iw);
                                std::size_t w_idx =
                                    (((static_cast<std::size_t>(oc) * in_c_ + ic) * k_ + kh) * k_ +
                                     kw);
                                grad_weights[w_idx] += go * val;
                                grad_input.at(n, ic, ih, iw) += go * weights_[w_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    for (std::size_t i = 0; i < weights_.size(); ++i) {
        weights_[i] -= learning_rate * grad_weights[i];
    }
    for (int oc = 0; oc < out_c_; ++oc) {
        bias_[oc] -= learning_rate * grad_bias[oc];
    }

    return grad_input;
}

Tensor4D ReLULayer::forward(const Tensor4D &input) const {
    Tensor4D output(input.n, input.c, input.h, input.w);
    std::size_t total = input.data.size();
    for (std::size_t i = 0; i < total; ++i) {
        output.data[i] = std::max(0.0f, input.data[i]);
    }
    return output;
}

Tensor4D ReLULayer::backward(const Tensor4D &input, const Tensor4D &grad_output) const {
    Tensor4D grad_input(input.n, input.c, input.h, input.w);
    std::size_t total = input.data.size();
    for (std::size_t i = 0; i < total; ++i) {
        grad_input.data[i] = input.data[i] > 0.0f ? grad_output.data[i] : 0.0f;
    }
    return grad_input;
}

MaxPool2DLayer::MaxPool2DLayer(int kernel_size, int stride)
    : k_(kernel_size), stride_(stride) {}

Tensor4D MaxPool2DLayer::forward(const Tensor4D &input) const {
    int out_h = (input.h - k_) / stride_ + 1;
    int out_w = (input.w - k_) / stride_ + 1;

    Tensor4D output(input.n, input.c, out_h, out_w);

    for (int n = 0; n < input.n; ++n) {
        for (int c = 0; c < input.c; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (int kh = 0; kh < k_; ++kh) {
                        for (int kw = 0; kw < k_; ++kw) {
                            int ih = oh * stride_ + kh;
                            int iw = ow * stride_ + kw;
                            if (ih < 0 || ih >= input.h || iw < 0 || iw >= input.w) {
                                continue;
                            }
                            float val = input.at(n, c, ih, iw);
                            if (val > max_val) {
                                max_val = val;
                            }
                        }
                    }
                    output.at(n, c, oh, ow) = max_val;
                }
            }
        }
    }

    return output;
}

UpSample2DLayer::UpSample2DLayer(int scale) : scale_(scale) {}

Tensor4D UpSample2DLayer::forward(const Tensor4D &input) const {
    int out_h = input.h * scale_;
    int out_w = input.w * scale_;

    Tensor4D output(input.n, input.c, out_h, out_w);

    for (int n = 0; n < input.n; ++n) {
        for (int c = 0; c < input.c; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    int ih = oh / scale_;
                    int iw = ow / scale_;
                    output.at(n, c, oh, ow) = input.at(n, c, ih, iw);
                }
            }
        }
    }

    return output;
}

Tensor4D UpSample2DLayer::backward(const Tensor4D &input, const Tensor4D &grad_output) const {
    Tensor4D grad_input(input.n, input.c, input.h, input.w);

    int out_h = grad_output.h;
    int out_w = grad_output.w;

    for (int n = 0; n < input.n; ++n) {
        for (int c = 0; c < input.c; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    int ih = oh / scale_;
                    int iw = ow / scale_;
                    grad_input.at(n, c, ih, iw) += grad_output.at(n, c, oh, ow);
                }
            }
        }
    }

    return grad_input;
}

float mse_loss(const Tensor4D &output, const Tensor4D &target) {
    if (output.n != target.n || output.c != target.c || output.h != target.h ||
        output.w != target.w) {
        throw std::runtime_error("mse_loss: tensor shapes do not match");
    }

    std::size_t total = output.data.size();
    float sum = 0.0f;
    for (std::size_t i = 0; i < total; ++i) {
        float diff = output.data[i] - target.data[i];
        sum += diff * diff;
    }

    return sum / static_cast<float>(total);
}

float mse_loss_with_grad(const Tensor4D &output, const Tensor4D &target,
                         Tensor4D &grad_output) {
    if (output.n != target.n || output.c != target.c || output.h != target.h ||
        output.w != target.w) {
        throw std::runtime_error("mse_loss_with_grad: tensor shapes do not match");
    }

    grad_output = Tensor4D(output.n, output.c, output.h, output.w);

    std::size_t total = output.data.size();
    float sum = 0.0f;
    float scale = 2.0f / static_cast<float>(total);
    for (std::size_t i = 0; i < total; ++i) {
        float diff = output.data[i] - target.data[i];
        sum += diff * diff;
        grad_output.data[i] = scale * diff;
    }

    return sum / static_cast<float>(total);
}
