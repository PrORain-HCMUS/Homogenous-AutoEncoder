#ifndef LAYER_H
#define LAYER_H

#include <algorithm>
#include <cstddef>
#include <iosfwd>
#include <vector>

struct Tensor4D {
    int n = 0;
    int c = 0;
    int h = 0;
    int w = 0;
    std::vector<float> data;

    Tensor4D() = default;

    Tensor4D(int n_, int c_, int h_, int w_)
        : n(n_), c(c_), h(h_), w(w_),
          data(static_cast<std::size_t>(n_) * c_ * h_ * w_, 0.0f) {}

    inline float &at(int ni, int ci, int hi, int wi) {
        std::size_t idx = (((static_cast<std::size_t>(ni) * c + ci) * h + hi) * w + wi);
        return data[idx];
    }

    inline const float &at(int ni, int ci, int hi, int wi) const {
        std::size_t idx = (((static_cast<std::size_t>(ni) * c + ci) * h + hi) * w + wi);
        return data[idx];
    }
};

class Conv2DLayer {
public:
    Conv2DLayer(int in_channels, int out_channels, int kernel_size,
                int stride = 1, int padding = 1);

    Tensor4D forward(const Tensor4D &input) const;

    Tensor4D backward(const Tensor4D &input, const Tensor4D &grad_output,
                      float learning_rate);

    void save(std::ostream &os) const;

    void load(std::istream &is);

private:
    int in_c_;
    int out_c_;
    int k_;
    int stride_;
    int padding_;
    std::vector<float> weights_;
    std::vector<float> bias_;
};

class ReLULayer {
public:
    Tensor4D forward(const Tensor4D &input) const;

    Tensor4D backward(const Tensor4D &input, const Tensor4D &grad_output) const;
};

class MaxPool2DLayer {
public:
    explicit MaxPool2DLayer(int kernel_size = 2, int stride = 2);

    Tensor4D forward(const Tensor4D &input) const;

    Tensor4D backward(const Tensor4D &input, const Tensor4D &grad_output) const;

private:
    int k_;
    int stride_;
};

class UpSample2DLayer {
public:
    explicit UpSample2DLayer(int scale = 2);

    Tensor4D forward(const Tensor4D &input) const;

    Tensor4D backward(const Tensor4D &input, const Tensor4D &grad_output) const;

private:
    int scale_;
};

float mse_loss(const Tensor4D &output, const Tensor4D &target);

float mse_loss_with_grad(const Tensor4D &output, const Tensor4D &target,
                         Tensor4D &grad_output);

#endif  // LAYER_H
