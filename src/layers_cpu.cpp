#include "layer.h"

#include <cmath>
#include <istream>
#include <ostream>
#include <random>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

Conv2DLayer::Conv2DLayer(int in_channels, int out_channels, int kernel_size,
                         int stride, int padding)
    : in_c_(in_channels),
      out_c_(out_channels),
      k_(kernel_size),
      stride_(stride),
      padding_(padding),
      weights_(static_cast<std::size_t>(out_channels) * in_channels * kernel_size * kernel_size),
      bias_(out_channels, 0.0f) {
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.01f);
    for (auto &w : weights_) {
        w = dist(rng);
    }
}

Tensor4D Conv2DLayer::forward(const Tensor4D &input) const {
    const int out_h = (input.h + 2 * padding_ - k_) / stride_ + 1;
    const int out_w = (input.w + 2 * padding_ - k_) / stride_ + 1;
    const int batch_size = input.n;
    const int input_h = input.h;
    const int input_w = input.w;

    Tensor4D output(batch_size, out_c_, out_h, out_w);

    const std::size_t input_c_stride = static_cast<std::size_t>(input_h) * input_w;
    const std::size_t input_n_stride = static_cast<std::size_t>(in_c_) * input_c_stride;
    const std::size_t out_c_stride = static_cast<std::size_t>(out_h) * out_w;
    const std::size_t out_n_stride = static_cast<std::size_t>(out_c_) * out_c_stride;
    const std::size_t weight_ic_stride = static_cast<std::size_t>(k_) * k_;
    const std::size_t weight_oc_stride = static_cast<std::size_t>(in_c_) * weight_ic_stride;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(dynamic)
#endif
    for (int n = 0; n < batch_size; ++n) {
        for (int oc = 0; oc < out_c_; ++oc) {
            const float bias_val = bias_[oc];
            const std::size_t out_base = n * out_n_stride + oc * out_c_stride;
            const std::size_t w_oc_base = oc * weight_oc_stride;
            
            for (int oh = 0; oh < out_h; ++oh) {
                const int ih_base = oh * stride_ - padding_;
                const std::size_t out_row = out_base + oh * out_w;
                
                for (int ow = 0; ow < out_w; ++ow) {
                    float sum = bias_val;
                    const int iw_base = ow * stride_ - padding_;
                    
                    for (int ic = 0; ic < in_c_; ++ic) {
                        const std::size_t input_ic_base = n * input_n_stride + ic * input_c_stride;
                        const std::size_t w_ic_base = w_oc_base + ic * weight_ic_stride;
                        
                        for (int kh = 0; kh < k_; ++kh) {
                            const int ih = ih_base + kh;
                            if (ih < 0 || ih >= input_h) continue;
                            
                            const std::size_t input_row = input_ic_base + ih * input_w;
                            const std::size_t w_kh_base = w_ic_base + kh * k_;
                            
                            if (k_ == 3) {
                                const int iw0 = iw_base, iw1 = iw_base + 1, iw2 = iw_base + 2;
                                if (iw0 >= 0 && iw0 < input_w) sum += input.data[input_row + iw0] * weights_[w_kh_base];
                                if (iw1 >= 0 && iw1 < input_w) sum += input.data[input_row + iw1] * weights_[w_kh_base + 1];
                                if (iw2 >= 0 && iw2 < input_w) sum += input.data[input_row + iw2] * weights_[w_kh_base + 2];
                            } else {
                                for (int kw = 0; kw < k_; ++kw) {
                                    const int iw = iw_base + kw;
                                    if (iw < 0 || iw >= input_w) continue;
                                    sum += input.data[input_row + iw] * weights_[w_kh_base + kw];
                                }
                            }
                        }
                    }
                    output.data[out_row + ow] = sum;
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
    const int out_h = (input.h - k_) / stride_ + 1;
    const int out_w = (input.w - k_) / stride_ + 1;
    const int batch_size = input.n;
    const int channels = input.c;
    const int input_h = input.h;
    const int input_w = input.w;

    Tensor4D grad_input(batch_size, channels, input_h, input_w);

    const std::size_t input_c_stride = static_cast<std::size_t>(input_h) * input_w;
    const std::size_t input_n_stride = static_cast<std::size_t>(channels) * input_c_stride;
    const std::size_t out_c_stride = static_cast<std::size_t>(out_h) * out_w;
    const std::size_t out_n_stride = static_cast<std::size_t>(channels) * out_c_stride;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int n = 0; n < batch_size; ++n) {
        for (int c = 0; c < channels; ++c) {
            const std::size_t input_base = n * input_n_stride + c * input_c_stride;
            const std::size_t out_base = n * out_n_stride + c * out_c_stride;
            
            for (int oh = 0; oh < out_h; ++oh) {
                const int ih_base = oh * stride_;
                const std::size_t out_row = out_base + oh * out_w;
                
                for (int ow = 0; ow < out_w; ++ow) {
                    const int iw_base = ow * stride_;
                    float max_val = -std::numeric_limits<float>::infinity();
                    int max_idx = -1;
                    
                    if (k_ == 2) {
                        const std::size_t idx00 = input_base + ih_base * input_w + iw_base;
                        const std::size_t idx01 = idx00 + 1;
                        const std::size_t idx10 = idx00 + input_w;
                        const std::size_t idx11 = idx10 + 1;
                        
                        max_val = input.data[idx00]; max_idx = idx00;
                        if (input.data[idx01] > max_val) { max_val = input.data[idx01]; max_idx = idx01; }
                        if (input.data[idx10] > max_val) { max_val = input.data[idx10]; max_idx = idx10; }
                        if (input.data[idx11] > max_val) { max_val = input.data[idx11]; max_idx = idx11; }
                    } else {
                        for (int kh = 0; kh < k_; ++kh) {
                            const int ih = ih_base + kh;
                            const std::size_t input_row = input_base + ih * input_w;
                            for (int kw = 0; kw < k_; ++kw) {
                                const std::size_t idx = input_row + iw_base + kw;
                                if (input.data[idx] > max_val) {
                                    max_val = input.data[idx];
                                    max_idx = idx;
                                }
                            }
                        }
                    }
                    
                    if (max_idx >= 0) {
#ifdef _OPENMP
#pragma omp atomic
#endif
                        grad_input.data[max_idx] += grad_output.data[out_row + ow];
                    }
                }
            }
        }
    }

    return grad_input;
}

Tensor4D Conv2DLayer::backward(const Tensor4D &input, const Tensor4D &grad_output,
                               float learning_rate) {
    const int out_h = grad_output.h;
    const int out_w = grad_output.w;
    const int batch_size = input.n;
    const int input_h = input.h;
    const int input_w = input.w;

    Tensor4D grad_input(batch_size, in_c_, input_h, input_w);
    
#ifdef _OPENMP
    const int num_threads = omp_get_max_threads();
#else
    const int num_threads = 1;
#endif
    std::vector<std::vector<float>> thread_grad_weights(num_threads, std::vector<float>(weights_.size(), 0.0f));
    std::vector<std::vector<float>> thread_grad_bias(num_threads, std::vector<float>(out_c_, 0.0f));

    const std::size_t input_c_stride = static_cast<std::size_t>(input_h) * input_w;
    const std::size_t input_n_stride = static_cast<std::size_t>(in_c_) * input_c_stride;
    const std::size_t out_c_stride = static_cast<std::size_t>(out_h) * out_w;
    const std::size_t out_n_stride = static_cast<std::size_t>(out_c_) * out_c_stride;
    const std::size_t weight_ic_stride = static_cast<std::size_t>(k_) * k_;
    const std::size_t weight_oc_stride = static_cast<std::size_t>(in_c_) * weight_ic_stride;

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
#else
        int tid = 0;
#endif
        std::vector<float> &local_grad_weights = thread_grad_weights[tid];
        std::vector<float> &local_grad_bias = thread_grad_bias[tid];

#ifdef _OPENMP
#pragma omp for collapse(2) schedule(dynamic)
#endif
        for (int n = 0; n < batch_size; ++n) {
            for (int oc = 0; oc < out_c_; ++oc) {
                const std::size_t go_base = n * out_n_stride + oc * out_c_stride;
                const std::size_t w_oc_base = oc * weight_oc_stride;
                
                for (int oh = 0; oh < out_h; ++oh) {
                    const int ih_base = oh * stride_ - padding_;
                    const std::size_t go_row = go_base + oh * out_w;
                    
                    for (int ow = 0; ow < out_w; ++ow) {
                        const float go = grad_output.data[go_row + ow];
                        local_grad_bias[oc] += go;
                        
                        const int iw_base = ow * stride_ - padding_;
                        
                        for (int ic = 0; ic < in_c_; ++ic) {
                            const std::size_t input_ic_base = n * input_n_stride + ic * input_c_stride;
                            const std::size_t w_ic_base = w_oc_base + ic * weight_ic_stride;
                            
                            for (int kh = 0; kh < k_; ++kh) {
                                const int ih = ih_base + kh;
                                if (ih < 0 || ih >= input_h) continue;
                                
                                const std::size_t input_row = input_ic_base + ih * input_w;
                                const std::size_t w_kh_base = w_ic_base + kh * k_;
                                
                                for (int kw = 0; kw < k_; ++kw) {
                                    const int iw = iw_base + kw;
                                    if (iw < 0 || iw >= input_w) continue;
                                    
                                    const float val = input.data[input_row + iw];
                                    const std::size_t w_idx = w_kh_base + kw;
                                    
                                    local_grad_weights[w_idx] += go * val;
                                    
#ifdef _OPENMP
#pragma omp atomic
#endif
                                    grad_input.data[input_row + iw] += go * weights_[w_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    std::vector<float> grad_weights(weights_.size(), 0.0f);
    std::vector<float> grad_bias(out_c_, 0.0f);
    
    for (int t = 0; t < num_threads; ++t) {
        for (std::size_t i = 0; i < weights_.size(); ++i) {
            grad_weights[i] += thread_grad_weights[t][i];
        }
        for (int oc = 0; oc < out_c_; ++oc) {
            grad_bias[oc] += thread_grad_bias[t][oc];
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
    const std::size_t total = input.data.size();
    const float* __restrict__ in_ptr = input.data.data();
    float* __restrict__ out_ptr = output.data.data();
    
#ifdef _OPENMP
#pragma omp parallel for simd schedule(static)
#endif
    for (std::size_t i = 0; i < total; ++i) {
        out_ptr[i] = in_ptr[i] > 0.0f ? in_ptr[i] : 0.0f;
    }
    return output;
}

Tensor4D ReLULayer::backward(const Tensor4D &input, const Tensor4D &grad_output) const {
    Tensor4D grad_input(input.n, input.c, input.h, input.w);
    const std::size_t total = input.data.size();
    const float* __restrict__ in_ptr = input.data.data();
    const float* __restrict__ grad_out_ptr = grad_output.data.data();
    float* __restrict__ grad_in_ptr = grad_input.data.data();
    
#ifdef _OPENMP
#pragma omp parallel for simd schedule(static)
#endif
    for (std::size_t i = 0; i < total; ++i) {
        grad_in_ptr[i] = in_ptr[i] > 0.0f ? grad_out_ptr[i] : 0.0f;
    }
    return grad_input;
}

MaxPool2DLayer::MaxPool2DLayer(int kernel_size, int stride)
    : k_(kernel_size), stride_(stride) {}

Tensor4D MaxPool2DLayer::forward(const Tensor4D &input) const {
    const int out_h = (input.h - k_) / stride_ + 1;
    const int out_w = (input.w - k_) / stride_ + 1;
    const int batch_size = input.n;
    const int channels = input.c;
    const int input_h = input.h;
    const int input_w = input.w;

    Tensor4D output(batch_size, channels, out_h, out_w);

    const std::size_t input_c_stride = static_cast<std::size_t>(input_h) * input_w;
    const std::size_t input_n_stride = static_cast<std::size_t>(channels) * input_c_stride;
    const std::size_t out_c_stride = static_cast<std::size_t>(out_h) * out_w;
    const std::size_t out_n_stride = static_cast<std::size_t>(channels) * out_c_stride;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int n = 0; n < batch_size; ++n) {
        for (int c = 0; c < channels; ++c) {
            const std::size_t input_base = n * input_n_stride + c * input_c_stride;
            const std::size_t out_base = n * out_n_stride + c * out_c_stride;
            
            for (int oh = 0; oh < out_h; ++oh) {
                const int ih_base = oh * stride_;
                const std::size_t out_row = out_base + oh * out_w;
                
                for (int ow = 0; ow < out_w; ++ow) {
                    const int iw_base = ow * stride_;
                    float max_val = -std::numeric_limits<float>::infinity();
                    
                    if (k_ == 2) {
                        const std::size_t idx00 = input_base + ih_base * input_w + iw_base;
                        const std::size_t idx01 = idx00 + 1;
                        const std::size_t idx10 = idx00 + input_w;
                        const std::size_t idx11 = idx10 + 1;
                        
                        max_val = input.data[idx00];
                        if (input.data[idx01] > max_val) max_val = input.data[idx01];
                        if (input.data[idx10] > max_val) max_val = input.data[idx10];
                        if (input.data[idx11] > max_val) max_val = input.data[idx11];
                    } else {
                        for (int kh = 0; kh < k_; ++kh) {
                            const int ih = ih_base + kh;
                            const std::size_t input_row = input_base + ih * input_w;
                            for (int kw = 0; kw < k_; ++kw) {
                                const float val = input.data[input_row + iw_base + kw];
                                if (val > max_val) max_val = val;
                            }
                        }
                    }
                    output.data[out_row + ow] = max_val;
                }
            }
        }
    }

    return output;
}

UpSample2DLayer::UpSample2DLayer(int scale) : scale_(scale) {}

Tensor4D UpSample2DLayer::forward(const Tensor4D &input) const {
    const int out_h = input.h * scale_;
    const int out_w = input.w * scale_;
    const int batch_size = input.n;
    const int channels = input.c;
    const int input_h = input.h;
    const int input_w = input.w;

    Tensor4D output(batch_size, channels, out_h, out_w);

    const std::size_t input_c_stride = static_cast<std::size_t>(input_h) * input_w;
    const std::size_t input_n_stride = static_cast<std::size_t>(channels) * input_c_stride;
    const std::size_t out_c_stride = static_cast<std::size_t>(out_h) * out_w;
    const std::size_t out_n_stride = static_cast<std::size_t>(channels) * out_c_stride;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int n = 0; n < batch_size; ++n) {
        for (int c = 0; c < channels; ++c) {
            const std::size_t input_base = n * input_n_stride + c * input_c_stride;
            const std::size_t out_base = n * out_n_stride + c * out_c_stride;
            
            for (int oh = 0; oh < out_h; ++oh) {
                const int ih = oh / scale_;
                const std::size_t input_row = input_base + ih * input_w;
                const std::size_t out_row = out_base + oh * out_w;
                
                for (int ow = 0; ow < out_w; ++ow) {
                    const int iw = ow / scale_;
                    output.data[out_row + ow] = input.data[input_row + iw];
                }
            }
        }
    }

    return output;
}

Tensor4D UpSample2DLayer::backward(const Tensor4D &input, const Tensor4D &grad_output) const {
    const int out_h = grad_output.h;
    const int out_w = grad_output.w;
    const int batch_size = input.n;
    const int channels = input.c;
    const int input_h = input.h;
    const int input_w = input.w;

    Tensor4D grad_input(batch_size, channels, input_h, input_w);

    const std::size_t input_c_stride = static_cast<std::size_t>(input_h) * input_w;
    const std::size_t input_n_stride = static_cast<std::size_t>(channels) * input_c_stride;
    const std::size_t out_c_stride = static_cast<std::size_t>(out_h) * out_w;
    const std::size_t out_n_stride = static_cast<std::size_t>(channels) * out_c_stride;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int n = 0; n < batch_size; ++n) {
        for (int c = 0; c < channels; ++c) {
            const std::size_t input_base = n * input_n_stride + c * input_c_stride;
            const std::size_t out_base = n * out_n_stride + c * out_c_stride;
            
            for (int oh = 0; oh < out_h; ++oh) {
                const int ih = oh / scale_;
                const std::size_t input_row = input_base + ih * input_w;
                const std::size_t out_row = out_base + oh * out_w;
                
                for (int ow = 0; ow < out_w; ++ow) {
                    const int iw = ow / scale_;
#ifdef _OPENMP
#pragma omp atomic
#endif
                    grad_input.data[input_row + iw] += grad_output.data[out_row + ow];
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

    const std::size_t total = output.data.size();
    const float* __restrict__ out_ptr = output.data.data();
    const float* __restrict__ tgt_ptr = target.data.data();
    
    float sum = 0.0f;
#ifdef _OPENMP
#pragma omp parallel for simd reduction(+:sum) schedule(static)
#endif
    for (std::size_t i = 0; i < total; ++i) {
        const float diff = out_ptr[i] - tgt_ptr[i];
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

    const std::size_t total = output.data.size();
    const float scale = 2.0f / static_cast<float>(total);
    const float* __restrict__ out_ptr = output.data.data();
    const float* __restrict__ tgt_ptr = target.data.data();
    float* __restrict__ grad_ptr = grad_output.data.data();
    
    float sum = 0.0f;
#ifdef _OPENMP
#pragma omp parallel for simd reduction(+:sum) schedule(static)
#endif
    for (std::size_t i = 0; i < total; ++i) {
        const float diff = out_ptr[i] - tgt_ptr[i];
        sum += diff * diff;
        grad_ptr[i] = scale * diff;
    }

    return sum / static_cast<float>(total);
}
