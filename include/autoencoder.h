#ifndef AUTOENCODER_H
#define AUTOENCODER_H

#include <string>

#include "layer.h"

class Autoencoder {
public:
    Autoencoder();

    Tensor4D forward(const Tensor4D &input) const;

    float train_step(const Tensor4D &input, const Tensor4D &target,
                     float learning_rate);

    Tensor4D encode(const Tensor4D &input) const;

    bool save_weights(const std::string &path) const;

    bool load_weights(const std::string &path);

private:
    Conv2DLayer conv1_;
    ReLULayer relu1_;
    MaxPool2DLayer pool1_;

    Conv2DLayer conv2_;
    ReLULayer relu2_;
    MaxPool2DLayer pool2_;

    Conv2DLayer conv3_;
    ReLULayer relu3_;
    UpSample2DLayer up1_;

    Conv2DLayer conv4_;
    ReLULayer relu4_;
    UpSample2DLayer up2_;

    Conv2DLayer conv5_;
};

#endif  // AUTOENCODER_H
