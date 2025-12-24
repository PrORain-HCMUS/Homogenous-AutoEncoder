#ifndef DATASET_H
#define DATASET_H

#include <string>
#include <vector>
#include <random>

// Data augmentation configuration
struct AugmentConfig {
    bool horizontal_flip = true;
    bool random_crop = true;
    int crop_padding = 4;
    bool color_jitter = true;
    float brightness_range = 0.2f;
    float contrast_range = 0.0f;
    float saturation_range = 0.0f;
    bool cutout = false;
    int cutout_size = 8;
    bool gaussian_noise = false;
    float noise_std = 0.03f;
    bool random_grayscale = false;
    float grayscale_prob = 0.15f;
    
    AugmentConfig() = default;
    AugmentConfig(bool flip, bool crop, int pad = 4, bool jitter = true, float bright = 0.2f)
        : horizontal_flip(flip), random_crop(crop), crop_padding(pad), 
          color_jitter(jitter), brightness_range(bright), cutout(true), cutout_size(8) {}
};

struct CifarBatch {
    std::vector<float> images;
    std::vector<int> labels;
    int num_images = 0;
};

class CIFAR10Dataset {
public:
    static constexpr int IMAGE_HEIGHT = 32;
    static constexpr int IMAGE_WIDTH = 32;
    static constexpr int IMAGE_CHANNELS = 3;
    static constexpr int IMAGE_SIZE = IMAGE_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH;

    explicit CIFAR10Dataset(const std::string &data_dir);

    const CifarBatch &train() const { return train_; }
    const CifarBatch &test() const { return test_; }

    static void augment_image(float* image, const AugmentConfig& config, std::mt19937& rng);
    static void augment_batch(float* batch, int batch_size, const AugmentConfig& config, std::mt19937& rng);

private:
    CifarBatch train_;
    CifarBatch test_;

    static CifarBatch load_batch(const std::string &file_path);
    static void horizontal_flip_image(float* image);
    static void random_crop_image(float* image, int padding, std::mt19937& rng);
};

#endif // DATASET_H
