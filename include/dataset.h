#ifndef DATASET_H
#define DATASET_H

#include <string>
#include <vector>
#include <random>

// Data augmentation configuration
struct AugmentConfig {
    bool horizontal_flip = true;      // Random horizontal flip (50% chance)
    bool random_crop = true;          // Random crop with padding
    int crop_padding = 4;             // Padding for random crop (32 -> 40 -> crop back to 32)
    
    AugmentConfig() = default;
    AugmentConfig(bool flip, bool crop, int pad = 4)
        : horizontal_flip(flip), random_crop(crop), crop_padding(pad) {}
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

    // Apply data augmentation to a single image (in-place)
    // image: pointer to CHW format image (3 * 32 * 32 floats)
    // rng: random number generator
    static void augment_image(float* image, const AugmentConfig& config, std::mt19937& rng);
    
    // Apply augmentation to a batch of images
    static void augment_batch(float* batch, int batch_size, const AugmentConfig& config, std::mt19937& rng);

private:
    CifarBatch train_;
    CifarBatch test_;

    static CifarBatch load_batch(const std::string &file_path);
    
    // Helper functions for augmentation
    static void horizontal_flip_image(float* image);
    static void random_crop_image(float* image, int padding, std::mt19937& rng);
};

#endif // DATASET_H
