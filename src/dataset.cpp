#include "dataset.h"

#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cstring>

namespace {
constexpr int RECORD_BYTES = 1 + CIFAR10Dataset::IMAGE_SIZE;

CifarBatch load_cifar10_batch_impl(const std::string &file_path) {
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Failed to open CIFAR-10 batch: " + file_path);
    }

    std::streamsize file_size = file.tellg();
    if (file_size % RECORD_BYTES != 0) {
        throw std::runtime_error("Invalid CIFAR-10 batch file size: " + file_path);
    }

    int num_records = static_cast<int>(file_size / RECORD_BYTES);
    std::vector<unsigned char> buffer(static_cast<size_t>(file_size));

    file.seekg(0, std::ios::beg);
    if (!file.read(reinterpret_cast<char *>(buffer.data()), file_size)) {
        throw std::runtime_error("Failed to read CIFAR-10 batch: " + file_path);
    }

    CifarBatch batch;
    batch.num_images = num_records;
    batch.labels.resize(num_records);
    batch.images.resize(static_cast<size_t>(num_records) * CIFAR10Dataset::IMAGE_SIZE);

    for (int i = 0; i < num_records; ++i) {
        const unsigned char *record = buffer.data() + i * RECORD_BYTES;
        batch.labels[i] = static_cast<int>(record[0]);

        const unsigned char *pixels = record + 1;
        for (int j = 0; j < CIFAR10Dataset::IMAGE_SIZE; ++j) {
            batch.images[static_cast<size_t>(i) * CIFAR10Dataset::IMAGE_SIZE + j] =
                static_cast<float>(pixels[j]) / 255.0f;
        }
    }

    return batch;
}

void append_batch(CifarBatch &dst, const CifarBatch &src) {
    if (src.num_images == 0) return;
    if (dst.num_images == 0) {
        dst = src;
        return;
    }

    dst.images.reserve(dst.images.size() + src.images.size());
    dst.labels.reserve(dst.labels.size() + src.labels.size());

    dst.images.insert(dst.images.end(), src.images.begin(), src.images.end());
    dst.labels.insert(dst.labels.end(), src.labels.begin(), src.labels.end());

    dst.num_images = static_cast<int>(dst.labels.size());
}

}

CifarBatch CIFAR10Dataset::load_batch(const std::string &file_path) {
    return load_cifar10_batch_impl(file_path);
}

CIFAR10Dataset::CIFAR10Dataset(const std::string &data_dir) {
    for (int i = 1; i <= 5; ++i) {
        std::string path = data_dir + "/data_batch_" + std::to_string(i) + ".bin";
        CifarBatch batch = load_cifar10_batch_impl(path);
        append_batch(train_, batch);
    }

    std::string test_path = data_dir + "/test_batch.bin";
    test_ = load_cifar10_batch_impl(test_path);
}

// ============================================================
// DATA AUGMENTATION IMPLEMENTATION
// ============================================================
// Designed for CIFAR-10 with 10 classes:
// airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
//
// Augmentations chosen:
// 1. Random Horizontal Flip (50%): Safe for all classes
//    - Objects like cars, animals, planes look natural when flipped
//    - Ships and trucks are symmetric
// 2. Random Crop with Padding: Simulates slight translation/scale variation
//    - Pad image to 40x40, then crop random 32x32 region
//    - Helps model learn position-invariant features
// ============================================================

void CIFAR10Dataset::horizontal_flip_image(float* image) {
    // Image is in CHW format: 3 channels x 32 height x 32 width
    // Flip each row horizontally for each channel
    constexpr int H = IMAGE_HEIGHT;
    constexpr int W = IMAGE_WIDTH;
    
    for (int c = 0; c < IMAGE_CHANNELS; ++c) {
        float* channel = image + c * H * W;
        for (int h = 0; h < H; ++h) {
            float* row = channel + h * W;
            // Reverse the row
            for (int w = 0; w < W / 2; ++w) {
                std::swap(row[w], row[W - 1 - w]);
            }
        }
    }
}

void CIFAR10Dataset::random_crop_image(float* image, int padding, std::mt19937& rng) {
    // Pad image with zeros, then crop random 32x32 region
    // This simulates slight translation augmentation
    constexpr int H = IMAGE_HEIGHT;  // 32
    constexpr int W = IMAGE_WIDTH;   // 32
    const int padded_H = H + 2 * padding;  // 40 if padding=4
    const int padded_W = W + 2 * padding;  // 40 if padding=4
    
    // Create padded image (zero-padded)
    std::vector<float> padded(IMAGE_CHANNELS * padded_H * padded_W, 0.0f);
    
    // Copy original image to center of padded image
    for (int c = 0; c < IMAGE_CHANNELS; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                int src_idx = c * H * W + h * W + w;
                int dst_idx = c * padded_H * padded_W + (h + padding) * padded_W + (w + padding);
                padded[dst_idx] = image[src_idx];
            }
        }
    }
    
    // Random crop offset
    std::uniform_int_distribution<int> dist_h(0, 2 * padding);  // 0 to 8
    std::uniform_int_distribution<int> dist_w(0, 2 * padding);  // 0 to 8
    int offset_h = dist_h(rng);
    int offset_w = dist_w(rng);
    
    // Extract cropped region back to original image
    for (int c = 0; c < IMAGE_CHANNELS; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                int src_idx = c * padded_H * padded_W + (h + offset_h) * padded_W + (w + offset_w);
                int dst_idx = c * H * W + h * W + w;
                image[dst_idx] = padded[src_idx];
            }
        }
    }
}

void CIFAR10Dataset::augment_image(float* image, const AugmentConfig& config, std::mt19937& rng) {
    // Apply random horizontal flip (50% probability)
    if (config.horizontal_flip) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        if (dist(rng) < 0.5f) {
            horizontal_flip_image(image);
        }
    }
    
    // Apply random crop with padding
    if (config.random_crop && config.crop_padding > 0) {
        random_crop_image(image, config.crop_padding, rng);
    }
    
    // Apply color jitter (random brightness)
    if (config.color_jitter && config.brightness_range > 0.0f) {
        std::uniform_real_distribution<float> dist(-config.brightness_range, config.brightness_range);
        float brightness_delta = dist(rng);
        
        for (int i = 0; i < IMAGE_SIZE; ++i) {
            image[i] = std::min(1.0f, std::max(0.0f, image[i] + brightness_delta));
        }
    }
    
    // Apply Cutout (random erasing) - 50% probability
    if (config.cutout && config.cutout_size > 0) {
        std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
        if (prob_dist(rng) < 0.5f) {
            // Random position for cutout center
            std::uniform_int_distribution<int> pos_h(0, IMAGE_HEIGHT - 1);
            std::uniform_int_distribution<int> pos_w(0, IMAGE_WIDTH - 1);
            int center_h = pos_h(rng);
            int center_w = pos_w(rng);
            
            int half_size = config.cutout_size / 2;
            int h_start = std::max(0, center_h - half_size);
            int h_end = std::min(IMAGE_HEIGHT, center_h + half_size);
            int w_start = std::max(0, center_w - half_size);
            int w_end = std::min(IMAGE_WIDTH, center_w + half_size);
            
            // Zero out the region across all channels
            for (int c = 0; c < IMAGE_CHANNELS; ++c) {
                for (int h = h_start; h < h_end; ++h) {
                    for (int w = w_start; w < w_end; ++w) {
                        image[c * IMAGE_HEIGHT * IMAGE_WIDTH + h * IMAGE_WIDTH + w] = 0.0f;
                    }
                }
            }
        }
    }
}

void CIFAR10Dataset::augment_batch(float* batch, int batch_size, const AugmentConfig& config, std::mt19937& rng) {
    for (int i = 0; i < batch_size; ++i) {
        float* image = batch + static_cast<size_t>(i) * IMAGE_SIZE;
        augment_image(image, config, rng);
    }
}
