#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>

struct Image {
    std::vector<float> data; // Flattened 3x32x32 image, normalized [0-1]
    int label;
};

class CIFAR10Dataset {
public:
    std::vector<Image> train_images;
    std::vector<Image> test_images;

    CIFAR10Dataset(const std::string& data_dir);
    
    // Đọc file binary batch
    void load_batch(const std::string& filename, std::vector<Image>& dataset);
    
    // Đọc toàn bộ dữ liệu (5 train batches + 1 test batch)
    void load_data();
    
    // Normalization: Chia cho 255.0
    void normalize();

private:
    std::string data_dir;
};

#endif