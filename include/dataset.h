#ifndef DATASET_H
#define DATASET_H

#include <string>
#include <vector>

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

private:
    CifarBatch train_;
    CifarBatch test_;

    static CifarBatch load_batch(const std::string &file_path);
};

#endif // DATASET_H
