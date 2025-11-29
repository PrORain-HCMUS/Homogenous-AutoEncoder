#include "dataset.h"

#include <fstream>
#include <stdexcept>

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
