#include "../include/dataset.h"

// Implement CIFAR10Dataset methods
CIFAR10Dataset::CIFAR10Dataset(const std::string& dir) : data_dir(dir) {}

void CIFAR10Dataset::load_batch(const std::string& filename, std::vector<Image>& dataset) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        exit(1);
    }

    // Mỗi batch file chứa 10,000 ảnh
    for (int i = 0; i < 10000; ++i) {
        Image img;
        unsigned char label_byte;
        file.read((char*)&label_byte, 1);
        img.label = (int)label_byte;

        std::vector<unsigned char> buffer(3072);
        file.read((char*)buffer.data(), 3072);

        img.data.resize(3072);
        // CIFAR-10 binary format: RRR...GGG...BBB...
        // Chuẩn hóa về [0, 1]
        for (int j = 0; j < 3072; ++j) {
            img.data[j] = (float)buffer[j] / 255.0f;
        }
        dataset.push_back(img);
    }
    file.close();
    std::cout << "Loaded " << filename << std::endl;
}

void CIFAR10Dataset::load_data() {
    // Load 5 training batches
    for (int i = 1; i <= 5; ++i) {
        // Lưu ý: Đường dẫn phải khớp với thư mục bạn giải nén
        // Nếu file nằm ngay trong data/: "data/data_batch_1.bin"
        std::string path = data_dir + "data_batch_" + std::to_string(i) + ".bin";
        load_batch(path, train_images);
    }
    // Load test batch
    load_batch(data_dir + "test_batch.bin", test_images);
    
    std::cout << "Data loading complete. Train: " << train_images.size() 
              << ", Test: " << test_images.size() << std::endl;
}

void CIFAR10Dataset::normalize() {
    // Đã normalize trong lúc load
}