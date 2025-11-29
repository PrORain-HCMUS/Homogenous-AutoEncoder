#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <random>
#include "../include/dataset.h"
#include "../include/autoencoder.h"

// Hyperparameters chung
const int BATCH_SIZE = 32;
const float LEARNING_RATE = 0.001f;

// Hàm tính MSE Loss và Gradient của Loss
float compute_mse_loss_grad(const std::vector<float>& pred, const std::vector<float>& target, std::vector<float>& grad_output) {
    float loss = 0.0f;
    int size = pred.size();
    grad_output.resize(size);
    
    for (int i = 0; i < size; ++i) {
        float diff = pred[i] - target[i];
        loss += diff * diff;
        // Gradient MSE: dL/dx = 2 * (pred - target) / N
        grad_output[i] = 2.0f * diff / size;
    }
    return loss / size;
}

int main(int argc, char** argv) {
    // 1. Load Data
    std::cout << "Loading CIFAR-10 data..." << std::endl;
    std::string data_path = (argc > 1) ? argv[1] : "data/";
    
    CIFAR10Dataset dataset(data_path);
    dataset.load_data();

    // 2. Initialize Model
    std::cout << "Initializing Autoencoder on CPU..." << std::endl;
    Autoencoder model;

    // ========================================================================
    // [CONFIGURATION AREA] - CHỌN 1 TRONG 2 CHẾ ĐỘ BÊN DƯỚI (COMMENT/UNCOMMENT)
    // ========================================================================

    // --- MODE 1: QUICK TEST (Dùng để debug xem loss có giảm không) ---
    // Chạy trên 64 ảnh (2 batches), chạy 10 epochs để thấy rõ loss giảm.
    // -----------------------------------------------------------------
    int num_train = 64; 
    int num_epochs = 2; 
    std::cout << ">> RUNNING MODE: QUICK TEST (64 images, 2 epochs)" << std::endl;


    // --- MODE 2: FULL TRAINING (Dùng để đo thời gian baseline) ---
    // Chạy trên 50,000 ảnh, chỉ chạy 1 epoch vì CPU rất chậm.
    // -----------------------------------------------------------------
    // int num_train = dataset.train_images.size();
    // int num_epochs = 1;
    // std::cout << ">> RUNNING MODE: FULL TRAINING (50000 images, 1 epoch)" << std::endl;

    // ========================================================================
    // END CONFIGURATION
    // ========================================================================

    // 3. Training Loop
    std::cout << "Start Training..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Tạo danh sách indices để shuffle
    std::vector<int> indices(num_train);
    for(int i=0; i<num_train; ++i) indices[i] = i;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float total_loss = 0.0f;
        
        // Shuffle data
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);

        // Batch Loop
        for (int i = 0; i < num_train; i += BATCH_SIZE) {
            int current_batch_size = std::min(BATCH_SIZE, num_train - i);
            
            for (int j = 0; j < current_batch_size; ++j) {
                int img_idx = indices[i + j];
                const std::vector<float>& input_img = dataset.train_images[img_idx].data;
                std::vector<float> output_img;
                std::vector<float> loss_grad;

                // A. Forward Pass
                model.forward(input_img, output_img);

                // B. Compute Loss & Gradient
                float loss = compute_mse_loss_grad(output_img, input_img, loss_grad);
                total_loss += loss;

                // C. Backward Pass & Update
                model.backward(loss_grad, LEARNING_RATE);
            }

            // In tiến độ mỗi 100 batches (hoặc mỗi batch nếu đang test ít dữ liệu)
            if (num_train <= 1000 || (i / BATCH_SIZE) % 100 == 0) {
                std::cout << "Epoch " << epoch + 1 << "/" << num_epochs 
                          << ", Batch " << i / BATCH_SIZE 
                          << ", Loss: " << total_loss / (i + 1) << "\r" << std::flush;
            }
        }
        
        // Xuống dòng khi hết epoch
        std::cout << std::endl;
        if (num_train <= 1000) {
             std::cout << "-> Avg Epoch Loss: " << total_loss / num_train << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    std::cout << "Training finished in " << elapsed.count() << " seconds." << std::endl;

    return 0;
}