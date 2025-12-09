#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <random>
#include <string> 
#include <fstream>
#include <iomanip>
#include <thread>
#include <mutex>
#include <filesystem>
#include <cstdlib> // Dùng cho system()
#include "../include/dataset.h"
#include "../include/autoencoder.h"

namespace fs = std::filesystem;

// ======================= HYPERPARAMETERS =======================
// Với 80 threads, Batch size phải đủ lớn để giảm tranh chấp Mutex
const int BATCH_SIZE = 4096; 
// Tăng LR theo Batch size (Linear Scaling)
const float LEARNING_RATE = 0.004f; 
const long CHECKPOINT_INTERVAL = 1000;

std::mutex training_mutex; 
std::string detailed_log_file = "loss_history.csv"; // File log chi tiết để vẽ biểu đồ

// ======================= HELPER FUNCTIONS =======================

float compute_mse_loss_grad(const std::vector<float>& pred, const std::vector<float>& target, std::vector<float>& grad_output) {
    float loss = 0.0f;
    size_t size = pred.size();
    grad_output.resize(size);
    
    for (size_t i = 0; i < size; ++i) {
        float diff = pred[i] - target[i];
        loss += diff * diff;
        grad_output[i] = 2.0f * diff / size;
    }
    return loss / size;
}

// Hàm gọi Python để vẽ biểu đồ từ file CSV
void save_loss_plot(const std::string& csv_path, const std::string& output_img_path) {
    // Tạo một script python tạm thời
    std::string python_script = 
        "import matplotlib.pyplot as plt\n"
        "import pandas as pd\n"
        "import sys\n"
        "\n"
        "try:\n"
        "    data = pd.read_csv('" + csv_path + "')\n"
        "    plt.figure(figsize=(10, 6))\n"
        "    plt.plot(data['samples'], data['loss'], label='Training Loss')\n"
        "    plt.xlabel('Images Trained')\n"
        "    plt.ylabel('Loss')\n"
        "    plt.title('Training Loss Progress')\n"
        "    plt.grid(True)\n"
        "    plt.legend()\n"
        "    plt.savefig('" + output_img_path + "')\n"
        "    plt.close()\n"
        "    print('>> Plot saved: " + output_img_path + "')\n"
        "except Exception as e:\n"
        "    print(f'>> Error plotting: {e}')\n";

    // Lưu script ra file
    std::ofstream script_file("plot_script_temp.py");
    script_file << python_script;
    script_file.close();

    // Gọi python để chạy script
    // Lưu ý: Server cần cài sẵn python3, pandas, matplotlib
    int ret = system("python3 plot_script_temp.py"); 
    if (ret != 0) {
        // Fallback nếu lệnh python3 không chạy được
        system("python plot_script_temp.py");
    }
}

// ======================= MAIN =======================

int main(int argc, char** argv) {
    // Tối ưu số lượng worker
    unsigned int hw_threads = std::thread::hardware_concurrency();
    int max_workers = (hw_threads > 0) ? hw_threads : 4; 

    std::cout << "==================================================" << std::endl;
    std::cout << ">> Server CPU Cores: " << hw_threads << std::endl;
    std::cout << ">> Batch Size: " << BATCH_SIZE << std::endl;
    std::cout << ">> Learning Rate: " << LEARNING_RATE << std::endl;
    std::cout << "==================================================" << std::endl;

    // Load Data
    std::cout << "Loading CIFAR-10 data..." << std::endl;
    std::string data_path = (argc > 1) ? argv[1] : "data/";
    CIFAR10Dataset dataset(data_path);
    dataset.load_data();

    std::cout << "Initializing Autoencoder on CPU..." << std::endl;
    Autoencoder model;

    // Config Paths
    int num_train = dataset.train_images.size();
    int num_epochs = 20;
    std::string model_save_path = "weights/cpu_model.bin"; 
    std::string log_filename = "cpu_model_training_log.csv";
    std::string checkpoint_dir = "checkpoints/";
    
    fs::create_directories("weights");
    fs::create_directories(checkpoint_dir);

    // Khởi tạo file log chi tiết cho việc vẽ biểu đồ
    {
        std::ofstream detail_log(detailed_log_file);
        detail_log << "samples,loss" << std::endl; // Header
    }

    // Khởi tạo file log tổng (epoch log)
    std::ofstream log_file(log_filename);
    log_file << "epoch,avg_loss,epoch_time_sec,batch_size,workers,images_per_sec" << std::endl;

    long global_images_trained = 0;
    long last_checkpoint_trigger = 0;

    std::cout << "Start Training..." << std::endl;
    auto total_start_time = std::chrono::high_resolution_clock::now();

    std::vector<int> indices(num_train);
    for(int i=0; i<num_train; ++i) indices[i] = i;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        auto epoch_start_time = std::chrono::high_resolution_clock::now();
        float total_loss = 0.0f;
        
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);

        int total_batches = (num_train + BATCH_SIZE - 1) / BATCH_SIZE;

        for (int i = 0; i < num_train; i += BATCH_SIZE) {
            int current_batch_size = std::min(BATCH_SIZE, num_train - i);
            float batch_loss_accum = 0.0f;

            // Logic chia việc cho workers
            // Với 80 cores, ta chia càng đều càng tốt
            int current_workers = std::min(max_workers, current_batch_size); 

            std::vector<std::thread> workers;
            int items_per_worker = current_batch_size / current_workers;
            int remainder = current_batch_size % current_workers;
            int start_idx_in_batch = 0;

            for (int t = 0; t < current_workers; ++t) {
                int count = items_per_worker + (t < remainder ? 1 : 0);
                int end_idx_in_batch = start_idx_in_batch + count;
                
                if (count > 0) {
                    workers.emplace_back([&, start_idx_in_batch, end_idx_in_batch]() {
                        float local_loss = 0.0f;
                        std::vector<float> output_img; output_img.reserve(3072);
                        std::vector<float> loss_grad;  loss_grad.reserve(3072);

                        for (int k = start_idx_in_batch; k < end_idx_in_batch; ++k) {
                            int img_idx = indices[i + k];
                            const std::vector<float>& input_img = dataset.train_images[img_idx].data;
                            
                            // Forward (Không cần lock)
                            model.forward(input_img, output_img);
                            
                            // Tính Loss & Gradient (Không cần lock)
                            local_loss += compute_mse_loss_grad(output_img, input_img, loss_grad);

                            // Backward & Update Weight (CẦN LOCK)
                            // Đây là điểm nghẽn (bottleneck) khi dùng nhiều threads
                            {
                                std::lock_guard<std::mutex> lock(training_mutex);
                                model.backward(loss_grad, LEARNING_RATE);
                            }
                        }

                        // Tổng hợp loss local
                        {
                            std::lock_guard<std::mutex> lock(training_mutex);
                            batch_loss_accum += local_loss;
                        }
                    });
                }
                start_idx_in_batch = end_idx_in_batch;
            }

            for (auto& w : workers)
                if (w.joinable()) w.join();

            // Cập nhật thống kê sau Batch
            {
                std::lock_guard<std::mutex> lock(training_mutex); // An toàn luồng chính
                total_loss += batch_loss_accum;
                global_images_trained += current_batch_size;
            }
            
            float current_avg_loss = batch_loss_accum / current_batch_size;

            // ================== CHECKPOINT & PLOTTING LOGIC ==================
            // Kiểm tra xem đã vượt qua ngưỡng checkpoint tiếp theo chưa
            if (global_images_trained - last_checkpoint_trigger >= CHECKPOINT_INTERVAL) {
                
                last_checkpoint_trigger = global_images_trained;

                // 1. Lưu checkpoint model
                std::string ckpt_file = checkpoint_dir + 
                    "checkpoint_" + std::to_string(global_images_trained) + ".bin";
                model.save_weights(ckpt_file);

                // 2. Ghi loss hiện tại vào file log chi tiết
                std::ofstream detail_log(detailed_log_file, std::ios::app); // Append mode
                detail_log << global_images_trained << "," << current_avg_loss << std::endl;
                detail_log.close();

                // 3. Vẽ biểu đồ và lưu cùng chỗ với checkpoint
                std::string plot_file = checkpoint_dir + 
                    "loss_plot_" + std::to_string(global_images_trained) + ".png";
                
                // Gọi hàm vẽ (chạy trên luồng chính để tránh lỗi system call)
                save_loss_plot(detailed_log_file, plot_file);
            }
            // ================================================================

            int current_batch_idx = (i / BATCH_SIZE) + 1;
            std::cout << "Epoch " << epoch + 1 << "/" << num_epochs 
                      << " [Batch " << std::setw(2) << current_batch_idx << "/" << total_batches << "] "
                      << "Loss: " << std::fixed << std::setprecision(4) << current_avg_loss
                      << "\r" << std::flush;
        }
        
        // ... (Logic tính toán thời gian epoch giữ nguyên) ...
        auto epoch_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> epoch_duration = epoch_end_time - epoch_start_time;
        double epoch_seconds = epoch_duration.count();
        float avg_loss = total_loss / num_train;
        double images_per_sec = num_train / epoch_seconds;

        std::cout << std::string(80, ' ') << "\r"; 
        std::cout << "Epoch [" << std::setw(3) << epoch + 1 << "/" << num_epochs << "] "
                  << "Loss: " << std::fixed << std::setprecision(6) << avg_loss << " | "
                  << "Speed: " << (int)images_per_sec << " img/s"
                  << std::endl;

        log_file << (epoch + 1) << "," << avg_loss << "," << epoch_seconds << ","
                 << BATCH_SIZE << "," << max_workers << "," << images_per_sec << std::endl;
    }

    // Kết thúc
    auto total_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = total_end_time - total_start_time;
    
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Training finished in " << elapsed.count() << " seconds." << std::endl;
    std::cout << "Saving final model..." << std::endl;
    model.save_weights(model_save_path);
    log_file.close();

    // Dọn dẹp file script tạm
    fs::remove("plot_script_temp.py");

    return 0;
}