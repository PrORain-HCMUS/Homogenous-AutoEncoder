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
#include <filesystem>     //// ==== NEW ====
#include "../include/dataset.h"
#include "../include/autoencoder.h"

namespace fs = std::filesystem;

// ========================================================================
const int BATCH_SIZE = 1024; 
const float LEARNING_RATE = 0.001f;
std::mutex training_mutex; 

float compute_mse_loss_grad(const std::vector<float>& pred, const std::vector<float>& target, std::vector<float>& grad_output) {
    float loss = 0.0f;
    int size = pred.size();
    grad_output.resize(size);
    
    for (int i = 0; i < size; ++i) {
        float diff = pred[i] - target[i];
        loss += diff * diff;
        grad_output[i] = 2.0f * diff / size;
    }
    return loss / size;
}

int main(int argc, char** argv) {

    unsigned int hw_threads = std::thread::hardware_concurrency();
    int max_workers = (hw_threads > 0) ? hw_threads : 4; 

    std::cout << "==================================================" << std::endl;
    std::cout << ">> Server CPU Cores: " << hw_threads << std::endl;
    std::cout << ">> Max Available Workers: " << max_workers << std::endl;
    std::cout << "==================================================" << std::endl;

    std::cout << "Loading CIFAR-10 data..." << std::endl;
    std::string data_path = (argc > 1) ? argv[1] : "data/";
    
    CIFAR10Dataset dataset(data_path);
    dataset.load_data();

    std::cout << "Initializing Autoencoder on CPU..." << std::endl;
    Autoencoder model;

    // ------------------ TRAIN CONFIG ------------------ 
    int num_train = dataset.train_images.size();
    int num_epochs = 20;
    std::string model_save_path = "weights/cpu_model.bin"; 
    std::string log_filename = "cpu_model_training_log.csv";

    std::cout << ">> RUNNING MODE: FULL TRAINING (50000 images)" << std::endl;

    //// ==== NEW CHECKPOINT CODE ====
    std::string checkpoint_dir = "checkpoints/";
    fs::create_directories(checkpoint_dir);
    long global_images_trained = 0;
    const long CHECKPOINT_INTERVAL = 1000;
    //// ==== END NEW CODE ====

    // ---------------------------------------------------
    std::ofstream log_file(log_filename);
    if (!log_file.is_open()) {
        std::cerr << "Error: Could not create log file!" << std::endl;
        return 1;
    }
    log_file << "epoch,avg_loss,epoch_time_sec,batch_size,workers,images_per_sec" << std::endl;
    
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

        int processed_count = 0;
        int total_batches = (num_train + BATCH_SIZE - 1) / BATCH_SIZE;

        for (int i = 0; i < num_train; i += BATCH_SIZE) {
            int current_batch_size = std::min(BATCH_SIZE, num_train - i);
            float batch_loss_accum = 0.0f;

            int needed_workers = current_batch_size / 4; 
            if (needed_workers < 1) needed_workers = 1;
            int current_workers = std::min(max_workers, needed_workers);

            std::vector<std::thread> workers;
            int items_per_worker = current_batch_size / current_workers;
            int remainder = current_batch_size % current_workers;
            int start_idx_in_batch = 0;

            for (int t = 0; t < current_workers; ++t) {
                int end_idx_in_batch = start_idx_in_batch + items_per_worker + (t < remainder ? 1 : 0);
                
                if (start_idx_in_batch < end_idx_in_batch) {
                    workers.emplace_back([&, start_idx_in_batch, end_idx_in_batch]() {
                        float local_loss = 0.0f;
                        std::vector<float> output_img; output_img.reserve(3072);
                        std::vector<float> loss_grad;  loss_grad.reserve(3072);

                        for (int k = start_idx_in_batch; k < end_idx_in_batch; ++k) {
                            int img_idx = indices[i + k];
                            const std::vector<float>& input_img = dataset.train_images[img_idx].data;
                            
                            model.forward(input_img, output_img);
                            local_loss += compute_mse_loss_grad(output_img, input_img, loss_grad);

                            {
                                std::lock_guard<std::mutex> lock(training_mutex);
                                model.backward(loss_grad, LEARNING_RATE);
                            }
                        }

                        {
                            std::lock_guard<std::mutex> lock(training_mutex);
                            total_loss += local_loss;
                            batch_loss_accum += local_loss;
                        }
                    });
                }
                start_idx_in_batch = end_idx_in_batch;
            }

            for (auto& w : workers)
                if (w.joinable()) w.join();

            processed_count += current_batch_size;

            //// ==== NEW CHECKPOINT CODE ====
            global_images_trained += current_batch_size;

            if (global_images_trained % CHECKPOINT_INTERVAL == 0) {
                std::string ckpt_file = checkpoint_dir + 
                    "checkpoint_" + std::to_string(global_images_trained) + ".bin";

                model.save_weights(ckpt_file);

                std::cout << "\n📌 Saved checkpoint: " << ckpt_file 
                          << " (trained " << global_images_trained << " images)\n";
            }
            //// ==== END NEW CODE ====

            int current_batch_idx = (i / BATCH_SIZE) + 1;
            std::cout << "Epoch " << epoch + 1 << "/" << num_epochs 
                      << " [Batch " << std::setw(2) << current_batch_idx << "/" << total_batches << "] "
                      << "Loss: " << std::fixed << std::setprecision(4) 
                      << (batch_loss_accum / current_batch_size)
                      << " | Workers: " << current_workers 
                      << "\r" << std::flush;
        }
        
        auto epoch_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> epoch_duration = epoch_end_time - epoch_start_time;
        double epoch_seconds = epoch_duration.count();
        float avg_loss = total_loss / num_train;
        double images_per_sec = num_train / epoch_seconds;

        std::cout << std::string(80, ' ') << "\r"; 
        std::cout << "Epoch [" << std::setw(3) << epoch + 1 << "/" << num_epochs << "] "
                  << "Loss: " << std::fixed << std::setprecision(6) << avg_loss << " | "
                  << "Time: " << std::setprecision(2) << epoch_seconds << "s | "
                  << "Speed: " << (int)images_per_sec << " img/s"
                  << std::endl;

        log_file << (epoch + 1) << "," 
                 << avg_loss << "," 
                 << epoch_seconds << ","
                 << BATCH_SIZE << "," 
                 << max_workers << ","
                 << images_per_sec << std::endl;
    }

    auto total_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = total_end_time - total_start_time;
    
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "Training finished in " << elapsed.count() << " seconds." << std::endl;
    
    std::cout << "Saving final model to " << model_save_path << "..." << std::endl;
    model.save_weights(model_save_path);
    log_file.close();

    return 0;
}
