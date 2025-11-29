#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <iomanip>

#include "dataset.h"
#include "gpu_autoencoder.h"
#include "gpu_layer.h"
#include "cuda_utils.h"

#ifdef WITH_SVM
#include "svm_wrapper.h"
#endif

void print_gpu_info() {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    
    if (device_count == 0) {
        std::cerr << "No CUDA-capable devices found!" << std::endl;
        return;
    }
    
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "GPU " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total memory: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    }
}

int main(int argc, char** argv) {
    std::string data_dir = "data";
    int epochs = 20;
    int batch_size = 64;
    float learning_rate = 1e-3f;
    std::string log_path = "gpu_phase2_log.csv";
    int max_train_images = 0;
    std::string weights_load_path;
    std::string weights_save_path = "autoencoder_gpu.weights";
    bool verify_mode = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (arg == "--epochs" && i + 1 < argc) {
            epochs = std::stoi(argv[++i]);
        } else if (arg == "--batch" && i + 1 < argc) {
            batch_size = std::stoi(argv[++i]);
        } else if (arg == "--lr" && i + 1 < argc) {
            learning_rate = std::stof(argv[++i]);
        } else if (arg == "--log" && i + 1 < argc) {
            log_path = argv[++i];
        } else if (arg == "--max-images" && i + 1 < argc) {
            max_train_images = std::stoi(argv[++i]);
        } else if (arg == "--load-weights" && i + 1 < argc) {
            weights_load_path = argv[++i];
        } else if (arg == "--save-weights" && i + 1 < argc) {
            weights_save_path = argv[++i];
        } else if (arg == "--verify") {
            verify_mode = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --data <dir>         CIFAR-10 data directory (default: data)\n"
                      << "  --epochs <n>         Number of training epochs (default: 20)\n"
                      << "  --batch <n>          Batch size (default: 64)\n"
                      << "  --lr <f>             Learning rate (default: 0.001)\n"
                      << "  --log <file>         CSV log file path\n"
                      << "  --max-images <n>     Max training images (0=all)\n"
                      << "  --load-weights <f>   Load weights from file\n"
                      << "  --save-weights <f>   Save weights to file\n"
                      << "  --verify             Run GPU/CPU verification mode\n"
                      << "  --help               Show this help\n";
            return 0;
        }
    }

    std::cout << "=== GPU Autoencoder Training (Phase 2) ===" << std::endl;
    print_gpu_info();
    std::cout << std::endl;

    CUDA_CHECK(cudaSetDevice(0));

    std::cout << "Loading CIFAR-10 dataset from: " << data_dir << std::endl;
    CIFAR10Dataset dataset(data_dir);
    const auto& train = dataset.train();
    const auto& test = dataset.test();

    std::cout << "  Training images: " << train.num_images << std::endl;
    std::cout << "  Test images: " << test.num_images << std::endl;

    if (train.num_images == 0) {
        std::cerr << "Error: No training images loaded!" << std::endl;
        return 1;
    }

    int effective_train = train.num_images;
    if (max_train_images > 0 && max_train_images < effective_train) {
        effective_train = max_train_images;
        std::cout << "  Using first " << effective_train << " images for training" << std::endl;
    }

    int num_batches = effective_train / batch_size;
    if (num_batches == 0) {
        std::cerr << "Error: batch_size too large for dataset" << std::endl;
        return 1;
    }

    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Batches per epoch: " << num_batches << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // ========================================================================
    std::cout << "Initializing GPU autoencoder..." << std::endl;
    GPUAutoencoder autoencoder;

    if (!weights_load_path.empty()) {
        std::cout << "Loading weights from: " << weights_load_path << std::endl;
        if (!autoencoder.load_weights(weights_load_path)) {
            std::cerr << "Warning: Failed to load weights, starting fresh" << std::endl;
        }
    }

    std::ofstream log_file;
    if (!log_path.empty()) {
        log_file.open(log_path);
        if (log_file) {
            log_file << "epoch,batch,loss,epoch_time_sec,batch_time_ms" << std::endl;
        }
    }

    std::vector<int> indices(effective_train);
    for (int i = 0; i < effective_train; ++i) {
        indices[i] = i;
    }
    std::mt19937 rng(42);

    GPUTensor4D gpu_batch(batch_size, 3, 32, 32);
    GPUTensor4D gpu_output;

    std::vector<float> h_batch(static_cast<size_t>(batch_size) * 3 * 32 * 32);

    std::cout << "\n=== Starting Training ===" << std::endl;
    std::cout << "Epochs: " << epochs << ", LR: " << learning_rate << std::endl;
    std::cout << std::endl;

    auto total_start = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);

        float epoch_loss = 0.0f;
        auto epoch_start = std::chrono::high_resolution_clock::now();

        for (int batch = 0; batch < num_batches; ++batch) {
            auto batch_start = std::chrono::high_resolution_clock::now();

            for (int b = 0; b < batch_size; ++b) {
                int img_idx = indices[batch * batch_size + b];
                const float* src = train.images.data() + static_cast<size_t>(img_idx) * 3 * 32 * 32;
                float* dst = h_batch.data() + static_cast<size_t>(b) * 3 * 32 * 32;
                std::copy(src, src + 3 * 32 * 32, dst);
            }

            gpu_batch.copy_from_host(h_batch.data());

            float loss = autoencoder.train_step(gpu_batch, gpu_batch, learning_rate);
            epoch_loss += loss;

            auto batch_end = std::chrono::high_resolution_clock::now();
            double batch_ms = std::chrono::duration<double, std::milli>(batch_end - batch_start).count();

            if (log_file && (batch % 50 == 0 || batch == num_batches - 1)) {
                log_file << epoch << "," << batch << "," << loss << ",," << batch_ms << std::endl;
            }

            if (batch % 100 == 0 || batch == num_batches - 1) {
                std::cout << "\r  Epoch " << (epoch + 1) << "/" << epochs
                          << " | Batch " << (batch + 1) << "/" << num_batches
                          << " | Loss: " << std::fixed << std::setprecision(4) << loss
                          << " | " << std::setprecision(1) << batch_ms << " ms/batch"
                          << std::flush;
            }
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_sec = std::chrono::duration<double>(epoch_end - epoch_start).count();
        float avg_loss = epoch_loss / num_batches;

        std::cout << std::endl;
        std::cout << "  Epoch " << (epoch + 1) << " complete: "
                  << "Avg Loss = " << std::fixed << std::setprecision(4) << avg_loss
                  << ", Time = " << std::setprecision(1) << epoch_sec << " sec" << std::endl;

        if (log_file) {
            log_file << epoch << ",," << avg_loss << "," << epoch_sec << "," << std::endl;
        }
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(total_end - total_start).count();

    std::cout << "\n=== Training Complete ===" << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(1) << total_sec << " seconds" << std::endl;

    if (!weights_save_path.empty()) {
        std::cout << "Saving weights to: " << weights_save_path << std::endl;
        if (autoencoder.save_weights(weights_save_path)) {
            std::cout << "  Success!" << std::endl;
        } else {
            std::cerr << "  Failed to save weights!" << std::endl;
        }
    }

    #ifdef WITH_SVM
    std::cout << "\n=== Feature Extraction & SVM Training ===" << std::endl;
    
    const int feature_dim = 128 * 8 * 8;
    std::vector<float> train_features(static_cast<size_t>(effective_train) * feature_dim);
    std::vector<int> train_labels(effective_train);
    
    GPUTensor4D single_image(1, 3, 32, 32);
    GPUTensor4D latent;
    std::vector<float> h_latent(feature_dim);
    
    std::cout << "Extracting training features..." << std::endl;
    for (int i = 0; i < effective_train; ++i) {
        const float* src = train.images.data() + static_cast<size_t>(i) * 3 * 32 * 32;
        single_image.copy_from_host(src);
        autoencoder.encode(single_image, latent);
        latent.copy_to_host(h_latent.data());
        std::copy(h_latent.begin(), h_latent.end(), 
                  train_features.begin() + static_cast<size_t>(i) * feature_dim);
        train_labels[i] = train.labels[i];
        
        if ((i + 1) % 1000 == 0) {
            std::cout << "\r  Processed " << (i + 1) << "/" << effective_train << std::flush;
        }
    }
    std::cout << std::endl;
    
    std::vector<float> test_features(static_cast<size_t>(test.num_images) * feature_dim);
    std::vector<int> test_labels(test.num_images);
    
    std::cout << "Extracting test features..." << std::endl;
    for (int i = 0; i < test.num_images; ++i) {
        const float* src = test.images.data() + static_cast<size_t>(i) * 3 * 32 * 32;
        single_image.copy_from_host(src);
        autoencoder.encode(single_image, latent);
        latent.copy_to_host(h_latent.data());
        std::copy(h_latent.begin(), h_latent.end(),
                  test_features.begin() + static_cast<size_t>(i) * feature_dim);
        test_labels[i] = test.labels[i];
        
        if ((i + 1) % 1000 == 0) {
            std::cout << "\r  Processed " << (i + 1) << "/" << test.num_images << std::flush;
        }
    }
    std::cout << std::endl;
    
    std::cout << "Training SVM classifier..." << std::endl;
    SVMWrapper svm;
    svm.train(train_features.data(), train_labels.data(), effective_train, feature_dim);
    
    std::cout << "Evaluating on test set..." << std::endl;
    float accuracy = svm.evaluate(test_features.data(), test_labels.data(), 
                                   test.num_images, feature_dim);
    std::cout << "Test Accuracy: " << std::fixed << std::setprecision(2) 
              << (accuracy * 100.0f) << "%" << std::endl;
#endif

    if (log_file) log_file.close();
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
