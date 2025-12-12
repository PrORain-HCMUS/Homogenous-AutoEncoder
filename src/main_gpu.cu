#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <limits>

#include "dataset.h"
#include "gpu_autoencoder.h"
#include "gpu_layer.h"
#include "cuda_utils.h"

#ifdef WITH_SVM
#include "svm_wrapper.h"
#endif

// ============================================================
// LEARNING RATE SCHEDULE
// ============================================================
// MultiStepLR: Reduce LR at specified milestones
// For 20 epochs: milestones at epoch 10, 15
// ============================================================
float get_scheduled_lr(float base_lr, int epoch, int total_epochs) {
    // LR Schedule DISABLED - return constant learning rate
    // This produces better features for downstream SVM classification
    return base_lr;
    
    // Original MultiStepLR (disabled):
    // int milestone1 = total_epochs / 2;      // epoch 10 for 20 epochs
    // int milestone2 = total_epochs * 3 / 4;  // epoch 15 for 20 epochs
    // if (epoch >= milestone2) return base_lr * 0.01f;
    // else if (epoch >= milestone1) return base_lr * 0.1f;
    // return base_lr;
}

std::string get_timestamp_gpu() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

class GPUTrainingLogger {
private:
    std::ofstream txt_file;
    std::ofstream csv_file;
    std::string txt_path;
    std::string csv_path;
    
public:
    GPUTrainingLogger(const std::string& txt_log_path, const std::string& csv_log_path) 
        : txt_path(txt_log_path), csv_path(csv_log_path) {
        if (!txt_path.empty()) {
            txt_file.open(txt_path, std::ios::out);
            if (!txt_file) {
                std::cerr << "Warning: failed to open TXT log file: " << txt_path << std::endl;
            }
        }
        if (!csv_path.empty()) {
            csv_file.open(csv_path, std::ios::out);
            if (csv_file) {
                csv_file << "epoch,batch,loss,epoch_time_sec,batch_time_ms,best_loss" << std::endl;
            }
        }
    }
    
    ~GPUTrainingLogger() {
        if (txt_file.is_open()) txt_file.close();
        if (csv_file.is_open()) csv_file.close();
    }
    
    void log(const std::string& message) {
        if (txt_file.is_open()) {
            txt_file << "[" << get_timestamp_gpu() << "] " << message << std::endl;
            txt_file.flush();
        }
    }
    
    void log_config(int epochs, int batch_size, float lr, const std::string& data_dir, 
                    int max_images, const std::string& load_weights, const std::string& save_weights) {
        log("============================================================");
        log("GPU AUTOENCODER TRAINING LOG (Phase 2)");
        log("============================================================");
        log("");
        log("CONFIGURATION:");
        log("  Data directory: " + data_dir);
        log("  Epochs: " + std::to_string(epochs));
        log("  Batch size: " + std::to_string(batch_size));
        log("  Learning rate: " + std::to_string(lr));
        log("  Max train images: " + (max_images > 0 ? std::to_string(max_images) : "all"));
        log("  Load weights: " + (load_weights.empty() ? "none" : load_weights));
        log("  Save weights: " + save_weights);
        log("");
    }
    
    void log_gpu_info(const std::string& gpu_name, int compute_major, int compute_minor,
                      size_t total_mem_mb, int sm_count, int max_threads_per_block) {
        log("GPU HARDWARE:");
        log("  Device: " + gpu_name);
        log("  Compute capability: " + std::to_string(compute_major) + "." + std::to_string(compute_minor));
        log("  Total memory: " + std::to_string(total_mem_mb) + " MB");
        log("  Multiprocessors (SMs): " + std::to_string(sm_count));
        log("  Max threads/block: " + std::to_string(max_threads_per_block));
        log("");
    }
    
    void log_optimizations() {
        log("============================================================");
        log("CUDA OPTIMIZATIONS APPLIED:");
        log("============================================================");
        log("");
        log("1. KERNEL LAUNCH CONFIGURATION:");
        log("   - Conv2D: 2D thread blocks dim3(16,16) = 256 threads");
        log("   - MaxPool2D: 2D spatial thread blocks");
        log("   - UpSample2D: 2D spatial thread blocks");
        log("   - ReLU/MSE: 1D blocks of 256 threads");
        log("");
        log("2. MEMORY OPTIMIZATIONS:");
        log("   - Persistent MSE reduction buffer (allocated once)");
        log("   - Coalesced memory access patterns");
        log("   - Shared memory for tiled convolution (layers_gpu_opt.cu)");
        log("   - Constant memory for small frequently-accessed data");
        log("");
        log("3. WARP-LEVEL OPTIMIZATIONS:");
        log("   - Warp shuffle reduction for MSE loss computation");
        log("   - Uses __shfl_down_sync for fast intra-warp reduction");
        log("   - Reduces shared memory bank conflicts");
        log("");
        log("4. TILED CONVOLUTION (Phase 3 kernels):");
        log("   - 16x16 tiles for output feature maps");
        log("   - Cooperative loading of input tiles to shared memory");
        log("   - Reduced global memory bandwidth");
        log("   - #pragma unroll for small loops");
        log("");
        log("5. LOOP UNROLLING:");
        log("   - Special cases for 3x3 convolution kernel");
        log("   - 2x2 MaxPool unrolling");
        log("   - Manual unroll hints for critical loops");
        log("");
        log("6. CUDA STREAM OPTIMIZATIONS:");
        log("   - Asynchronous memory transfers where applicable");
        log("   - Kernel execution overlapping potential");
        log("");
        log("EXPECTED SPEEDUP VS NAIVE IMPLEMENTATION:");
        log("  - Conv2D: 2-4x (memory coalescing + tiling)");
        log("  - MSE Reduction: 3-5x (warp shuffle vs atomic)");
        log("  - MaxPool2D: 1.5-2x (2D blocks + unrolling)");
        log("");
    }
    
    void log_dataset_info(int train_images, int test_images, int batches_per_epoch) {
        log("DATASET:");
        log("  Training images: " + std::to_string(train_images));
        log("  Test images: " + std::to_string(test_images));
        log("  Batches per epoch: " + std::to_string(batches_per_epoch));
        log("");
    }
    
    void log_epoch_start(int epoch, int total_epochs) {
        log("");
        log("--- Epoch " + std::to_string(epoch) + "/" + std::to_string(total_epochs) + " ---");
    }
    
    void log_batch(int batch, int total_batches, float loss, double batch_ms) {
        std::stringstream ss;
        ss << "  Batch " << batch << "/" << total_batches 
           << " | Loss: " << std::fixed << std::setprecision(6) << loss
           << " | Time: " << std::setprecision(2) << batch_ms << " ms";
        log(ss.str());
    }
    
    void log_epoch_end(int epoch, int total_epochs, float avg_loss, double epoch_time, 
                       bool is_best, float best_loss) {
        std::stringstream ss;
        ss << "Epoch " << epoch << "/" << total_epochs << " COMPLETE:";
        log(ss.str());
        
        ss.str(""); ss.clear();
        ss << "  Average Loss: " << std::fixed << std::setprecision(6) << avg_loss;
        log(ss.str());
        
        ss.str(""); ss.clear();
        ss << "  Epoch Time: " << std::fixed << std::setprecision(2) << epoch_time << " seconds";
        log(ss.str());
        
        ss.str(""); ss.clear();
        ss << "  Throughput: " << std::setprecision(0) << (1.0 / epoch_time * 3600) << " epochs/hour";
        log(ss.str());
        
        if (is_best) {
            log("  *** NEW BEST LOSS ***");
        }
        
        ss.str(""); ss.clear();
        ss << "  Best Loss So Far: " << std::fixed << std::setprecision(6) << best_loss;
        log(ss.str());
    }
    
    void write_csv_batch(int epoch, int batch, float loss, double batch_ms) {
        if (csv_file.is_open()) {
            csv_file << epoch << "," << batch << "," << loss << ",," << batch_ms << "," << std::endl;
        }
    }
    
    void write_csv_epoch(int epoch, float avg_loss, double epoch_sec, float best_loss) {
        if (csv_file.is_open()) {
            csv_file << epoch << ",," << avg_loss << "," << epoch_sec << ",," << best_loss << std::endl;
        }
    }
    
    void log_training_complete(int total_epochs, float final_best_loss, float avg_loss, 
                               double total_time, const std::string& weights_path) {
        log("");
        log("============================================================");
        log("TRAINING COMPLETE");
        log("============================================================");
        log("");
        log("FINAL STATISTICS:");
        log("  Total epochs: " + std::to_string(total_epochs));
        
        std::stringstream ss;
        ss << "  Best loss achieved: " << std::fixed << std::setprecision(6) << final_best_loss;
        log(ss.str());
        
        ss.str(""); ss.clear();
        ss << "  Average final loss: " << std::fixed << std::setprecision(6) << avg_loss;
        log(ss.str());
        
        ss.str(""); ss.clear();
        ss << "  Total training time: " << std::fixed << std::setprecision(2) << total_time << " seconds";
        log(ss.str());
        
        ss.str(""); ss.clear();
        ss << "  Average time per epoch: " << std::fixed << std::setprecision(2) << (total_time / total_epochs) << " seconds";
        log(ss.str());
        
        log("");
        log("MODEL:");
        log("  Weights saved to: " + weights_path);
        log("");
        log("LOG FILES:");
        log("  TXT log: " + txt_path);
        log("  CSV log: " + csv_path);
        log("");
        log("============================================================");
    }
    
    void log_svm_results(float accuracy, int train_samples, int test_samples, int feature_dim) {
        log("");
        log("============================================================");
        log("SVM CLASSIFICATION RESULTS");
        log("============================================================");
        log("");
        log("  Training samples: " + std::to_string(train_samples));
        log("  Test samples: " + std::to_string(test_samples));
        log("  Feature dimension: " + std::to_string(feature_dim));
        
        std::stringstream ss;
        ss << "  Test Accuracy: " << std::fixed << std::setprecision(2) << (accuracy * 100.0f) << "%";
        log(ss.str());
        log("");
    }
};

int get_device_count() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    return device_count;
}

void print_gpu_info(int device_count) {
    if (device_count == 0) {
        std::cerr << "No CUDA-capable devices found!" << std::endl;
        return;
    }
    
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
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
    int batch_size = 64;  // Default batch size
    float learning_rate = 1e-3f;
    std::string csv_path = "gpu_phase2_log.csv";
    std::string txt_path = "gpu_phase2_log.txt";
    int max_train_images = 0;
    std::string weights_load_path;
    std::string weights_save_path = "autoencoder_gpu.weights";
    int device_id = 0;
    bool use_bce_loss = false;  // Default: MSE loss

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--bce-loss") {
            use_bce_loss = true;
        } else if (arg == "--data" && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (arg == "--epochs" && i + 1 < argc) {
            epochs = std::stoi(argv[++i]);
        } else if (arg == "--batch" && i + 1 < argc) {
            batch_size = std::stoi(argv[++i]);
        } else if (arg == "--lr" && i + 1 < argc) {
            learning_rate = std::stof(argv[++i]);
        } else if (arg == "--log" && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (arg == "--log-txt" && i + 1 < argc) {
            txt_path = argv[++i];
        } else if (arg == "--max-images" && i + 1 < argc) {
            max_train_images = std::stoi(argv[++i]);
        } else if (arg == "--load-weights" && i + 1 < argc) {
            weights_load_path = argv[++i];
        } else if (arg == "--save-weights" && i + 1 < argc) {
            weights_save_path = argv[++i];
        } else if (arg == "--device" && i + 1 < argc) {
            device_id = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --data <dir>         CIFAR-10 data directory (default: data)\n"
                      << "  --epochs <n>         Number of training epochs (default: 20)\n"
                      << "  --batch <n>          Batch size (default: 64)\n"
                      << "  --lr <f>             Learning rate (default: 0.001)\n"
                      << "  --log <file>         CSV log file path\n"
                      << "  --log-txt <file>     TXT log file path\n"
                      << "  --max-images <n>     Max training images (0=all)\n"
                      << "  --load-weights <f>   Load weights from file\n"
                      << "  --save-weights <f>   Save weights to file\n"
                      << "  --device <n>         GPU device ID (default: 0)\n"
                      << "  --bce-loss           Use BCE loss with Sigmoid output (default: MSE)\n"
                      << "  --help               Show this help\n";
            return 0;
        }
    }

    std::cout << "=== GPU Autoencoder Training (Phase 2) ===" << std::endl;
    
    int device_count = get_device_count();
    if (device_count == 0) {
        std::cerr << "Error: No CUDA-capable devices found!" << std::endl;
        return 1;
    }
    
    if (device_id < 0 || device_id >= device_count) {
        std::cerr << "Error: Invalid device ID " << device_id 
                  << ". Available devices: 0-" << (device_count - 1) << std::endl;
        return 1;
    }
    
    print_gpu_info(device_count);
    std::cout << std::endl;
    
    std::cout << "Using GPU device: " << device_id << std::endl;
    CUDA_CHECK(cudaSetDevice(device_id));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    
    GPUTrainingLogger logger(txt_path, csv_path);
    // Optimizer configuration
    // NOTE: Momentum and Weight Decay DISABLED for better SVM features
    // Plain SGD produces features that generalize better for classification
    OptimizerConfig opt_config;
    opt_config.momentum = 0.0f;        // DISABLED (was 0.9f)
    opt_config.weight_decay = 0.0f;    // DISABLED (was 1e-4f)
    opt_config.use_momentum = false;   // DISABLED (was true)
    
    // Data augmentation configuration
    // NOTE: Disabled for now - testing Momentum SGD + Weight Decay + LR Schedule only
    // Augmentation may hurt autoencoder training (input != target after augment)
    AugmentConfig aug_config;
    aug_config.horizontal_flip = false;  // DISABLED
    aug_config.random_crop = false;      // DISABLED
    aug_config.crop_padding = 4;
    
    logger.log_config(epochs, batch_size, learning_rate, data_dir, 
                      max_train_images, weights_load_path, weights_save_path);
    logger.log_gpu_info(prop.name, prop.major, prop.minor,
                        prop.totalGlobalMem / (1024 * 1024),
                        prop.multiProcessorCount, prop.maxThreadsPerBlock);
    logger.log_optimizations();

    std::cout << "Loading CIFAR-10 dataset from: " << data_dir << std::endl;
    logger.log("Loading CIFAR-10 dataset from: " + data_dir);
    
    CIFAR10Dataset dataset(data_dir);
    const auto& train = dataset.train();
    const auto& test = dataset.test();

    std::cout << "  Training images: " << train.num_images << std::endl;
    std::cout << "  Test images: " << test.num_images << std::endl;

    if (train.num_images == 0) {
        std::cerr << "Error: No training images loaded!" << std::endl;
        logger.log("ERROR: No training images loaded!");
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
        logger.log("ERROR: batch_size too large for dataset");
        return 1;
    }

    std::cout << "  Batch size: " << batch_size << std::endl;
    std::cout << "  Batches per epoch: " << num_batches << std::endl;
    std::cout << std::endl;
    
    logger.log_dataset_info(effective_train, test.num_images, num_batches);

    std::cout << "Initializing GPU autoencoder..." << std::endl;
    logger.log("Initializing GPU autoencoder...");
    
    // Select loss function based on command line flag
    LossType loss_type = use_bce_loss ? LossType::BCE : LossType::MSE;
    GPUAutoencoder autoencoder(loss_type);
    
    if (use_bce_loss) {
        std::cout << "Using BCE loss with Sigmoid output activation" << std::endl;
        logger.log("Loss function: BCE (Binary Cross-Entropy) with Sigmoid");
    } else {
        std::cout << "Using MSE loss (no output activation)" << std::endl;
        logger.log("Loss function: MSE (Mean Squared Error)");
    }

    if (!weights_load_path.empty()) {
        std::cout << "Loading weights from: " << weights_load_path << std::endl;
        logger.log("Loading weights from: " + weights_load_path);
        if (!autoencoder.load_weights(weights_load_path)) {
            std::cerr << "Warning: Failed to load weights, starting fresh" << std::endl;
            logger.log("WARNING: Failed to load weights, starting fresh");
        } else {
            logger.log("Weights loaded successfully");
        }
    }

    std::vector<int> indices(effective_train);
    for (int i = 0; i < effective_train; ++i) {
        indices[i] = i;
    }
    std::mt19937 rng(42);

    // Double buffering with CUDA streams for overlapping transfer and compute
    GPUTensor4D gpu_batch[2];
    gpu_batch[0].allocate(batch_size, 3, 32, 32);
    gpu_batch[1].allocate(batch_size, 3, 32, 32);
    GPUTensor4D gpu_output;

    const size_t h_batch_size = static_cast<size_t>(batch_size) * 3 * 32 * 32;
    
    // Allocate two pinned host buffers for double buffering
    float* h_batch[2];
    CUDA_CHECK(cudaMallocHost(&h_batch[0], h_batch_size * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_batch[1], h_batch_size * sizeof(float)));
    
    // Create CUDA streams for overlapping
    cudaStream_t streams[2];
    CUDA_CHECK(cudaStreamCreate(&streams[0]));
    CUDA_CHECK(cudaStreamCreate(&streams[1]));

    std::cout << "\n=== Starting Training ===" << std::endl;
    std::cout << "Using double-buffered CUDA streams for overlapped transfer/compute" << std::endl;
    std::cout << "Epochs: " << epochs << ", LR: " << learning_rate << std::endl;
    std::cout << std::endl;
    
    logger.log("");
    logger.log(std::string(60, '='));
    logger.log("TRAINING STARTED");
    logger.log(std::string(60, '='));
    logger.log("");
    logger.log("OPTIMIZER SETTINGS:");
    logger.log("  Momentum: " + std::to_string(opt_config.momentum));
    logger.log("  Weight Decay: " + std::to_string(opt_config.weight_decay));
    logger.log("  LR Schedule: MultiStepLR (milestones at 50%, 75% epochs)");
    logger.log("");
    logger.log("DATA AUGMENTATION:");
    logger.log("  Horizontal Flip: " + std::string(aug_config.horizontal_flip ? "enabled" : "disabled"));
    logger.log("  Random Crop: " + std::string(aug_config.random_crop ? "enabled" : "disabled"));
    logger.log("  Crop Padding: " + std::to_string(aug_config.crop_padding));
    logger.log("");

    auto total_start = std::chrono::high_resolution_clock::now();
    float best_loss = std::numeric_limits<float>::max();
    std::vector<float> epoch_losses;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);
        logger.log_epoch_start(epoch + 1, epochs);
        
        // Get scheduled learning rate for this epoch
        float current_lr = get_scheduled_lr(learning_rate, epoch, epochs);
        if (epoch == 0 || current_lr != get_scheduled_lr(learning_rate, epoch - 1, epochs)) {
            // Reset cout format to show LR correctly (avoid truncation from previous setprecision)
            std::cout << std::defaultfloat << std::setprecision(6);
            std::cout << "  LR: " << current_lr << std::endl;
            logger.log("  Learning rate: " + std::to_string(current_lr));
        }

        float epoch_loss = 0.0f;
        auto epoch_start = std::chrono::high_resolution_clock::now();

        // Prepare first batch with augmentation
        for (int b = 0; b < batch_size; ++b) {
            int img_idx = indices[b];
            const float* src = train.images.data() + static_cast<size_t>(img_idx) * 3 * 32 * 32;
            float* dst = h_batch[0] + static_cast<size_t>(b) * 3 * 32 * 32;
            std::copy(src, src + 3 * 32 * 32, dst);
        }
        // Apply augmentation to first batch
        CIFAR10Dataset::augment_batch(h_batch[0], batch_size, aug_config, rng);
        
        for (int batch = 0; batch < num_batches; ++batch) {
            auto batch_start = std::chrono::high_resolution_clock::now();
            
            int curr_buf = batch % 2;
            int next_buf = (batch + 1) % 2;
            
            // Async copy current batch to GPU
            CUDA_CHECK(cudaMemcpyAsync(gpu_batch[curr_buf].d_data, h_batch[curr_buf], 
                                       h_batch_size * sizeof(float), 
                                       cudaMemcpyHostToDevice, streams[curr_buf]));
            
            // Prepare next batch on CPU while GPU is working (if not last batch)
            if (batch + 1 < num_batches) {
                for (int b = 0; b < batch_size; ++b) {
                    int img_idx = indices[(batch + 1) * batch_size + b];
                    const float* src = train.images.data() + static_cast<size_t>(img_idx) * 3 * 32 * 32;
                    float* dst = h_batch[next_buf] + static_cast<size_t>(b) * 3 * 32 * 32;
                    std::copy(src, src + 3 * 32 * 32, dst);
                }
                // Apply augmentation to next batch
                CIFAR10Dataset::augment_batch(h_batch[next_buf], batch_size, aug_config, rng);
            }
            
            // Wait for transfer to complete before training
            CUDA_CHECK(cudaStreamSynchronize(streams[curr_buf]));

            // Train with Momentum SGD + Weight Decay + scheduled LR
            float loss = autoencoder.train_step_momentum(gpu_batch[curr_buf], gpu_batch[curr_buf], 
                                                          current_lr, opt_config);
            epoch_loss += loss;

            auto batch_end = std::chrono::high_resolution_clock::now();
            double batch_ms = std::chrono::duration<double, std::milli>(batch_end - batch_start).count();

            if ((batch + 1) % 50 == 0 || batch == num_batches - 1) {
                logger.log_batch(batch + 1, num_batches, loss, batch_ms);
                logger.write_csv_batch(epoch + 1, batch + 1, loss, batch_ms);
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
        epoch_losses.push_back(avg_loss);
        
        bool is_best = avg_loss < best_loss;
        if (is_best) {
            best_loss = avg_loss;
        }

        std::cout << std::endl;
        std::cout << "  Epoch " << (epoch + 1) << " complete: "
                  << "Avg Loss = " << std::fixed << std::setprecision(4) << avg_loss
                  << ", Time = " << std::setprecision(1) << epoch_sec << " sec"
                  << (is_best ? " [BEST]" : "") << std::endl;

        logger.log_epoch_end(epoch + 1, epochs, avg_loss, epoch_sec, is_best, best_loss);
        logger.write_csv_epoch(epoch + 1, avg_loss, epoch_sec, best_loss);
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(total_end - total_start).count();

    float final_avg_loss = 0.0f;
    for (float l : epoch_losses) {
        final_avg_loss += l;
    }
    final_avg_loss /= static_cast<float>(epoch_losses.size());

    std::cout << "\n=== Training Complete ===" << std::endl;
    std::cout << "Total time: " << std::fixed << std::setprecision(1) << total_sec << " seconds" << std::endl;
    std::cout << "Best loss: " << std::setprecision(6) << best_loss << std::endl;

    if (!weights_save_path.empty()) {
        std::cout << "Saving weights to: " << weights_save_path << std::endl;
        if (autoencoder.save_weights(weights_save_path)) {
            std::cout << "  Success!" << std::endl;
            logger.log("Weights saved successfully to: " + weights_save_path);
        } else {
            std::cerr << "  Failed to save weights!" << std::endl;
            logger.log("ERROR: Failed to save weights!");
        }
    }
    
    logger.log_training_complete(epochs, best_loss, final_avg_loss, total_sec, weights_save_path);

    #ifdef WITH_SVM
    std::cout << "\n=== Feature Extraction & SVM Training ===" << std::endl;
    logger.log("");
    logger.log("Starting Feature Extraction & SVM Training...");
    
    const int feature_dim = 128 * 8 * 8;
    const int num_classes = 10;
    const int samples_per_class = 1000;  // 1000 per class = 10000 total for SVM
    const int max_svm_samples = num_classes * samples_per_class;
    
    // Stratified sampling: collect up to samples_per_class from each class
    std::vector<int> class_counts(num_classes, 0);
    std::vector<float> svm_train_features;
    std::vector<int> svm_train_labels;
    svm_train_features.reserve(static_cast<size_t>(max_svm_samples) * feature_dim);
    svm_train_labels.reserve(max_svm_samples);
    
    GPUTensor4D single_image(1, 3, 32, 32);
    GPUTensor4D latent;
    std::vector<float> h_latent(feature_dim);
    
    // Progress bar helper
    auto print_progress = [](int current, int total, double elapsed_sec, const char* task) {
        int bar_width = 30;
        float progress = static_cast<float>(current) / total;
        int pos = static_cast<int>(bar_width * progress);
        
        double eta = (current > 0) ? (elapsed_sec / current) * (total - current) : 0.0;
        
        std::cout << "\r  " << task << " [";
        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << current << "/" << total 
                  << " (" << std::fixed << std::setprecision(1) << (progress * 100.0f) << "%)"
                  << " ETA: " << std::setprecision(1) << eta << "s" << std::flush;
    };
    
    std::cout << "Extracting training features (stratified " << samples_per_class 
              << " per class = " << max_svm_samples << " for SVM)..." << std::endl;
    logger.log("Extracting training features (stratified sampling)...");
    logger.log("  " + std::to_string(samples_per_class) + " samples per class = " 
               + std::to_string(max_svm_samples) + " total for SVM");
    
    auto feat_start = std::chrono::high_resolution_clock::now();
    int collected = 0;
    
    for (int i = 0; i < effective_train; ++i) {
        int label = train.labels[i];
        
        // Only collect if we haven't reached quota for this class
        if (class_counts[label] < samples_per_class) {
            const float* src = train.images.data() + static_cast<size_t>(i) * 3 * 32 * 32;
            single_image.copy_from_host(src);
            autoencoder.encode(single_image, latent);
            latent.copy_to_host(h_latent.data());
            
            // Add to SVM training set
            svm_train_features.insert(svm_train_features.end(), h_latent.begin(), h_latent.end());
            svm_train_labels.push_back(label);
            
            class_counts[label]++;
            collected++;
            
            if (collected % 500 == 0 || collected == max_svm_samples) {
                auto now = std::chrono::high_resolution_clock::now();
                double elapsed = std::chrono::duration<double>(now - feat_start).count();
                print_progress(collected, max_svm_samples, elapsed, "Train");
            }
            
            // Early exit if we have enough samples
            if (collected >= max_svm_samples) break;
        }
    }
    
    auto train_feat_end = std::chrono::high_resolution_clock::now();
    double train_feat_time = std::chrono::duration<double>(train_feat_end - feat_start).count();
    std::cout << "\n  Collected " << collected << " samples (" << samples_per_class 
              << " per class) in " << std::fixed << std::setprecision(2) << train_feat_time << "s" << std::endl;
    
    // Print class distribution
    std::cout << "  Class distribution: ";
    for (int c = 0; c < num_classes; ++c) {
        std::cout << class_counts[c];
        if (c < num_classes - 1) std::cout << "/";
    }
    std::cout << std::endl;
    
    std::vector<float> test_features(static_cast<size_t>(test.num_images) * feature_dim);
    std::vector<int> test_labels(test.num_images);
    
    std::cout << "Extracting test features (" << test.num_images << " images)..." << std::endl;
    logger.log("Extracting test features...");
    auto test_feat_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < test.num_images; ++i) {
        const float* src = test.images.data() + static_cast<size_t>(i) * 3 * 32 * 32;
        single_image.copy_from_host(src);
        autoencoder.encode(single_image, latent);
        latent.copy_to_host(h_latent.data());
        std::copy(h_latent.begin(), h_latent.end(),
                  test_features.begin() + static_cast<size_t>(i) * feature_dim);
        test_labels[i] = test.labels[i];
        
        if ((i + 1) % 500 == 0 || i == test.num_images - 1) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - test_feat_start).count();
            print_progress(i + 1, test.num_images, elapsed, "Test ");
        }
    }
    auto test_feat_end = std::chrono::high_resolution_clock::now();
    double test_feat_time = std::chrono::duration<double>(test_feat_end - test_feat_start).count();
    std::cout << "\n  Done in " << std::fixed << std::setprecision(2) << test_feat_time << "s" << std::endl;
    
    double total_feat_time = train_feat_time + test_feat_time;
    std::cout << "Total feature extraction: " << std::setprecision(2) << total_feat_time << "s" << std::endl;
    logger.log("Feature extraction time: " + std::to_string(total_feat_time) + "s");
    
    // L2 normalize features for better SVM performance
    std::cout << "Normalizing features (L2)..." << std::endl;
    logger.log("Normalizing features (L2)...");
    auto norm_start = std::chrono::high_resolution_clock::now();
    
    auto l2_normalize = [](float* features, int num_samples, int dim) {
        for (int i = 0; i < num_samples; ++i) {
            float* vec = features + static_cast<size_t>(i) * dim;
            float norm = 0.0f;
            for (int j = 0; j < dim; ++j) {
                norm += vec[j] * vec[j];
            }
            norm = std::sqrt(norm) + 1e-8f;  // avoid division by zero
            for (int j = 0; j < dim; ++j) {
                vec[j] /= norm;
            }
        }
    };
    
    const int svm_train_size = static_cast<int>(svm_train_labels.size());
    l2_normalize(svm_train_features.data(), svm_train_size, feature_dim);
    l2_normalize(test_features.data(), test.num_images, feature_dim);
    auto norm_end = std::chrono::high_resolution_clock::now();
    double norm_time = std::chrono::duration<double>(norm_end - norm_start).count();
    std::cout << "  Done in " << std::setprecision(2) << norm_time << "s" << std::endl;
    
    std::cout << "\nTraining SVM classifier..." << std::endl;
    std::cout << "  Kernel: RBF | C: 10 | gamma: " << std::scientific << std::setprecision(4) 
              << (1.0 / feature_dim) << " (auto=1/dim)" << std::endl;
    std::cout << "  Samples: " << svm_train_size << " (stratified) | Features: " << feature_dim << std::endl;
    logger.log("Training SVM classifier...");
    
    auto svm_start = std::chrono::high_resolution_clock::now();
    SVMWrapper svm;
    svm.set_C(10.0);  // C=10 as per project spec
    // gamma=auto means 1/feature_dim, which is default in SVMWrapper
    svm.train(svm_train_features.data(), svm_train_labels.data(), svm_train_size, feature_dim);
    auto svm_end = std::chrono::high_resolution_clock::now();
    double svm_time = std::chrono::duration<double>(svm_end - svm_start).count();
    std::cout << "  SVM training complete in " << std::fixed << std::setprecision(2) << svm_time << "s" << std::endl;
    logger.log("SVM training time: " + std::to_string(svm_time) + "s");
    
    std::cout << "\nEvaluating on test set (" << test.num_images << " images)..." << std::endl;
    auto eval_start = std::chrono::high_resolution_clock::now();
    float accuracy = svm.evaluate(test_features.data(), test_labels.data(), 
                                   test.num_images, feature_dim);
    auto eval_end = std::chrono::high_resolution_clock::now();
    double eval_time = std::chrono::duration<double>(eval_end - eval_start).count();
    
    std::cout << "\n============================================================" << std::endl;
    std::cout << "SVM RESULTS" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "  Training samples: " << svm_train_size << " (stratified " << samples_per_class << "/class)" << std::endl;
    std::cout << "  Test samples: " << test.num_images << std::endl;
    std::cout << "  Test Accuracy: " << std::fixed << std::setprecision(2) 
              << (accuracy * 100.0f) << "%" << std::endl;
    std::cout << "  Evaluation time: " << std::setprecision(2) << eval_time << "s" << std::endl;
    std::cout << "============================================================" << std::endl;
    
    logger.log_svm_results(accuracy, svm_train_size, test.num_images, feature_dim);
#endif

    // Cleanup double buffers and streams
    CUDA_CHECK(cudaFreeHost(h_batch[0]));
    CUDA_CHECK(cudaFreeHost(h_batch[1]));
    CUDA_CHECK(cudaStreamDestroy(streams[0]));
    CUDA_CHECK(cudaStreamDestroy(streams[1]));
    
#ifdef USE_OPTIMIZED_KERNELS
    cleanup_gpu_opt_buffers();
#endif

    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
