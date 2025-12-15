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

float get_scheduled_lr(float base_lr, int epoch, int total_epochs) {
    const int warmup_epochs = 5;
    if (epoch < warmup_epochs) {
        // Linear warmup
        return base_lr * static_cast<float>(epoch + 1) / static_cast<float>(warmup_epochs);
    }
    // Cosine annealing after warmup
    const float min_lr = base_lr * 0.001f;
    int adjusted_epoch = epoch - warmup_epochs;
    int adjusted_total = total_epochs - warmup_epochs;
    if (adjusted_total <= 0) return base_lr;
    float cos_val = cosf(3.14159265f * static_cast<float>(adjusted_epoch) / static_cast<float>(adjusted_total));
    return min_lr + 0.5f * (base_lr - min_lr) * (1.0f + cos_val);
}

std::string get_timestamp_gpu() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

class GPUTrainingLogger {
    std::ofstream txt_file, csv_file;
    std::string txt_path, csv_path;
public:
    GPUTrainingLogger(const std::string& txt_log_path, const std::string& csv_log_path) : txt_path(txt_log_path), csv_path(csv_log_path) {
        if (!txt_path.empty()) { txt_file.open(txt_path, std::ios::out); }
        if (!csv_path.empty()) { csv_file.open(csv_path, std::ios::out); if (csv_file) csv_file << "epoch,batch,loss,epoch_time_sec,batch_time_ms,best_loss\n"; }
    }
    ~GPUTrainingLogger() { if (txt_file.is_open()) txt_file.close(); if (csv_file.is_open()) csv_file.close(); }
    void log(const std::string& message) { if (txt_file.is_open()) { txt_file << "[" << get_timestamp_gpu() << "] " << message << "\n"; txt_file.flush(); } }
    void log_config(int epochs, int batch_size, float lr, const std::string& data_dir, int max_images, const std::string& load_weights, const std::string& save_weights) {
        log("GPU AUTOENCODER TRAINING LOG"); log("Data: " + data_dir + ", Epochs: " + std::to_string(epochs) + ", Batch: " + std::to_string(batch_size) + ", LR: " + std::to_string(lr));
        log("Max images: " + (max_images > 0 ? std::to_string(max_images) : "all") + ", Load: " + (load_weights.empty() ? "none" : load_weights) + ", Save: " + save_weights);
    }
    void log_gpu_info(const std::string& gpu_name, int major, int minor, size_t mem_mb, int sm_count, int max_threads) {
        log("GPU: " + gpu_name + " (SM " + std::to_string(major) + "." + std::to_string(minor) + "), " + std::to_string(mem_mb) + " MB, " + std::to_string(sm_count) + " SMs");
    }
    void log_optimizations() { log("CUDA optimizations: tiled conv, warp shuffle, vectorized ops"); }
    void log_dataset_info(int train, int test, int batches) { log("Dataset: " + std::to_string(train) + " train, " + std::to_string(test) + " test, " + std::to_string(batches) + " batches/epoch"); }
    void log_epoch_start(int epoch, int total) { log("--- Epoch " + std::to_string(epoch) + "/" + std::to_string(total) + " ---"); }
    void log_batch(int batch, int total, float loss, double ms) { std::stringstream ss; ss << "  Batch " << batch << "/" << total << " | Loss: " << std::fixed << std::setprecision(6) << loss << " | " << std::setprecision(2) << ms << " ms"; log(ss.str()); }
    void log_epoch_end(int epoch, int total, float avg_loss, double sec, bool is_best, float best_loss) {
        std::stringstream ss; ss << "Epoch " << epoch << "/" << total << " done: Loss=" << std::fixed << std::setprecision(6) << avg_loss << ", Time=" << std::setprecision(2) << sec << "s" << (is_best ? " [BEST]" : "");
        log(ss.str()); ss.str(""); ss << "Best: " << std::fixed << std::setprecision(6) << best_loss; log(ss.str());
    }
    void write_csv_batch(int epoch, int batch, float loss, double ms) { if (csv_file.is_open()) csv_file << epoch << "," << batch << "," << loss << ",," << ms << ",\n"; }
    void write_csv_epoch(int epoch, float avg_loss, double sec, float best) { if (csv_file.is_open()) csv_file << epoch << ",," << avg_loss << "," << sec << ",," << best << "\n"; }
    void log_training_complete(int epochs, float best_loss, float avg_loss, double total_time, const std::string& weights_path) {
        std::stringstream ss; ss << "TRAINING COMPLETE: " << epochs << " epochs, Best=" << std::fixed << std::setprecision(6) << best_loss << ", Avg=" << avg_loss << ", Time=" << std::setprecision(2) << total_time << "s";
        log(ss.str()); log("Weights: " + weights_path);
    }
    void log_svm_results(float accuracy, int train_samples, int test_samples, int feature_dim) {
        std::stringstream ss; ss << "SVM: Train=" << train_samples << ", Test=" << test_samples << ", Dim=" << feature_dim << ", Accuracy=" << std::fixed << std::setprecision(2) << (accuracy * 100.0f) << "%";
        log(ss.str());
    }
};

// ============== GPU L2 Normalization ==============
__global__ void l2_normalize_kernel(float* features, int num_samples, int dim) {
    int sample_idx = blockIdx.x;
    if (sample_idx >= num_samples) return;
    
    float* vec = features + static_cast<size_t>(sample_idx) * dim;
    
    // Compute norm using shared memory reduction
    extern __shared__ float sdata[];
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        local_sum += vec[j] * vec[j];
    }
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    
    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    
    float norm = sqrtf(sdata[0]) + 1e-8f;
    
    // Normalize
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        vec[j] /= norm;
    }
}

void gpu_l2_normalize(float* d_features, int num_samples, int dim) {
    int block_size = 256;
    l2_normalize_kernel<<<num_samples, block_size, block_size * sizeof(float)>>>(d_features, num_samples, dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============== Batch Feature Extraction ==============
void extract_features_batch(GPUAutoencoder& autoencoder, const float* images, float* features,
                           int num_images, int batch_size, int feature_dim, float* h_batch, GPUTensor4D& gpu_batch, GPUTensor4D& latent) {
    const size_t image_size = 3 * 32 * 32;
    
    for (int start = 0; start < num_images; start += batch_size) {
        int current_batch = std::min(batch_size, num_images - start);
        
        // Copy batch to pinned memory
        for (int b = 0; b < current_batch; ++b) {
            const float* src = images + static_cast<size_t>(start + b) * image_size;
            std::copy(src, src + image_size, h_batch + static_cast<size_t>(b) * image_size);
        }
        
        // Reallocate if batch size changed (last batch)
        if (current_batch != gpu_batch.n) {
            gpu_batch.allocate(current_batch, 3, 32, 32);
        }
        
        // Copy to GPU
        CUDA_CHECK(cudaMemcpy(gpu_batch.d_data, h_batch, current_batch * image_size * sizeof(float), cudaMemcpyHostToDevice));
        
        // Encode batch
        autoencoder.encode(gpu_batch, latent);
        
        // Copy features back
        CUDA_CHECK(cudaMemcpy(features + static_cast<size_t>(start) * feature_dim, 
                             latent.d_data, current_batch * feature_dim * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Progress
        if ((start + current_batch) % 5000 == 0 || start + current_batch == num_images) {
            std::cout << "\r  Extracted: " << (start + current_batch) << "/" << num_images << " images" << std::flush;
        }
    }
    std::cout << std::endl;
}

int get_device_count() { int count = 0; CUDA_CHECK(cudaGetDeviceCount(&count)); return count; }

void print_gpu_info(int device_count) {
    if (device_count == 0) { std::cerr << "No CUDA devices found!\n"; return; }
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p, i));
        std::cout << "GPU " << i << ": " << p.name << " (SM " << p.major << "." << p.minor << ", " << (p.totalGlobalMem >> 20) << " MB, " << p.multiProcessorCount << " SMs)\n";
    }
}

int main(int argc, char** argv) {
    std::string data_dir = "data", csv_path = "gpu_phase2_log.csv", txt_path = "gpu_phase2_log.txt", weights_load_path, weights_save_path = "autoencoder_gpu.weights";
    int epochs = 20, batch_size = 64, max_train_images = 0, device_id = 0;
    float learning_rate = 1e-3f;
    bool use_bce_loss = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--bce-loss") use_bce_loss = true;
        else if (arg == "--data" && i + 1 < argc) data_dir = argv[++i];
        else if (arg == "--epochs" && i + 1 < argc) epochs = std::stoi(argv[++i]);
        else if (arg == "--batch" && i + 1 < argc) batch_size = std::stoi(argv[++i]);
        else if (arg == "--lr" && i + 1 < argc) learning_rate = std::stof(argv[++i]);
        else if (arg == "--log" && i + 1 < argc) csv_path = argv[++i];
        else if (arg == "--log-txt" && i + 1 < argc) txt_path = argv[++i];
        else if (arg == "--max-images" && i + 1 < argc) max_train_images = std::stoi(argv[++i]);
        else if (arg == "--load-weights" && i + 1 < argc) weights_load_path = argv[++i];
        else if (arg == "--save-weights" && i + 1 < argc) weights_save_path = argv[++i];
        else if (arg == "--device" && i + 1 < argc) device_id = std::stoi(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [--data <dir>] [--epochs <n>] [--batch <n>] [--lr <f>] [--log <csv>] [--log-txt <txt>] [--max-images <n>] [--load-weights <f>] [--save-weights <f>] [--device <n>] [--bce-loss]\n";
            return 0;
        }
    }

    std::cout << "=== GPU Autoencoder Training ===\n";
    int device_count = get_device_count();
    if (device_count == 0) { std::cerr << "No CUDA devices!\n"; return 1; }
    if (device_id < 0 || device_id >= device_count) { std::cerr << "Invalid device ID\n"; return 1; }
    print_gpu_info(device_count);
    std::cout << "Using GPU " << device_id << "\n";
    CUDA_CHECK(cudaSetDevice(device_id));

    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    GPUTrainingLogger logger(txt_path, csv_path);
    OptimizerConfig opt_config; opt_config.weight_decay = 1e-4f; opt_config.beta1 = 0.9f; opt_config.beta2 = 0.999f; opt_config.use_adamw = true;
    AugmentConfig aug_config; aug_config.horizontal_flip = true; aug_config.random_crop = true; aug_config.crop_padding = 4;
    logger.log_config(epochs, batch_size, learning_rate, data_dir, max_train_images, weights_load_path, weights_save_path);
    logger.log_gpu_info(prop.name, prop.major, prop.minor, prop.totalGlobalMem >> 20, prop.multiProcessorCount, prop.maxThreadsPerBlock);
    logger.log_optimizations();

    std::cout << "Loading CIFAR-10 from: " << data_dir << "\n";
    CIFAR10Dataset dataset(data_dir);
    const auto& train = dataset.train();
    const auto& test = dataset.test();
    std::cout << "  Train: " << train.num_images << ", Test: " << test.num_images << "\n";
    if (train.num_images == 0) { std::cerr << "No training images!\n"; return 1; }

    int effective_train = (max_train_images > 0 && max_train_images < train.num_images) ? max_train_images : train.num_images;
    int num_batches = effective_train / batch_size;
    if (num_batches == 0) { std::cerr << "Batch size too large!\n"; return 1; }
    std::cout << "  Batch: " << batch_size << ", Batches/epoch: " << num_batches << "\n";
    logger.log_dataset_info(effective_train, test.num_images, num_batches);

    std::cout << "Initializing autoencoder...\n";
    LossType loss_type = use_bce_loss ? LossType::BCE : LossType::MSE;
    GPUAutoencoder autoencoder(loss_type);
    std::cout << "Loss: " << (use_bce_loss ? "BCE+Sigmoid" : "MSE") << "\n";
    logger.log(std::string("Loss: ") + (use_bce_loss ? "BCE" : "MSE"));

    if (!weights_load_path.empty()) {
        std::cout << "Loading weights: " << weights_load_path << "\n";
        if (!autoencoder.load_weights(weights_load_path)) std::cerr << "Failed to load weights\n";
    }

    std::vector<int> indices(effective_train); for (int i = 0; i < effective_train; ++i) indices[i] = i;
    std::mt19937 rng(42);

    GPUTensor4D gpu_batch[2]; gpu_batch[0].allocate(batch_size, 3, 32, 32); gpu_batch[1].allocate(batch_size, 3, 32, 32);
    GPUTensor4D gpu_output;
    const size_t h_batch_size = static_cast<size_t>(batch_size) * 3 * 32 * 32;
    float* h_batch[2]; CUDA_CHECK(cudaMallocHost(&h_batch[0], h_batch_size * sizeof(float))); CUDA_CHECK(cudaMallocHost(&h_batch[1], h_batch_size * sizeof(float)));
    cudaStream_t streams[2]; CUDA_CHECK(cudaStreamCreate(&streams[0])); CUDA_CHECK(cudaStreamCreate(&streams[1]));

    std::cout << "\n=== Training ===\nEpochs: " << epochs << ", LR: " << learning_rate << "\n\n";
    logger.log("TRAINING STARTED");

    auto total_start = std::chrono::high_resolution_clock::now();
    float best_loss = std::numeric_limits<float>::max();
    std::vector<float> epoch_losses;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);
        logger.log_epoch_start(epoch + 1, epochs);
        float current_lr = get_scheduled_lr(learning_rate, epoch, epochs);
        float epoch_loss = 0.0f;
        auto epoch_start = std::chrono::high_resolution_clock::now();

        for (int b = 0; b < batch_size; ++b) {
            const float* src = train.images.data() + static_cast<size_t>(indices[b]) * 3 * 32 * 32;
            std::copy(src, src + 3 * 32 * 32, h_batch[0] + static_cast<size_t>(b) * 3 * 32 * 32);
        }
        CIFAR10Dataset::augment_batch(h_batch[0], batch_size, aug_config, rng);

        for (int batch = 0; batch < num_batches; ++batch) {
            auto batch_start = std::chrono::high_resolution_clock::now();
            int curr_buf = batch % 2, next_buf = (batch + 1) % 2;
            CUDA_CHECK(cudaMemcpyAsync(gpu_batch[curr_buf].d_data, h_batch[curr_buf], h_batch_size * sizeof(float), cudaMemcpyHostToDevice, streams[curr_buf]));

            if (batch + 1 < num_batches) {
                for (int b = 0; b < batch_size; ++b) {
                    const float* src = train.images.data() + static_cast<size_t>(indices[(batch + 1) * batch_size + b]) * 3 * 32 * 32;
                    std::copy(src, src + 3 * 32 * 32, h_batch[next_buf] + static_cast<size_t>(b) * 3 * 32 * 32);
                }
                CIFAR10Dataset::augment_batch(h_batch[next_buf], batch_size, aug_config, rng);
            }
            CUDA_CHECK(cudaStreamSynchronize(streams[curr_buf]));
            float loss = autoencoder.train_step_momentum(gpu_batch[curr_buf], gpu_batch[curr_buf], current_lr, opt_config);
            opt_config.step++;
            epoch_loss += loss;

            auto batch_end = std::chrono::high_resolution_clock::now();
            double batch_ms = std::chrono::duration<double, std::milli>(batch_end - batch_start).count();
            if ((batch + 1) % 50 == 0 || batch == num_batches - 1) { logger.log_batch(batch + 1, num_batches, loss, batch_ms); logger.write_csv_batch(epoch + 1, batch + 1, loss, batch_ms); }
            if (batch % 100 == 0 || batch == num_batches - 1) std::cout << "\r  Epoch " << (epoch + 1) << "/" << epochs << " | Batch " << (batch + 1) << "/" << num_batches << " | Loss: " << std::fixed << std::setprecision(4) << loss << " | " << std::setprecision(1) << batch_ms << " ms" << std::flush;
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        double epoch_sec = std::chrono::duration<double>(epoch_end - epoch_start).count();
        float avg_loss = epoch_loss / num_batches;
        epoch_losses.push_back(avg_loss);
        bool is_best = avg_loss < best_loss; if (is_best) best_loss = avg_loss;
        std::cout << "\n  Epoch " << (epoch + 1) << ": Avg=" << std::fixed << std::setprecision(4) << avg_loss << ", " << std::setprecision(1) << epoch_sec << "s" << (is_best ? " [BEST]" : "") << "\n";
        logger.log_epoch_end(epoch + 1, epochs, avg_loss, epoch_sec, is_best, best_loss);
        logger.write_csv_epoch(epoch + 1, avg_loss, epoch_sec, best_loss);
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_sec = std::chrono::duration<double>(total_end - total_start).count();
    float final_avg = 0.0f; for (float l : epoch_losses) final_avg += l; final_avg /= static_cast<float>(epoch_losses.size());

    std::cout << "\n=== Training Complete ===\nTime: " << std::fixed << std::setprecision(1) << total_sec << "s, Best: " << std::setprecision(6) << best_loss << "\n";
    if (!weights_save_path.empty()) { std::cout << "Saving: " << weights_save_path << "\n"; autoencoder.save_weights(weights_save_path); }
    logger.log_training_complete(epochs, best_loss, final_avg, total_sec, weights_save_path);

#ifdef WITH_SVM
    std::cout << "\n=== SVM Training (Optimized Batch Extraction) ===\n";
    const int feature_dim = 128 * 8 * 8;  // 8192 features
    const int svm_batch_size = 64;        // Batch size for feature extraction
    
    // Allocate pinned memory for batch processing
    float* h_svm_batch = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_svm_batch, static_cast<size_t>(svm_batch_size) * 3 * 32 * 32 * sizeof(float)));
    GPUTensor4D svm_gpu_batch(svm_batch_size, 3, 32, 32);
    GPUTensor4D svm_latent;
    
    // Extract ALL training features using batch processing
    std::cout << "Extracting train features (" << effective_train << " images, batch=" << svm_batch_size << ")...\n";
    std::vector<float> all_train_features(static_cast<size_t>(effective_train) * feature_dim);
    auto feat_start = std::chrono::high_resolution_clock::now();
    
    extract_features_batch(autoencoder, train.images.data(), all_train_features.data(),
                          effective_train, svm_batch_size, feature_dim, h_svm_batch, svm_gpu_batch, svm_latent);
    
    auto train_feat_end = std::chrono::high_resolution_clock::now();
    double train_feat_time = std::chrono::duration<double>(train_feat_end - feat_start).count();
    std::cout << "  Done in " << std::fixed << std::setprecision(2) << train_feat_time << "s";
    std::cout << " (" << static_cast<int>(effective_train / train_feat_time) << " images/sec)\n";
    
    // Extract test features using batch processing
    std::cout << "Extracting test features (" << test.num_images << " images)...\n";
    std::vector<float> test_features(static_cast<size_t>(test.num_images) * feature_dim);
    std::vector<int> test_labels(test.num_images);
    for (int i = 0; i < test.num_images; ++i) test_labels[i] = test.labels[i];
    
    auto test_start = std::chrono::high_resolution_clock::now();
    extract_features_batch(autoencoder, test.images.data(), test_features.data(),
                          test.num_images, svm_batch_size, feature_dim, h_svm_batch, svm_gpu_batch, svm_latent);
    
    double test_feat_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - test_start).count();
    std::cout << "  Done in " << std::fixed << std::setprecision(2) << test_feat_time << "s";
    std::cout << " (" << static_cast<int>(test.num_images / test_feat_time) << " images/sec)\n";
    
    // GPU L2 Normalization (much faster than CPU)
    std::cout << "L2 normalizing features on GPU...\n";
    float* d_train_features = nullptr;
    float* d_test_features = nullptr;
    size_t train_feat_bytes = static_cast<size_t>(effective_train) * feature_dim * sizeof(float);
    size_t test_feat_bytes = static_cast<size_t>(test.num_images) * feature_dim * sizeof(float);
    
    CUDA_CHECK(cudaMalloc(&d_train_features, train_feat_bytes));
    CUDA_CHECK(cudaMalloc(&d_test_features, test_feat_bytes));
    CUDA_CHECK(cudaMemcpy(d_train_features, all_train_features.data(), train_feat_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_test_features, test_features.data(), test_feat_bytes, cudaMemcpyHostToDevice));
    
    auto norm_start = std::chrono::high_resolution_clock::now();
    gpu_l2_normalize(d_train_features, effective_train, feature_dim);
    gpu_l2_normalize(d_test_features, test.num_images, feature_dim);
    double norm_time = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - norm_start).count();
    std::cout << "  Done in " << std::fixed << std::setprecision(4) << norm_time << "s\n";
    
    // Copy normalized features back
    CUDA_CHECK(cudaMemcpy(all_train_features.data(), d_train_features, train_feat_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(test_features.data(), d_test_features, test_feat_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_train_features));
    CUDA_CHECK(cudaFree(d_test_features));
    
    // Sample balanced training set for SVM (1000 per class = 10000 total)
    const int num_classes = 10, samples_per_class = 1000;
    std::vector<int> class_counts(num_classes, 0);
    std::vector<float> svm_train_features;
    std::vector<int> svm_train_labels;
    svm_train_features.reserve(static_cast<size_t>(num_classes * samples_per_class) * feature_dim);
    svm_train_labels.reserve(num_classes * samples_per_class);
    
    for (int i = 0; i < effective_train; ++i) {
        int label = train.labels[i];
        if (class_counts[label] < samples_per_class) {
            const float* feat = all_train_features.data() + static_cast<size_t>(i) * feature_dim;
            svm_train_features.insert(svm_train_features.end(), feat, feat + feature_dim);
            svm_train_labels.push_back(label);
            class_counts[label]++;
        }
    }
    const int svm_train_size = static_cast<int>(svm_train_labels.size());
    std::cout << "Selected " << svm_train_size << " balanced samples for SVM training\n";
    
    std::cout << "Training SVM (C=10, RBF)...\n";
    SVMWrapper svm; svm.set_C(10.0);
    svm.train(svm_train_features.data(), svm_train_labels.data(), svm_train_size, feature_dim);
    float accuracy = svm.evaluate(test_features.data(), test_labels.data(), test.num_images, feature_dim);
    std::cout << "Accuracy: " << std::fixed << std::setprecision(2) << (accuracy * 100.0f) << "%\n";
    logger.log_svm_results(accuracy, svm_train_size, test.num_images, feature_dim);
    
    // Cleanup SVM resources
    CUDA_CHECK(cudaFreeHost(h_svm_batch));
#endif

    CUDA_CHECK(cudaFreeHost(h_batch[0])); CUDA_CHECK(cudaFreeHost(h_batch[1]));
    CUDA_CHECK(cudaStreamDestroy(streams[0])); CUDA_CHECK(cudaStreamDestroy(streams[1]));
#ifdef USE_OPTIMIZED_KERNELS
    cleanup_gpu_opt_buffers();
#endif
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}
