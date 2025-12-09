// Phase 4: Feature Extraction + SVM Training
// This is a standalone executable for Phase 4 SVM integration

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
#include <cmath>

#include "dataset.h"
#include "gpu_autoencoder.h"
#include "gpu_layer.h"
#include "cuda_utils.h"
#include "svm_wrapper.h"

std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

void log_msg(std::ofstream& log_file, const std::string& msg) {
    std::cout << "[" << get_timestamp() << "] " << msg << std::endl;
    if (log_file.is_open()) {
        log_file << "[" << get_timestamp() << "] " << msg << std::endl;
        log_file.flush();
    }
}

void print_progress(int current, int total, double elapsed_sec, const char* task) {
    int bar_width = 40;
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
}

void l2_normalize(float* features, int num_samples, int dim) {
    for (int i = 0; i < num_samples; ++i) {
        float* vec = features + static_cast<size_t>(i) * dim;
        float norm = 0.0f;
        for (int j = 0; j < dim; ++j) {
            norm += vec[j] * vec[j];
        }
        norm = std::sqrt(norm) + 1e-8f;
        for (int j = 0; j < dim; ++j) {
            vec[j] /= norm;
        }
    }
}

int main(int argc, char** argv) {
    std::string data_dir = "data";
    std::string weights_path = "phase3_opt.weights";
    std::string log_path = "phase4_svm.txt";
    std::string csv_path = "phase4_results.csv";
    int max_train_samples = 0;  // 0 = use all 50000
    double svm_C = 10.0;
    double svm_gamma = 0.0;  // 0 = auto (1/feature_dim)
    bool save_features = false;
    bool extract_only = false;  // Skip SVM training, only extract features
    int device_id = 0;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (arg == "--weights" && i + 1 < argc) {
            weights_path = argv[++i];
        } else if (arg == "--log" && i + 1 < argc) {
            log_path = argv[++i];
        } else if (arg == "--csv" && i + 1 < argc) {
            csv_path = argv[++i];
        } else if (arg == "--max-train" && i + 1 < argc) {
            max_train_samples = std::stoi(argv[++i]);
        } else if (arg == "--svm-c" && i + 1 < argc) {
            svm_C = std::stod(argv[++i]);
        } else if (arg == "--svm-gamma" && i + 1 < argc) {
            svm_gamma = std::stod(argv[++i]);
        } else if (arg == "--save-features") {
            save_features = true;
        } else if (arg == "--extract-only") {
            extract_only = true;
            save_features = true;  // Automatically save features when extract-only
        } else if (arg == "--device" && i + 1 < argc) {
            device_id = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Phase 4: SVM Integration\n"
                      << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --data <dir>        CIFAR-10 data directory (default: data)\n"
                      << "  --weights <file>    Trained weights from Phase 3 (default: phase3_opt.weights)\n"
                      << "  --log <file>        Log file path (default: phase4_svm.txt)\n"
                      << "  --csv <file>        CSV results file (default: phase4_results.csv)\n"
                      << "  --max-train <n>     Max training samples for SVM (0=all 50000)\n"
                      << "  --svm-c <f>         SVM C parameter (default: 10.0)\n"
                      << "  --svm-gamma <f>     SVM gamma (0=auto=1/dim, default: 0)\n"
                      << "  --save-features     Save extracted features to files\n"
                      << "  --extract-only      Only extract features, skip SVM training\n"
                      << "  --device <n>        GPU device ID (default: 0)\n"
                      << "  --help              Show this help\n";
            return 0;
        }
    }

    std::ofstream log_file(log_path);
    
    log_msg(log_file, "============================================================");
    log_msg(log_file, "PHASE 4: SVM INTEGRATION AND ANALYSIS");
    log_msg(log_file, "============================================================");
    log_msg(log_file, "");
    
    // GPU setup
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(device_id));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    log_msg(log_file, "GPU: " + std::string(prop.name));
    log_msg(log_file, "  Memory: " + std::to_string(prop.totalGlobalMem / (1024*1024)) + " MB");
    log_msg(log_file, "");
    
    // Configuration
    log_msg(log_file, "CONFIGURATION:");
    log_msg(log_file, "  Data directory: " + data_dir);
    log_msg(log_file, "  Weights file: " + weights_path);
    log_msg(log_file, "  SVM C: " + std::to_string(svm_C));
    log_msg(log_file, "  SVM gamma: " + (svm_gamma > 0 ? std::to_string(svm_gamma) : "auto (1/dim)"));
    log_msg(log_file, "  Max train samples: " + (max_train_samples > 0 ? std::to_string(max_train_samples) : "all"));
    log_msg(log_file, "");
    
    // Load dataset
    log_msg(log_file, "Loading CIFAR-10 dataset...");
    CIFAR10Dataset dataset(data_dir);
    const auto& train = dataset.train();
    const auto& test = dataset.test();
    
    log_msg(log_file, "  Training images: " + std::to_string(train.num_images));
    log_msg(log_file, "  Test images: " + std::to_string(test.num_images));
    log_msg(log_file, "");
    
    if (train.num_images == 0) {
        log_msg(log_file, "ERROR: No training images loaded!");
        return 1;
    }
    
    // Load autoencoder weights
    log_msg(log_file, "Loading autoencoder weights from: " + weights_path);
    GPUAutoencoder autoencoder;
    if (!autoencoder.load_weights(weights_path)) {
        log_msg(log_file, "ERROR: Failed to load weights!");
        return 1;
    }
    log_msg(log_file, "  Weights loaded successfully!");
    log_msg(log_file, "");
    
    // Feature extraction parameters
    const int feature_dim = 128 * 8 * 8;  // 8192
    const int num_train = (max_train_samples > 0 && max_train_samples < train.num_images) 
                          ? max_train_samples : train.num_images;
    const int num_test = test.num_images;
    
    log_msg(log_file, "============================================================");
    log_msg(log_file, "STEP 1: FEATURE EXTRACTION");
    log_msg(log_file, "============================================================");
    log_msg(log_file, "  Feature dimension: " + std::to_string(feature_dim));
    log_msg(log_file, "  Training samples: " + std::to_string(num_train));
    log_msg(log_file, "  Test samples: " + std::to_string(num_test));
    log_msg(log_file, "");
    
    // Allocate feature buffers
    std::vector<float> train_features(static_cast<size_t>(num_train) * feature_dim);
    std::vector<int> train_labels(num_train);
    std::vector<float> test_features(static_cast<size_t>(num_test) * feature_dim);
    std::vector<int> test_labels(num_test);
    
    GPUTensor4D single_image(1, 3, 32, 32);
    GPUTensor4D latent;
    std::vector<float> h_latent(feature_dim);
    
    // Extract training features
    log_msg(log_file, "Extracting training features...");
    auto train_start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_train; ++i) {
        const float* src = train.images.data() + static_cast<size_t>(i) * 3 * 32 * 32;
        single_image.copy_from_host(src);
        autoencoder.encode(single_image, latent);
        latent.copy_to_host(h_latent.data());
        
        std::copy(h_latent.begin(), h_latent.end(),
                  train_features.begin() + static_cast<size_t>(i) * feature_dim);
        train_labels[i] = train.labels[i];
        
        if ((i + 1) % 1000 == 0 || i == num_train - 1) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - train_start).count();
            print_progress(i + 1, num_train, elapsed, "Train features");
        }
    }
    
    auto train_end = std::chrono::high_resolution_clock::now();
    double train_feat_time = std::chrono::duration<double>(train_end - train_start).count();
    std::cout << std::endl;
    log_msg(log_file, "  Training features extracted in " + std::to_string(train_feat_time) + "s");
    log_msg(log_file, "  Throughput: " + std::to_string(num_train / train_feat_time) + " images/sec");
    
    // Extract test features
    log_msg(log_file, "Extracting test features...");
    auto test_start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < num_test; ++i) {
        const float* src = test.images.data() + static_cast<size_t>(i) * 3 * 32 * 32;
        single_image.copy_from_host(src);
        autoencoder.encode(single_image, latent);
        latent.copy_to_host(h_latent.data());
        
        std::copy(h_latent.begin(), h_latent.end(),
                  test_features.begin() + static_cast<size_t>(i) * feature_dim);
        test_labels[i] = test.labels[i];
        
        if ((i + 1) % 1000 == 0 || i == num_test - 1) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - test_start).count();
            print_progress(i + 1, num_test, elapsed, "Test features ");
        }
    }
    
    auto test_end = std::chrono::high_resolution_clock::now();
    double test_feat_time = std::chrono::duration<double>(test_end - test_start).count();
    std::cout << std::endl;
    log_msg(log_file, "  Test features extracted in " + std::to_string(test_feat_time) + "s");
    
    double total_feat_time = train_feat_time + test_feat_time;
    log_msg(log_file, "");
    log_msg(log_file, "FEATURE EXTRACTION SUMMARY:");
    log_msg(log_file, "  Total time: " + std::to_string(total_feat_time) + "s");
    log_msg(log_file, "  Total images: " + std::to_string(num_train + num_test));
    log_msg(log_file, "  Average: " + std::to_string((num_train + num_test) / total_feat_time) + " images/sec");
    log_msg(log_file, "");
    
    // L2 normalize features
    log_msg(log_file, "Normalizing features (L2)...");
    auto norm_start = std::chrono::high_resolution_clock::now();
    l2_normalize(train_features.data(), num_train, feature_dim);
    l2_normalize(test_features.data(), num_test, feature_dim);
    auto norm_end = std::chrono::high_resolution_clock::now();
    double norm_time = std::chrono::duration<double>(norm_end - norm_start).count();
    log_msg(log_file, "  Normalization done in " + std::to_string(norm_time) + "s");
    log_msg(log_file, "");
    
    // Save features if requested
    if (save_features) {
        log_msg(log_file, "Saving features to files...");
        
        std::ofstream train_feat_file("train_features.bin", std::ios::binary);
        train_feat_file.write(reinterpret_cast<char*>(train_features.data()), 
                              train_features.size() * sizeof(float));
        train_feat_file.close();
        
        std::ofstream test_feat_file("test_features.bin", std::ios::binary);
        test_feat_file.write(reinterpret_cast<char*>(test_features.data()),
                             test_features.size() * sizeof(float));
        test_feat_file.close();
        
        log_msg(log_file, "  Features saved!");
        log_msg(log_file, "");
    }
    
    // Skip SVM if extract_only mode
    if (extract_only) {
        log_msg(log_file, "============================================================");
        log_msg(log_file, "EXTRACT-ONLY MODE: Skipping SVM training");
        log_msg(log_file, "============================================================");
        log_msg(log_file, "Features saved to:");
        log_msg(log_file, "  - train_features.bin (" + std::to_string(num_train) + " x " + std::to_string(feature_dim) + ")");
        log_msg(log_file, "  - test_features.bin (" + std::to_string(num_test) + " x " + std::to_string(feature_dim) + ")");
        log_msg(log_file, "");
        log_msg(log_file, "Use ThunderSVM or other SVM library to train on these features.");
        log_msg(log_file, "============================================================");
        
        // Cleanup and exit
#ifdef USE_OPTIMIZED_KERNELS
        cleanup_gpu_opt_buffers();
#endif
        CUDA_CHECK(cudaDeviceReset());
        return 0;
    }
    
    // SVM Training
    log_msg(log_file, "============================================================");
    log_msg(log_file, "STEP 2: SVM TRAINING");
    log_msg(log_file, "============================================================");
    
    double effective_gamma = (svm_gamma > 0) ? svm_gamma : (1.0 / feature_dim);
    
    log_msg(log_file, "SVM Parameters:");
    log_msg(log_file, "  Kernel: RBF (Radial Basis Function)");
    log_msg(log_file, "  C: " + std::to_string(svm_C));
    std::stringstream gamma_ss;
    gamma_ss << std::scientific << std::setprecision(6) << effective_gamma;
    log_msg(log_file, "  gamma: " + gamma_ss.str() + (svm_gamma <= 0 ? " (auto=1/dim)" : ""));
    log_msg(log_file, "  Training samples: " + std::to_string(num_train));
    log_msg(log_file, "  Feature dimension: " + std::to_string(feature_dim));
    log_msg(log_file, "");
    
    log_msg(log_file, "Training SVM classifier...");
    auto svm_start = std::chrono::high_resolution_clock::now();
    
    SVMWrapper svm;
    svm.set_C(svm_C);
    if (svm_gamma > 0) {
        svm.set_gamma(svm_gamma);
    }
    svm.train(train_features.data(), train_labels.data(), num_train, feature_dim);
    
    auto svm_end = std::chrono::high_resolution_clock::now();
    double svm_time = std::chrono::duration<double>(svm_end - svm_start).count();
    log_msg(log_file, "  SVM training completed in " + std::to_string(svm_time) + "s");
    log_msg(log_file, "");
    
    // Save SVM model
    std::string svm_model_path = "phase4_svm.model";
    if (svm.save_model(svm_model_path)) {
        log_msg(log_file, "  SVM model saved to: " + svm_model_path);
    }
    log_msg(log_file, "");
    
    // Evaluation
    log_msg(log_file, "============================================================");
    log_msg(log_file, "STEP 3: EVALUATION");
    log_msg(log_file, "============================================================");
    
    log_msg(log_file, "Evaluating on test set...");
    auto eval_start = std::chrono::high_resolution_clock::now();
    
    std::vector<int> predictions = svm.predict(test_features.data(), num_test, feature_dim);
    float accuracy = svm.evaluate(test_features.data(), test_labels.data(), num_test, feature_dim);
    
    auto eval_end = std::chrono::high_resolution_clock::now();
    double eval_time = std::chrono::duration<double>(eval_end - eval_start).count();
    log_msg(log_file, "  Evaluation completed in " + std::to_string(eval_time) + "s");
    log_msg(log_file, "");
    
    // Print confusion matrix
    print_confusion_matrix(std::vector<int>(test_labels.begin(), test_labels.end()), 
                           predictions, 10);
    
    // Final results
    log_msg(log_file, "");
    log_msg(log_file, "============================================================");
    log_msg(log_file, "FINAL RESULTS");
    log_msg(log_file, "============================================================");
    log_msg(log_file, "");
    
    std::stringstream acc_ss;
    acc_ss << std::fixed << std::setprecision(2) << (accuracy * 100.0f);
    log_msg(log_file, "TEST ACCURACY: " + acc_ss.str() + "%");
    log_msg(log_file, "");
    log_msg(log_file, "TIMING BREAKDOWN:");
    log_msg(log_file, "  Feature extraction (train): " + std::to_string(train_feat_time) + "s");
    log_msg(log_file, "  Feature extraction (test):  " + std::to_string(test_feat_time) + "s");
    log_msg(log_file, "  Feature normalization:      " + std::to_string(norm_time) + "s");
    log_msg(log_file, "  SVM training:               " + std::to_string(svm_time) + "s");
    log_msg(log_file, "  SVM evaluation:             " + std::to_string(eval_time) + "s");
    
    double total_time = total_feat_time + norm_time + svm_time + eval_time;
    log_msg(log_file, "  ----------------------------------------");
    log_msg(log_file, "  TOTAL:                      " + std::to_string(total_time) + "s");
    log_msg(log_file, "");
    
    // Check target
    bool target_met = (accuracy >= 0.60f && accuracy <= 0.65f);
    log_msg(log_file, "TARGET CHECK:");
    log_msg(log_file, "  Expected accuracy: 60-65%");
    log_msg(log_file, "  Achieved accuracy: " + acc_ss.str() + "%");
    log_msg(log_file, "  Status: " + std::string(target_met ? "TARGET MET!" : 
                      (accuracy < 0.60f ? "Below target" : "Above target (excellent!)")));
    log_msg(log_file, "");
    log_msg(log_file, "============================================================");
    
    // Write CSV results
    std::ofstream csv_file(csv_path);
    if (csv_file.is_open()) {
        csv_file << "metric,value" << std::endl;
        csv_file << "test_accuracy," << accuracy << std::endl;
        csv_file << "train_samples," << num_train << std::endl;
        csv_file << "test_samples," << num_test << std::endl;
        csv_file << "feature_dim," << feature_dim << std::endl;
        csv_file << "svm_c," << svm_C << std::endl;
        csv_file << "svm_gamma," << effective_gamma << std::endl;
        csv_file << "feature_extraction_time," << total_feat_time << std::endl;
        csv_file << "svm_training_time," << svm_time << std::endl;
        csv_file << "evaluation_time," << eval_time << std::endl;
        csv_file << "total_time," << total_time << std::endl;
        csv_file.close();
        log_msg(log_file, "Results saved to: " + csv_path);
    }
    
    // Cleanup
#ifdef USE_OPTIMIZED_KERNELS
    cleanup_gpu_opt_buffers();
#endif
    CUDA_CHECK(cudaDeviceReset());
    
    return 0;
}
