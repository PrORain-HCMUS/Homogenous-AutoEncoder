#include "svm_wrapper.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

// CUDA error check macro
#ifndef CUDA_CHECK_SVM
#define CUDA_CHECK_SVM(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
    } \
} while(0)
#endif

// ============== GPU KNN Kernels ==============

// Compute pairwise L2 distances: dist[i,j] = ||test[i] - train[j]||^2
__global__ void compute_distances_kernel(
    const float* __restrict__ test_features,   // [num_test x dim]
    const float* __restrict__ train_features,  // [num_train x dim]
    float* __restrict__ distances,             // [num_test x num_train]
    int num_test, int num_train, int dim
) {
    int test_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int train_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (test_idx >= num_test || train_idx >= num_train) return;
    
    const float* test_vec = test_features + static_cast<size_t>(test_idx) * dim;
    const float* train_vec = train_features + static_cast<size_t>(train_idx) * dim;
    
    float dist = 0.0f;
    for (int d = 0; d < dim; ++d) {
        float diff = test_vec[d] - train_vec[d];
        dist += diff * diff;
    }
    
    distances[static_cast<size_t>(test_idx) * num_train + train_idx] = dist;
}

// Find K nearest neighbors and vote for class
__global__ void knn_vote_kernel(
    const float* __restrict__ distances,  // [num_test x num_train]
    const int* __restrict__ train_labels, // [num_train]
    int* __restrict__ predictions,        // [num_test]
    int num_test, int num_train, int K, int num_classes
) {
    int test_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (test_idx >= num_test) return;
    
    const float* dists = distances + static_cast<size_t>(test_idx) * num_train;
    
    // Simple selection of K nearest (for small K, this is efficient)
    // Store (distance, train_idx) pairs
    extern __shared__ float shared_mem[];
    float* min_dists = shared_mem;  // K floats per thread
    int* min_indices = (int*)(min_dists + K * blockDim.x);  // K ints per thread
    
    float* my_dists = min_dists + threadIdx.x * K;
    int* my_indices = min_indices + threadIdx.x * K;
    
    // Initialize with large values
    for (int k = 0; k < K; ++k) {
        my_dists[k] = 1e30f;
        my_indices[k] = -1;
    }
    
    // Find K nearest
    for (int j = 0; j < num_train; ++j) {
        float d = dists[j];
        // Insert into sorted list if smaller than max
        if (d < my_dists[K-1]) {
            my_dists[K-1] = d;
            my_indices[K-1] = j;
            // Bubble sort to maintain sorted order
            for (int k = K-2; k >= 0; --k) {
                if (my_dists[k+1] < my_dists[k]) {
                    float tmp_d = my_dists[k];
                    int tmp_i = my_indices[k];
                    my_dists[k] = my_dists[k+1];
                    my_indices[k] = my_indices[k+1];
                    my_dists[k+1] = tmp_d;
                    my_indices[k+1] = tmp_i;
                } else break;
            }
        }
    }
    
    // Vote for class
    int votes[10] = {0};  // CIFAR-10 has 10 classes
    for (int k = 0; k < K; ++k) {
        if (my_indices[k] >= 0) {
            int label = train_labels[my_indices[k]];
            if (label >= 0 && label < num_classes) {
                votes[label]++;
            }
        }
    }
    
    // Find majority
    int best_class = 0;
    int best_votes = votes[0];
    for (int c = 1; c < num_classes; ++c) {
        if (votes[c] > best_votes) {
            best_votes = votes[c];
            best_class = c;
        }
    }
    
    predictions[test_idx] = best_class;
}

// ============== SVMWrapper Implementation (GPU KNN) ==============

struct SVMWrapper::Impl {
    double C = 10.0;
    double gamma = 0.0;
    int kernel_type = 2;
    int K = 5;  // Number of neighbors
    int num_classes = 10;
    
    // Training data stored on GPU
    float* d_train_features = nullptr;
    int* d_train_labels = nullptr;
    int num_train = 0;
    int feature_dim = 0;
    bool trained = false;
    
    ~Impl() {
        if (d_train_features) cudaFree(d_train_features);
        if (d_train_labels) cudaFree(d_train_labels);
    }
};

SVMWrapper::SVMWrapper() : impl_(new Impl()) {}

SVMWrapper::~SVMWrapper() {
    delete impl_;
}

void SVMWrapper::set_kernel(int kernel_type) {
    impl_->kernel_type = kernel_type;
}

void SVMWrapper::set_C(double C) {
    impl_->C = C;
    // Map C to K: higher C = more precise = more neighbors
    if (C >= 100) impl_->K = 11;
    else if (C >= 10) impl_->K = 7;
    else impl_->K = 5;
}

void SVMWrapper::set_gamma(double gamma) {
    impl_->gamma = gamma;
}

void SVMWrapper::train(const float* features, const int* labels,
                       int num_samples, int feature_dim) {
    std::cout << "Training GPU-KNN classifier on " << num_samples << " samples, "
              << feature_dim << " features, K=" << impl_->K << "..." << std::endl;
    
    impl_->num_train = num_samples;
    impl_->feature_dim = feature_dim;
    
    // Free old data
    if (impl_->d_train_features) cudaFree(impl_->d_train_features);
    if (impl_->d_train_labels) cudaFree(impl_->d_train_labels);
    
    // Allocate and copy training data to GPU
    size_t feat_bytes = static_cast<size_t>(num_samples) * feature_dim * sizeof(float);
    size_t label_bytes = num_samples * sizeof(int);
    
    CUDA_CHECK_SVM(cudaMalloc(&impl_->d_train_features, feat_bytes));
    CUDA_CHECK_SVM(cudaMalloc(&impl_->d_train_labels, label_bytes));
    CUDA_CHECK_SVM(cudaMemcpy(impl_->d_train_features, features, feat_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK_SVM(cudaMemcpy(impl_->d_train_labels, labels, label_bytes, cudaMemcpyHostToDevice));
    
    impl_->trained = true;
    std::cout << "  GPU-KNN training complete (data stored on GPU)" << std::endl;
}

std::vector<int> SVMWrapper::predict(const float* features, int num_samples,
                                      int feature_dim) const {
    std::vector<int> predictions(num_samples, -1);
    
    if (!impl_->trained) {
        std::cerr << "Error: Model not trained!" << std::endl;
        return predictions;
    }
    
    // Allocate GPU memory for test features and results
    float* d_test_features = nullptr;
    float* d_distances = nullptr;
    int* d_predictions = nullptr;
    
    size_t test_feat_bytes = static_cast<size_t>(num_samples) * feature_dim * sizeof(float);
    size_t dist_bytes = static_cast<size_t>(num_samples) * impl_->num_train * sizeof(float);
    
    CUDA_CHECK_SVM(cudaMalloc(&d_test_features, test_feat_bytes));
    CUDA_CHECK_SVM(cudaMalloc(&d_distances, dist_bytes));
    CUDA_CHECK_SVM(cudaMalloc(&d_predictions, num_samples * sizeof(int)));
    
    CUDA_CHECK_SVM(cudaMemcpy(d_test_features, features, test_feat_bytes, cudaMemcpyHostToDevice));
    
    // Compute pairwise distances
    dim3 block_dist(16, 16);
    dim3 grid_dist((num_samples + 15) / 16, (impl_->num_train + 15) / 16);
    compute_distances_kernel<<<grid_dist, block_dist>>>(
        d_test_features, impl_->d_train_features, d_distances,
        num_samples, impl_->num_train, feature_dim
    );
    CUDA_CHECK_SVM(cudaGetLastError());
    
    // KNN voting
    int block_knn = 64;
    int grid_knn = (num_samples + block_knn - 1) / block_knn;
    size_t shared_mem = block_knn * impl_->K * (sizeof(float) + sizeof(int));
    knn_vote_kernel<<<grid_knn, block_knn, shared_mem>>>(
        d_distances, impl_->d_train_labels, d_predictions,
        num_samples, impl_->num_train, impl_->K, impl_->num_classes
    );
    CUDA_CHECK_SVM(cudaGetLastError());
    
    // Copy predictions back
    CUDA_CHECK_SVM(cudaMemcpy(predictions.data(), d_predictions, num_samples * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Cleanup
    cudaFree(d_test_features);
    cudaFree(d_distances);
    cudaFree(d_predictions);
    
    return predictions;
}

float SVMWrapper::evaluate(const float* features, const int* labels,
                           int num_samples, int feature_dim) const {
    std::vector<int> predictions = predict(features, num_samples, feature_dim);
    
    int correct = 0;
    for (int i = 0; i < num_samples; ++i) {
        if (predictions[i] == labels[i]) {
            ++correct;
        }
    }
    
    return static_cast<float>(correct) / num_samples;
}

bool SVMWrapper::save_model(const std::string& path) const {
    if (!impl_->trained) return false;
    
    // Save training data to file
    std::vector<float> h_features(static_cast<size_t>(impl_->num_train) * impl_->feature_dim);
    std::vector<int> h_labels(impl_->num_train);
    
    CUDA_CHECK_SVM(cudaMemcpy(h_features.data(), impl_->d_train_features,
                              h_features.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK_SVM(cudaMemcpy(h_labels.data(), impl_->d_train_labels,
                              h_labels.size() * sizeof(int), cudaMemcpyDeviceToHost));
    
    std::ofstream file(path, std::ios::binary);
    if (!file) return false;
    
    file.write(reinterpret_cast<const char*>(&impl_->num_train), sizeof(int));
    file.write(reinterpret_cast<const char*>(&impl_->feature_dim), sizeof(int));
    file.write(reinterpret_cast<const char*>(&impl_->K), sizeof(int));
    file.write(reinterpret_cast<const char*>(h_features.data()), h_features.size() * sizeof(float));
    file.write(reinterpret_cast<const char*>(h_labels.data()), h_labels.size() * sizeof(int));
    
    std::cout << "GPU-KNN model saved to: " << path << std::endl;
    return true;
}

bool SVMWrapper::load_model(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    
    file.read(reinterpret_cast<char*>(&impl_->num_train), sizeof(int));
    file.read(reinterpret_cast<char*>(&impl_->feature_dim), sizeof(int));
    file.read(reinterpret_cast<char*>(&impl_->K), sizeof(int));
    
    std::vector<float> h_features(static_cast<size_t>(impl_->num_train) * impl_->feature_dim);
    std::vector<int> h_labels(impl_->num_train);
    
    file.read(reinterpret_cast<char*>(h_features.data()), h_features.size() * sizeof(float));
    file.read(reinterpret_cast<char*>(h_labels.data()), h_labels.size() * sizeof(int));
    
    // Allocate and copy to GPU
    if (impl_->d_train_features) cudaFree(impl_->d_train_features);
    if (impl_->d_train_labels) cudaFree(impl_->d_train_labels);
    
    CUDA_CHECK_SVM(cudaMalloc(&impl_->d_train_features, h_features.size() * sizeof(float)));
    CUDA_CHECK_SVM(cudaMalloc(&impl_->d_train_labels, h_labels.size() * sizeof(int)));
    CUDA_CHECK_SVM(cudaMemcpy(impl_->d_train_features, h_features.data(),
                              h_features.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_SVM(cudaMemcpy(impl_->d_train_labels, h_labels.data(),
                              h_labels.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    impl_->trained = true;
    std::cout << "GPU-KNN model loaded from: " << path << std::endl;
    return true;
}

// ============== Confusion Matrix (unchanged) ==============

void print_confusion_matrix(const std::vector<int>& true_labels,
                            const std::vector<int>& pred_labels,
                            int num_classes) {
    if (true_labels.size() != pred_labels.size()) {
        std::cerr << "Error: label vectors have different sizes" << std::endl;
        return;
    }
    
    std::vector<std::vector<int>> matrix(num_classes, std::vector<int>(num_classes, 0));
    
    for (size_t i = 0; i < true_labels.size(); ++i) {
        int t = true_labels[i];
        int p = pred_labels[i];
        if (t >= 0 && t < num_classes && p >= 0 && p < num_classes) {
            matrix[t][p]++;
        }
    }
    
    std::cout << "\nConfusion Matrix:\n";
    std::cout << "True\\Pred ";
    for (int j = 0; j < num_classes; ++j) {
        std::cout << std::setw(5) << j;
    }
    std::cout << "\n";
    
    for (int i = 0; i < num_classes; ++i) {
        std::cout << std::setw(9) << i << " ";
        for (int j = 0; j < num_classes; ++j) {
            std::cout << std::setw(5) << matrix[i][j];
        }
        std::cout << "\n";
    }
    
    std::cout << "\nPer-class accuracy:\n";
    for (int i = 0; i < num_classes; ++i) {
        int total = 0;
        for (int j = 0; j < num_classes; ++j) {
            total += matrix[i][j];
        }
        float acc = (total > 0) ? static_cast<float>(matrix[i][i]) / total : 0.0f;
        std::cout << "  Class " << i << ": " << std::fixed << std::setprecision(1) 
                  << (acc * 100) << "%" << std::endl;
    }
}
