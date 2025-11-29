#include "svm_wrapper.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>

#ifdef WITH_LIBSVM
#include "svm.h"  // LIBSVM header

struct SVMWrapper::Impl {
    svm_model* model = nullptr;
    svm_parameter param;
    double gamma_value = 0.0;
    
    Impl() {
        param.svm_type = C_SVC;
        param.kernel_type = RBF;
        param.degree = 3;
        param.gamma = 0;
        param.coef0 = 0;
        param.C = 10.0;
        param.nu = 0.5;
        param.eps = 1e-3;
        param.cache_size = 200;
        param.shrinking = 1;
        param.probability = 0;
        param.nr_weight = 0;
        param.weight_label = nullptr;
        param.weight = nullptr;
    }
    
    ~Impl() {
        if (model) {
            svm_free_and_destroy_model(&model);
        }
    }
};

SVMWrapper::SVMWrapper() : impl_(new Impl()) {}

SVMWrapper::~SVMWrapper() {
    delete impl_;
}

void SVMWrapper::set_kernel(int kernel_type) {
    impl_->param.kernel_type = kernel_type;
}

void SVMWrapper::set_C(double C) {
    impl_->param.C = C;
}

void SVMWrapper::set_gamma(double gamma) {
    impl_->gamma_value = gamma;
}

void SVMWrapper::train(const float* features, const int* labels,
                       int num_samples, int feature_dim) {
    std::cout << "Training SVM on " << num_samples << " samples, "
              << feature_dim << " features..." << std::endl;
    
    if (impl_->gamma_value <= 0) {
        impl_->param.gamma = 1.0 / feature_dim;
    } else {
        impl_->param.gamma = impl_->gamma_value;
    }
    
    svm_problem prob;
    prob.l = num_samples;
    prob.y = new double[num_samples];
    prob.x = new svm_node*[num_samples];
    
    for (int i = 0; i < num_samples; ++i) {
        prob.y[i] = static_cast<double>(labels[i]);
        
        prob.x[i] = new svm_node[feature_dim + 1];
        
        const float* sample = features + static_cast<size_t>(i) * feature_dim;
        for (int j = 0; j < feature_dim; ++j) {
            prob.x[i][j].index = j + 1;
            prob.x[i][j].value = static_cast<double>(sample[j]);
        }
        prob.x[i][feature_dim].index = -1;
    }
    
    const char* error_msg = svm_check_parameter(&prob, &impl_->param);
    if (error_msg) {
        std::cerr << "SVM parameter error: " << error_msg << std::endl;
        for (int i = 0; i < num_samples; ++i) delete[] prob.x[i];
        delete[] prob.x;
        delete[] prob.y;
        return;
    }
    
    impl_->model = svm_train(&prob, &impl_->param);
    
    if (impl_->model) {
        std::cout << "SVM training complete. Support vectors: "
                  << impl_->model->l << std::endl;
    }
    
    for (int i = 0; i < num_samples; ++i) {
        delete[] prob.x[i];
    }
    delete[] prob.x;
    delete[] prob.y;
}

std::vector<int> SVMWrapper::predict(const float* features, int num_samples,
                                      int feature_dim) const {
    std::vector<int> predictions(num_samples, -1);
    
    if (!impl_->model) {
        std::cerr << "Error: SVM model not trained!" << std::endl;
        return predictions;
    }
    
    std::vector<svm_node> x(feature_dim + 1);
    
    for (int i = 0; i < num_samples; ++i) {
        const float* sample = features + static_cast<size_t>(i) * feature_dim;
        
        for (int j = 0; j < feature_dim; ++j) {
            x[j].index = j + 1;
            x[j].value = static_cast<double>(sample[j]);
        }
        x[feature_dim].index = -1;
        
        double pred = svm_predict(impl_->model, x.data());
        predictions[i] = static_cast<int>(pred);
    }
    
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
    
    float accuracy = static_cast<float>(correct) / num_samples;
    return accuracy;
}

bool SVMWrapper::save_model(const std::string& path) const {
    if (!impl_->model) return false;
    return svm_save_model(path.c_str(), impl_->model) == 0;
}

bool SVMWrapper::load_model(const std::string& path) {
    if (impl_->model) {
        svm_free_and_destroy_model(&impl_->model);
    }
    impl_->model = svm_load_model(path.c_str());
    return impl_->model != nullptr;
}

#else

struct SVMWrapper::Impl {
    double C = 10.0;
    double gamma = 0.0;
    int kernel_type = 2;
    bool trained = false;
    
    std::vector<std::vector<float>> class_centroids;
    int num_classes = 10;
    int feature_dim = 0;
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
}

void SVMWrapper::set_gamma(double gamma) {
    impl_->gamma = gamma;
}

void SVMWrapper::train(const float* features, const int* labels,
                       int num_samples, int feature_dim) {
    std::cout << "[STUB] SVM training (LIBSVM not available)" << std::endl;
    std::cout << "  Using nearest-centroid classifier as fallback" << std::endl;
    
    impl_->feature_dim = feature_dim;
    impl_->class_centroids.resize(impl_->num_classes);
    
    std::vector<int> class_counts(impl_->num_classes, 0);
    for (int c = 0; c < impl_->num_classes; ++c) {
        impl_->class_centroids[c].resize(feature_dim, 0.0f);
    }
    
    for (int i = 0; i < num_samples; ++i) {
        int label = labels[i];
        if (label >= 0 && label < impl_->num_classes) {
            const float* sample = features + static_cast<size_t>(i) * feature_dim;
            for (int j = 0; j < feature_dim; ++j) {
                impl_->class_centroids[label][j] += sample[j];
            }
            class_counts[label]++;
        }
    }
    
    for (int c = 0; c < impl_->num_classes; ++c) {
        if (class_counts[c] > 0) {
            for (int j = 0; j < feature_dim; ++j) {
                impl_->class_centroids[c][j] /= class_counts[c];
            }
        }
    }
    
    impl_->trained = true;
    std::cout << "  Training complete (stub)" << std::endl;
}

std::vector<int> SVMWrapper::predict(const float* features, int num_samples,
                                      int feature_dim) const {
    std::vector<int> predictions(num_samples, -1);
    
    if (!impl_->trained) {
        std::cerr << "Error: Model not trained!" << std::endl;
        return predictions;
    }
    
    for (int i = 0; i < num_samples; ++i) {
        const float* sample = features + static_cast<size_t>(i) * feature_dim;
        
        float min_dist = 1e30f;
        int best_class = 0;
        
        for (int c = 0; c < impl_->num_classes; ++c) {
            float dist = 0.0f;
            for (int j = 0; j < feature_dim; ++j) {
                float d = sample[j] - impl_->class_centroids[c][j];
                dist += d * d;
            }
            if (dist < min_dist) {
                min_dist = dist;
                best_class = c;
            }
        }
        
        predictions[i] = best_class;
    }
    
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

bool SVMWrapper::save_model(const std::string& /*path*/) const {
    std::cerr << "[STUB] save_model not implemented" << std::endl;
    return false;
}

bool SVMWrapper::load_model(const std::string& /*path*/) {
    std::cerr << "[STUB] load_model not implemented" << std::endl;
    return false;
}

#endif  // WITH_LIBSVM

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
