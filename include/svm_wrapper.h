#ifndef SVM_WRAPPER_H
#define SVM_WRAPPER_H

#include <string>
#include <vector>

class SVMWrapper {
public:
    SVMWrapper();
    ~SVMWrapper();

    void train(const float* features, const int* labels, 
               int num_samples, int feature_dim);

    std::vector<int> predict(const float* features, int num_samples, int feature_dim) const;

    float evaluate(const float* features, const int* labels,
                   int num_samples, int feature_dim) const;

    bool save_model(const std::string& path) const;
    bool load_model(const std::string& path);

    void set_kernel(int kernel_type);
    void set_C(double C);
    void set_gamma(double gamma);

private:
    struct Impl;
    Impl* impl_;
};

void print_confusion_matrix(const std::vector<int>& true_labels,
                            const std::vector<int>& pred_labels,
                            int num_classes);

#endif  // SVM_WRAPPER_H
