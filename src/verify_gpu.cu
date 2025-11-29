#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#include "dataset.h"
#include "autoencoder.h"
#include "gpu_autoencoder.h"
#include "layer.h"
#include "gpu_layer.h"
#include "cuda_utils.h"

static bool g_verbose = false;

bool tensors_equal(const Tensor4D& cpu, const GPUTensor4D& gpu, float tolerance = 1e-4f) {
    if (cpu.n != gpu.n || cpu.c != gpu.c || cpu.h != gpu.h || cpu.w != gpu.w) {
        std::cerr << "Shape mismatch: CPU(" << cpu.n << "," << cpu.c << "," << cpu.h << "," << cpu.w
                  << ") vs GPU(" << gpu.n << "," << gpu.c << "," << gpu.h << "," << gpu.w << ")" << std::endl;
        return false;
    }
    
    std::vector<float> gpu_data(gpu.size());
    gpu.copy_to_host(gpu_data.data());
    
    float max_diff = 0.0f;
    size_t diff_count = 0;
    size_t first_diff_idx = 0;
    float first_cpu_val = 0.0f, first_gpu_val = 0.0f;
    
    for (size_t i = 0; i < cpu.data.size(); ++i) {
        float diff = std::abs(cpu.data[i] - gpu_data[i]);
        if (diff > tolerance) {
            if (diff_count == 0) {
                first_diff_idx = i;
                first_cpu_val = cpu.data[i];
                first_gpu_val = gpu_data[i];
            }
            ++diff_count;
            if (diff > max_diff) max_diff = diff;
        }
    }
    
    if (diff_count > 0) {
        std::cerr << "  Differences: " << diff_count << "/" << cpu.data.size()
                  << " (max diff: " << max_diff << ")" << std::endl;
        if (g_verbose) {
            std::cerr << "  First mismatch at index " << first_diff_idx 
                      << ": CPU=" << first_cpu_val << " GPU=" << first_gpu_val << std::endl;
        }
        return false;
    }
    return true;
}

bool verify_conv2d() {
    std::cout << "\n=== Verifying Conv2D ===" << std::endl;
    
    Conv2DLayer cpu_conv(3, 64, 3, 1, 1);
    GPUConv2DLayer gpu_conv(3, 64, 3, 1, 1);
    
    Tensor4D cpu_input(2, 3, 32, 32);
    for (auto& v : cpu_input.data) v = static_cast<float>(rand()) / RAND_MAX;
    
    GPUTensor4D gpu_input;
    tensor_cpu_to_gpu(cpu_input, gpu_input);
    
    Tensor4D cpu_output = cpu_conv.forward(cpu_input);
    
    GPUTensor4D gpu_output;
    gpu_conv.forward(gpu_input, gpu_output);
    
    bool passed = tensors_equal(cpu_output, gpu_output, 1e-3f);
    std::cout << "  Conv2D Forward: " << (passed ? "PASS" : "FAIL") << std::endl;
    return passed;
}

bool verify_relu() {
    std::cout << "\n=== Verifying ReLU ===" << std::endl;
    
    ReLULayer cpu_relu;
    GPUReLULayer gpu_relu;
    
    Tensor4D cpu_input(2, 64, 16, 16);
    for (auto& v : cpu_input.data) v = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    
    GPUTensor4D gpu_input;
    tensor_cpu_to_gpu(cpu_input, gpu_input);
    
    Tensor4D cpu_output = cpu_relu.forward(cpu_input);
    
    GPUTensor4D gpu_output;
    gpu_relu.forward(gpu_input, gpu_output);
    
    bool passed = tensors_equal(cpu_output, gpu_output);
    std::cout << "  ReLU Forward: " << (passed ? "PASS" : "FAIL") << std::endl;
    return passed;
}

bool verify_maxpool() {
    std::cout << "\n=== Verifying MaxPool2D ===" << std::endl;
    
    MaxPool2DLayer cpu_pool(2, 2);
    GPUMaxPool2DLayer gpu_pool(2, 2);
    
    Tensor4D cpu_input(2, 64, 32, 32);
    for (auto& v : cpu_input.data) v = static_cast<float>(rand()) / RAND_MAX;
    
    GPUTensor4D gpu_input;
    tensor_cpu_to_gpu(cpu_input, gpu_input);
    
    Tensor4D cpu_output = cpu_pool.forward(cpu_input);
    
    GPUTensor4D gpu_output;
    gpu_pool.forward(gpu_input, gpu_output);
    
    bool passed = tensors_equal(cpu_output, gpu_output);
    std::cout << "  MaxPool2D Forward: " << (passed ? "PASS" : "FAIL") << std::endl;
    return passed;
}

bool verify_upsample() {
    std::cout << "\n=== Verifying UpSample2D ===" << std::endl;
    
    UpSample2DLayer cpu_up(2);
    GPUUpSample2DLayer gpu_up(2);
    
    Tensor4D cpu_input(2, 128, 8, 8);
    for (auto& v : cpu_input.data) v = static_cast<float>(rand()) / RAND_MAX;
    
    GPUTensor4D gpu_input;
    tensor_cpu_to_gpu(cpu_input, gpu_input);
    
    Tensor4D cpu_output = cpu_up.forward(cpu_input);
    
    GPUTensor4D gpu_output;
    gpu_up.forward(gpu_input, gpu_output);
    
    bool passed = tensors_equal(cpu_output, gpu_output);
    std::cout << "  UpSample2D Forward: " << (passed ? "PASS" : "FAIL") << std::endl;
    return passed;
}

bool verify_mse_loss() {
    std::cout << "\n=== Verifying MSE Loss ===" << std::endl;
    
    Tensor4D cpu_output(2, 3, 32, 32);
    Tensor4D cpu_target(2, 3, 32, 32);
    for (auto& v : cpu_output.data) v = static_cast<float>(rand()) / RAND_MAX;
    for (auto& v : cpu_target.data) v = static_cast<float>(rand()) / RAND_MAX;
    
    GPUTensor4D gpu_output, gpu_target;
    tensor_cpu_to_gpu(cpu_output, gpu_output);
    tensor_cpu_to_gpu(cpu_target, gpu_target);
    
    float cpu_loss = mse_loss(cpu_output, cpu_target);
    float gpu_loss = gpu_mse_loss(gpu_output, gpu_target);
    
    float diff = std::abs(cpu_loss - gpu_loss);
    bool passed = diff < 1e-4f;
    if (passed) {
        std::cout << "  MSE Loss: PASS (CPU=" << cpu_loss << ", GPU=" << gpu_loss << ")" << std::endl;
    } else {
        std::cout << "  MSE Loss: FAIL (CPU=" << cpu_loss << ", GPU=" << gpu_loss 
                  << ", diff=" << diff << ")" << std::endl;
    }
    return passed;
}

void benchmark_forward(const std::string& data_dir, int batch_size = 32, int iterations = 10) {
    std::cout << "\n=== Forward Pass Benchmark ===" << std::endl;
    
    Tensor4D cpu_input(batch_size, 3, 32, 32);
    for (auto& v : cpu_input.data) v = static_cast<float>(rand()) / RAND_MAX;
    
    GPUTensor4D gpu_input;
    tensor_cpu_to_gpu(cpu_input, gpu_input);
    
    Autoencoder cpu_ae;
    GPUAutoencoder gpu_ae;
    
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        cpu_ae.forward(cpu_input);
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    
    GPUTensor4D gpu_output;
    CUDA_CHECK(cudaDeviceSynchronize());
    auto gpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        gpu_ae.forward(gpu_input, gpu_output);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
    
    std::cout << "  Batch size: " << batch_size << ", Iterations: " << iterations << std::endl;
    std::cout << "  CPU time: " << cpu_ms << " ms (" << cpu_ms / iterations << " ms/iter)" << std::endl;
    std::cout << "  GPU time: " << gpu_ms << " ms (" << gpu_ms / iterations << " ms/iter)" << std::endl;
    std::cout << "  Speedup: " << cpu_ms / gpu_ms << "x" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "=== GPU vs CPU Verification Tool ===" << std::endl;
    
    int device_id = 0;
    std::string data_dir = "data";
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) {
            device_id = std::atoi(argv[++i]);
        } else if (arg == "--verbose" || arg == "-v") {
            g_verbose = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options] [data_dir]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --device N    Select GPU device (default: 0)" << std::endl;
            std::cout << "  --verbose, -v Show detailed mismatch info" << std::endl;
            std::cout << "  --help, -h    Show this help" << std::endl;
            return 0;
        } else if (arg[0] != '-') {
            data_dir = arg;
        }
    }
    
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }
    
    if (device_id < 0 || device_id >= device_count) {
        std::cerr << "Invalid device ID " << device_id << ". Available: 0-" << device_count - 1 << std::endl;
        return 1;
    }
    
    CUDA_CHECK(cudaSetDevice(device_id));
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    std::cout << "GPU: " << prop.name << " (device " << device_id << ")" << std::endl;
    
    srand(42);
    
    int failures = 0;
    
    if (!verify_relu()) failures++;
    if (!verify_maxpool()) failures++;
    if (!verify_upsample()) failures++;
    if (!verify_mse_loss()) failures++;
    if (!verify_conv2d()) failures++;
    
    benchmark_forward(data_dir, 32, 10);
    benchmark_forward(data_dir, 64, 10);
    
    std::cout << "\n=== Verification Complete ===" << std::endl;
    if (failures > 0) {
        std::cerr << "FAILED: " << failures << " verification(s) failed" << std::endl;
    } else {
        std::cout << "PASSED: All verifications passed" << std::endl;
    }
    
    CUDA_CHECK(cudaDeviceReset());
    return failures > 0 ? 1 : 0;
}
