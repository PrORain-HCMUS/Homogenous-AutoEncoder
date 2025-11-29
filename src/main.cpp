#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "autoencoder.h"
#include "dataset.h"

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char **argv) {
    std::string data_dir = "data";
    int epochs = 5;
    int batch_size = 32;
    float learning_rate = 1e-3f;
    std::string log_path;
    int max_train_images = 1000;
    bool use_openmp = false;

    if (argc > 1) {
        data_dir = argv[1];
    }
    if (argc > 2) {
        epochs = std::stoi(argv[2]);
    }
    if (argc > 3) {
        batch_size = std::stoi(argv[3]);
    }
    if (argc > 4) {
        learning_rate = std::stof(argv[4]);
    }
    if (argc > 5) {
        log_path = argv[5];
    }
    if (argc > 6) {
        max_train_images = std::stoi(argv[6]);
    }
    if (argc > 7) {
        use_openmp = (std::stoi(argv[7]) != 0);
    }

    try {
        CIFAR10Dataset dataset(data_dir);
        const auto &train = dataset.train();
        const auto &test = dataset.test();

        std::cout << "Loaded CIFAR-10: train = " << train.num_images
                  << ", test = " << test.num_images << std::endl;

        if (train.num_images == 0) {
            std::cerr << "No training images loaded. Check data directory: " << data_dir
                      << std::endl;
            return 1;
        }

        int effective_train = train.num_images;
        if (max_train_images > 0 && max_train_images < effective_train) {
            effective_train = max_train_images;
        }

        int num_batches = effective_train / batch_size;
        if (num_batches == 0) {
            std::cerr << "Not enough training images for batch_size = " << batch_size
                      << std::endl;
            return 1;
        }

        std::ofstream log_file;
        if (!log_path.empty()) {
            log_file.open(log_path, std::ios::out);
            if (!log_file) {
                std::cerr << "Warning: failed to open log file: " << log_path << std::endl;
            } else {
                log_file << "epoch,loss,time_sec" << std::endl;
            }
        }

        if (use_openmp) {
#ifdef _OPENMP
            int max_threads = omp_get_max_threads();
            int num_threads = (max_threads > 2) ? (max_threads - 2) : 1;
            omp_set_num_threads(num_threads);
            std::cout << "OpenMP enabled with " << num_threads
                      << " threads (max-2 from " << max_threads << ")" << std::endl;
#else
            std::cout << "Warning: OpenMP requested (use_openmp=1) but program was not "
                         "compiled with -fopenmp. Running single-threaded."
                      << std::endl;
#endif
        }

        Autoencoder model;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            auto start = std::chrono::high_resolution_clock::now();
            float epoch_loss = 0.0f;
            int batches = 0;

            // Create shuffled indices for this epoch
            std::vector<int> indices(effective_train);
            for (int i = 0; i < effective_train; ++i) {
                indices[i] = i;
            }
            std::mt19937 rng(1234 + epoch);
            std::shuffle(indices.begin(), indices.end(), rng);

            for (int b = 0; b < num_batches; ++b) {
                Tensor4D input(batch_size, CIFAR10Dataset::IMAGE_CHANNELS,
                               CIFAR10Dataset::IMAGE_HEIGHT, CIFAR10Dataset::IMAGE_WIDTH);
                Tensor4D target(batch_size, CIFAR10Dataset::IMAGE_CHANNELS,
                                CIFAR10Dataset::IMAGE_HEIGHT, CIFAR10Dataset::IMAGE_WIDTH);

                const std::size_t image_size = static_cast<std::size_t>(CIFAR10Dataset::IMAGE_SIZE);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
                for (int i = 0; i < batch_size; ++i) {
                    const int img_idx = indices[b * batch_size + i];
                    const float* __restrict__ src = &train.images[static_cast<std::size_t>(img_idx) * image_size];
                    float* __restrict__ input_ptr = input.data.data() + static_cast<std::size_t>(i) * image_size;
                    float* __restrict__ target_ptr = target.data.data() + static_cast<std::size_t>(i) * image_size;
                    
                    std::memcpy(input_ptr, src, image_size * sizeof(float));
                    std::memcpy(target_ptr, src, image_size * sizeof(float));
                }

                float loss = model.train_step(input, target, learning_rate);
                epoch_loss += loss;
                ++batches;
            }

            if (batches > 0) {
                epoch_loss /= static_cast<float>(batches);
            }

            auto end = std::chrono::high_resolution_clock::now();
            double seconds =
                std::chrono::duration<double>(end - start).count();

            std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                      << " - loss: " << epoch_loss
                      << " - time: " << seconds << " s" << std::endl;

            if (log_file) {
                log_file << (epoch + 1) << "," << epoch_loss << "," << seconds << std::endl;
            }
        }

        const std::string weights_path = "cpu_phase1_weights.bin";
        if (!model.save_weights(weights_path)) {
            std::cerr << "Warning: failed to save weights to " << weights_path << std::endl;
        } else {
            std::cout << "Saved CPU model weights to " << weights_path << std::endl;
        }
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
