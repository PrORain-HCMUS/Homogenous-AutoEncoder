#include <algorithm>
#include <chrono>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "autoencoder.h"
#include "dataset.h"

#ifdef _OPENMP
#include <omp.h>
#endif

std::string get_timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

class TrainingLogger {
public:
    TrainingLogger(const std::string& log_path, const std::string& csv_path = "")
        : log_path_(log_path), csv_path_(csv_path) {
        if (!log_path_.empty()) {
            log_file_.open(log_path_, std::ios::out);
            if (log_file_) {
                log_file_ << "========================================\n";
                log_file_ << "CPU AUTOENCODER TRAINING LOG\n";
                log_file_ << "========================================\n";
                log_file_ << "Start Time: " << get_timestamp() << "\n\n";
            }
        }
        if (!csv_path_.empty()) {
            csv_file_.open(csv_path_, std::ios::out);
            if (csv_file_) {
                csv_file_ << "epoch,batch,loss,epoch_time_sec,batch_time_ms,avg_loss\n";
            }
        }
    }

    ~TrainingLogger() {
        if (log_file_) {
            log_file_ << "\n========================================\n";
            log_file_ << "End Time: " << get_timestamp() << "\n";
            log_file_ << "========================================\n";
            log_file_.close();
        }
        if (csv_file_) csv_file_.close();
    }

    void log_config(int epochs, int batch_size, float lr, 
                    const std::string& data_dir, bool use_openmp, int max_images) {
        if (!log_file_) return;
        log_file_ << "--- Configuration ---\n";
        log_file_ << "Data Directory: " << data_dir << "\n";
        log_file_ << "Epochs: " << epochs << "\n";
        log_file_ << "Batch Size: " << batch_size << "\n";
        log_file_ << "Learning Rate: " << lr << "\n";
        log_file_ << "Max Train Images: " << (max_images > 0 ? std::to_string(max_images) : "all") << "\n";
        log_file_ << "OpenMP: " << (use_openmp ? "enabled" : "disabled") << "\n";
#ifdef _OPENMP
        if (use_openmp) {
            log_file_ << "OpenMP Threads: " << omp_get_max_threads() << "\n";
        }
#endif
        log_file_ << "\n";
    }

    void log_optimizations() {
        if (!log_file_) return;
        log_file_ << "--- CPU Optimizations Applied ---\n";
        log_file_ << "1. OpenMP parallelization for Conv2D forward/backward\n";
        log_file_ << "2. Loop unrolling for 3x3 kernel convolutions\n";
        log_file_ << "3. Precomputed stride offsets for memory access\n";
        log_file_ << "4. SIMD vectorization for ReLU and MSE operations\n";
        log_file_ << "5. OpenMP parallelization for MaxPool2D/UpSample2D\n";
        log_file_ << "6. Direct memcpy for batch loading\n";
        log_file_ << "7. Thread-local gradient accumulators\n";
        log_file_ << "\n";
    }

    void log_dataset_info(int effective_train, int test_count, int num_batches) {
        if (!log_file_) return;
        log_file_ << "--- Dataset Info ---\n";
        log_file_ << "Effective Training: " << effective_train << "\n";
        log_file_ << "Test Images: " << test_count << "\n";
        log_file_ << "Batches per Epoch: " << num_batches << "\n";
        log_file_ << "\n";
    }

    void log(const std::string& msg) {
        if (log_file_) {
            log_file_ << msg << "\n";
        }
    }

    void log_epoch_start(int epoch, int total_epochs) {
        if (!log_file_) return;
        log_file_ << "--- Epoch " << epoch << "/" << total_epochs << " ---\n";
    }

    void log_batch(int batch, int num_batches, float loss, double batch_time_sec) {
        if (csv_file_ && (batch % 10 == 0 || batch == num_batches - 1)) {
            csv_file_ << "," << batch << "," << std::fixed << std::setprecision(6)
                      << loss << ",," << (batch_time_sec * 1000.0) << ",\n";
        }
    }

    void log_epoch_end(int epoch, int total_epochs, float avg_loss, double epoch_time_sec, 
                        bool is_best, float best_loss) {
        if (log_file_) {
            log_file_ << "Epoch " << epoch << "/" << total_epochs;
            if (is_best) log_file_ << " [BEST]";
            log_file_ << "\n";
            log_file_ << "  Avg Loss: " << std::fixed << std::setprecision(6) << avg_loss << "\n";
            log_file_ << "  Best Loss: " << std::setprecision(6) << best_loss << "\n";
            log_file_ << "  Time: " << std::setprecision(2) << epoch_time_sec << " seconds\n";
            log_file_ << "  Throughput: " << std::setprecision(1) 
                      << (1.0 / epoch_time_sec * 60.0) << " epochs/min\n\n";
        }
    }

    void write_csv(int epoch, float avg_loss, double epoch_time_sec) {
        if (csv_file_) {
            csv_file_ << epoch << ",,," << std::fixed << std::setprecision(2) 
                      << epoch_time_sec << ",," << std::setprecision(6) << avg_loss << "\n";
        }
    }

    void log_training_complete(int total_epochs, float best_loss, float avg_loss, double total_time) {
        if (!log_file_) return;
        log_file_ << "=== Training Complete ===\n";
        log_file_ << "Total Epochs: " << total_epochs << "\n";
        log_file_ << "Best Loss: " << std::fixed << std::setprecision(6) << best_loss << "\n";
        log_file_ << "Average Loss: " << std::setprecision(6) << avg_loss << "\n";
        log_file_ << "Total Training Time: " << std::setprecision(2) 
                  << total_time << " seconds\n";
    }

private:
    std::string log_path_;
    std::string csv_path_;
    std::ofstream log_file_;
    std::ofstream csv_file_;
};

int main(int argc, char **argv) {
    std::string data_dir = "data";
    int epochs = 5;
    int batch_size = 32;
    float learning_rate = 1e-3f;
    std::string log_path = "cpu_training.log";
    std::string csv_path = "cpu_training.csv";
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
    if (argc > 8) {
        csv_path = argv[8];
    }

    try {
        TrainingLogger logger(log_path, csv_path);
        
        logger.log_config(epochs, batch_size, learning_rate, data_dir, 
                          use_openmp, max_train_images);
        
        logger.log_optimizations();
        
        CIFAR10Dataset dataset(data_dir);
        const auto &train = dataset.train();
        const auto &test = dataset.test();

        std::cout << "Loaded CIFAR-10: train = " << train.num_images
                  << ", test = " << test.num_images << std::endl;
        logger.log("Loaded CIFAR-10: train = " + std::to_string(train.num_images) + 
                   ", test = " + std::to_string(test.num_images));

        if (train.num_images == 0) {
            std::cerr << "No training images loaded. Check data directory: " << data_dir
                      << std::endl;
            logger.log("ERROR: No training images loaded. Check data directory: " + data_dir);
            return 1;
        }

        int effective_train = train.num_images;
        if (max_train_images > 0 && max_train_images < effective_train) {
            effective_train = max_train_images;
            logger.log("Limited training to " + std::to_string(effective_train) + " images");
        }

        int num_batches = effective_train / batch_size;
        if (num_batches == 0) {
            std::cerr << "Not enough training images for batch_size = " << batch_size
                      << std::endl;
            logger.log("ERROR: Not enough training images for batch_size = " + std::to_string(batch_size));
            return 1;
        }

        logger.log_dataset_info(effective_train, test.num_images, num_batches);

        if (use_openmp) {
#ifdef _OPENMP
            int max_threads = omp_get_max_threads();
            int num_threads = (max_threads > 2) ? (max_threads - 2) : 1;
            omp_set_num_threads(num_threads);
            std::cout << "OpenMP enabled with " << num_threads
                      << " threads (max-2 from " << max_threads << ")" << std::endl;
            logger.log("OpenMP enabled with " + std::to_string(num_threads) + 
                       " threads (max-2 from " + std::to_string(max_threads) + ")");
#else
            std::cout << "Warning: OpenMP requested (use_openmp=1) but program was not "
                         "compiled with -fopenmp. Running single-threaded."
                      << std::endl;
            logger.log("WARNING: OpenMP requested but not compiled. Running single-threaded.");
#endif
        } else {
            logger.log("OpenMP disabled. Running single-threaded.");
        }

        logger.log("\n" + std::string(60, '='));
        logger.log("TRAINING STARTED");
        logger.log(std::string(60, '=') + "\n");

        Autoencoder model;
        auto training_start = std::chrono::high_resolution_clock::now();
        float best_loss = std::numeric_limits<float>::max();
        std::vector<float> epoch_losses;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            auto start = std::chrono::high_resolution_clock::now();
            logger.log_epoch_start(epoch + 1, epochs);
            
            float epoch_loss = 0.0f;
            int batches = 0;

            std::vector<int> indices(effective_train);
            for (int i = 0; i < effective_train; ++i) {
                indices[i] = i;
            }
            std::mt19937 rng(1234 + epoch);
            std::shuffle(indices.begin(), indices.end(), rng);

            for (int b = 0; b < num_batches; ++b) {
                auto batch_start = std::chrono::high_resolution_clock::now();
                
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
                
                auto batch_end = std::chrono::high_resolution_clock::now();
                double batch_time = std::chrono::duration<double>(batch_end - batch_start).count();
                
                if ((b + 1) % 10 == 0 || b == num_batches - 1) {
                    logger.log_batch(b + 1, num_batches, loss, batch_time);
                }
            }

            if (batches > 0) {
                epoch_loss /= static_cast<float>(batches);
            }
            epoch_losses.push_back(epoch_loss);

            auto end = std::chrono::high_resolution_clock::now();
            double seconds = std::chrono::duration<double>(end - start).count();

            bool is_best = epoch_loss < best_loss;
            if (is_best) {
                best_loss = epoch_loss;
            }

            logger.log_epoch_end(epoch + 1, epochs, epoch_loss, seconds, is_best, best_loss);
            
            logger.write_csv(epoch + 1, epoch_loss, seconds);

            std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                      << " - loss: " << epoch_loss
                      << " - time: " << seconds << " s"
                      << (is_best ? " [BEST]" : "") << std::endl;
        }

        auto training_end = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(training_end - training_start).count();
        
        float avg_loss = 0.0f;
        for (float l : epoch_losses) {
            avg_loss += l;
        }
        avg_loss /= static_cast<float>(epoch_losses.size());
        
        logger.log_training_complete(epochs, best_loss, avg_loss, total_time);

        const std::string weights_path = "cpu_phase1_weights.bin";
        if (!model.save_weights(weights_path)) {
            std::cerr << "Warning: failed to save weights to " << weights_path << std::endl;
            logger.log("WARNING: Failed to save weights to " + weights_path);
        } else {
            std::cout << "Saved CPU model weights to " << weights_path << std::endl;
            logger.log("Saved CPU model weights to " + weights_path);
        }
        
        logger.log("\nLog files saved to:");
        logger.log("  TXT: " + log_path);
        logger.log("  CSV: " + csv_path);
        
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}