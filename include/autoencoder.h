#ifndef AUTOENCODER_H
#define AUTOENCODER_H

#include "layer.h"
#include <vector>
#include <memory> // Dùng smart pointer để quản lý layer dễ hơn

class Autoencoder {
public:
    std::vector<Layer*> layers; // Danh sách các layer tuần tự

    Autoencoder();
    ~Autoencoder();

    // Forward toàn mạng: Input Image -> Reconstructed Image
    void forward(const std::vector<float>& input, std::vector<float>& output);

    // Backward toàn mạng: Tính gradient từ Loss -> Input
    void backward(const std::vector<float>& loss_grad, float learning_rate);

    // Chỉ chạy Encoder để lấy đặc trưng (cho Phase 4 SVM)
    void get_latent(const std::vector<float>& input, std::vector<float>& latent_out);

    void save_weights(const std::string& filepath);

        // (Gợi ý: Bạn cũng nên khai báo sẵn hàm load để dùng cho Phase 4 sau này)
    // void load_weights(const std::string& filepath);
};

#endif