# CSC14120 - Parallel Programming: Unsupervised Feature Learning with Autoencoder

**Đại học Khoa học Tự nhiên, ĐHQG-HCM**
**Khoa Công nghệ Thông tin**

Đây là mã nguồn cho đồ án cuối kỳ môn Lập trình Song song. [cite_start]Dự án cài đặt một hệ thống Convolutional Autoencoder để học đặc trưng không giám sát trên bộ dữ liệu CIFAR-10, được tăng tốc bằng CUDA và sử dụng LIBSVM để phân loại ảnh[cite: 4, 9, 23].

## 1. Cấu trúc thư mục (Directory Structure)

[cite_start]Dự án được tổ chức theo cấu trúc sau để tách biệt rõ ràng giữa mã nguồn, dữ liệu và thư viện[cite: 569, 579]:

```text
CSC14120_FinalProject/
├── data/                        # Chứa dữ liệu CIFAR-10
│   ├── data_batch_1.bin
│   ├── ...
│   └── test_batch.bin
├── include/                     # Các file header (.h) định nghĩa class và cấu trúc
│   ├── dataset.h                # Class xử lý đọc file binary CIFAR-10
│   ├── layer.h                  # Class cơ sở và các layer (Conv2D, ReLU, MaxPool...)
│   ├── autoencoder.h            # Định nghĩa kiến trúc mạng Encoder-Decoder
│   ├── cuda_utils.h             # Các macro kiểm tra lỗi CUDA (CUDA_CHECK)
│   └── svm_wrapper.h            # Wrapper giao tiếp với thư viện LIBSVM
├── src/                         # Mã nguồn cài đặt (.cpp, .cu)
│   ├── main.cpp                 # Chương trình chính, điều khiển luồng (CPU/GPU)
│   ├── dataset.cpp              # Cài đặt đọc dữ liệu và chuẩn hóa
│   ├── layers_cpu.cpp           # Cài đặt thuật toán trên CPU (Phase 1)
│   ├── layers_gpu.cu            # Cài đặt CUDA Kernels cơ bản (Phase 2)
│   ├── layers_gpu_opt.cu        # Cài đặt CUDA Kernels tối ưu (Phase 3 - Shared Memory...)
│   ├── autoencoder.cpp          # Quản lý luồng forward/backward
│   └── svm_wrapper.cpp          # Tích hợp huấn luyện và dự đoán SVM
├── lib/                         # Thư viện bên thứ 3
│   └── libsvm/                  # Source code của LIBSVM (cần giải nén vào đây)
├── weights/                     # Lưu trữ trọng số mô hình sau khi train
│   ├── encoder_weights.bin
│   └── decoder_weights.bin
├── notebooks/                   # Báo cáo và phân tích kết quả
│   └── FinalReport.ipynb        # Jupyter Notebook chạy trên Colab
├── Makefile                     # Script biên dịch tự động (g++ và nvcc)
└── README.md                    # Tài liệu hướng dẫn này

```

Chạy phase 1:
mingw32-make
./build/project_run data/
