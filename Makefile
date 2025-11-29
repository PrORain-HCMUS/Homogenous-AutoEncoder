# Makefile
CXX = g++
NVCC = nvcc
CXXFLAGS = -std=c++11 -O3 -Wall -I./include
NVCCFLAGS = -O3 -I./include -arch=sm_60 # Chỉnh sm_XX tùy GPU của bạn (GTX 1060 là sm_61, T4 là sm_75)

SRC_DIR = src
BUILD_DIR = build
OBJ_DIR = build/obj

# File C++ (Host)
SRCS_CPP = $(SRC_DIR)/main.cpp $(SRC_DIR)/dataset.cpp $(SRC_DIR)/autoencoder.cpp $(SRC_DIR)/layers_cpu.cpp 
# File CUDA (Device) - Thêm file này
SRCS_CU = $(SRC_DIR)/layers_gpu.cu

OBJS_CPP = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS_CPP))
OBJS_CU = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.cu.o, $(SRCS_CU))

TARGET = $(BUILD_DIR)/project_run

all: $(TARGET)

$(TARGET): $(OBJS_CPP) $(OBJS_CU)
	@mkdir -p $(BUILD_DIR)
	# Link bằng NVCC để đảm bảo thư viện CUDA được load
	$(NVCC) $(NVCCFLAGS) -o $@ $^ 

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.cu.o: $(SRC_DIR)/%.cu
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)