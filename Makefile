CXX = g++
CXXFLAGS = -std=c++11 -O3 -Wall -I./include

# Thư mục
SRC_DIR = src
BUILD_DIR = build
OBJ_DIR = build/obj

# Các file nguồn
SRCS = $(SRC_DIR)/main.cpp $(SRC_DIR)/dataset.cpp $(SRC_DIR)/layers_cpu.cpp $(SRC_DIR)/autoencoder.cpp
OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(SRCS))

# Tên file thực thi
TARGET = $(BUILD_DIR)/project_run

# Quy tắc biên dịch
all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean