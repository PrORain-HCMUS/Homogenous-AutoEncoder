UNAME_S := $(shell uname -s 2>/dev/null || echo Windows)

ifeq ($(UNAME_S),Darwin)
    BREW_GCC := $(shell which g++-14 2>/dev/null || which g++-13 2>/dev/null || which g++-12 2>/dev/null || which g++-11 2>/dev/null)
    ifneq ($(BREW_GCC),)
        CXX = $(BREW_GCC)
        OPENMP_CFLAGS = -fopenmp
        OPENMP_LDFLAGS = -fopenmp
        HAS_OPENMP = 1
    else
        CXX = clang++
        LIBOMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
        LIBOMP_INCLUDE := $(wildcard $(LIBOMP_PREFIX)/include/omp.h)
        ifneq ($(LIBOMP_INCLUDE),)
            OPENMP_CFLAGS = -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include
            OPENMP_LDFLAGS = -L$(LIBOMP_PREFIX)/lib -lomp
            HAS_OPENMP = 1
        else
            OPENMP_CFLAGS = 
            OPENMP_LDFLAGS =
            HAS_OPENMP = 0
        endif
    endif
else ifeq ($(UNAME_S),Linux)
    CXX = g++
    OPENMP_CFLAGS = -fopenmp
    OPENMP_LDFLAGS = -fopenmp
    HAS_OPENMP = 1
else
    CXX = g++
    OPENMP_CFLAGS = -fopenmp
    OPENMP_LDFLAGS = -fopenmp
    HAS_OPENMP = 1
endif

NVCC = nvcc

CXXFLAGS = -O3 -std=c++17 -Wall -Wextra
NVCCFLAGS = -O2 -std=c++17 -arch=sm_75 --expt-relaxed-constexpr

ifeq ($(UNAME_S),Darwin)
    CXXFLAGS += -march=native
else ifeq ($(UNAME_S),Linux)
    CXXFLAGS += -march=native
endif

INCLUDES = -Iinclude

LIBSVM_DIR = external/libsvm
LIBSVM_INCLUDE = -I$(LIBSVM_DIR)
LIBSVM_LIB = $(LIBSVM_DIR)/svm.o

CPU_SRC = src/main.cpp \
          src/dataset.cpp \
          src/layers_cpu.cpp \
          src/autoencoder.cpp

GPU_SRC = src/layers_gpu.cu \
          src/gpu_autoencoder.cu

GPU_OPT_SRC = src/layers_gpu_opt.cu

GPU_MAIN = src/main_gpu.cu

GPU_SHARED_SRC = src/dataset.cpp

SVM_SRC = src/svm_wrapper.cpp

ifeq ($(OS),Windows_NT)
    EXE_EXT = .exe
else
    EXE_EXT =
endif

all: cpu_train

cpu_train: $(CPU_SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@$(EXE_EXT) $(CPU_SRC)

cpu_train_omp: $(CPU_SRC)
ifeq ($(HAS_OPENMP),1)
	$(CXX) $(CXXFLAGS) $(OPENMP_CFLAGS) $(INCLUDES) -o $@$(EXE_EXT) $(CPU_SRC) $(OPENMP_LDFLAGS)
else
	@echo "WARNING: OpenMP not available on this system."
	@echo "On macOS, install with: brew install libomp  OR  brew install gcc"
	@echo "Building without OpenMP parallelization..."
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@$(EXE_EXT) $(CPU_SRC)
endif

info:
	@echo "=== Build Configuration ==="
	@echo "OS: $(UNAME_S)"
	@echo "Compiler: $(CXX)"
	@echo "CXXFLAGS: $(CXXFLAGS)"
	@echo "OpenMP available: $(HAS_OPENMP)"
	@echo "OPENMP_CFLAGS: $(OPENMP_CFLAGS)"
	@echo "OPENMP_LDFLAGS: $(OPENMP_LDFLAGS)"
	@echo "==========================="

gpu_train: $(GPU_MAIN) $(GPU_SRC) $(GPU_SHARED_SRC)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@$(EXE_EXT) $(GPU_MAIN) $(GPU_SRC) $(GPU_SHARED_SRC)

gpu_train_opt: $(GPU_MAIN) $(GPU_SRC) $(GPU_OPT_SRC) $(GPU_SHARED_SRC)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -DUSE_OPTIMIZED_KERNELS -o $@$(EXE_EXT) \
		$(GPU_MAIN) $(GPU_SRC) $(GPU_OPT_SRC) $(GPU_SHARED_SRC)

full_pipeline: $(GPU_MAIN) $(GPU_SRC) $(GPU_SHARED_SRC) $(SVM_SRC) $(LIBSVM_LIB)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIBSVM_INCLUDE) -DWITH_SVM -o $@$(EXE_EXT) \
		$(GPU_MAIN) $(GPU_SRC) $(GPU_SHARED_SRC) $(SVM_SRC) $(LIBSVM_LIB)

$(LIBSVM_DIR)/svm.o:
	@echo "Building LIBSVM..."
	@if [ -d $(LIBSVM_DIR) ]; then \
		cd $(LIBSVM_DIR) && make lib; \
	else \
		echo "LIBSVM not found. Run: git submodule add https://github.com/cjlin1/libsvm external/libsvm"; \
	fi

verify_gpu: src/verify_gpu.cu $(GPU_SRC) $(CPU_SRC)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@$(EXE_EXT) src/verify_gpu.cu $(GPU_SRC) \
		src/dataset.cpp src/layers_cpu.cpp src/autoencoder.cpp

clean:
ifeq ($(OS),Windows_NT)
	del /Q cpu_train.exe cpu_train_omp.exe gpu_train.exe gpu_train_opt.exe full_pipeline.exe verify_gpu.exe 2>nul || exit 0
	del /Q *.o src\*.o 2>nul || exit 0
else
	rm -f cpu_train cpu_train_omp gpu_train gpu_train_opt full_pipeline verify_gpu
	rm -f cpu_train.exe cpu_train_omp.exe gpu_train.exe gpu_train_opt.exe full_pipeline.exe verify_gpu.exe
	rm -f *.o src/*.o
endif

clean_all: clean
ifeq ($(OS),Windows_NT)
	del /Q *.weights *.csv 2>nul || exit 0
else
	rm -f *.weights *.csv
	@if [ -d $(LIBSVM_DIR) ]; then cd $(LIBSVM_DIR) && make clean; fi
endif

gpu_info:
	@nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

.PHONY: all clean clean_all gpu_info info
