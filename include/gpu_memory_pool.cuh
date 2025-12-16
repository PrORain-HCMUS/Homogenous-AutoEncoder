#ifndef GPU_MEMORY_POOL_CUH
#define GPU_MEMORY_POOL_CUH

#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

class GPUMemoryPool {
    struct Block {
        float* ptr = nullptr;
        size_t size = 0;
        bool in_use = false;
    };
    
    std::vector<Block> blocks_;
    size_t total_allocated_ = 0;
    size_t peak_usage_ = 0;
    
public:
    GPUMemoryPool() = default;
    
    ~GPUMemoryPool() {
        for (auto& block : blocks_) {
            if (block.ptr) cudaFree(block.ptr);
        }
    }
    
    float* acquire(size_t size) {
        for (auto& block : blocks_) {
            if (!block.in_use && block.size >= size) {
                block.in_use = true;
                return block.ptr;
            }
        }
        
        Block new_block;
        new_block.size = size;
        new_block.in_use = true;
        cudaMalloc(&new_block.ptr, size * sizeof(float));
        cudaMemset(new_block.ptr, 0, size * sizeof(float));
        blocks_.push_back(new_block);
        
        total_allocated_ += size * sizeof(float);
        peak_usage_ = std::max(peak_usage_, total_allocated_);
        
        return new_block.ptr;
    }
    
    void release(float* ptr) {
        for (auto& block : blocks_) {
            if (block.ptr == ptr) {
                block.in_use = false;
                return;
            }
        }
    }
    
    void release_all() {
        for (auto& block : blocks_) {
            block.in_use = false;
        }
    }
    
    size_t get_total_allocated() const { return total_allocated_; }
    size_t get_peak_usage() const { return peak_usage_; }
    size_t get_num_blocks() const { return blocks_.size(); }
    
    void print_stats() const {
        printf("GPU Memory Pool Stats:\n");
        printf("  Total allocated: %.2f MB\n", total_allocated_ / (1024.0 * 1024.0));
        printf("  Peak usage: %.2f MB\n", peak_usage_ / (1024.0 * 1024.0));
        printf("  Num blocks: %zu\n", blocks_.size());
    }
};

class TensorView {
public:
    float* d_data = nullptr;
    int n = 0, c = 0, h = 0, w = 0;
    GPUMemoryPool* pool = nullptr;
    
    TensorView() = default;
    
    TensorView(GPUMemoryPool* p, int n_, int c_, int h_, int w_) 
        : pool(p), n(n_), c(c_), h(h_), w(w_) {
        d_data = pool->acquire(size());
    }
    
    ~TensorView() {
        if (pool && d_data) pool->release(d_data);
    }
    
    TensorView(const TensorView&) = delete;
    TensorView& operator=(const TensorView&) = delete;
    
    TensorView(TensorView&& other) noexcept 
        : d_data(other.d_data), n(other.n), c(other.c), h(other.h), w(other.w), pool(other.pool) {
        other.d_data = nullptr;
        other.pool = nullptr;
    }
    
    TensorView& operator=(TensorView&& other) noexcept {
        if (this != &other) {
            if (pool && d_data) pool->release(d_data);
            d_data = other.d_data;
            n = other.n; c = other.c; h = other.h; w = other.w;
            pool = other.pool;
            other.d_data = nullptr;
            other.pool = nullptr;
        }
        return *this;
    }
    
    size_t size() const { return static_cast<size_t>(n) * c * h * w; }
    size_t bytes() const { return size() * sizeof(float); }
    
    void reshape(int n_, int c_, int h_, int w_) {
        size_t new_size = static_cast<size_t>(n_) * c_ * h_ * w_;
        if (new_size > size()) {
            if (pool && d_data) pool->release(d_data);
            d_data = pool->acquire(new_size);
        }
        n = n_; c = c_; h = h_; w = w_;
    }
};

#endif
