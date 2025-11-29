#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(expr)                                                            \
    do {                                                                            \
        cudaError_t err__ = (expr);                                                \
        if (err__ != cudaSuccess) {                                                \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)              \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
        }                                                                           \
    } while (0)
#else
#define CUDA_CHECK(expr) (expr)
#endif

#endif  // CUDA_UTILS_H
