#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "core/utils/tensor.cuh"

namespace ltm {
namespace ops {

/**
 * @brief Perform matrix multiplication C = alpha * (A @ B) + beta * C
 * 
 * Uses CUTLASS for optimized GEMM computation on tensor cores.
 * 
 * @tparam T Data type (float or half)
 * @param A Input matrix A
 * @param B Input matrix B
 * @param C Output matrix C
 * @param transpose_a Whether to transpose matrix A
 * @param transpose_b Whether to transpose matrix B
 * @param alpha Scaling factor for A @ B
 * @param beta Scaling factor for C
 * @param stream CUDA stream
 */
template<typename T>
void matmul(
    const Tensor<T>& A,
    const Tensor<T>& B,
    Tensor<T>& C,
    bool transpose_a = false,
    bool transpose_b = false,
    float alpha = 1.0f,
    float beta = 0.0f,
    cudaStream_t stream = nullptr
);

/**
 * @brief Fused matrix multiplication and GELU activation
 * 
 * Computes C = GELU(A @ B) in a single kernel for better performance.
 * 
 * @tparam T Data type (float or half)
 * @param A Input matrix A
 * @param B Input matrix B
 * @param C Output matrix C
 * @param transpose_a Whether to transpose matrix A
 * @param transpose_b Whether to transpose matrix B
 * @param stream CUDA stream
 */
template<typename T>
void mmaGelu(
    const Tensor<T>& A,
    const Tensor<T>& B,
    Tensor<T>& C,
    bool transpose_a = false,
    bool transpose_b = false,
    cudaStream_t stream = nullptr
);

/**
 * @brief Fused matrix multiplication and dropout
 * 
 * Computes C = Dropout(A @ B) in a single kernel for better performance.
 * 
 * @tparam T Data type (float or half)
 * @param A Input matrix A
 * @param B Input matrix B
 * @param C Output matrix C
 * @param dropout_prob Dropout probability
 * @param seed Random seed for dropout
 * @param transpose_a Whether to transpose matrix A
 * @param transpose_b Whether to transpose matrix B
 * @param stream CUDA stream
 */
template<typename T>
void mmaDropout(
    const Tensor<T>& A,
    const Tensor<T>& B,
    Tensor<T>& C,
    float dropout_prob,
    unsigned long long seed,
    bool transpose_a = false,
    bool transpose_b = false,
    cudaStream_t stream = nullptr
);

/**
 * @brief Configuration for MMA operations
 */
struct MMAConfig {
    // Thread block dimensions
    static constexpr int BLOCK_M = 128;
    static constexpr int BLOCK_N = 128;
    static constexpr int BLOCK_K = 32;
    
    // Warp dimensions
    static constexpr int WARP_M = 64;
    static constexpr int WARP_N = 64;
    static constexpr int WARP_K = 32;
    
    // Instruction dimensions
    static constexpr int INST_M = 16;
    static constexpr int INST_N = 8;
    static constexpr int INST_K = 16;
    
    // Pipeline stages
    static constexpr int NUM_STAGES = 3;
    
    // Shared memory configuration
    static constexpr int SMEM_BYTES_PER_STAGE = 
        (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * sizeof(half);
    static constexpr int SMEM_BYTES_TOTAL = 
        SMEM_BYTES_PER_STAGE * NUM_STAGES;
    
    // Performance tuning
    static constexpr bool USE_TENSOR_CORES = true;
    static constexpr bool SPLIT_K_SERIAL = false;
    static constexpr int MIN_BLOCKS_PER_SM = 1;
};

/**
 * @brief Get optimal grid dimensions for MMA operations
 * 
 * @param m Number of rows in output
 * @param n Number of columns in output
 * @return dim3 Grid dimensions
 */
inline dim3 getMMAGridDim(int m, int n) {
    return dim3(
        (m + MMAConfig::BLOCK_M - 1) / MMAConfig::BLOCK_M,
        (n + MMAConfig::BLOCK_N - 1) / MMAConfig::BLOCK_N,
        1
    );
}

/**
 * @brief Get block dimensions for MMA operations
 * 
 * @return dim3 Block dimensions
 */
inline dim3 getMMABlockDim() {
    return dim3(
        MMAConfig::BLOCK_M / MMAConfig::INST_M * 
        MMAConfig::BLOCK_N / MMAConfig::INST_N,
        1,
        1
    );
}

/**
 * @brief Check if tensor cores can be used for given dimensions
 * 
 * @param m Number of rows
 * @param n Number of columns
 * @param k Inner dimension
 * @return bool True if tensor cores can be used
 */
inline bool canUseTensorCores(int m, int n, int k) {
    return (
        m % MMAConfig::INST_M == 0 &&
        n % MMAConfig::INST_N == 0 &&
        k % MMAConfig::INST_K == 0
    );
}

} // namespace ops
} // namespace ltm
