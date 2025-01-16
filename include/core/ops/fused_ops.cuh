#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include "core/utils/tensor.cuh"

namespace ltm {
namespace ops {

/**
 * @brief Fused layer normalization and residual connection
 * 
 * Computes: output = LayerNorm(input + residual) * gamma + beta
 * Fuses the residual connection and layer normalization for better performance.
 * 
 * @tparam T Data type (float or half)
 * @param input Input tensor [batch_size, hidden_dim]
 * @param residual Residual tensor [batch_size, hidden_dim]
 * @param gamma Scale parameter [hidden_dim]
 * @param beta Bias parameter [hidden_dim]
 * @param output Output tensor [batch_size, hidden_dim]
 * @param stream CUDA stream
 */
template<typename T>
void layerNormResidual(
    const Tensor<T>& input,
    const Tensor<T>& residual,
    const Tensor<T>& gamma,
    const Tensor<T>& beta,
    Tensor<T>& output,
    cudaStream_t stream = nullptr
);

/**
 * @brief Fused dropout and residual connection
 * 
 * Computes: output = dropout(input) + residual
 * Fuses dropout and residual connection for better performance.
 * 
 * @tparam T Data type (float or half)
 * @param input Input tensor
 * @param residual Residual tensor
 * @param output Output tensor
 * @param dropout_prob Dropout probability
 * @param seed Random seed for dropout
 * @param stream CUDA stream
 */
template<typename T>
void dropoutResidual(
    const Tensor<T>& input,
    const Tensor<T>& residual,
    Tensor<T>& output,
    float dropout_prob,
    unsigned long long seed,
    cudaStream_t stream = nullptr
);

/**
 * @brief Fused bias addition and GELU activation
 * 
 * Computes: output = GELU(input + bias)
 * Fuses bias addition and GELU activation for better performance.
 * 
 * @tparam T Data type (float or half)
 * @param input Input tensor [batch_size, hidden_dim]
 * @param bias Bias tensor [hidden_dim]
 * @param output Output tensor [batch_size, hidden_dim]
 * @param stream CUDA stream
 */
template<typename T>
void biasGeluFused(
    const Tensor<T>& input,
    const Tensor<T>& bias,
    Tensor<T>& output,
    cudaStream_t stream = nullptr
);

/**
 * @brief Fused bias addition and ReLU activation
 * 
 * Computes: output = ReLU(input + bias)
 * Fuses bias addition and ReLU activation for better performance.
 * 
 * @tparam T Data type (float or half)
 * @param input Input tensor [batch_size, hidden_dim]
 * @param bias Bias tensor [hidden_dim]
 * @param output Output tensor [batch_size, hidden_dim]
 * @param stream CUDA stream
 */
template<typename T>
void biasReluFused(
    const Tensor<T>& input,
    const Tensor<T>& bias,
    Tensor<T>& output,
    cudaStream_t stream = nullptr
);

/**
 * @brief Tensor addition with scaling factors
 * 
 * Computes: output = alpha * input1 + beta * input2
 * 
 * @tparam T Data type (float or half)
 * @param input1 First input tensor
 * @param input2 Second input tensor
 * @param output Output tensor
 * @param alpha Scaling factor for input1
 * @param beta Scaling factor for input2
 * @param stream CUDA stream
 */
template<typename T>
void tensorAdd(
    const Tensor<T>& input1,
    const Tensor<T>& input2,
    Tensor<T>& output,
    float alpha = 1.0f,
    float beta = 1.0f,
    cudaStream_t stream = nullptr
);

/**
 * @brief Element-wise multiplication
 * 
 * Computes: output = input1 * input2
 * 
 * @tparam T Data type (float or half)
 * @param input1 First input tensor
 * @param input2 Second input tensor
 * @param output Output tensor
 * @param stream CUDA stream
 */
template<typename T>
void elementwiseMul(
    const Tensor<T>& input1,
    const Tensor<T>& input2,
    Tensor<T>& output,
    cudaStream_t stream = nullptr
);

/**
 * @brief Configuration for fused operations
 */
struct FusedOpsConfig {
    // Thread block configuration
    static constexpr int BLOCK_SIZE = 256;
    
    // Layer normalization
    static constexpr float LAYERNORM_EPS = 1e-5f;
    static constexpr int MIN_ELEMENTS_PER_THREAD = 4;
    
    // Shared memory configuration
    static constexpr int MAX_SHARED_MEM = 48 * 1024;  // 48 KB
    static constexpr int MIN_SHARED_MEM = 16 * 1024;  // 16 KB
    
    // Performance tuning
    static constexpr bool USE_VECTORIZED_LOAD = true;
    static constexpr bool USE_VECTORIZED_STORE = true;
    static constexpr int UNROLL_FACTOR = 4;
    
    // Dropout configuration
    static constexpr int THREADS_PER_ROW = 32;
    static constexpr int ROWS_PER_BLOCK = 4;
};

/**
 * @brief Get optimal block size for fused operations
 * 
 * @param hidden_dim Hidden dimension size
 * @return int Optimal block size
 */
inline int getOptimalBlockSize(int hidden_dim) {
    // Choose block size based on hidden dimension
    if (hidden_dim <= 128) return 128;
    if (hidden_dim <= 256) return 256;
    if (hidden_dim <= 512) return 512;
    return 1024;
}

/**
 * @brief Check if tensor dimensions are compatible with fused operations
 * 
 * @param input Input tensor
 * @param residual Residual tensor
 * @return bool True if dimensions are compatible
 */
template<typename T>
inline bool checkDimensions(
    const Tensor<T>& input,
    const Tensor<T>& residual
) {
    return (
        input.shape() == residual.shape() &&
        input.shape().size() == 2  // Expect [batch_size, hidden_dim]
    );
}

/**
 * @brief Calculate required shared memory size for layer normalization
 * 
 * @param block_size Thread block size
 * @return size_t Required shared memory size in bytes
 */
inline size_t getLayerNormSharedMemSize(int block_size) {
    // Need space for mean and variance
    return 2 * block_size * sizeof(float);
}

} // namespace ops
} // namespace ltm
