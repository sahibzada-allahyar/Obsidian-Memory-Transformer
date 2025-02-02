#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include "core/utils/tensor.cuh"
#include "core/utils/cuda_utils.cuh"

namespace ltm {
namespace memory {

/**
 * @brief Configuration for compression gate
 */
struct CompressionConfig {
    int input_dim = 768;              // Input dimension
    int compressed_dim = 64;          // Compressed dimension
    float compression_ratio = 0.25f;  // Target compression ratio
    bool use_attention = true;        // Use attention for compression
    bool learn_compression = true;    // Learn compression parameters
    bool use_residual = true;        // Use residual connection
    float dropout_prob = 0.1f;        // Dropout probability
    bool use_layer_norm = true;      // Apply layer normalization
    int num_heads = 1;               // Number of attention heads for compression
    bool use_gating = true;          // Use gating mechanism
};

/**
 * @brief Compression gate for reducing input dimension
 * 
 * Implements a trainable compression mechanism that reduces the dimensionality
 * of input states while preserving important information. Uses attention and
 * gating mechanisms to selectively compress information.
 * 
 * @tparam T Data type (float or half)
 */
template<typename T>
class CompressionGate {
public:
    /**
     * @brief Create compression gate
     * 
     * @param config Compression configuration
     */
    explicit CompressionGate(const CompressionConfig& config);

    /**
     * @brief Initialize parameters
     * 
     * @param stream CUDA stream
     */
    void initialize(cudaStream_t stream = nullptr);

    /**
     * @brief Forward pass
     * 
     * @param input Input tensor [batch_size, seq_len, input_dim]
     * @param output Output tensor [batch_size, seq_len, compressed_dim]
     * @param attention_mask Optional attention mask
     * @param stream CUDA stream
     */
    void forward(
        const Tensor<T>& input,
        Tensor<T>& output,
        const Tensor<T>* attention_mask = nullptr,
        cudaStream_t stream = nullptr
    );

    /**
     * @brief Backward pass
     * 
     * @param grad_output Gradient w.r.t. output [batch_size, seq_len, compressed_dim]
     * @param grad_input Gradient w.r.t. input [batch_size, seq_len, input_dim]
     * @param stream CUDA stream
     */
    void backward(
        const Tensor<T>& grad_output,
        Tensor<T>& grad_input,
        cudaStream_t stream = nullptr
    );

    /**
     * @brief Update parameters
     * 
     * @param learning_rate Learning rate
     * @param stream CUDA stream
     */
    void updateParameters(float learning_rate, cudaStream_t stream = nullptr);

    /**
     * @brief Get compression configuration
     * 
     * @return const CompressionConfig& Configuration
     */
    const CompressionConfig& getConfig() const { return config_; }

    /**
     * @brief Set compression configuration
     * 
     * @param config New configuration
     */
    void setConfig(const CompressionConfig& config) { config_ = config; }

    /**
     * @brief Get compression statistics
     * 
     * @return std::vector<float> Compression statistics
     */
    std::vector<float> getStats() const;

    /**
     * @brief Save compression gate parameters
     * 
     * @param path Path to save parameters
     */
    void save(const std::string& path) const;

    /**
     * @brief Load compression gate parameters
     * 
     * @param path Path to load parameters from
     */
    void load(const std::string& path);

private:
    // Model parameters
    Tensor<T> query_weight_;     // [input_dim, compressed_dim]
    Tensor<T> key_weight_;       // [input_dim, compressed_dim]
    Tensor<T> value_weight_;     // [input_dim, compressed_dim]
    Tensor<T> gate_weight_;      // [input_dim, compressed_dim]
    Tensor<T> output_weight_;    // [compressed_dim, compressed_dim]
    
    // Layer normalization parameters
    Tensor<T> layer_norm_weight_;  // [input_dim]
    Tensor<T> layer_norm_bias_;    // [input_dim]
    
    // Parameter gradients
    Tensor<T> query_weight_grad_;
    Tensor<T> key_weight_grad_;
    Tensor<T> value_weight_grad_;
    Tensor<T> gate_weight_grad_;
    Tensor<T> output_weight_grad_;
    Tensor<T> layer_norm_weight_grad_;
    Tensor<T> layer_norm_bias_grad_;
    
    // Intermediate buffers
    Tensor<T> attention_scores_;
    Tensor<T> gate_scores_;
    Tensor<T> normalized_input_;
    Tensor<T> compressed_state_;
    
    // Configuration
    CompressionConfig config_;
    
    // Helper functions
    void initializeParameters(cudaStream_t stream);
    
    void computeAttention(
        const Tensor<T>& input,
        const Tensor<T>* mask,
        cudaStream_t stream
    );
    
    void computeGating(
        const Tensor<T>& input,
        cudaStream_t stream
    );
    
    void layerNorm(
        const Tensor<T>& input,
        Tensor<T>& output,
        cudaStream_t stream
    );
};

// Explicit instantiations
extern template class CompressionGate<float>;
extern template class CompressionGate<half>;

} // namespace memory
} // namespace ltm
