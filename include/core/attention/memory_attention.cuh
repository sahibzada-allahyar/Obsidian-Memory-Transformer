#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include "core/utils/tensor.cuh"
#include "core/utils/cuda_utils.cuh"
#include "core/ltm/memory_bank.cuh"

namespace ltm {
namespace attention {

/**
 * @brief Configuration for memory attention
 */
struct MemoryAttentionConfig {
    int hidden_dim = 768;            // Hidden dimension
    int num_heads = 12;              // Number of attention heads
    int head_dim = 64;               // Dimension per head
    float dropout_prob = 0.1f;       // Attention dropout probability
    bool use_bias = true;           // Use bias in projections
    bool scale_by_dim = true;       // Scale attention by sqrt(head_dim)
    bool use_rotary = true;         // Use rotary position embeddings
    bool use_alibi = false;         // Use ALiBi position bias
    bool use_memory_compression = true;  // Compress memory before attention
    float memory_compression_ratio = 0.5f;  // Memory compression ratio
    bool use_memory_gating = true;  // Gate memory attention contribution
    int max_memory_length = 16384;  // Maximum memory context length
};

/**
 * @brief Memory attention module
 * 
 * Implements attention between the current input context and the memory bank,
 * allowing the model to access and integrate information from long-term memory.
 * 
 * @tparam T Data type (float or half)
 */
template<typename T>
class MemoryAttention {
public:
    /**
     * @brief Create memory attention module
     * 
     * @param config Attention configuration
     * @param memory_bank Memory bank reference
     */
    MemoryAttention(
        const MemoryAttentionConfig& config,
        memory::MemoryBank<T>& memory_bank
    );

    /**
     * @brief Initialize parameters
     * 
     * @param stream CUDA stream
     */
    void initialize(cudaStream_t stream = nullptr);

    /**
     * @brief Forward pass
     * 
     * @param input Input tensor [batch_size, seq_len, hidden_dim]
     * @param output Output tensor [batch_size, seq_len, hidden_dim]
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
     * @param grad_output Gradient w.r.t. output [batch_size, seq_len, hidden_dim]
     * @param grad_input Gradient w.r.t. input [batch_size, seq_len, hidden_dim]
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
     * @brief Get attention configuration
     * 
     * @return const MemoryAttentionConfig& Configuration
     */
    const MemoryAttentionConfig& getConfig() const { return config_; }

    /**
     * @brief Set attention configuration
     * 
     * @param config New configuration
     */
    void setConfig(const MemoryAttentionConfig& config) { config_ = config; }

    /**
     * @brief Get attention statistics
     * 
     * @return std::vector<float> Attention statistics
     */
    std::vector<float> getStats() const;

    /**
     * @brief Save attention parameters
     * 
     * @param path Path to save parameters
     */
    void save(const std::string& path) const;

    /**
     * @brief Load attention parameters
     * 
     * @param path Path to load parameters from
     */
    void load(const std::string& path);

private:
    // Model parameters
    Tensor<T> query_weight_;     // [hidden_dim, hidden_dim]
    Tensor<T> key_weight_;       // [hidden_dim, hidden_dim]
    Tensor<T> value_weight_;     // [hidden_dim, hidden_dim]
    Tensor<T> output_weight_;    // [hidden_dim, hidden_dim]
    
    // Optional bias parameters
    Tensor<T> query_bias_;       // [hidden_dim]
    Tensor<T> key_bias_;         // [hidden_dim]
    Tensor<T> value_bias_;       // [hidden_dim]
    Tensor<T> output_bias_;      // [hidden_dim]
    
    // Memory compression parameters
    Tensor<T> memory_proj_;      // [hidden_dim, compressed_dim]
    Tensor<T> memory_gate_;      // [hidden_dim, 1]
    
    // Parameter gradients
    Tensor<T> query_weight_grad_;
    Tensor<T> key_weight_grad_;
    Tensor<T> value_weight_grad_;
    Tensor<T> output_weight_grad_;
    Tensor<T> query_bias_grad_;
    Tensor<T> key_bias_grad_;
    Tensor<T> value_bias_grad_;
    Tensor<T> output_bias_grad_;
    Tensor<T> memory_proj_grad_;
    Tensor<T> memory_gate_grad_;
    
    // Intermediate buffers
    Tensor<T> query_;           // [batch_size, seq_len, hidden_dim]
    Tensor<T> key_;            // [batch_size, mem_len, hidden_dim]
    Tensor<T> value_;          // [batch_size, mem_len, hidden_dim]
    Tensor<T> attention_scores_;  // [batch_size, num_heads, seq_len, mem_len]
    Tensor<T> attention_probs_;   // [batch_size, num_heads, seq_len, mem_len]
    Tensor<T> attention_output_;  // [batch_size, seq_len, hidden_dim]
    Tensor<T> memory_gate_scores_;  // [batch_size, seq_len, 1]
    
    // Configuration and state
    MemoryAttentionConfig config_;
    memory::MemoryBank<T>& memory_bank_;
    
    // Helper functions
    void initializeParameters(cudaStream_t stream);
    
    void projectQKV(
        const Tensor<T>& input,
        cudaStream_t stream
    );
    
    void computeAttentionScores(
        const Tensor<T>* mask,
        cudaStream_t stream
    );
    
    void applyAttention(cudaStream_t stream);
    
    void projectOutput(
        Tensor<T>& output,
        cudaStream_t stream
    );
    
    void compressMemory(cudaStream_t stream);
    
    void computeMemoryGating(
        const Tensor<T>& input,
        cudaStream_t stream
    );
    
    void applyPositionEmbeddings(cudaStream_t stream);
};

// Explicit instantiations
extern template class MemoryAttention<float>;
extern template class MemoryAttention<half>;

} // namespace attention
} // namespace ltm
