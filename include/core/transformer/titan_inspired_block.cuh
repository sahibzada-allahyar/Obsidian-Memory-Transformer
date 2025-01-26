#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include "core/utils/tensor.cuh"
#include "core/utils/cuda_utils.cuh"
#include "core/attention/flash_attention.cuh"
#include "core/attention/memory_attention.cuh"
#include "core/ltm/memory_bank.cuh"
#include "core/ltm/compression_gate.cuh"

namespace ltm {
namespace transformer {

/**
 * @brief Configuration for Titan-inspired transformer block
 */
struct TitanBlockConfig {
    // Model dimensions
    int hidden_dim = 768;            // Hidden dimension
    int ffn_dim = 3072;             // Feed-forward dimension
    int num_heads = 12;              // Number of attention heads
    int head_dim = 64;              // Dimension per head
    
    // Memory configuration
    int memory_slots = 512;         // Number of memory slots
    int memory_dim = 64;           // Memory slot dimension
    float memory_update_rate = 0.9f;  // Memory update rate
    bool use_memory_compression = true;  // Use memory compression
    float memory_compression_ratio = 0.5f;  // Memory compression ratio
    
    // Attention configuration
    bool use_flash_attention = true;  // Use flash attention
    bool use_alibi = false;         // Use ALiBi position bias
    bool use_rotary = true;         // Use rotary embeddings
    float dropout_prob = 0.1f;      // Dropout probability
    
    // Architecture configuration
    bool use_parallel_attention = true;  // Run attention layers in parallel
    bool use_memory_gating = true;     // Use gating for memory integration
    bool use_layer_norm = true;        // Use layer normalization
    bool use_bias = true;              // Use bias terms
    bool fuse_operations = true;       // Fuse compatible operations
    
    // Training configuration
    bool learn_memory = true;          // Learn memory parameters
    bool learn_compression = true;     // Learn compression parameters
    bool gradient_checkpointing = true;  // Use gradient checkpointing
};

/**
 * @brief Titan-inspired transformer block
 * 
 * Implements a transformer block with long-term memory capabilities,
 * inspired by Google's Titan architecture. Integrates flash attention,
 * memory bank, and compression mechanisms.
 * 
 * @tparam T Data type (float or half)
 */
template<typename T>
class TitanBlock {
public:
    /**
     * @brief Create Titan-inspired transformer block
     * 
     * @param config Block configuration
     */
    explicit TitanBlock(const TitanBlockConfig& config);

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
     * @brief Get block configuration
     * 
     * @return const TitanBlockConfig& Configuration
     */
    const TitanBlockConfig& getConfig() const { return config_; }

    /**
     * @brief Set block configuration
     * 
     * @param config New configuration
     */
    void setConfig(const TitanBlockConfig& config) { config_ = config; }

    /**
     * @brief Get memory bank
     * 
     * @return memory::MemoryBank<T>& Memory bank reference
     */
    memory::MemoryBank<T>& getMemoryBank() { return *memory_bank_; }

    /**
     * @brief Get block statistics
     * 
     * @return std::vector<float> Block statistics
     */
    std::vector<float> getStats() const;

    /**
     * @brief Save block parameters
     * 
     * @param path Path to save parameters
     */
    void save(const std::string& path) const;

    /**
     * @brief Load block parameters
     * 
     * @param path Path to load parameters from
     */
    void load(const std::string& path);

private:
    // Core components
    std::unique_ptr<attention::FlashAttention<T>> self_attention_;
    std::unique_ptr<attention::MemoryAttention<T>> memory_attention_;
    std::unique_ptr<memory::MemoryBank<T>> memory_bank_;
    std::unique_ptr<memory::CompressionGate<T>> compression_gate_;
    
    // Feed-forward parameters
    Tensor<T> ffn_weight1_;    // [hidden_dim, ffn_dim]
    Tensor<T> ffn_weight2_;    // [ffn_dim, hidden_dim]
    Tensor<T> ffn_bias1_;      // [ffn_dim]
    Tensor<T> ffn_bias2_;      // [hidden_dim]
    
    // Layer normalization parameters
    Tensor<T> norm1_weight_;   // [hidden_dim]
    Tensor<T> norm1_bias_;     // [hidden_dim]
    Tensor<T> norm2_weight_;   // [hidden_dim]
    Tensor<T> norm2_bias_;     // [hidden_dim]
    
    // Memory gating parameters
    Tensor<T> memory_gate_weight_;  // [hidden_dim, hidden_dim]
    Tensor<T> memory_gate_bias_;    // [hidden_dim]
    
    // Parameter gradients
    Tensor<T> ffn_weight1_grad_;
    Tensor<T> ffn_weight2_grad_;
    Tensor<T> ffn_bias1_grad_;
    Tensor<T> ffn_bias2_grad_;
    Tensor<T> norm1_weight_grad_;
    Tensor<T> norm1_bias_grad_;
    Tensor<T> norm2_weight_grad_;
    Tensor<T> norm2_bias_grad_;
    Tensor<T> memory_gate_weight_grad_;
    Tensor<T> memory_gate_bias_grad_;
    
    // Intermediate buffers
    Tensor<T> attention_output_;
    Tensor<T> memory_output_;
    Tensor<T> ffn_intermediate_;
    Tensor<T> normalized_input_;
    Tensor<T> memory_gate_scores_;
    
    // Configuration
    TitanBlockConfig config_;
    
    // Helper functions
    void initializeParameters(cudaStream_t stream);
    
    void computeSelfAttention(
        const Tensor<T>& input,
        const Tensor<T>* mask,
        cudaStream_t stream
    );
    
    void computeMemoryAttention(
        const Tensor<T>& input,
        const Tensor<T>* mask,
        cudaStream_t stream
    );
    
    void computeFFN(
        const Tensor<T>& input,
        cudaStream_t stream
    );
    
    void layerNorm(
        const Tensor<T>& input,
        const Tensor<T>& weight,
        const Tensor<T>& bias,
        Tensor<T>& output,
        cudaStream_t stream
    );
    
    void computeMemoryGating(
        const Tensor<T>& input,
        cudaStream_t stream
    );
    
    void integrateMemory(
        const Tensor<T>& input,
        Tensor<T>& output,
        cudaStream_t stream
    );
};

// Explicit instantiations
extern template class TitanBlock<float>;
extern template class TitanBlock<half>;

} // namespace transformer
} // namespace ltm
