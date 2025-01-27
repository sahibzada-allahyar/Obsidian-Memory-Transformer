#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include "core/utils/tensor.cuh"
#include "core/utils/cuda_utils.cuh"

namespace ltm {
namespace attention {

/**
 * @brief Configuration for flash attention
 */
struct FlashAttentionConfig {
    int hidden_dim = 768;            // Hidden dimension
    int num_heads = 12;              // Number of attention heads
    int head_dim = 64;               // Dimension per head
    float dropout_prob = 0.1f;       // Attention dropout probability
    bool use_bias = true;           // Use bias in projections
    bool scale_by_dim = true;       // Scale attention by sqrt(head_dim)
    bool causal = false;            // Use causal attention mask
    int block_size = 64;            // Block size for tiling
    int chunk_size = 1024;          // Chunk size for memory efficiency
    bool use_alibi = false;         // Use ALiBi position bias
    bool use_rope = true;           // Use rotary position embeddings
    bool fuse_qkv = true;          // Fuse QKV projections
    bool fuse_softmax = true;      // Fuse softmax computation
    bool use_flash_v2 = true;      // Use FlashAttention v2 optimizations
};

/**
 * @brief Flash attention module
 * 
 * Implements efficient attention computation with minimal memory overhead
 * by processing attention in chunks and avoiding materializing the full
 * attention matrix.
 * 
 * @tparam T Data type (float or half)
 */
template<typename T>
class FlashAttention {
public:
    /**
     * @brief Create flash attention module
     * 
     * @param config Attention configuration
     */
    explicit FlashAttention(const FlashAttentionConfig& config);

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
     * @return const FlashAttentionConfig& Configuration
     */
    const FlashAttentionConfig& getConfig() const { return config_; }

    /**
     * @brief Set attention configuration
     * 
     * @param config New configuration
     */
    void setConfig(const FlashAttentionConfig& config) { config_ = config; }

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
    Tensor<T> qkv_weight_;      // [3, hidden_dim, hidden_dim]
    Tensor<T> output_weight_;   // [hidden_dim, hidden_dim]
    
    // Optional bias parameters
    Tensor<T> qkv_bias_;        // [3, hidden_dim]
    Tensor<T> output_bias_;     // [hidden_dim]
    
    // Parameter gradients
    Tensor<T> qkv_weight_grad_;
    Tensor<T> output_weight_grad_;
    Tensor<T> qkv_bias_grad_;
    Tensor<T> output_bias_grad_;
    
    // Intermediate buffers
    Tensor<T> qkv_;            // [batch_size, seq_len, 3, hidden_dim]
    Tensor<T> query_;          // [batch_size, num_heads, seq_len, head_dim]
    Tensor<T> key_;           // [batch_size, num_heads, seq_len, head_dim]
    Tensor<T> value_;         // [batch_size, num_heads, seq_len, head_dim]
    Tensor<T> attention_output_;  // [batch_size, seq_len, hidden_dim]
    
    // Tiling state
    struct TileInfo {
        int block_size;       // Block size for tiling
        int num_blocks;       // Number of blocks
        int chunk_size;       // Chunk size for memory efficiency
        int num_chunks;       // Number of chunks
        float scaling;        // Attention scaling factor
    };
    TileInfo tile_info_;
    
    // Configuration
    FlashAttentionConfig config_;
    
    // Helper functions
    void initializeParameters(cudaStream_t stream);
    
    void projectQKV(
        const Tensor<T>& input,
        cudaStream_t stream
    );
    
    void computeAttention(
        const Tensor<T>* mask,
        cudaStream_t stream
    );
    
    void projectOutput(
        Tensor<T>& output,
        cudaStream_t stream
    );
    
    void applyPositionEmbeddings(cudaStream_t stream);
    
    void computeBlockMaxima(
        const Tensor<T>& block,
        Tensor<T>& maxima,
        cudaStream_t stream
    );
    
    void computeBlockSoftmax(
        const Tensor<T>& block,
        const Tensor<T>& maxima,
        Tensor<T>& softmax,
        cudaStream_t stream
    );
    
    void computeChunkAttention(
        const Tensor<T>& query_chunk,
        const Tensor<T>& key_chunk,
        const Tensor<T>& value_chunk,
        Tensor<T>& output_chunk,
        const Tensor<T>* mask_chunk,
        cudaStream_t stream
    );
    
    void updateOutputChunk(
        const Tensor<T>& chunk_output,
        Tensor<T>& final_output,
        int chunk_idx,
        cudaStream_t stream
    );
};

// Explicit instantiations
extern template class FlashAttention<float>;
extern template class FlashAttention<half>;

} // namespace attention
} // namespace ltm
