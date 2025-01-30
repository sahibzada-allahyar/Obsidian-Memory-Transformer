#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include "core/utils/tensor.cuh"
#include "core/utils/cuda_utils.cuh"

namespace ltm {
namespace memory {

/**
 * @brief Configuration for memory bank
 */
struct MemoryBankConfig {
    int num_slots = 512;              // Number of memory slots
    int slot_dim = 64;                // Dimension of each memory slot
    float update_rate = 0.9f;         // Memory update rate (alpha)
    int update_interval = 8;          // Steps between memory updates
    bool use_attention_scores = true; // Use attention scores for updates
    float prune_threshold = 0.1f;     // Threshold for pruning unused slots
    bool use_dynamic_slots = false;   // Dynamically adjust number of slots
    int min_slots = 128;              // Minimum number of slots if dynamic
    int max_slots = 1024;             // Maximum number of slots if dynamic
};

/**
 * @brief Memory bank for storing compressed context representations
 * 
 * Implements a trainable memory bank that stores compressed states from
 * previous segments, providing the model with long-term recall capability.
 * 
 * @tparam T Data type (float or half)
 */
template<typename T>
class MemoryBank {
public:
    /**
     * @brief Create memory bank
     * 
     * @param batch_size Batch size
     * @param num_heads Number of attention heads
     * @param num_slots Number of memory slots
     * @param slot_dim Dimension of each slot
     * @param update_interval Steps between memory updates
     */
    MemoryBank(
        int batch_size,
        int num_heads,
        int num_slots,
        int slot_dim,
        int update_interval
    );

    /**
     * @brief Initialize memory bank
     * 
     * @param stream CUDA stream
     */
    void initialize(cudaStream_t stream = nullptr);

    /**
     * @brief Reset memory bank to initial state
     * 
     * @param stream CUDA stream
     */
    void reset(cudaStream_t stream = nullptr);

    /**
     * @brief Store compressed state in memory bank
     * 
     * @param state Compressed state to store [batch_size, seq_len, hidden_dim]
     * @param attention_scores Optional attention scores for update weighting
     * @param stream CUDA stream
     */
    void store(
        const Tensor<T>& state,
        const Tensor<T>* attention_scores = nullptr,
        cudaStream_t stream = nullptr
    );

    /**
     * @brief Retrieve relevant memory slots
     * 
     * @param query Query tensor [batch_size, seq_len, hidden_dim]
     * @param output Output tensor [batch_size, num_slots, hidden_dim]
     * @param stream CUDA stream
     */
    void retrieve(
        const Tensor<T>& query,
        Tensor<T>& output,
        cudaStream_t stream = nullptr
    );

    /**
     * @brief Update memory slots
     * 
     * @param new_values New values for memory slots
     * @param indices Indices of slots to update
     * @param stream CUDA stream
     */
    void update(
        const Tensor<T>& new_values,
        const Tensor<int>& indices,
        cudaStream_t stream = nullptr
    );

    /**
     * @brief Prune unused memory slots
     * 
     * @param usage_threshold Usage threshold for pruning
     * @param stream CUDA stream
     */
    void prune(
        float usage_threshold,
        cudaStream_t stream = nullptr
    );

    /**
     * @brief Get current memory bank state
     * 
     * @return const Tensor<T>& Memory bank tensor
     */
    const Tensor<T>& getMemoryBank() const { return memory_bank_; }

    /**
     * @brief Get memory bank configuration
     * 
     * @return const MemoryBankConfig& Configuration
     */
    const MemoryBankConfig& getConfig() const { return config_; }

    /**
     * @brief Set memory bank configuration
     * 
     * @param config New configuration
     */
    void setConfig(const MemoryBankConfig& config) { config_ = config; }

    /**
     * @brief Get memory usage statistics
     * 
     * @return std::vector<float> Usage statistics for each slot
     */
    std::vector<float> getUsageStats() const;

    /**
     * @brief Save memory bank state
     * 
     * @param path Path to save state
     */
    void save(const std::string& path) const;

    /**
     * @brief Load memory bank state
     * 
     * @param path Path to load state from
     */
    void load(const std::string& path);

private:
    // Memory bank state
    Tensor<T> memory_bank_;      // [batch_size, num_slots, slot_dim]
    Tensor<T> usage_counts_;     // [batch_size, num_slots]
    Tensor<int> age_;           // [batch_size, num_slots]
    
    // Memory update state
    int update_counter_ = 0;
    Tensor<T> temp_storage_;
    
    // Configuration
    MemoryBankConfig config_;
    int batch_size_;
    int num_heads_;
    int num_slots_;
    int slot_dim_;
    int update_interval_;
    
    // Helper functions
    void updateUsageCounts(
        const Tensor<T>& attention_scores,
        cudaStream_t stream
    );
    
    void updateAges(cudaStream_t stream);
    
    void findUnusedSlots(
        Tensor<int>& unused_indices,
        int& num_unused,
        cudaStream_t stream
    );
    
    void compactMemory(cudaStream_t stream);
};

// Explicit instantiations
extern template class MemoryBank<float>;
extern template class MemoryBank<half>;

} // namespace memory
} // namespace ltm
