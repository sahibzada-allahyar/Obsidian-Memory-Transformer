#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "core/ltm/memory_bank.cuh"
#include "core/utils/cuda_utils.cuh"

namespace cg = cooperative_groups;

namespace ltm {
namespace memory {

namespace {

// Kernel for initializing memory bank
template<typename T>
__global__ void initializeMemoryKernel(
    T* __restrict__ memory_bank,
    T* __restrict__ usage_counts,
    int* __restrict__ age,
    const int num_slots,
    const int slot_dim,
    const int total_size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size) {
        // Initialize memory slots to small random values
        const float val = (static_cast<float>(idx % 17) - 8.0f) * 0.01f;
        memory_bank[idx] = cuda_cast<T>(val);
    }
    
    // Initialize usage counts and ages
    if (idx < num_slots) {
        usage_counts[idx] = cuda_cast<T>(0.0f);
        age[idx] = 0;
    }
}

// Kernel for storing compressed state
template<typename T>
__global__ void storeStateKernel(
    T* __restrict__ memory_bank,
    const T* __restrict__ state,
    const T* __restrict__ attention_scores,
    const float update_rate,
    const int num_slots,
    const int slot_dim
) {
    const int slot_idx = blockIdx.x;
    const int dim_idx = threadIdx.x;
    
    if (slot_idx >= num_slots || dim_idx >= slot_dim) return;
    
    // Get memory slot and state
    const int mem_offset = slot_idx * slot_dim + dim_idx;
    const int state_offset = dim_idx;
    
    // Compute update weight
    float weight = update_rate;
    if (attention_scores != nullptr) {
        weight *= static_cast<float>(attention_scores[slot_idx]);
    }
    
    // Update memory slot
    const float old_val = type2float(memory_bank[mem_offset]);
    const float new_val = type2float(state[state_offset]);
    memory_bank[mem_offset] = cuda_cast<T>(
        old_val * (1.0f - weight) + new_val * weight
    );
}

// Kernel for retrieving memory slots
template<typename T>
__global__ void retrieveKernel(
    const T* __restrict__ memory_bank,
    const T* __restrict__ query,
    T* __restrict__ output,
    T* __restrict__ usage_counts,
    const int num_slots,
    const int slot_dim,
    const int query_length
) {
    extern __shared__ char shared_mem[];
    T* shared_query = reinterpret_cast<T*>(shared_mem);
    T* shared_scores = reinterpret_cast<T*>(shared_mem + slot_dim * sizeof(T));
    
    const int slot_idx = blockIdx.x;
    const int query_idx = blockIdx.y;
    const int dim_idx = threadIdx.x;
    
    if (slot_idx >= num_slots || query_idx >= query_length || dim_idx >= slot_dim) return;
    
    // Load query into shared memory
    if (dim_idx < slot_dim) {
        shared_query[dim_idx] = query[query_idx * slot_dim + dim_idx];
    }
    __syncthreads();
    
    // Compute attention scores
    if (dim_idx < slot_dim) {
        const float query_val = type2float(shared_query[dim_idx]);
        const float memory_val = type2float(
            memory_bank[slot_idx * slot_dim + dim_idx]
        );
        shared_scores[dim_idx] = cuda_cast<T>(query_val * memory_val);
    }
    __syncthreads();
    
    // Reduce scores
    for (int stride = slot_dim/2; stride > 0; stride >>= 1) {
        if (dim_idx < stride) {
            shared_scores[dim_idx] = cuda_cast<T>(
                type2float(shared_scores[dim_idx]) +
                type2float(shared_scores[dim_idx + stride])
            );
        }
        __syncthreads();
    }
    
    // Write output and update usage counts
    if (dim_idx == 0) {
        const float score = type2float(shared_scores[0]) / sqrtf(slot_dim);
        const int out_idx = query_idx * num_slots + slot_idx;
        output[out_idx] = cuda_cast<T>(score);
        
        // Update usage count
        atomicAdd(
            reinterpret_cast<float*>(&usage_counts[slot_idx]),
            fabsf(score)
        );
    }
}

// Kernel for updating memory slots
template<typename T>
__global__ void updateSlotsKernel(
    T* __restrict__ memory_bank,
    const T* __restrict__ new_values,
    const int* __restrict__ indices,
    const int num_updates,
    const int slot_dim
) {
    const int update_idx = blockIdx.x;
    const int dim_idx = threadIdx.x;
    
    if (update_idx >= num_updates || dim_idx >= slot_dim) return;
    
    const int slot_idx = indices[update_idx];
    const int mem_offset = slot_idx * slot_dim + dim_idx;
    const int val_offset = update_idx * slot_dim + dim_idx;
    
    memory_bank[mem_offset] = new_values[val_offset];
}

// Kernel for updating ages
template<typename T>
__global__ void updateAgesKernel(
    int* __restrict__ age,
    const T* __restrict__ usage_counts,
    const int num_slots
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_slots) return;
    
    // Increment age if slot is unused
    if (type2float(usage_counts[idx]) < 1e-6f) {
        age[idx]++;
    } else {
        age[idx] = 0;
    }
}

// Kernel for finding unused slots
template<typename T>
__global__ void findUnusedSlotsKernel(
    const T* __restrict__ usage_counts,
    const int* __restrict__ age,
    int* __restrict__ unused_indices,
    int* __restrict__ num_unused,
    const float threshold,
    const int max_age,
    const int num_slots
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_slots) return;
    
    // Check if slot is unused
    if (type2float(usage_counts[idx]) < threshold && age[idx] > max_age) {
        const int pos = atomicAdd(num_unused, 1);
        unused_indices[pos] = idx;
    }
}

} // anonymous namespace

template<typename T>
MemoryBank<T>::MemoryBank(
    int batch_size,
    int num_heads,
    int num_slots,
    int slot_dim,
    int update_interval
) : batch_size_(batch_size),
    num_heads_(num_heads),
    num_slots_(num_slots),
    slot_dim_(slot_dim),
    update_interval_(update_interval) {
    // Initialize tensors
    memory_bank_ = Tensor<T>({batch_size, num_slots, slot_dim});
    usage_counts_ = Tensor<T>({batch_size, num_slots});
    age_ = Tensor<int>({batch_size, num_slots});
    temp_storage_ = Tensor<T>({batch_size, num_slots, slot_dim});
    
    // Set default config
    config_.num_slots = num_slots;
    config_.slot_dim = slot_dim;
}

template<typename T>
void MemoryBank<T>::initialize(cudaStream_t stream) {
    const int total_size = batch_size_ * num_slots_ * slot_dim_;
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    initializeMemoryKernel<T><<<num_blocks, block_size, 0, stream>>>(
        memory_bank_.data(),
        usage_counts_.data(),
        age_.data(),
        num_slots_,
        slot_dim_,
        total_size
    );
    
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void MemoryBank<T>::reset(cudaStream_t stream) {
    initialize(stream);
    update_counter_ = 0;
}

template<typename T>
void MemoryBank<T>::store(
    const Tensor<T>& state,
    const Tensor<T>* attention_scores,
    cudaStream_t stream
) {
    // Only update memory every update_interval_ steps
    if (++update_counter_ % update_interval_ != 0) return;
    
    const dim3 grid(num_slots_);
    const dim3 block(slot_dim_);
    
    storeStateKernel<T><<<grid, block, 0, stream>>>(
        memory_bank_.data(),
        state.data(),
        attention_scores ? attention_scores->data() : nullptr,
        config_.update_rate,
        num_slots_,
        slot_dim_
    );
    
    CUDA_CHECK(cudaGetLastError());
    
    // Update usage statistics
    if (attention_scores) {
        updateUsageCounts(*attention_scores, stream);
    }
    updateAges(stream);
    
    // Prune unused slots if needed
    if (config_.use_dynamic_slots) {
        prune(config_.prune_threshold, stream);
    }
}

template<typename T>
void MemoryBank<T>::retrieve(
    const Tensor<T>& query,
    Tensor<T>& output,
    cudaStream_t stream
) {
    const int query_length = query.shape()[1];
    const dim3 grid(num_slots_, query_length);
    const dim3 block(slot_dim_);
    const int shared_mem_size = 2 * slot_dim_ * sizeof(T);
    
    retrieveKernel<T><<<grid, block, shared_mem_size, stream>>>(
        memory_bank_.data(),
        query.data(),
        output.data(),
        usage_counts_.data(),
        num_slots_,
        slot_dim_,
        query_length
    );
    
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void MemoryBank<T>::update(
    const Tensor<T>& new_values,
    const Tensor<int>& indices,
    cudaStream_t stream
) {
    const int num_updates = indices.shape()[0];
    const dim3 grid(num_updates);
    const dim3 block(slot_dim_);
    
    updateSlotsKernel<T><<<grid, block, 0, stream>>>(
        memory_bank_.data(),
        new_values.data(),
        indices.data(),
        num_updates,
        slot_dim_
    );
    
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void MemoryBank<T>::updateUsageCounts(
    const Tensor<T>& attention_scores,
    cudaStream_t stream
) {
    // Reset usage counts
    usage_counts_.fill(0);
    
    // Update based on attention scores
    // Implementation depends on attention score format
}

template<typename T>
void MemoryBank<T>::updateAges(cudaStream_t stream) {
    const int block_size = 256;
    const int num_blocks = (num_slots_ + block_size - 1) / block_size;
    
    updateAgesKernel<T><<<num_blocks, block_size, 0, stream>>>(
        age_.data(),
        usage_counts_.data(),
        num_slots_
    );
    
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void MemoryBank<T>::prune(float usage_threshold, cudaStream_t stream) {
    const int block_size = 256;
    const int num_blocks = (num_slots_ + block_size - 1) / block_size;
    
    // Find unused slots
    Tensor<int> unused_indices({num_slots_});
    Tensor<int> num_unused({1});
    num_unused.fill(0);
    
    findUnusedSlotsKernel<T><<<num_blocks, block_size, 0, stream>>>(
        usage_counts_.data(),
        age_.data(),
        unused_indices.data(),
        num_unused.data(),
        usage_threshold,
        update_interval_ * 10,  // Max age threshold
        num_slots_
    );
    
    CUDA_CHECK(cudaGetLastError());
    
    // Compact memory if needed
    int host_num_unused;
    CUDA_CHECK(cudaMemcpyAsync(
        &host_num_unused,
        num_unused.data(),
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream
    ));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    if (host_num_unused > 0) {
        compactMemory(stream);
    }
}

// Kernel for compacting memory bank
template<typename T>
__global__ void compactMemoryKernel(
    T* __restrict__ memory_bank,
    T* __restrict__ temp_storage,
    const int* __restrict__ unused_indices,
    const int num_unused,
    const int num_slots,
    const int slot_dim
) {
    const int slot_idx = blockIdx.x;
    const int dim_idx = threadIdx.x;
    
    if (slot_idx >= (num_slots - num_unused) || dim_idx >= slot_dim) return;
    
    // Find next valid slot index
    int valid_slot = slot_idx;
    for (int i = 0; i < num_unused; ++i) {
        if (unused_indices[i] <= valid_slot) {
            valid_slot++;
        }
    }
    
    // Copy valid slot to temporary storage
    const int src_offset = valid_slot * slot_dim + dim_idx;
    const int dst_offset = slot_idx * slot_dim + dim_idx;
    temp_storage[dst_offset] = memory_bank[src_offset];
}

template<typename T>
void MemoryBank<T>::compactMemory(cudaStream_t stream) {
    // Get number of unused slots
    int num_unused;
    CUDA_CHECK(cudaMemcpyAsync(
        &num_unused,
        num_unused_.data(),
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream
    ));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Launch compaction kernel
    const dim3 grid(num_slots_ - num_unused);
    const dim3 block(slot_dim_);
    
    compactMemoryKernel<T><<<grid, block, 0, stream>>>(
        memory_bank_.data(),
        temp_storage_.data(),
        unused_indices_.data(),
        num_unused,
        num_slots_,
        slot_dim_
    );
    
    // Copy compacted memory back to main storage
    CUDA_CHECK(cudaMemcpyAsync(
        memory_bank_.data(),
        temp_storage_.data(),
        (num_slots_ - num_unused) * slot_dim_ * sizeof(T),
        cudaMemcpyDeviceToDevice,
        stream
    ));
    
    // Update memory bank size
    num_slots_ -= num_unused;
    
    CUDA_CHECK(cudaGetLastError());
}

// Explicit instantiations
template class MemoryBank<float>;
template class MemoryBank<half>;

} // namespace memory
} // namespace ltm
