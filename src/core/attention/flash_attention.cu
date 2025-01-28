#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "core/attention/flash_attention.cuh"
#include "core/utils/cuda_utils.cuh"
#include "core/ops/mma_ops.cuh"

namespace cg = cooperative_groups;

namespace ltm {
namespace attention {

namespace {

// Kernel for initializing parameters with Kaiming initialization
template<typename T>
__global__ void initializeParametersKernel(
    T* __restrict__ weight,
    const int fan_in,
    const int fan_out,
    const unsigned long long seed
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int size = fan_in * fan_out;
    if (idx >= size) return;
    
    // Initialize curand state
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);
    
    // Kaiming initialization
    const float std = sqrtf(2.0f / fan_in);
    const float val = curand_normal(&state) * std;
    weight[idx] = cuda_cast<T>(val);
}

// Kernel for fused QKV projection
template<typename T>
__global__ void fusedQKVProjectionKernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ output,
    const int batch_size,
    const int seq_len,
    const int hidden_dim
) {
    extern __shared__ char shared_mem[];
    T* shared_input = reinterpret_cast<T*>(shared_mem);
    T* shared_weight = shared_input + blockDim.x;
    
    const int batch_idx = blockIdx.z;
    const int seq_idx = blockIdx.y;
    const int head_idx = blockIdx.x;
    const int dim_idx = threadIdx.x;
    
    // Load input and weight to shared memory
    if (dim_idx < hidden_dim) {
        const int input_idx = (batch_idx * seq_len + seq_idx) * hidden_dim + dim_idx;
        shared_input[dim_idx] = input[input_idx];
        
        for (int i = 0; i < 3; ++i) {
            const int weight_idx = (i * hidden_dim + head_idx) * hidden_dim + dim_idx;
            shared_weight[i * hidden_dim + dim_idx] = weight[weight_idx];
        }
    }
    __syncthreads();
    
    // Compute Q, K, V projections
    if (dim_idx < hidden_dim) {
        for (int i = 0; i < 3; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < hidden_dim; ++j) {
                sum += type2float(shared_input[j]) * 
                      type2float(shared_weight[i * hidden_dim + j]);
            }
            
            if (bias != nullptr) {
                sum += type2float(bias[i * hidden_dim + dim_idx]);
            }
            
            const int out_idx = ((batch_idx * 3 + i) * seq_len + seq_idx) * hidden_dim + dim_idx;
            output[out_idx] = cuda_cast<T>(sum);
        }
    }
}

// Kernel for computing block maxima
template<typename T>
__global__ void blockMaximaKernel(
    const T* __restrict__ block,
    T* __restrict__ maxima,
    const int seq_len,
    const int block_size
) {
    extern __shared__ float shared_max[];
    
    const int tid = threadIdx.x;
    const int block_idx = blockIdx.x;
    const int block_start = block_idx * block_size;
    
    // Initialize local maximum
    float local_max = -INFINITY;
    
    // Compute local maximum
    for (int i = tid; i < block_size; i += blockDim.x) {
        const int idx = block_start + i;
        if (idx < seq_len) {
            local_max = max(local_max, type2float(block[idx]));
        }
    }
    
    // Store in shared memory
    shared_max[tid] = local_max;
    __syncthreads();
    
    // Reduce within block
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }
    
    // Write block maximum
    if (tid == 0) {
        maxima[block_idx] = cuda_cast<T>(shared_max[0]);
    }
}

// Kernel for computing block softmax
template<typename T>
__global__ void blockSoftmaxKernel(
    const T* __restrict__ block,
    const T* __restrict__ maxima,
    T* __restrict__ output,
    const int seq_len,
    const int block_size
) {
    extern __shared__ float shared_sum[];
    
    const int tid = threadIdx.x;
    const int block_idx = blockIdx.x;
    const int block_start = block_idx * block_size;
    
    // Get block maximum
    const float block_max = type2float(maxima[block_idx]);
    
    // Compute exponentials and local sum
    float local_sum = 0.0f;
    float local_vals[32];  // Cache for local values
    int local_count = 0;
    
    for (int i = tid; i < block_size; i += blockDim.x) {
        const int idx = block_start + i;
        if (idx < seq_len) {
            const float val = expf(type2float(block[idx]) - block_max);
            local_vals[local_count++] = val;
            local_sum += val;
        }
    }
    
    // Store in shared memory
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    // Reduce within block
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }
    
    // Compute softmax values
    const float inv_sum = 1.0f / shared_sum[0];
    
    // Write normalized values
    for (int i = 0; i < local_count; ++i) {
        const int idx = block_start + tid + i * blockDim.x;
        if (idx < seq_len) {
            output[idx] = cuda_cast<T>(local_vals[i] * inv_sum);
        }
    }
}

// Kernel for computing chunk attention
template<typename T>
__global__ void chunkAttentionKernel(
    const T* __restrict__ query,
    const T* __restrict__ key,
    const T* __restrict__ value,
    T* __restrict__ output,
    const T* __restrict__ mask,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const int chunk_size
) {
    extern __shared__ char shared_mem[];
    T* shared_query = reinterpret_cast<T*>(shared_mem);
    T* shared_key = shared_query + chunk_size * head_dim;
    T* shared_value = shared_key + chunk_size * head_dim;
    
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int chunk_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    const int chunk_start = chunk_idx * chunk_size;
    const int chunk_end = min(chunk_start + chunk_size, seq_len);
    const int chunk_len = chunk_end - chunk_start;
    
    // Load chunk data to shared memory
    for (int i = tid; i < chunk_len * head_dim; i += blockDim.x) {
        const int seq_idx = i / head_dim;
        const int dim_idx = i % head_dim;
        const int q_offset = ((batch_idx * num_heads + head_idx) * seq_len + chunk_start + seq_idx) * head_dim + dim_idx;
        const int k_offset = ((batch_idx * num_heads + head_idx) * seq_len + chunk_start + seq_idx) * head_dim + dim_idx;
        const int v_offset = ((batch_idx * num_heads + head_idx) * seq_len + chunk_start + seq_idx) * head_dim + dim_idx;
        
        shared_query[i] = query[q_offset];
        shared_key[i] = key[k_offset];
        shared_value[i] = value[v_offset];
    }
    __syncthreads();
    
    // Compute attention scores and output for this chunk
    for (int i = tid; i < chunk_len; i += blockDim.x) {
        float output_val = 0.0f;
        float sum = 0.0f;
        
        for (int j = 0; j < chunk_len; ++j) {
            float score = 0.0f;
            
            // Compute attention score
            for (int k = 0; k < head_dim; ++k) {
                score += type2float(shared_query[i * head_dim + k]) *
                        type2float(shared_key[j * head_dim + k]);
            }
            score /= sqrtf(head_dim);
            
            // Apply mask if provided
            if (mask != nullptr) {
                const int mask_idx = batch_idx * seq_len * seq_len +
                                   (chunk_start + i) * seq_len +
                                   (chunk_start + j);
                score += type2float(mask[mask_idx]);
            }
            
            // Apply softmax and compute weighted sum
            const float alpha = expf(score);
            sum += alpha;
            
            for (int k = 0; k < head_dim; ++k) {
                output_val += alpha * type2float(shared_value[j * head_dim + k]);
            }
        }
        
        // Write normalized output
        const float inv_sum = 1.0f / sum;
        const int out_offset = ((batch_idx * num_heads + head_idx) * seq_len + chunk_start + i) * head_dim;
        
        for (int k = 0; k < head_dim; ++k) {
            output[out_offset + k] = cuda_cast<T>(output_val * inv_sum);
        }
    }
}

} // anonymous namespace

template<typename T>
FlashAttention<T>::FlashAttention(const FlashAttentionConfig& config)
    : config_(config) {
    // Initialize tensors
    qkv_weight_ = Tensor<T>({3, config.hidden_dim, config.hidden_dim});
    output_weight_ = Tensor<T>({config.hidden_dim, config.hidden_dim});
    
    if (config.use_bias) {
        qkv_bias_ = Tensor<T>({3, config.hidden_dim});
        output_bias_ = Tensor<T>({config.hidden_dim});
    }
    
    // Initialize tiling info
    tile_info_.block_size = config.block_size;
    tile_info_.chunk_size = config.chunk_size;
    tile_info_.scaling = 1.0f / sqrtf(config.head_dim);
}

template<typename T>
void FlashAttention<T>::initialize(cudaStream_t stream) {
    // Initialize parameters with Kaiming initialization
    const int block_size = 256;
    
    // Initialize QKV weights
    const int qkv_size = 3 * config_.hidden_dim * config_.hidden_dim;
    const int qkv_blocks = (qkv_size + block_size - 1) / block_size;
    
    initializeParametersKernel<T><<<qkv_blocks, block_size, 0, stream>>>(
        qkv_weight_.data(),
        config_.hidden_dim,
        3 * config_.hidden_dim,
        1234ULL
    );
    
    // Initialize output weights
    const int out_size = config_.hidden_dim * config_.hidden_dim;
    const int out_blocks = (out_size + block_size - 1) / block_size;
    
    initializeParametersKernel<T><<<out_blocks, block_size, 0, stream>>>(
        output_weight_.data(),
        config_.hidden_dim,
        config_.hidden_dim,
        1235ULL
    );
    
    // Initialize bias terms
    if (config_.use_bias) {
        qkv_bias_.fill(0);
        output_bias_.fill(0);
    }
    
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void FlashAttention<T>::forward(
    const Tensor<T>& input,
    Tensor<T>& output,
    const Tensor<T>* attention_mask,
    cudaStream_t stream
) {
    const int batch_size = input.shape()[0];
    const int seq_len = input.shape()[1];
    
    // Project input to Q, K, V
    if (config_.fuse_qkv) {
        projectQKV(input, stream);
    }
    
    // Apply position embeddings if enabled
    if (config_.use_rope) {
        applyPositionEmbeddings(stream);
    }
    
    // Compute attention in chunks
    computeAttention(attention_mask, stream);
    
    // Project output
    projectOutput(output, stream);
}

template<typename T>
void FlashAttention<T>::projectQKV(
    const Tensor<T>& input,
    cudaStream_t stream
) {
    const int batch_size = input.shape()[0];
    const int seq_len = input.shape()[1];
    
    const dim3 grid(
        config_.num_heads,
        seq_len,
        batch_size
    );
    const dim3 block(config_.hidden_dim);
    const int shared_mem_size = (config_.hidden_dim + 3 * config_.hidden_dim) * sizeof(T);
    
    fusedQKVProjectionKernel<T><<<grid, block, shared_mem_size, stream>>>(
        input.data(),
        qkv_weight_.data(),
        config_.use_bias ? qkv_bias_.data() : nullptr,
        qkv_.data(),
        batch_size,
        seq_len,
        config_.hidden_dim
    );
}

template<typename T>
void FlashAttention<T>::computeAttention(
    const Tensor<T>* mask,
    cudaStream_t stream
) {
    const int batch_size = query_.shape()[0];
    const int seq_len = query_.shape()[1];
    
    // Process attention in chunks
    const int num_chunks = (seq_len + tile_info_.chunk_size - 1) / tile_info_.chunk_size;
    
    for (int i = 0; i < num_chunks; ++i) {
        const int chunk_start = i * tile_info_.chunk_size;
        const int chunk_end = min(chunk_start + tile_info_.chunk_size, seq_len);
        const int chunk_len = chunk_end - chunk_start;
        
        // Extract chunk tensors
        Tensor<T> query_chunk = query_.slice(chunk_start, chunk_end);
        Tensor<T> key_chunk = key_.slice(chunk_start, chunk_end);
        Tensor<T> value_chunk = value_.slice(chunk_start, chunk_end);
        
        // Compute chunk attention
        computeChunkAttention(
            query_chunk,
            key_chunk,
            value_chunk,
            attention_output_,
            mask,
            stream
        );
    }
}

template<typename T>
void FlashAttention<T>::computeChunkAttention(
    const Tensor<T>& query_chunk,
    const Tensor<T>& key_chunk,
    const Tensor<T>& value_chunk,
    Tensor<T>& output_chunk,
    const Tensor<T>* mask_chunk,
    cudaStream_t stream
) {
    const int batch_size = query_chunk.shape()[0];
    const int chunk_size = query_chunk.shape()[1];
    
    const dim3 grid(
        (chunk_size + tile_info_.block_size - 1) / tile_info_.block_size,
        config_.num_heads,
        batch_size
    );
    const dim3 block(256);
    const int shared_mem_size = 3 * chunk_size * config_.head_dim * sizeof(T);
    
    chunkAttentionKernel<T><<<grid, block, shared_mem_size, stream>>>(
        query_chunk.data(),
        key_chunk.data(),
        value_chunk.data(),
        output_chunk.data(),
        mask_chunk ? mask_chunk->data() : nullptr,
        batch_size,
        config_.num_heads,
        chunk_size,
        config_.head_dim,
        tile_info_.block_size
    );
}

template<typename T>
void FlashAttention<T>::projectOutput(
    Tensor<T>& output,
    cudaStream_t stream
) {
    ops::matmul(
        attention_output_,
        output_weight_,
        output,
        false,
        false,
        1.0f,
        0.0f,
        stream
    );
    
    if (config_.use_bias) {
        ops::tensorAdd(
            output,
            output_bias_,
            output,
            1.0f,
            1.0f,
            stream
        );
    }
}

// Explicit instantiations
template class FlashAttention<float>;
template class FlashAttention<half>;

} // namespace attention
} // namespace ltm
