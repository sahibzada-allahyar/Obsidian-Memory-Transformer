#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "core/attention/memory_attention.cuh"
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

// Kernel for computing memory attention scores
template<typename T>
__global__ void memoryAttentionScoresKernel(
    const T* __restrict__ query,
    const T* __restrict__ key,
    T* __restrict__ scores,
    const T* __restrict__ mask,
    const int batch_size,
    const int seq_len,
    const int mem_len,
    const int num_heads,
    const int head_dim
) {
    extern __shared__ char shared_mem[];
    T* shared_query = reinterpret_cast<T*>(shared_mem);
    T* shared_key = shared_query + head_dim;
    
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int query_idx = blockIdx.x;
    const int key_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads ||
        query_idx >= seq_len || key_idx >= mem_len) return;
    
    // Load query and key into shared memory
    if (key_idx < head_dim) {
        const int q_offset = ((batch_idx * num_heads + head_idx) * seq_len + query_idx) * head_dim + key_idx;
        const int k_offset = ((batch_idx * num_heads + head_idx) * mem_len + key_idx) * head_dim + threadIdx.x;
        shared_query[key_idx] = query[q_offset];
        shared_key[key_idx] = key[k_offset];
    }
    __syncthreads();
    
    // Compute attention score
    if (key_idx < mem_len) {
        float score = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            score += type2float(shared_query[i]) * type2float(shared_key[i]);
        }
        score /= sqrtf(head_dim);
        
        // Apply mask if provided
        if (mask != nullptr) {
            const int mask_idx = batch_idx * seq_len * mem_len + query_idx * mem_len + key_idx;
            score += type2float(mask[mask_idx]);
        }
        
        // Write output
        const int out_idx = ((batch_idx * num_heads + head_idx) * seq_len + query_idx) * mem_len + key_idx;
        scores[out_idx] = cuda_cast<T>(score);
    }
}

// Kernel for computing memory gating values
template<typename T>
__global__ void memoryGatingKernel(
    const T* __restrict__ input,
    const T* __restrict__ gate_weight,
    T* __restrict__ gate_scores,
    const int batch_size,
    const int seq_len,
    const int hidden_dim
) {
    const int batch_idx = blockIdx.y;
    const int seq_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len) return;
    
    // Compute gating score
    float score = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        const int input_idx = (batch_idx * seq_len + seq_idx) * hidden_dim + i;
        score += type2float(input[input_idx]) * type2float(gate_weight[i]);
    }
    
    // Reduce within block
    extern __shared__ float s_mem[];
    s_mem[tid] = score;
    __syncthreads();
    
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_mem[tid] += s_mem[tid + stride];
        }
        __syncthreads();
    }
    
    // Write output
    if (tid == 0) {
        const int out_idx = batch_idx * seq_len + seq_idx;
        gate_scores[out_idx] = cuda_cast<T>(sigmoid(s_mem[0]));
    }
}

// Kernel for applying rotary position embeddings
template<typename T>
__global__ void rotaryPositionEmbeddingsKernel(
    T* __restrict__ query,
    T* __restrict__ key,
    const int batch_size,
    const int seq_len,
    const int num_heads,
    const int head_dim
) {
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.x;
    const int dim_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads ||
        seq_idx >= seq_len || dim_idx >= head_dim) return;
    
    // Compute position encoding
    const float inv_freq = 1.0f / powf(10000.0f, (2.0f * (dim_idx / 2)) / head_dim);
    const float pos = static_cast<float>(seq_idx) * inv_freq;
    const float sin_pos = sinf(pos);
    const float cos_pos = cosf(pos);
    
    // Apply rotation
    const int offset = ((batch_idx * num_heads + head_idx) * seq_len + seq_idx) * head_dim + dim_idx;
    if (dim_idx % 2 == 0) {
        const float q_val = type2float(query[offset]);
        const float k_val = type2float(key[offset]);
        query[offset] = cuda_cast<T>(q_val * cos_pos);
        key[offset] = cuda_cast<T>(k_val * cos_pos);
    } else {
        const float q_val = type2float(query[offset - 1]);
        const float k_val = type2float(key[offset - 1]);
        query[offset] = cuda_cast<T>(q_val * sin_pos);
        key[offset] = cuda_cast<T>(k_val * sin_pos);
    }
}

} // anonymous namespace

template<typename T>
MemoryAttention<T>::MemoryAttention(
    const MemoryAttentionConfig& config,
    memory::MemoryBank<T>& memory_bank
) : config_(config),
    memory_bank_(memory_bank) {
    // Initialize tensors
    query_weight_ = Tensor<T>({config.hidden_dim, config.hidden_dim});
    key_weight_ = Tensor<T>({config.hidden_dim, config.hidden_dim});
    value_weight_ = Tensor<T>({config.hidden_dim, config.hidden_dim});
    output_weight_ = Tensor<T>({config.hidden_dim, config.hidden_dim});
    
    if (config.use_bias) {
        query_bias_ = Tensor<T>({config.hidden_dim});
        key_bias_ = Tensor<T>({config.hidden_dim});
        value_bias_ = Tensor<T>({config.hidden_dim});
        output_bias_ = Tensor<T>({config.hidden_dim});
    }
    
    if (config.use_memory_compression) {
        const int compressed_dim = static_cast<int>(config.hidden_dim * config.memory_compression_ratio);
        memory_proj_ = Tensor<T>({config.hidden_dim, compressed_dim});
    }
    
    if (config.use_memory_gating) {
        memory_gate_ = Tensor<T>({config.hidden_dim, 1});
    }
}

template<typename T>
void MemoryAttention<T>::initialize(cudaStream_t stream) {
    // Initialize parameters with Kaiming initialization
    const int block_size = 256;
    
    // Initialize attention weights
    const int attn_size = config_.hidden_dim * config_.hidden_dim;
    const int attn_blocks = (attn_size + block_size - 1) / block_size;
    
    initializeParametersKernel<T><<<attn_blocks, block_size, 0, stream>>>(
        query_weight_.data(),
        config_.hidden_dim,
        config_.hidden_dim,
        1234ULL
    );
    
    initializeParametersKernel<T><<<attn_blocks, block_size, 0, stream>>>(
        key_weight_.data(),
        config_.hidden_dim,
        config_.hidden_dim,
        1235ULL
    );
    
    initializeParametersKernel<T><<<attn_blocks, block_size, 0, stream>>>(
        value_weight_.data(),
        config_.hidden_dim,
        config_.hidden_dim,
        1236ULL
    );
    
    initializeParametersKernel<T><<<attn_blocks, block_size, 0, stream>>>(
        output_weight_.data(),
        config_.hidden_dim,
        config_.hidden_dim,
        1237ULL
    );
    
    // Initialize bias terms
    if (config_.use_bias) {
        query_bias_.fill(0);
        key_bias_.fill(0);
        value_bias_.fill(0);
        output_bias_.fill(0);
    }
    
    // Initialize memory compression
    if (config_.use_memory_compression) {
        const int compressed_dim = static_cast<int>(config_.hidden_dim * config_.memory_compression_ratio);
        const int comp_size = config_.hidden_dim * compressed_dim;
        const int comp_blocks = (comp_size + block_size - 1) / block_size;
        
        initializeParametersKernel<T><<<comp_blocks, block_size, 0, stream>>>(
            memory_proj_.data(),
            config_.hidden_dim,
            compressed_dim,
            1238ULL
        );
    }
    
    // Initialize memory gating
    if (config_.use_memory_gating) {
        const int gate_blocks = (config_.hidden_dim + block_size - 1) / block_size;
        
        initializeParametersKernel<T><<<gate_blocks, block_size, 0, stream>>>(
            memory_gate_.data(),
            config_.hidden_dim,
            1,
            1239ULL
        );
    }
    
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void MemoryAttention<T>::forward(
    const Tensor<T>& input,
    Tensor<T>& output,
    const Tensor<T>* attention_mask,
    cudaStream_t stream
) {
    const int batch_size = input.shape()[0];
    const int seq_len = input.shape()[1];
    
    // Project input to Q, K, V
    projectQKV(input, stream);
    
    // Apply position embeddings
    if (config_.use_rotary) {
        applyPositionEmbeddings(stream);
    }
    
    // Compress memory if enabled
    if (config_.use_memory_compression) {
        compressMemory(stream);
    }
    
    // Compute attention scores
    computeAttentionScores(attention_mask, stream);
    
    // Apply attention
    applyAttention(stream);
    
    // Apply memory gating if enabled
    if (config_.use_memory_gating) {
        computeMemoryGating(input, stream);
    }
    
    // Project output
    projectOutput(output, stream);
}

template<typename T>
void MemoryAttention<T>::projectQKV(
    const Tensor<T>& input,
    cudaStream_t stream
) {
    // Project input to Q, K, V using matmul
    ops::matmul(
        input,
        query_weight_,
        query_,
        false,
        false,
        1.0f,
        0.0f,
        stream
    );
    
    ops::matmul(
        input,
        key_weight_,
        key_,
        false,
        false,
        1.0f,
        0.0f,
        stream
    );
    
    ops::matmul(
        input,
        value_weight_,
        value_,
        false,
        false,
        1.0f,
        0.0f,
        stream
    );
    
    // Add bias if enabled
    if (config_.use_bias) {
        ops::tensorAdd(
            query_,
            query_bias_,
            query_,
            1.0f,
            1.0f,
            stream
        );
        
        ops::tensorAdd(
            key_,
            key_bias_,
            key_,
            1.0f,
            1.0f,
            stream
        );
        
        ops::tensorAdd(
            value_,
            value_bias_,
            value_,
            1.0f,
            1.0f,
            stream
        );
    }
}

template<typename T>
void MemoryAttention<T>::computeAttentionScores(
    const Tensor<T>* mask,
    cudaStream_t stream
) {
    const int batch_size = query_.shape()[0];
    const int seq_len = query_.shape()[1];
    const int mem_len = key_.shape()[1];
    
    const dim3 grid(
        seq_len,
        config_.num_heads,
        batch_size
    );
    const dim3 block(mem_len);
    const int shared_mem_size = 2 * config_.head_dim * sizeof(T);
    
    memoryAttentionScoresKernel<T><<<grid, block, shared_mem_size, stream>>>(
        query_.data(),
        key_.data(),
        attention_scores_.data(),
        mask ? mask->data() : nullptr,
        batch_size,
        seq_len,
        mem_len,
        config_.num_heads,
        config_.head_dim
    );
}

// Kernel for computing softmax
template<typename T>
__global__ void softmaxKernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int mem_len
) {
    extern __shared__ float shared_mem[];
    
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || seq_idx >= seq_len) return;
    
    // Load input to shared memory and find max
    float local_max = -INFINITY;
    for (int i = tid; i < mem_len; i += blockDim.x) {
        const int idx = ((batch_idx * num_heads + head_idx) * seq_len + seq_idx) * mem_len + i;
        const float val = type2float(input[idx]);
        local_max = max(local_max, val);
        shared_mem[i] = val;
    }
    
    // Reduce max within block
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride) {
            local_max = max(local_max, shared_mem[tid + stride]);
        }
    }
    __syncthreads();
    
    // Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < mem_len; i += blockDim.x) {
        const float val = expf(shared_mem[i] - local_max);
        shared_mem[i] = val;
        local_sum += val;
    }
    
    // Reduce sum within block
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
    }
    __syncthreads();
    
    // Normalize and write output
    const float inv_sum = 1.0f / shared_mem[0];
    for (int i = tid; i < mem_len; i += blockDim.x) {
        const int idx = ((batch_idx * num_heads + head_idx) * seq_len + seq_idx) * mem_len + i;
        output[idx] = cuda_cast<T>(shared_mem[i] * inv_sum);
    }
}

template<typename T>
void MemoryAttention<T>::applyAttention(cudaStream_t stream) {
    // Apply softmax to attention scores
    const int batch_size = attention_scores_.shape()[0];
    const int num_heads = attention_scores_.shape()[1];
    const int seq_len = attention_scores_.shape()[2];
    const int mem_len = attention_scores_.shape()[3];
    
    const dim3 grid(seq_len, num_heads, batch_size);
    const dim3 block(256);
    const int shared_mem_size = mem_len * sizeof(float);
    
    softmaxKernel<T><<<grid, block, shared_mem_size, stream>>>(
        attention_scores_.data(),
        attention_probs_.data(),
        batch_size,
        num_heads,
        seq_len,
        mem_len
    );
    
    CUDA_CHECK(cudaGetLastError());
    
    // Apply attention to values
    ops::matmul(
        attention_probs_,
        value_,
        attention_output_,
        false,
        false,
        1.0f,
        0.0f,
        stream
    );
}

template<typename T>
void MemoryAttention<T>::projectOutput(
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

template<typename T>
void MemoryAttention<T>::compressMemory(cudaStream_t stream) {
    if (!config_.use_memory_compression) return;
    
    // Project memory to compressed dimension
    ops::matmul(
        memory_bank_.getMemoryBank(),
        memory_proj_,
        key_,
        false,
        false,
        1.0f,
        0.0f,
        stream
    );
}

template<typename T>
void MemoryAttention<T>::computeMemoryGating(
    const Tensor<T>& input,
    cudaStream_t stream
) {
    if (!config_.use_memory_gating) return;
    
    const int batch_size = input.shape()[0];
    const int seq_len = input.shape()[1];
    const int block_size = 256;
    const dim3 grid(seq_len, batch_size);
    
    memoryGatingKernel<T><<<grid, block_size, block_size * sizeof(float), stream>>>(
        input.data(),
        memory_gate_.data(),
        memory_gate_scores_.data(),
        batch_size,
        seq_len,
        config_.hidden_dim
    );
}

template<typename T>
void MemoryAttention<T>::applyPositionEmbeddings(cudaStream_t stream) {
    if (!config_.use_rotary) return;
    
    const int batch_size = query_.shape()[0];
    const int seq_len = query_.shape()[1];
    
    const dim3 grid(
        seq_len,
        config_.num_heads,
        batch_size
    );
    const dim3 block(config_.head_dim);
    
    rotaryPositionEmbeddingsKernel<T><<<grid, block, 0, stream>>>(
        query_.data(),
        key_.data(),
        batch_size,
        seq_len,
        config_.num_heads,
        config_.head_dim
    );
}

// Explicit instantiations
template class MemoryAttention<float>;
template class MemoryAttention<half>;

} // namespace attention
} // namespace ltm
