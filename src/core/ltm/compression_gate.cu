#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "core/ltm/compression_gate.cuh"
#include "core/utils/cuda_utils.cuh"
#include "core/ops/mma_ops.cuh"

namespace cg = cooperative_groups;

namespace ltm {
namespace memory {

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

// Kernel for computing attention scores
template<typename T>
__global__ void computeAttentionScoresKernel(
    const T* __restrict__ query,
    const T* __restrict__ key,
    T* __restrict__ scores,
    const T* __restrict__ mask,
    const int batch_size,
    const int seq_len,
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
        query_idx >= seq_len || key_idx >= seq_len) return;
    
    // Load query and key into shared memory
    if (key_idx < head_dim) {
        const int q_offset = ((batch_idx * num_heads + head_idx) * seq_len + query_idx) * head_dim + key_idx;
        const int k_offset = ((batch_idx * num_heads + head_idx) * seq_len + key_idx) * head_dim + threadIdx.x;
        shared_query[key_idx] = query[q_offset];
        shared_key[key_idx] = key[k_offset];
    }
    __syncthreads();
    
    // Compute attention score
    if (key_idx < seq_len) {
        float score = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            score += type2float(shared_query[i]) * type2float(shared_key[i]);
        }
        score /= sqrtf(head_dim);
        
        // Apply mask if provided
        if (mask != nullptr) {
            const int mask_idx = batch_idx * seq_len * seq_len + query_idx * seq_len + key_idx;
            score += type2float(mask[mask_idx]);
        }
        
        // Write output
        const int out_idx = ((batch_idx * num_heads + head_idx) * seq_len + query_idx) * seq_len + key_idx;
        scores[out_idx] = cuda_cast<T>(score);
    }
}

// Kernel for computing gating values
template<typename T>
__global__ void computeGatingKernel(
    const T* __restrict__ input,
    const T* __restrict__ gate_weight,
    T* __restrict__ gate_scores,
    const int batch_size,
    const int seq_len,
    const int input_dim,
    const int compressed_dim
) {
    const int batch_idx = blockIdx.z;
    const int seq_idx = blockIdx.y;
    const int comp_idx = blockIdx.x;
    const int input_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len ||
        comp_idx >= compressed_dim || input_idx >= input_dim) return;
    
    // Compute gating score
    float score = 0.0f;
    const int input_offset = (batch_idx * seq_len + seq_idx) * input_dim + input_idx;
    const int weight_offset = comp_idx * input_dim + input_idx;
    
    score += type2float(input[input_offset]) * type2float(gate_weight[weight_offset]);
    
    // Reduce within warp
    score = warpReduceSum(score);
    
    // Write output
    if (threadIdx.x == 0) {
        const int out_idx = (batch_idx * seq_len + seq_idx) * compressed_dim + comp_idx;
        gate_scores[out_idx] = cuda_cast<T>(sigmoid(score));
    }
}

// Kernel for applying layer normalization
template<typename T>
__global__ void layerNormKernel(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ output,
    const int batch_size,
    const int seq_len,
    const int hidden_dim
) {
    extern __shared__ char shared_mem[];
    float* s_mean = reinterpret_cast<float*>(shared_mem);
    float* s_var = s_mean + blockDim.x;
    
    const int batch_seq_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (batch_seq_idx >= batch_size * seq_len) return;
    
    // Compute mean
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        const int idx = batch_seq_idx * hidden_dim + i;
        local_sum += type2float(input[idx]);
    }
    s_mean[tid] = local_sum;
    __syncthreads();
    
    // Reduce mean
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_mean[tid] += s_mean[tid + stride];
        }
        __syncthreads();
    }
    const float mean = s_mean[0] / hidden_dim;
    
    // Compute variance
    local_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        const int idx = batch_seq_idx * hidden_dim + i;
        const float val = type2float(input[idx]) - mean;
        local_sum += val * val;
    }
    s_var[tid] = local_sum;
    __syncthreads();
    
    // Reduce variance
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_var[tid] += s_var[tid + stride];
        }
        __syncthreads();
    }
    const float var = s_var[0] / hidden_dim;
    const float inv_std = rsqrtf(var + 1e-5f);
    
    // Normalize and scale
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        const int idx = batch_seq_idx * hidden_dim + i;
        const float normalized = (type2float(input[idx]) - mean) * inv_std;
        output[idx] = cuda_cast<T>(
            normalized * type2float(weight[i]) + type2float(bias[i])
        );
    }
}

} // anonymous namespace

template<typename T>
CompressionGate<T>::CompressionGate(const CompressionConfig& config)
    : config_(config) {
    // Initialize tensors
    query_weight_ = Tensor<T>({config.input_dim, config.compressed_dim});
    key_weight_ = Tensor<T>({config.input_dim, config.compressed_dim});
    value_weight_ = Tensor<T>({config.input_dim, config.compressed_dim});
    gate_weight_ = Tensor<T>({config.input_dim, config.compressed_dim});
    output_weight_ = Tensor<T>({config.compressed_dim, config.compressed_dim});
    
    if (config.use_layer_norm) {
        layer_norm_weight_ = Tensor<T>({config.input_dim});
        layer_norm_bias_ = Tensor<T>({config.input_dim});
    }
    
    // Initialize gradients
    if (config.learn_compression) {
        query_weight_grad_ = Tensor<T>({config.input_dim, config.compressed_dim});
        key_weight_grad_ = Tensor<T>({config.input_dim, config.compressed_dim});
        value_weight_grad_ = Tensor<T>({config.input_dim, config.compressed_dim});
        gate_weight_grad_ = Tensor<T>({config.input_dim, config.compressed_dim});
        output_weight_grad_ = Tensor<T>({config.compressed_dim, config.compressed_dim});
        
        if (config.use_layer_norm) {
            layer_norm_weight_grad_ = Tensor<T>({config.input_dim});
            layer_norm_bias_grad_ = Tensor<T>({config.input_dim});
        }
    }
}

template<typename T>
void CompressionGate<T>::initialize(cudaStream_t stream) {
    // Initialize parameters with Kaiming initialization
    const int block_size = 256;
    
    // Initialize attention weights
    const int attn_size = config_.input_dim * config_.compressed_dim;
    const int attn_blocks = (attn_size + block_size - 1) / block_size;
    
    initializeParametersKernel<T><<<attn_blocks, block_size, 0, stream>>>(
        query_weight_.data(),
        config_.input_dim,
        config_.compressed_dim,
        1234ULL
    );
    
    initializeParametersKernel<T><<<attn_blocks, block_size, 0, stream>>>(
        key_weight_.data(),
        config_.input_dim,
        config_.compressed_dim,
        1235ULL
    );
    
    initializeParametersKernel<T><<<attn_blocks, block_size, 0, stream>>>(
        value_weight_.data(),
        config_.input_dim,
        config_.compressed_dim,
        1236ULL
    );
    
    // Initialize gating weights
    if (config_.use_gating) {
        initializeParametersKernel<T><<<attn_blocks, block_size, 0, stream>>>(
            gate_weight_.data(),
            config_.input_dim,
            config_.compressed_dim,
            1237ULL
        );
    }
    
    // Initialize output weights
    const int out_size = config_.compressed_dim * config_.compressed_dim;
    const int out_blocks = (out_size + block_size - 1) / block_size;
    
    initializeParametersKernel<T><<<out_blocks, block_size, 0, stream>>>(
        output_weight_.data(),
        config_.compressed_dim,
        config_.compressed_dim,
        1238ULL
    );
    
    // Initialize layer norm parameters
    if (config_.use_layer_norm) {
        const int norm_blocks = (config_.input_dim + block_size - 1) / block_size;
        
        // Initialize to ones
        layer_norm_weight_.fill(cuda_cast<T>(1.0f));
        
        // Initialize to zeros
        layer_norm_bias_.fill(cuda_cast<T>(0.0f));
    }
    
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void CompressionGate<T>::forward(
    const Tensor<T>& input,
    Tensor<T>& output,
    const Tensor<T>* attention_mask,
    cudaStream_t stream
) {
    const int batch_size = input.shape()[0];
    const int seq_len = input.shape()[1];
    
    // Apply layer normalization if enabled
    if (config_.use_layer_norm) {
        layerNorm(input, normalized_input_, stream);
    }
    
    // Compute attention if enabled
    if (config_.use_attention) {
        computeAttention(
            config_.use_layer_norm ? normalized_input_ : input,
            attention_mask,
            stream
        );
    }
    
    // Compute gating if enabled
    if (config_.use_gating) {
        computeGating(
            config_.use_layer_norm ? normalized_input_ : input,
            stream
        );
    }
    
    // Final output projection
    ops::matmul(
        compressed_state_,
        output_weight_,
        output,
        false,
        false,
        1.0f,
        0.0f,
        stream
    );
    
    // Add residual if enabled
    if (config_.use_residual) {
        // Project input to compressed dimension first
        Tensor<T> residual({batch_size, seq_len, config_.compressed_dim});
        ops::matmul(
            input,
            value_weight_,
            residual,
            false,
            false,
            1.0f,
            0.0f,
            stream
        );
        
        // Add residual to output
        ops::tensorAdd(
            output,
            residual,
            output,
            1.0f,
            1.0f,
            stream
        );
    }
}

template<typename T>
void CompressionGate<T>::computeAttention(
    const Tensor<T>& input,
    const Tensor<T>* mask,
    cudaStream_t stream
) {
    const int batch_size = input.shape()[0];
    const int seq_len = input.shape()[1];
    const int head_dim = config_.compressed_dim / config_.num_heads;
    
    // Project input to Q, K, V
    Tensor<T> query({batch_size, seq_len, config_.compressed_dim});
    Tensor<T> key({batch_size, seq_len, config_.compressed_dim});
    Tensor<T> value({batch_size, seq_len, config_.compressed_dim});
    
    ops::matmul(input, query_weight_, query, false, false, 1.0f, 0.0f, stream);
    ops::matmul(input, key_weight_, key, false, false, 1.0f, 0.0f, stream);
    ops::matmul(input, value_weight_, value, false, false, 1.0f, 0.0f, stream);
    
    // Compute attention scores
    const dim3 grid(
        seq_len,
        config_.num_heads,
        batch_size
    );
    const dim3 block(seq_len);
    const int shared_mem_size = 2 * head_dim * sizeof(T);
    
    computeAttentionScoresKernel<T><<<grid, block, shared_mem_size, stream>>>(
        query.data(),
        key.data(),
        attention_scores_.data(),
        mask ? mask->data() : nullptr,
        batch_size,
        seq_len,
        config_.num_heads,
        head_dim
    );
    
    // Apply attention to values
    ops::matmul(
        attention_scores_,
        value,
        compressed_state_,
        false,
        false,
        1.0f,
        0.0f,
        stream
    );
}

template<typename T>
void CompressionGate<T>::computeGating(
    const Tensor<T>& input,
    cudaStream_t stream
) {
    const int batch_size = input.shape()[0];
    const int seq_len = input.shape()[1];
    
    const dim3 grid(
        config_.compressed_dim,
        seq_len,
        batch_size
    );
    const dim3 block(config_.input_dim);
    
    computeGatingKernel<T><<<grid, block, 0, stream>>>(
        input.data(),
        gate_weight_.data(),
        gate_scores_.data(),
        batch_size,
        seq_len,
        config_.input_dim,
        config_.compressed_dim
    );
    
    // Apply gating to compressed state
    ops::elementwiseMul(
        compressed_state_,
        gate_scores_,
        compressed_state_,
        stream
    );
}

template<typename T>
void CompressionGate<T>::layerNorm(
    const Tensor<T>& input,
    Tensor<T>& output,
    cudaStream_t stream
) {
    const int batch_size = input.shape()[0];
    const int seq_len = input.shape()[1];
    const int block_size = 256;
    const int num_blocks = batch_size * seq_len;
    const int shared_mem_size = 2 * block_size * sizeof(float);
    
    layerNormKernel<T><<<num_blocks, block_size, shared_mem_size, stream>>>(
        input.data(),
        layer_norm_weight_.data(),
        layer_norm_bias_.data(),
        output.data(),
        batch_size,
        seq_len,
        config_.input_dim
    );
}

// Explicit instantiations
template class CompressionGate<float>;
template class CompressionGate<half>;

} // namespace memory
} // namespace ltm
