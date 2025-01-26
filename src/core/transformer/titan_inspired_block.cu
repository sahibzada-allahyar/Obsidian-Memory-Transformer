#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "core/transformer/titan_inspired_block.cuh"
#include "core/utils/cuda_utils.cuh"
#include "core/ops/mma_ops.cuh"
#include "core/ops/fused_ops.cuh"

namespace cg = cooperative_groups;

namespace ltm {
namespace transformer {

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

// Kernel for fused feed-forward network with GELU activation
template<typename T>
__global__ void fusedFFNKernel(
    const T* __restrict__ input,
    const T* __restrict__ weight1,
    const T* __restrict__ bias1,
    const T* __restrict__ weight2,
    const T* __restrict__ bias2,
    T* __restrict__ output,
    const int batch_size,
    const int seq_len,
    const int hidden_dim,
    const int ffn_dim
) {
    extern __shared__ char shared_mem[];
    T* shared_input = reinterpret_cast<T*>(shared_mem);
    T* shared_weight = shared_input + hidden_dim;
    T* shared_intermediate = shared_weight + ffn_dim;
    
    const int batch_seq_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (batch_seq_idx >= batch_size * seq_len) return;
    
    // Load input to shared memory
    if (tid < hidden_dim) {
        shared_input[tid] = input[batch_seq_idx * hidden_dim + tid];
    }
    __syncthreads();
    
    // First linear layer + GELU
    float intermediate_val = 0.0f;
    for (int i = tid; i < ffn_dim; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < hidden_dim; ++j) {
            sum += type2float(shared_input[j]) * 
                   type2float(weight1[i * hidden_dim + j]);
        }
        if (bias1 != nullptr) {
            sum += type2float(bias1[i]);
        }
        
        // Apply GELU activation
        const float x = sum;
        const float cdf = 0.5f * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
        shared_intermediate[i] = cuda_cast<T>(x * cdf);
    }
    __syncthreads();
    
    // Second linear layer
    if (tid < hidden_dim) {
        float sum = 0.0f;
        for (int i = 0; i < ffn_dim; ++i) {
            sum += type2float(shared_intermediate[i]) *
                   type2float(weight2[tid * ffn_dim + i]);
        }
        if (bias2 != nullptr) {
            sum += type2float(bias2[tid]);
        }
        output[batch_seq_idx * hidden_dim + tid] = cuda_cast<T>(sum);
    }
}

// Kernel for memory gating
template<typename T>
__global__ void memoryGatingKernel(
    const T* __restrict__ input,
    const T* __restrict__ memory_output,
    const T* __restrict__ gate_weight,
    const T* __restrict__ gate_bias,
    T* __restrict__ output,
    const int batch_size,
    const int seq_len,
    const int hidden_dim
) {
    const int batch_idx = blockIdx.y;
    const int seq_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len || tid >= hidden_dim) return;
    
    const int offset = (batch_idx * seq_len + seq_idx) * hidden_dim + tid;
    
    // Compute gate value
    float gate_sum = 0.0f;
    for (int i = 0; i < hidden_dim; ++i) {
        gate_sum += type2float(input[offset]) * 
                   type2float(gate_weight[tid * hidden_dim + i]);
    }
    if (gate_bias != nullptr) {
        gate_sum += type2float(gate_bias[tid]);
    }
    const float gate = sigmoid(gate_sum);
    
    // Apply gating
    const float in_val = type2float(input[offset]);
    const float mem_val = type2float(memory_output[offset]);
    output[offset] = cuda_cast<T>(gate * mem_val + (1.0f - gate) * in_val);
}

} // anonymous namespace

template<typename T>
TitanBlock<T>::TitanBlock(const TitanBlockConfig& config)
    : config_(config) {
    // Initialize core components
    attention::FlashAttentionConfig flash_config;
    flash_config.hidden_dim = config.hidden_dim;
    flash_config.num_heads = config.num_heads;
    flash_config.head_dim = config.head_dim;
    flash_config.use_bias = config.use_bias;
    flash_config.use_rope = config.use_rotary;
    self_attention_ = std::make_unique<attention::FlashAttention<T>>(flash_config);
    
    attention::MemoryAttentionConfig memory_config;
    memory_config.hidden_dim = config.hidden_dim;
    memory_config.num_heads = config.num_heads;
    memory_config.head_dim = config.head_dim;
    memory_config.use_memory_compression = config.use_memory_compression;
    memory_config.memory_compression_ratio = config.memory_compression_ratio;
    memory_attention_ = std::make_unique<attention::MemoryAttention<T>>(memory_config);
    
    memory::MemoryBankConfig bank_config;
    bank_config.num_slots = config.memory_slots;
    bank_config.slot_dim = config.memory_dim;
    bank_config.update_rate = config.memory_update_rate;
    memory_bank_ = std::make_unique<memory::MemoryBank<T>>(bank_config);
    
    memory::CompressionConfig comp_config;
    comp_config.input_dim = config.hidden_dim;
    comp_config.compressed_dim = config.memory_dim;
    comp_config.compression_ratio = config.memory_compression_ratio;
    compression_gate_ = std::make_unique<memory::CompressionGate<T>>(comp_config);
    
    // Initialize tensors
    ffn_weight1_ = Tensor<T>({config.hidden_dim, config.ffn_dim});
    ffn_weight2_ = Tensor<T>({config.ffn_dim, config.hidden_dim});
    
    if (config.use_bias) {
        ffn_bias1_ = Tensor<T>({config.ffn_dim});
        ffn_bias2_ = Tensor<T>({config.hidden_dim});
    }
    
    if (config.use_layer_norm) {
        norm1_weight_ = Tensor<T>({config.hidden_dim});
        norm1_bias_ = Tensor<T>({config.hidden_dim});
        norm2_weight_ = Tensor<T>({config.hidden_dim});
        norm2_bias_ = Tensor<T>({config.hidden_dim});
    }
    
    if (config.use_memory_gating) {
        memory_gate_weight_ = Tensor<T>({config.hidden_dim, config.hidden_dim});
        memory_gate_bias_ = Tensor<T>({config.hidden_dim});
    }
}

template<typename T>
void TitanBlock<T>::initialize(cudaStream_t stream) {
    // Initialize core components
    self_attention_->initialize(stream);
    memory_attention_->initialize(stream);
    memory_bank_->initialize(stream);
    compression_gate_->initialize(stream);
    
    const int block_size = 256;
    
    // Initialize FFN weights
    const int ffn_size = config_.hidden_dim * config_.ffn_dim;
    const int ffn_blocks = (ffn_size + block_size - 1) / block_size;
    
    initializeParametersKernel<T><<<ffn_blocks, block_size, 0, stream>>>(
        ffn_weight1_.data(),
        config_.hidden_dim,
        config_.ffn_dim,
        1234ULL
    );
    
    initializeParametersKernel<T><<<ffn_blocks, block_size, 0, stream>>>(
        ffn_weight2_.data(),
        config_.ffn_dim,
        config_.hidden_dim,
        1235ULL
    );
    
    // Initialize bias terms
    if (config_.use_bias) {
        ffn_bias1_.fill(0);
        ffn_bias2_.fill(0);
    }
    
    // Initialize layer norm parameters
    if (config_.use_layer_norm) {
        norm1_weight_.fill(cuda_cast<T>(1.0f));
        norm1_bias_.fill(0);
        norm2_weight_.fill(cuda_cast<T>(1.0f));
        norm2_bias_.fill(0);
    }
    
    // Initialize memory gating parameters
    if (config_.use_memory_gating) {
        const int gate_size = config_.hidden_dim * config_.hidden_dim;
        const int gate_blocks = (gate_size + block_size - 1) / block_size;
        
        initializeParametersKernel<T><<<gate_blocks, block_size, 0, stream>>>(
            memory_gate_weight_.data(),
            config_.hidden_dim,
            config_.hidden_dim,
            1236ULL
        );
        
        memory_gate_bias_.fill(0);
    }
    
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void TitanBlock<T>::forward(
    const Tensor<T>& input,
    Tensor<T>& output,
    const Tensor<T>* attention_mask,
    cudaStream_t stream
) {
    const int batch_size = input.shape()[0];
    const int seq_len = input.shape()[1];
    
    // Apply first layer norm
    if (config_.use_layer_norm) {
        layerNorm(
            input,
            norm1_weight_,
            norm1_bias_,
            normalized_input_,
            stream
        );
    }
    
    // Compute attention in parallel if enabled
    if (config_.use_parallel_attention) {
        cudaStream_t attn_stream, mem_stream;
        CUDA_CHECK(cudaStreamCreate(&attn_stream));
        CUDA_CHECK(cudaStreamCreate(&mem_stream));
        
        // Self attention stream
        computeSelfAttention(
            config_.use_layer_norm ? normalized_input_ : input,
            attention_mask,
            attn_stream
        );
        
        // Memory attention stream
        computeMemoryAttention(
            config_.use_layer_norm ? normalized_input_ : input,
            attention_mask,
            mem_stream
        );
        
        CUDA_CHECK(cudaStreamSynchronize(attn_stream));
        CUDA_CHECK(cudaStreamSynchronize(mem_stream));
        
        CUDA_CHECK(cudaStreamDestroy(attn_stream));
        CUDA_CHECK(cudaStreamDestroy(mem_stream));
    } else {
        // Sequential attention
        computeSelfAttention(
            config_.use_layer_norm ? normalized_input_ : input,
            attention_mask,
            stream
        );
        
        computeMemoryAttention(
            config_.use_layer_norm ? normalized_input_ : input,
            attention_mask,
            stream
        );
    }
    
    // Integrate memory if enabled
    if (config_.use_memory_gating) {
        integrateMemory(input, output, stream);
    } else {
        // Simple addition of attention outputs
        ops::tensorAdd(
            self_attention_output_,
            memory_output_,
            output,
            1.0f,
            1.0f,
            stream
        );
    }
    
    // Apply second layer norm
    if (config_.use_layer_norm) {
        layerNorm(
            output,
            norm2_weight_,
            norm2_bias_,
            normalized_input_,
            stream
        );
    }
    
    // Compute FFN
    computeFFN(
        config_.use_layer_norm ? normalized_input_ : output,
        stream
    );
}

template<typename T>
void TitanBlock<T>::computeFFN(
    const Tensor<T>& input,
    cudaStream_t stream
) {
    const int batch_size = input.shape()[0];
    const int seq_len = input.shape()[1];
    
    const int block_size = 256;
    const dim3 grid(batch_size * seq_len);
    const int shared_mem_size = (config_.hidden_dim + config_.ffn_dim * 2) * sizeof(T);
    
    fusedFFNKernel<T><<<grid, block_size, shared_mem_size, stream>>>(
        input.data(),
        ffn_weight1_.data(),
        config_.use_bias ? ffn_bias1_.data() : nullptr,
        ffn_weight2_.data(),
        config_.use_bias ? ffn_bias2_.data() : nullptr,
        ffn_intermediate_.data(),
        batch_size,
        seq_len,
        config_.hidden_dim,
        config_.ffn_dim
    );
}

template<typename T>
void TitanBlock<T>::layerNorm(
    const Tensor<T>& input,
    const Tensor<T>& weight,
    const Tensor<T>& bias,
    Tensor<T>& output,
    cudaStream_t stream
) {
    ops::layerNormResidual(
        input,
        input,  // Use input as residual
        weight,
        bias,
        output,
        stream
    );
}

template<typename T>
void TitanBlock<T>::computeMemoryGating(
    const Tensor<T>& input,
    cudaStream_t stream
) {
    const int batch_size = input.shape()[0];
    const int seq_len = input.shape()[1];
    
    const dim3 grid(seq_len, batch_size);
    const dim3 block(config_.hidden_dim);
    
    memoryGatingKernel<T><<<grid, block, 0, stream>>>(
        input.data(),
        memory_output_.data(),
        memory_gate_weight_.data(),
        config_.use_bias ? memory_gate_bias_.data() : nullptr,
        memory_gate_scores_.data(),
        batch_size,
        seq_len,
        config_.hidden_dim
    );
}

template<typename T>
void TitanBlock<T>::integrateMemory(
    const Tensor<T>& input,
    Tensor<T>& output,
    cudaStream_t stream
) {
    // Compute gating scores
    computeMemoryGating(input, stream);
    
    // Apply gating to memory output
    const int batch_size = input.shape()[0];
    const int seq_len = input.shape()[1];
    
    const dim3 grid(seq_len, batch_size);
    const dim3 block(config_.hidden_dim);
    
    memoryGatingKernel<T><<<grid, block, 0, stream>>>(
        input.data(),
        memory_output_.data(),
        memory_gate_weight_.data(),
        config_.use_bias ? memory_gate_bias_.data() : nullptr,
        output.data(),
        batch_size,
        seq_len,
        config_.hidden_dim
    );
}

// Explicit instantiations
template class TitanBlock<float>;
template class TitanBlock<half>;

} // namespace transformer
} // namespace ltm
