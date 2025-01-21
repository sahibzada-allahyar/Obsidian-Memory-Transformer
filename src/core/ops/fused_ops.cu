#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "core/ops/fused_ops.cuh"
#include "core/utils/cuda_utils.cuh"

namespace cg = cooperative_groups;

namespace ltm {
namespace ops {

namespace {

// Kernel for fused layer normalization and residual connection
template<typename T>
__global__ void layerNormResidualKernel(
    const T* __restrict__ input,
    const T* __restrict__ residual,
    const T* __restrict__ gamma,
    const T* __restrict__ beta,
    T* __restrict__ output,
    const int batch_size,
    const int hidden_dim
) {
    extern __shared__ float s_mem[];
    float* s_mean = s_mem;
    float* s_var = &s_mem[blockDim.x];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    
    if (bid >= batch_size) return;
    
    // Step 1: Compute mean
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        const int idx = bid * hidden_dim + i;
        local_sum += static_cast<float>(input[idx]) + static_cast<float>(residual[idx]);
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
    
    // Step 2: Compute variance
    local_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        const int idx = bid * hidden_dim + i;
        const float val = static_cast<float>(input[idx]) + 
                         static_cast<float>(residual[idx]) - mean;
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
    const float rsqrt_var = rsqrtf(var + 1e-5f);
    
    // Step 3: Normalize and scale
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        const int idx = bid * hidden_dim + i;
        const float val = static_cast<float>(input[idx]) + 
                         static_cast<float>(residual[idx]);
        const float normalized = (val - mean) * rsqrt_var;
        output[idx] = static_cast<T>(
            normalized * static_cast<float>(gamma[i]) + static_cast<float>(beta[i])
        );
    }
}

// Kernel for fused dropout and residual connection
template<typename T>
__global__ void dropoutResidualKernel(
    const T* __restrict__ input,
    const T* __restrict__ residual,
    T* __restrict__ output,
    const float dropout_prob,
    const unsigned long long seed,
    const int size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // Generate random number using Philox algorithm
    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);
    const float rand = curand_uniform(&state);
    
    const float scale = 1.0f / (1.0f - dropout_prob);
    const float val = static_cast<float>(input[idx]);
    const float res = static_cast<float>(residual[idx]);
    
    output[idx] = static_cast<T>(
        (rand > dropout_prob ? val * scale : 0.0f) + res
    );
}

// Kernel for fused bias and activation
template<typename T, typename Act>
__global__ void biasActivationKernel(
    const T* __restrict__ input,
    const T* __restrict__ bias,
    T* __restrict__ output,
    const int batch_size,
    const int hidden_dim
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_size = batch_size * hidden_dim;
    
    if (tid >= total_size) return;
    
    const int bias_idx = tid % hidden_dim;
    const float val = static_cast<float>(input[tid]) + 
                     static_cast<float>(bias[bias_idx]);
    output[tid] = static_cast<T>(Act::forward(val));
}

// GELU activation functor
struct GELU {
    __device__ static float forward(float x) {
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        const float cdf = 0.5f * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
        return x * cdf;
    }
};

// ReLU activation functor
struct ReLU {
    __device__ static float forward(float x) {
        return x > 0.0f ? x : 0.0f;
    }
};

// Kernel for tensor addition
template<typename T>
__global__ void tensorAddKernel(
    const T* __restrict__ input1,
    const T* __restrict__ input2,
    T* __restrict__ output,
    const float alpha,
    const float beta,
    const int size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    const float val1 = static_cast<float>(input1[idx]);
    const float val2 = static_cast<float>(input2[idx]);
    output[idx] = static_cast<T>(alpha * val1 + beta * val2);
}

// Kernel for element-wise multiplication
template<typename T>
__global__ void elementwiseMulKernel(
    const T* __restrict__ input1,
    const T* __restrict__ input2,
    T* __restrict__ output,
    const int size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    const float val1 = static_cast<float>(input1[idx]);
    const float val2 = static_cast<float>(input2[idx]);
    output[idx] = static_cast<T>(val1 * val2);
}

} // anonymous namespace

template<typename T>
void tensorAdd(
    const Tensor<T>& input1,
    const Tensor<T>& input2,
    Tensor<T>& output,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    const int size = input1.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    tensorAddKernel<T><<<num_blocks, block_size, 0, stream>>>(
        input1.data(),
        input2.data(),
        output.data(),
        alpha,
        beta,
        size
    );
    
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void elementwiseMul(
    const Tensor<T>& input1,
    const Tensor<T>& input2,
    Tensor<T>& output,
    cudaStream_t stream
) {
    const int size = input1.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    elementwiseMulKernel<T><<<num_blocks, block_size, 0, stream>>>(
        input1.data(),
        input2.data(),
        output.data(),
        size
    );
    
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void layerNormResidual(
    const Tensor<T>& input,
    const Tensor<T>& residual,
    const Tensor<T>& gamma,
    const Tensor<T>& beta,
    Tensor<T>& output,
    cudaStream_t stream
) {
    const int batch_size = input.shape()[0];
    const int hidden_dim = input.shape()[1];
    
    const int block_size = 256;
    const int shared_mem_size = 2 * block_size * sizeof(float);
    
    layerNormResidualKernel<T><<<batch_size, block_size, shared_mem_size, stream>>>(
        input.data(),
        residual.data(),
        gamma.data(),
        beta.data(),
        output.data(),
        batch_size,
        hidden_dim
    );
    
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void dropoutResidual(
    const Tensor<T>& input,
    const Tensor<T>& residual,
    Tensor<T>& output,
    float dropout_prob,
    unsigned long long seed,
    cudaStream_t stream
) {
    const int size = input.numel();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    dropoutResidualKernel<T><<<num_blocks, block_size, 0, stream>>>(
        input.data(),
        residual.data(),
        output.data(),
        dropout_prob,
        seed,
        size
    );
    
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void biasGeluFused(
    const Tensor<T>& input,
    const Tensor<T>& bias,
    Tensor<T>& output,
    cudaStream_t stream
) {
    const int batch_size = input.shape()[0];
    const int hidden_dim = input.shape()[1];
    const int total_size = batch_size * hidden_dim;
    
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    biasActivationKernel<T, GELU><<<num_blocks, block_size, 0, stream>>>(
        input.data(),
        bias.data(),
        output.data(),
        batch_size,
        hidden_dim
    );
    
    CUDA_CHECK(cudaGetLastError());
}

template<typename T>
void biasReluFused(
    const Tensor<T>& input,
    const Tensor<T>& bias,
    Tensor<T>& output,
    cudaStream_t stream
) {
    const int batch_size = input.shape()[0];
    const int hidden_dim = input.shape()[1];
    const int total_size = batch_size * hidden_dim;
    
    const int block_size = 256;
    const int num_blocks = (total_size + block_size - 1) / block_size;
    
    biasActivationKernel<T, ReLU><<<num_blocks, block_size, 0, stream>>>(
        input.data(),
        bias.data(),
        output.data(),
        batch_size,
        hidden_dim
    );
    
    CUDA_CHECK(cudaGetLastError());
}

// Explicit instantiations
template void tensorAdd<float>(
    const Tensor<float>&, const Tensor<float>&,
    Tensor<float>&, float, float, cudaStream_t
);
template void tensorAdd<half>(
    const Tensor<half>&, const Tensor<half>&,
    Tensor<half>&, float, float, cudaStream_t
);

template void elementwiseMul<float>(
    const Tensor<float>&, const Tensor<float>&,
    Tensor<float>&, cudaStream_t
);
template void elementwiseMul<half>(
    const Tensor<half>&, const Tensor<half>&,
    Tensor<half>&, cudaStream_t
);

template void layerNormResidual<float>(
    const Tensor<float>&, const Tensor<float>&,
    const Tensor<float>&, const Tensor<float>&,
    Tensor<float>&, cudaStream_t
);
template void layerNormResidual<half>(
    const Tensor<half>&, const Tensor<half>&,
    const Tensor<half>&, const Tensor<half>&,
    Tensor<half>&, cudaStream_t
);

template void dropoutResidual<float>(
    const Tensor<float>&, const Tensor<float>&,
    Tensor<float>&, float, unsigned long long, cudaStream_t
);
template void dropoutResidual<half>(
    const Tensor<half>&, const Tensor<half>&,
    Tensor<half>&, float, unsigned long long, cudaStream_t
);

template void biasGeluFused<float>(
    const Tensor<float>&, const Tensor<float>&,
    Tensor<float>&, cudaStream_t
);
template void biasGeluFused<half>(
    const Tensor<half>&, const Tensor<half>&,
    Tensor<half>&, cudaStream_t
);

template void biasReluFused<float>(
    const Tensor<float>&, const Tensor<float>&,
    Tensor<float>&, cudaStream_t
);
template void biasReluFused<half>(
    const Tensor<half>&, const Tensor<half>&,
    Tensor<half>&, cudaStream_t
);

} // namespace ops
} // namespace ltm
