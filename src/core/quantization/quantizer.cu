#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include "core/quantization/quantizer.cuh"
#include "core/utils/cuda_utils.cuh"

namespace ltm {
namespace quantization {

// CUDA kernels for quantization
namespace {

// Kernel for computing min/max values
template<typename T>
__global__ void computeMinMaxKernel(
    const T* __restrict__ input,
    float* __restrict__ min_val,
    float* __restrict__ max_val,
    int size
) {
    extern __shared__ float shared_mem[];
    float* shared_min = shared_mem;
    float* shared_max = &shared_mem[blockDim.x];
    
    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize local min/max
    float thread_min = FLT_MAX;
    float thread_max = -FLT_MAX;
    
    // Process multiple elements per thread
    for (int i = gid; i < size; i += gridDim.x * blockDim.x) {
        float val = static_cast<float>(input[i]);
        thread_min = min(thread_min, val);
        thread_max = max(thread_max, val);
    }
    
    // Store in shared memory
    shared_min[tid] = thread_min;
    shared_max[tid] = thread_max;
    __syncthreads();
    
    // Reduce within block
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_min[tid] = min(shared_min[tid], shared_min[tid + s]);
            shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    
    // Write block result
    if (tid == 0) {
        atomicMin((int*)min_val, __float_as_int(shared_min[0]));
        atomicMax((int*)max_val, __float_as_int(shared_max[0]));
    }
}

// Kernel for linear quantization
template<typename T>
__global__ void linearQuantizeKernel(
    const T* __restrict__ input,
    int8_t* __restrict__ output,
    float scale,
    float zero_point,
    int size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = static_cast<float>(input[idx]);
        output[idx] = static_cast<int8_t>(round(val / scale + zero_point));
    }
}

// Kernel for linear dequantization
template<typename T>
__global__ void linearDequantizeKernel(
    const int8_t* __restrict__ input,
    T* __restrict__ output,
    float scale,
    float zero_point,
    int size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = (static_cast<float>(input[idx]) - zero_point) * scale;
        output[idx] = static_cast<T>(val);
    }
}

// Kernel for symmetric quantization
template<typename T>
__global__ void symmetricQuantizeKernel(
    const T* __restrict__ input,
    int8_t* __restrict__ output,
    float scale,
    int size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = static_cast<float>(input[idx]);
        output[idx] = static_cast<int8_t>(round(val / scale));
    }
}

// Kernel for symmetric dequantization
template<typename T>
__global__ void symmetricDequantizeKernel(
    const int8_t* __restrict__ input,
    T* __restrict__ output,
    float scale,
    int size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = static_cast<float>(input[idx]) * scale;
        output[idx] = static_cast<T>(val);
    }
}

// Kernel for per-channel quantization
template<typename T>
__global__ void perChannelQuantizeKernel(
    const T* __restrict__ input,
    int8_t* __restrict__ output,
    const float* __restrict__ scales,
    const float* __restrict__ zero_points,
    int size,
    int channels,
    int elements_per_channel
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        const int channel = (idx / elements_per_channel) % channels;
        float val = static_cast<float>(input[idx]);
        output[idx] = static_cast<int8_t>(
            round(val / scales[channel] + zero_points[channel])
        );
    }
}

// Kernel for per-channel dequantization
template<typename T>
__global__ void perChannelDequantizeKernel(
    const int8_t* __restrict__ input,
    T* __restrict__ output,
    const float* __restrict__ scales,
    const float* __restrict__ zero_points,
    int size,
    int channels,
    int elements_per_channel
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        const int channel = (idx / elements_per_channel) % channels;
        float val = (static_cast<float>(input[idx]) - zero_points[channel]) 
                   * scales[channel];
        output[idx] = static_cast<T>(val);
    }
}

} // anonymous namespace

template<typename T>
class Quantizer {
public:
    Quantizer(QuantizationConfig config) : config_(config) {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    ~Quantizer() {
        CUDA_CHECK(cudaStreamDestroy(stream_));
    }

    void quantize(const Tensor<T>& input, Tensor<int8_t>& output) {
        if (config_.per_channel) {
            quantizePerChannel(input, output);
        } else {
            quantizePerTensor(input, output);
        }
    }

    void dequantize(const Tensor<int8_t>& input, Tensor<T>& output) {
        if (config_.per_channel) {
            dequantizePerChannel(input, output);
        } else {
            dequantizePerTensor(input, output);
        }
    }

private:
    void quantizePerTensor(const Tensor<T>& input, Tensor<int8_t>& output) {
        // Compute min/max values
        float min_val = FLT_MAX;
        float max_val = -FLT_MAX;
        
        const int block_size = 256;
        const int num_blocks = (input.numel() + block_size - 1) / block_size;
        const int shared_mem_size = 2 * block_size * sizeof(float);
        
        computeMinMaxKernel<T><<<num_blocks, block_size, shared_mem_size, stream_>>>(
            input.data(),
            &min_val,
            &max_val,
            input.numel()
        );
        
        // Compute quantization parameters
        float scale, zero_point;
        if (config_.symmetric) {
            scale = max(abs(min_val), abs(max_val)) / 127.0f;
            zero_point = 0.0f;
            
            symmetricQuantizeKernel<T><<<num_blocks, block_size, 0, stream_>>>(
                input.data(),
                output.data(),
                scale,
                input.numel()
            );
        } else {
            scale = (max_val - min_val) / 255.0f;
            zero_point = -min_val / scale + 128.0f;
            
            linearQuantizeKernel<T><<<num_blocks, block_size, 0, stream_>>>(
                input.data(),
                output.data(),
                scale,
                zero_point,
                input.numel()
            );
        }
        
        // Store quantization parameters
        scales_.resize(1);
        zero_points_.resize(1);
        scales_[0] = scale;
        zero_points_[0] = zero_point;
    }

    void dequantizePerTensor(const Tensor<int8_t>& input, Tensor<T>& output) {
        const int block_size = 256;
        const int num_blocks = (input.numel() + block_size - 1) / block_size;
        
        if (config_.symmetric) {
            symmetricDequantizeKernel<T><<<num_blocks, block_size, 0, stream_>>>(
                input.data(),
                output.data(),
                scales_[0],
                input.numel()
            );
        } else {
            linearDequantizeKernel<T><<<num_blocks, block_size, 0, stream_>>>(
                input.data(),
                output.data(),
                scales_[0],
                zero_points_[0],
                input.numel()
            );
        }
    }

    void quantizePerChannel(const Tensor<T>& input, Tensor<int8_t>& output) {
        const int num_channels = input.shape()[config_.channel_axis];
        const int elements_per_channel = input.numel() / num_channels;
        
        // Compute per-channel scales and zero points
        scales_.resize(num_channels);
        zero_points_.resize(num_channels);
        
        for (int c = 0; c < num_channels; ++c) {
            float min_val = FLT_MAX;
            float max_val = -FLT_MAX;
            
            // Compute min/max for channel
            const int block_size = 256;
            const int num_blocks = (elements_per_channel + block_size - 1) / block_size;
            const int shared_mem_size = 2 * block_size * sizeof(float);
            
            computeMinMaxKernel<T><<<num_blocks, block_size, shared_mem_size, stream_>>>(
                input.data() + c * elements_per_channel,
                &min_val,
                &max_val,
                elements_per_channel
            );
            
            // Compute quantization parameters for channel
            if (config_.symmetric) {
                scales_[c] = max(abs(min_val), abs(max_val)) / 127.0f;
                zero_points_[c] = 0.0f;
            } else {
                scales_[c] = (max_val - min_val) / 255.0f;
                zero_points_[c] = -min_val / scales_[c] + 128.0f;
            }
        }
        
        // Quantize using per-channel parameters
        const int block_size = 256;
        const int num_blocks = (input.numel() + block_size - 1) / block_size;
        
        perChannelQuantizeKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            input.data(),
            output.data(),
            scales_.data(),
            zero_points_.data(),
            input.numel(),
            num_channels,
            elements_per_channel
        );
    }

    void dequantizePerChannel(const Tensor<int8_t>& input, Tensor<T>& output) {
        const int num_channels = input.shape()[config_.channel_axis];
        const int elements_per_channel = input.numel() / num_channels;
        
        const int block_size = 256;
        const int num_blocks = (input.numel() + block_size - 1) / block_size;
        
        perChannelDequantizeKernel<T><<<num_blocks, block_size, 0, stream_>>>(
            input.data(),
            output.data(),
            scales_.data(),
            zero_points_.data(),
            input.numel(),
            num_channels,
            elements_per_channel
        );
    }

    QuantizationConfig config_;
    cudaStream_t stream_;
    std::vector<float> scales_;
    std::vector<float> zero_points_;
};

// Explicit instantiations
template class Quantizer<float>;
template class Quantizer<half>;

} // namespace quantization
} // namespace ltm
