#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include "core/quantization/quantizer.cuh"
#include "core/utils/cuda_utils.cuh"

namespace ltm {
namespace quantization {

namespace {

// Kernel for collecting statistics
template<typename T>
__global__ void collectStatsKernel(
    const T* __restrict__ input,
    float* __restrict__ output,
    int size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = static_cast<float>(input[idx]);
    }
}

// Kernel for computing MSE
__global__ void computeMSEKernel(
    const float* __restrict__ original,
    const float* __restrict__ quantized,
    float* __restrict__ mse,
    int size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float diff = original[idx] - quantized[idx];
        mse[idx] = diff * diff;
    }
}

// Kernel for entropy calculation
__global__ void computeHistogramKernel(
    const float* __restrict__ input,
    int* __restrict__ histogram,
    float min_val,
    float max_val,
    int num_bins,
    int size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = input[idx];
        int bin = static_cast<int>((val - min_val) / (max_val - min_val) * num_bins);
        bin = max(0, min(bin, num_bins - 1));
        atomicAdd(&histogram[bin], 1);
    }
}

} // anonymous namespace

void MinMaxCalibrator::collectStats(const void* data, size_t size) {
    // Allocate device memory for stats
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));
    
    // Copy and convert data to float
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    collectStatsKernel<float><<<num_blocks, block_size>>>(
        static_cast<const float*>(data),
        d_data,
        size
    );
    
    // Find min/max using Thrust
    thrust::device_ptr<float> d_ptr(d_data);
    auto minmax = thrust::minmax_element(thrust::device, d_ptr, d_ptr + size);
    
    float local_min = *minmax.first;
    float local_max = *minmax.second;
    
    min_val_ = std::min(min_val_, local_min);
    max_val_ = std::max(max_val_, local_max);
    
    CUDA_CHECK(cudaFree(d_data));
}

void MinMaxCalibrator::computeRanges(float& min_val, float& max_val) {
    min_val = min_val_;
    max_val = max_val_;
}

void MinMaxCalibrator::reset() {
    min_val_ = FLT_MAX;
    max_val_ = -FLT_MAX;
}

void PercentileCalibrator::collectStats(const void* data, size_t size) {
    // Allocate device memory
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size * sizeof(float)));
    
    // Copy and convert data
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    collectStatsKernel<float><<<num_blocks, block_size>>>(
        static_cast<const float*>(data),
        d_data,
        size
    );
    
    // Copy to host for percentile computation
    std::vector<float> h_data(size);
    CUDA_CHECK(cudaMemcpy(
        h_data.data(),
        d_data,
        size * sizeof(float),
        cudaMemcpyDeviceToHost
    ));
    
    // Store values for later percentile computation
    values_.insert(values_.end(), h_data.begin(), h_data.end());
    
    CUDA_CHECK(cudaFree(d_data));
}

void PercentileCalibrator::computeRanges(float& min_val, float& max_val) {
    if (values_.empty()) {
        min_val = 0.0f;
        max_val = 0.0f;
        return;
    }
    
    // Sort values
    std::sort(values_.begin(), values_.end());
    
    // Compute percentile indices
    size_t lower_idx = static_cast<size_t>((100.0f - percentile_) * values_.size() / 100.0f);
    size_t upper_idx = static_cast<size_t>(percentile_ * values_.size() / 100.0f);
    
    // Get percentile values
    min_val = values_[lower_idx];
    max_val = values_[upper_idx];
}

void PercentileCalibrator::reset() {
    values_.clear();
}

void MSECalibrator::collectStats(const void* data, size_t size) {
    // Store data for MSE computation
    const float* float_data = static_cast<const float*>(data);
    values_.insert(values_.end(), float_data, float_data + size);
}

void MSECalibrator::computeRanges(float& min_val, float& max_val) {
    if (values_.empty()) {
        min_val = optimal_min_;
        max_val = optimal_max_;
        return;
    }
    
    // Allocate device memory
    float* d_original;
    float* d_quantized;
    float* d_mse;
    CUDA_CHECK(cudaMalloc(&d_original, values_.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_quantized, values_.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mse, values_.size() * sizeof(float)));
    
    // Copy original data to device
    CUDA_CHECK(cudaMemcpy(
        d_original,
        values_.data(),
        values_.size() * sizeof(float),
        cudaMemcpyHostToDevice
    ));
    
    // Grid/block configuration
    const int block_size = 256;
    const int num_blocks = (values_.size() + block_size - 1) / block_size;
    
    // Search for optimal range
    float best_mse = FLT_MAX;
    const int num_trials = 100;
    
    for (int i = 0; i < num_trials; ++i) {
        // Try different ranges
        float trial_min = thrust::reduce(
            thrust::device,
            thrust::device_pointer_cast(d_original),
            thrust::device_pointer_cast(d_original + values_.size()),
            FLT_MAX,
            thrust::minimum<float>()
        );
        
        float trial_max = thrust::reduce(
            thrust::device,
            thrust::device_pointer_cast(d_original),
            thrust::device_pointer_cast(d_original + values_.size()),
            -FLT_MAX,
            thrust::maximum<float>()
        );
        
        // Simulate quantization
        float scale = (trial_max - trial_min) / 255.0f;
        
        // Quantize and dequantize
        linearQuantizeKernel<float><<<num_blocks, block_size>>>(
            d_original,
            reinterpret_cast<int8_t*>(d_quantized),
            scale,
            -trial_min / scale + 128.0f,
            values_.size()
        );
        
        linearDequantizeKernel<float><<<num_blocks, block_size>>>(
            reinterpret_cast<int8_t*>(d_quantized),
            d_quantized,
            scale,
            -trial_min / scale + 128.0f,
            values_.size()
        );
        
        // Compute MSE
        computeMSEKernel<<<num_blocks, block_size>>>(
            d_original,
            d_quantized,
            d_mse,
            values_.size()
        );
        
        float total_mse = thrust::reduce(
            thrust::device,
            thrust::device_pointer_cast(d_mse),
            thrust::device_pointer_cast(d_mse + values_.size()),
            0.0f,
            thrust::plus<float>()
        );
        
        // Update best range
        if (total_mse < best_mse) {
            best_mse = total_mse;
            optimal_min_ = trial_min;
            optimal_max_ = trial_max;
        }
    }
    
    // Clean up
    CUDA_CHECK(cudaFree(d_original));
    CUDA_CHECK(cudaFree(d_quantized));
    CUDA_CHECK(cudaFree(d_mse));
    
    min_val = optimal_min_;
    max_val = optimal_max_;
}

void MSECalibrator::reset() {
    values_.clear();
    optimal_min_ = 0.0f;
    optimal_max_ = 0.0f;
}

} // namespace quantization
} // namespace ltm
