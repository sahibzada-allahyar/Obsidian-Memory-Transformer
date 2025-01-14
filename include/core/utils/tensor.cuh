#pragma once

#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "core/utils/cuda_utils.cuh"

namespace ltm {

/**
 * @brief CUDA tensor class with automatic memory management
 * 
 * Provides a high-level interface for managing GPU memory and tensor operations.
 * Supports both float and half precision types.
 */
template<typename T>
class Tensor {
public:
    /**
     * @brief Create tensor with given shape
     * 
     * @param shape Vector of dimensions
     */
    explicit Tensor(const std::vector<int>& shape)
        : shape_(shape), stride_(computeStrides(shape)) {
        allocateMemory();
    }

    /**
     * @brief Create tensor with given shape and data
     * 
     * @param shape Vector of dimensions
     * @param data Pointer to data (will be copied to device)
     * @param own_memory Whether tensor should own the memory
     */
    Tensor(
        const std::vector<int>& shape,
        const T* data,
        bool own_memory = true
    ) : shape_(shape),
        stride_(computeStrides(shape)),
        own_memory_(own_memory) {
        if (own_memory_) {
            allocateMemory();
            copyFromHost(data);
        } else {
            data_ = const_cast<T*>(data);
        }
    }

    /**
     * @brief Move constructor
     */
    Tensor(Tensor&& other) noexcept
        : data_(other.data_),
          shape_(std::move(other.shape_)),
          stride_(std::move(other.stride_)),
          own_memory_(other.own_memory_) {
        other.data_ = nullptr;
        other.own_memory_ = false;
    }

    /**
     * @brief Move assignment
     */
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            freeMemory();
            data_ = other.data_;
            shape_ = std::move(other.shape_);
            stride_ = std::move(other.stride_);
            own_memory_ = other.own_memory_;
            other.data_ = nullptr;
            other.own_memory_ = false;
        }
        return *this;
    }

    /**
     * @brief Destructor
     */
    ~Tensor() {
        freeMemory();
    }

    // Disable copy operations
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    /**
     * @brief Get raw pointer to device memory
     */
    T* data() const { return data_; }

    /**
     * @brief Get tensor shape
     */
    const std::vector<int>& shape() const { return shape_; }

    /**
     * @brief Get tensor strides
     */
    const std::vector<int>& stride() const { return stride_; }

    /**
     * @brief Get total number of elements
     */
    size_t numel() const {
        size_t n = 1;
        for (int dim : shape_) {
            n *= dim;
        }
        return n;
    }

    /**
     * @brief Get size of dimension
     */
    int size(int dim) const {
        return shape_[dim];
    }

    /**
     * @brief Copy data from host to device
     */
    void copyFromHost(const T* host_data) {
        CUDA_CHECK(cudaMemcpy(
            data_,
            host_data,
            numel() * sizeof(T),
            cudaMemcpyHostToDevice
        ));
    }

    /**
     * @brief Copy data from device to host
     */
    void copyToHost(T* host_data) const {
        CUDA_CHECK(cudaMemcpy(
            host_data,
            data_,
            numel() * sizeof(T),
            cudaMemcpyDeviceToHost
        ));
    }

    /**
     * @brief Copy data from another tensor
     */
    void copyFrom(const Tensor<T>& other) {
        if (numel() != other.numel()) {
            throw std::runtime_error("Tensor sizes don't match for copy");
        }
        CUDA_CHECK(cudaMemcpy(
            data_,
            other.data_,
            numel() * sizeof(T),
            cudaMemcpyDeviceToDevice
        ));
    }

    /**
     * @brief Fill tensor with value
     */
    void fill(T value) {
        CUDA_CHECK(cudaMemset(
            data_,
            value,
            numel() * sizeof(T)
        ));
    }

    /**
     * @brief Reshape tensor to new dimensions
     * 
     * @param new_shape New shape
     * @return Tensor& Reference to this tensor
     */
    Tensor& reshape(const std::vector<int>& new_shape) {
        size_t new_size = 1;
        for (int dim : new_shape) {
            new_size *= dim;
        }
        if (new_size != numel()) {
            throw std::runtime_error("Invalid reshape dimensions");
        }
        shape_ = new_shape;
        stride_ = computeStrides(new_shape);
        return *this;
    }

    /**
     * @brief Get view of tensor with different shape
     * 
     * @param new_shape New shape
     * @return Tensor New tensor sharing the same memory
     */
    Tensor view(const std::vector<int>& new_shape) const {
        size_t new_size = 1;
        for (int dim : new_shape) {
            new_size *= dim;
        }
        if (new_size != numel()) {
            throw std::runtime_error("Invalid view dimensions");
        }
        return Tensor(new_shape, data_, false);
    }

    /**
     * @brief Transpose tensor
     * 
     * @return Tensor Transposed tensor
     */
    Tensor transpose() const {
        if (shape_.size() != 2) {
            throw std::runtime_error("Transpose only supported for 2D tensors");
        }
        std::vector<int> transposed_shape = {shape_[1], shape_[0]};
        std::vector<int> transposed_stride = {stride_[1], stride_[0]};
        
        Tensor result(transposed_shape);
        
        // Launch efficient transpose kernel
        const int TILE_DIM = 32;
        const int BLOCK_ROWS = 8;
        dim3 grid((shape_[1] + TILE_DIM - 1) / TILE_DIM,
                  (shape_[0] + TILE_DIM - 1) / TILE_DIM);
        dim3 block(TILE_DIM, BLOCK_ROWS);
        
        // Shared memory size with padding to avoid bank conflicts
        const size_t shared_mem_size = (TILE_DIM * (TILE_DIM + 1)) * sizeof(T);
        
        transposeKernel<T><<<grid, block, shared_mem_size>>>(
            data_,
            result.data(),
            shape_[0], // rows
            shape_[1]  // cols
        );
        CUDA_CHECK(cudaGetLastError());
        
        return result;
    }

private:
    // Efficient matrix transpose kernel using shared memory tiling
    template<typename U>
    __global__ static void transposeKernel(
        const U* __restrict__ input,
        U* __restrict__ output,
        const int rows,
        const int cols
    ) {
        __shared__ U tile[32][33]; // +1 padding to avoid bank conflicts
        
        const int x = blockIdx.x * 32 + threadIdx.x;
        const int y = blockIdx.y * 32 + threadIdx.y;
        
        // Load tile into shared memory with coalesced reads
        for (int j = 0; j < 32; j += 8) {
            if (y + j < rows && x < cols) {
                tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * cols + x];
            }
        }
        __syncthreads();
        
        // Write transposed tile with coalesced writes
        const int out_x = blockIdx.y * 32 + threadIdx.x;
        const int out_y = blockIdx.x * 32 + threadIdx.y;
        
        for (int j = 0; j < 32; j += 8) {
            if (out_y + j < cols && out_x < rows) {
                output[(out_y + j) * rows + out_x] = tile[threadIdx.x][threadIdx.y + j];
            }
        }
    }

    T* data_ = nullptr;
    std::vector<int> shape_;
    std::vector<int> stride_;
    bool own_memory_ = true;

    /**
     * @brief Compute strides for given shape
     */
    static std::vector<int> computeStrides(const std::vector<int>& shape) {
        std::vector<int> stride(shape.size());
        int curr_stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            stride[i] = curr_stride;
            curr_stride *= shape[i];
        }
        return stride;
    }

    /**
     * @brief Allocate device memory
     */
    void allocateMemory() {
        if (numel() > 0) {
            CUDA_CHECK(cudaMalloc(&data_, numel() * sizeof(T)));
        }
    }

    /**
     * @brief Free device memory
     */
    void freeMemory() {
        if (own_memory_ && data_) {
            CUDA_CHECK(cudaFree(data_));
            data_ = nullptr;
        }
    }
};

// Common tensor types
using FloatTensor = Tensor<float>;
using HalfTensor = Tensor<half>;

/**
 * @brief Create tensor from host data
 * 
 * @tparam T Data type
 * @param shape Tensor shape
 * @param data Host data pointer
 * @return Tensor<T> New tensor with copied data
 */
template<typename T>
Tensor<T> tensorFromHost(
    const std::vector<int>& shape,
    const T* data
) {
    Tensor<T> tensor(shape);
    tensor.copyFromHost(data);
    return tensor;
}

/**
 * @brief Create tensor filled with zeros
 * 
 * @tparam T Data type
 * @param shape Tensor shape
 * @return Tensor<T> Zero-initialized tensor
 */
template<typename T>
Tensor<T> zeros(const std::vector<int>& shape) {
    Tensor<T> tensor(shape);
    tensor.fill(0);
    return tensor;
}

/**
 * @brief Create tensor filled with ones
 * 
 * @tparam T Data type
 * @param shape Tensor shape
 * @return Tensor<T> One-initialized tensor
 */
template<typename T>
Tensor<T> ones(const std::vector<int>& shape) {
    Tensor<T> tensor(shape);
    tensor.fill(1);
    return tensor;
}

} // namespace ltm
