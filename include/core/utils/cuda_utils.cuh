#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>
#include <sstream>

namespace ltm {

/**
 * @brief Check CUDA error and throw exception if any
 * 
 * @param error CUDA error code
 * @param file Source file name
 * @param line Source line number
 */
inline void checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        std::stringstream ss;
        ss << "CUDA error " << cudaGetErrorString(error)
           << " at " << file << ":" << line;
        throw std::runtime_error(ss.str());
    }
}

/**
 * @brief Check cuBLAS error and throw exception if any
 * 
 * @param error cuBLAS error code
 * @param file Source file name
 * @param line Source line number
 */
inline void checkCublasError(cublasStatus_t error, const char* file, int line) {
    if (error != CUBLAS_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "cuBLAS error " << error
           << " at " << file << ":" << line;
        throw std::runtime_error(ss.str());
    }
}

// Macro for CUDA error checking
#define CUDA_CHECK(err) checkCudaError(err, __FILE__, __LINE__)

// Macro for cuBLAS error checking
#define CUBLAS_CHECK(err) checkCublasError(err, __FILE__, __LINE__)

/**
 * @brief Get optimal block size for CUDA kernel
 * 
 * @param func Kernel function pointer
 * @param dynamic_smem_size Dynamic shared memory size
 * @return int Optimal block size
 */
template<typename F>
inline int getOptimalBlockSize(F func, size_t dynamic_smem_size = 0) {
    int min_grid_size;
    int block_size;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size,
        &block_size,
        func,
        dynamic_smem_size
    ));
    return block_size;
}

/**
 * @brief Get grid size for given problem size and block size
 * 
 * @param n Total number of elements
 * @param block_size Thread block size
 * @return int Grid size
 */
inline int getGridSize(int n, int block_size) {
    return (n + block_size - 1) / block_size;
}

/**
 * @brief Convert data type to float
 * 
 * @tparam T Input type
 * @param x Input value
 * @return float Converted value
 */
template<typename T>
__device__ __forceinline__ float type2float(T x);

// Specializations for different types
template<>
__device__ __forceinline__ float type2float<float>(float x) {
    return x;
}

template<>
__device__ __forceinline__ float type2float<half>(half x) {
    return __half2float(x);
}

/**
 * @brief Convert float to target type
 * 
 * @tparam T Target type
 * @param x Float value
 * @return T Converted value
 */
template<typename T>
__device__ __forceinline__ T cuda_cast(float x);

// Specializations for different types
template<>
__device__ __forceinline__ float cuda_cast<float>(float x) {
    return x;
}

template<>
__device__ __forceinline__ half cuda_cast<half>(float x) {
    return __float2half(x);
}

/**
 * @brief CUDA memory deleter for unique_ptr
 */
struct CudaDeleter {
    void operator()(void* ptr) const {
        if (ptr) {
            CUDA_CHECK(cudaFree(ptr));
        }
    }
};

/**
 * @brief Unique pointer for CUDA memory
 */
template<typename T>
using CudaUniquePtr = std::unique_ptr<T, CudaDeleter>;

/**
 * @brief Allocate CUDA memory with unique_ptr
 * 
 * @tparam T Data type
 * @param size Number of elements
 * @return CudaUniquePtr<T> Unique pointer to allocated memory
 */
template<typename T>
CudaUniquePtr<T> cudaMakeUnique(size_t size) {
    T* ptr;
    CUDA_CHECK(cudaMalloc(&ptr, size * sizeof(T)));
    return CudaUniquePtr<T>(ptr);
}

/**
 * @brief CUDA stream wrapper
 */
class CudaStream {
public:
    CudaStream() {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    ~CudaStream() {
        if (stream_) {
            CUDA_CHECK(cudaStreamDestroy(stream_));
        }
    }

    // Disable copy
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    // Enable move
    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_) {
        other.stream_ = nullptr;
    }

    CudaStream& operator=(CudaStream&& other) noexcept {
        if (this != &other) {
            if (stream_) {
                CUDA_CHECK(cudaStreamDestroy(stream_));
            }
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief Get raw CUDA stream
     */
    cudaStream_t get() const { return stream_; }

    /**
     * @brief Synchronize stream
     */
    void synchronize() const {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

private:
    cudaStream_t stream_ = nullptr;
};

/**
 * @brief CUDA event wrapper
 */
class CudaEvent {
public:
    CudaEvent() {
        CUDA_CHECK(cudaEventCreate(&event_));
    }

    ~CudaEvent() {
        if (event_) {
            CUDA_CHECK(cudaEventDestroy(event_));
        }
    }

    // Disable copy
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    // Enable move
    CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
        other.event_ = nullptr;
    }

    CudaEvent& operator=(CudaEvent&& other) noexcept {
        if (this != &other) {
            if (event_) {
                CUDA_CHECK(cudaEventDestroy(event_));
            }
            event_ = other.event_;
            other.event_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief Get raw CUDA event
     */
    cudaEvent_t get() const { return event_; }

    /**
     * @brief Record event on stream
     */
    void record(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaEventRecord(event_, stream));
    }

    /**
     * @brief Synchronize event
     */
    void synchronize() {
        CUDA_CHECK(cudaEventSynchronize(event_));
    }

private:
    cudaEvent_t event_ = nullptr;
};

/**
 * @brief Get available GPU memory
 * 
 * @param device_id GPU device ID
 * @return std::pair<size_t, size_t> Free and total memory in bytes
 */
inline std::pair<size_t, size_t> getGpuMemoryInfo(int device_id = 0) {
    size_t free_memory, total_memory;
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaMemGetInfo(&free_memory, &total_memory));
    return {free_memory, total_memory};
}

/**
 * @brief Set GPU device with error checking
 * 
 * @param device_id GPU device ID
 */
inline void setDevice(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
}

/**
 * @brief Get current GPU device
 * 
 * @return int Current device ID
 */
inline int getCurrentDevice() {
    int device_id;
    CUDA_CHECK(cudaGetDevice(&device_id));
    return device_id;
}

/**
 * @brief Get number of available GPUs
 * 
 * @return int Number of GPUs
 */
inline int getDeviceCount() {
    int count;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

} // namespace ltm
