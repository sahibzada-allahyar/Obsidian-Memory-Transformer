#pragma once

#include <cuda_runtime.h>
#include <vector>
#include "core/utils/tensor.cuh"

namespace ltm {
namespace quantization {

enum class QuantizationPrecision {
    INT8,
    INT4,
    FP16,
    BF16
};

enum class CalibrationMethod {
    MINMAX,
    PERCENTILE,
    MSE,
    ENTROPY
};

struct QuantizationConfig {
    // General settings
    bool enabled = true;
    QuantizationPrecision precision = QuantizationPrecision::INT8;
    bool per_channel = true;
    bool symmetric = true;
    int channel_axis = 0;
    
    // Calibration settings
    CalibrationMethod calibration_method = CalibrationMethod::MINMAX;
    float percentile = 99.9f;  // For percentile calibration
    int num_samples = 1000;    // Number of samples for calibration
    
    // Dynamic quantization
    bool use_dynamic_ranges = false;
    int window_size = 1024;    // For dynamic range estimation
    
    // Mixed precision settings
    bool enable_mixed_precision = false;
    float sensitivity_threshold = 0.1f;  // For mixed precision decisions
    
    // Performance settings
    bool use_cuda_graphs = true;
    int num_cuda_streams = 4;
    
    // Optimization flags
    bool fuse_quantize_dequantize = true;
    bool cache_quantization_params = true;
};

// Forward declarations
template<typename T> class Quantizer;

// Interface for quantized tensors
template<typename T>
class QuantizedTensor {
public:
    QuantizedTensor(const std::vector<int>& shape, const QuantizationConfig& config)
        : shape_(shape), config_(config) {
        // Calculate size
        size_t num_elements = 1;
        for (int dim : shape) {
            num_elements *= dim;
        }
        
        // Allocate storage
        CUDA_CHECK(cudaMalloc(&data_, num_elements * sizeof(int8_t)));
        
        // Allocate space for quantization parameters
        if (config.per_channel) {
            int num_channels = shape[config.channel_axis];
            scales_.resize(num_channels);
            zero_points_.resize(num_channels);
        } else {
            scales_.resize(1);
            zero_points_.resize(1);
        }
    }

    ~QuantizedTensor() {
        if (data_) {
            CUDA_CHECK(cudaFree(data_));
        }
    }

    // Getters
    int8_t* data() const { return data_; }
    const std::vector<int>& shape() const { return shape_; }
    const std::vector<float>& scales() const { return scales_; }
    const std::vector<float>& zeroPoints() const { return zero_points_; }
    const QuantizationConfig& config() const { return config_; }

    // Utility functions
    size_t numel() const {
        size_t n = 1;
        for (int dim : shape_) {
            n *= dim;
        }
        return n;
    }

    void setQuantizationParams(
        const std::vector<float>& scales,
        const std::vector<float>& zero_points
    ) {
        scales_ = scales;
        zero_points_ = zero_points;
    }

private:
    int8_t* data_ = nullptr;
    std::vector<int> shape_;
    std::vector<float> scales_;
    std::vector<float> zero_points_;
    QuantizationConfig config_;
    
    friend class Quantizer<T>;
};

// Interface for quantization calibration
class QuantizationCalibrator {
public:
    virtual ~QuantizationCalibrator() = default;
    
    virtual void collectStats(const void* data, size_t size) = 0;
    virtual void computeRanges(float& min_val, float& max_val) = 0;
    virtual void reset() = 0;
};

// MinMax calibrator
class MinMaxCalibrator : public QuantizationCalibrator {
public:
    MinMaxCalibrator() : min_val_(FLT_MAX), max_val_(-FLT_MAX) {}
    
    void collectStats(const void* data, size_t size) override;
    void computeRanges(float& min_val, float& max_val) override;
    void reset() override;

private:
    float min_val_;
    float max_val_;
};

// Percentile calibrator
class PercentileCalibrator : public QuantizationCalibrator {
public:
    explicit PercentileCalibrator(float percentile = 99.9f)
        : percentile_(percentile) {}
    
    void collectStats(const void* data, size_t size) override;
    void computeRanges(float& min_val, float& max_val) override;
    void reset() override;

private:
    float percentile_;
    std::vector<float> values_;
};

// MSE calibrator
class MSECalibrator : public QuantizationCalibrator {
public:
    void collectStats(const void* data, size_t size) override;
    void computeRanges(float& min_val, float& max_val) override;
    void reset() override;

private:
    std::vector<float> values_;
    float optimal_min_ = 0.0f;
    float optimal_max_ = 0.0f;
};

// Main quantizer interface
template<typename T>
class Quantizer {
public:
    explicit Quantizer(const QuantizationConfig& config);
    ~Quantizer();
    
    // Quantization
    void quantize(const Tensor<T>& input, QuantizedTensor<T>& output);
    void dequantize(const QuantizedTensor<T>& input, Tensor<T>& output);
    
    // Calibration
    void calibrate(const std::vector<Tensor<T>>& calibration_data);
    void resetCalibration();
    
    // Dynamic quantization
    void updateDynamicRanges(const Tensor<T>& input);
    
    // Mixed precision
    void analyzeSensitivity(const Tensor<T>& input, const Tensor<T>& grad);
    bool shouldQuantize(const std::string& layer_name) const;
    
    // Utility functions
    const QuantizationConfig& config() const { return config_; }
    void setConfig(const QuantizationConfig& config) { config_ = config; }
    
    // Stream management
    void setStream(cudaStream_t stream) { stream_ = stream; }
    cudaStream_t getStream() const { return stream_; }

private:
    // Implementation details in quantizer.cu
    QuantizationConfig config_;
    cudaStream_t stream_;
    std::unique_ptr<QuantizationCalibrator> calibrator_;
    
    // Cached parameters
    std::vector<float> scales_;
    std::vector<float> zero_points_;
    
    // Mixed precision state
    std::unordered_map<std::string, float> layer_sensitivity_;
};

} // namespace quantization
} // namespace ltm
