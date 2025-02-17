#include <cuda_runtime.h>
#include <nccl.h>
#include <vector>
#include <memory>
#include "core/parallel/tensor_parallel.hpp"
#include "core/utils/cuda_utils.cuh"

namespace ltm {
namespace parallel {

class TensorParallelContext {
public:
    TensorParallelContext(
        int world_size,
        int rank,
        ncclComm_t nccl_comm,
        int device_id
    ) : world_size_(world_size),
        rank_(rank),
        nccl_comm_(nccl_comm) {
        CUDA_CHECK(cudaSetDevice(device_id));
        CUDA_CHECK(cudaStreamCreate(&compute_stream_));
        CUDA_CHECK(cudaStreamCreate(&comm_stream_));
        
        // Create events for stream synchronization
        CUDA_CHECK(cudaEventCreate(&compute_event_));
        CUDA_CHECK(cudaEventCreate(&comm_event_));
    }

    ~TensorParallelContext() {
        CUDA_CHECK(cudaEventDestroy(compute_event_));
        CUDA_CHECK(cudaEventDestroy(comm_event_));
        CUDA_CHECK(cudaStreamDestroy(compute_stream_));
        CUDA_CHECK(cudaStreamDestroy(comm_stream_));
    }

    // Split tensor along specified dimension
    template<typename T>
    void splitTensor(
        const Tensor<T>& input,
        Tensor<T>& local_output,
        int split_dim
    ) {
        const auto& shape = input.shape();
        std::vector<int> local_shape = shape;
        local_shape[split_dim] /= world_size_;
        
        // Calculate offset for this rank
        size_t offset = rank_ * input.numel() / world_size_;
        
        // Copy local portion
        CUDA_CHECK(cudaMemcpyAsync(
            local_output.data(),
            static_cast<const T*>(input.data()) + offset,
            local_output.numel() * sizeof(T),
            cudaMemcpyDeviceToDevice,
            compute_stream_
        ));
    }

    // Gather split tensor
    template<typename T>
    void gatherTensor(
        const Tensor<T>& local_input,
        Tensor<T>& output,
        int split_dim
    ) {
        ncclAllGather(
            local_input.data(),
            output.data(),
            local_input.numel(),
            ncclDataType<T>(),
            nccl_comm_,
            comm_stream_
        );
        
        // Wait for gather to complete
        CUDA_CHECK(cudaStreamSynchronize(comm_stream_));
    }

    // All-reduce across devices
    template<typename T>
    void allReduce(void* data, size_t count) {
        ncclAllReduce(
            data,
            data,
            count,
            ncclDataType<T>(),
            ncclSum,
            nccl_comm_,
            comm_stream_
        );
        
        // Wait for reduction to complete
        CUDA_CHECK(cudaStreamSynchronize(comm_stream_));
    }

    cudaStream_t computeStream() const { return compute_stream_; }
    cudaStream_t commStream() const { return comm_stream_; }
    int worldSize() const { return world_size_; }
    int rank() const { return rank_; }

private:
    int world_size_;
    int rank_;
    ncclComm_t nccl_comm_;
    
    cudaStream_t compute_stream_;
    cudaStream_t comm_stream_;
    cudaEvent_t compute_event_;
    cudaEvent_t comm_event_;
    
    // Helper to get NCCL data type
    template<typename T>
    static ncclDataType_t ncclDataType();
};

// Template specializations for NCCL data types
template<> ncclDataType_t TensorParallelContext::ncclDataType<float>() { return ncclFloat32; }
template<> ncclDataType_t TensorParallelContext::ncclDataType<half>() { return ncclFloat16; }

class TensorParallelLinear {
public:
    TensorParallelLinear(
        TensorParallelContext& ctx,
        int input_dim,
        int output_dim
    ) : ctx_(ctx) {
        // Split weight matrix across devices
        const int local_output_dim = output_dim / ctx.worldSize();
        weight_ = Tensor<float>({local_output_dim, input_dim});
        bias_ = Tensor<float>({local_output_dim});
        
        // Initialize parameters
        initializeParameters();
    }

    void forward(
        const Tensor<float>& input,
        Tensor<float>& output
    ) {
        // Local matrix multiplication
        matmul(input, weight_, local_output_, ctx_.computeStream());
        
        // Add bias
        addBias(local_output_, bias_, ctx_.computeStream());
        
        // Gather results from all devices
        ctx_.gatherTensor(local_output_, output, 1);
    }

    void backward(
        const Tensor<float>& grad_output,
        const Tensor<float>& input,
        Tensor<float>& grad_input,
        bool compute_grad_input = true
    ) {
        // Split gradient
        Tensor<float> local_grad_output;
        ctx_.splitTensor(grad_output, local_grad_output, 1);
        
        // Compute weight gradients
        matmul(
            local_grad_output.transpose(),
            input,
            weight_grad_,
            ctx_.computeStream()
        );
        
        // Compute bias gradients
        reduceSumAlongDim(local_grad_output, bias_grad_, 0, ctx_.computeStream());
        
        if (compute_grad_input) {
            // Compute input gradients
            matmul(
                local_grad_output,
                weight_.transpose(),
                local_grad_input_,
                ctx_.computeStream()
            );
            
            // All-reduce input gradients
            ctx_.allReduce<float>(
                local_grad_input_.data(),
                local_grad_input_.numel()
            );
            
            grad_input = local_grad_input_;
        }
    }

    void updateParameters(float learning_rate) {
        // Update weights
        axpy(
            -learning_rate,
            weight_grad_,
            weight_,
            ctx_.computeStream()
        );
        
        // Update bias
        axpy(
            -learning_rate,
            bias_grad_,
            bias_,
            ctx_.computeStream()
        );
    }

private:
    void initializeParameters() {
        // Initialize weights using Kaiming initialization
        float std = sqrt(2.0f / weight_.shape()[1]);
        initializeNormal(weight_, 0.0f, std, ctx_.computeStream());
        
        // Initialize bias to zero
        initializeZero(bias_, ctx_.computeStream());
    }

    TensorParallelContext& ctx_;
    Tensor<float> weight_;
    Tensor<float> bias_;
    Tensor<float> weight_grad_;
    Tensor<float> bias_grad_;
    Tensor<float> local_output_;
    Tensor<float> local_grad_input_;
};

class TensorParallelAttention {
public:
    TensorParallelAttention(
        TensorParallelContext& ctx,
        int hidden_dim,
        int num_heads
    ) : ctx_(ctx),
        hidden_dim_(hidden_dim),
        num_heads_(num_heads) {
        // Split attention heads across devices
        const int local_num_heads = num_heads / ctx.worldSize();
        const int head_dim = hidden_dim / num_heads;
        
        // Initialize projection matrices
        query_proj_ = std::make_unique<TensorParallelLinear>(
            ctx, hidden_dim, local_num_heads * head_dim);
        key_proj_ = std::make_unique<TensorParallelLinear>(
            ctx, hidden_dim, local_num_heads * head_dim);
        value_proj_ = std::make_unique<TensorParallelLinear>(
            ctx, hidden_dim, local_num_heads * head_dim);
        output_proj_ = std::make_unique<TensorParallelLinear>(
            ctx, local_num_heads * head_dim, hidden_dim);
    }

    void forward(
        const Tensor<float>& input,
        Tensor<float>& output,
        const Tensor<float>* attention_mask = nullptr
    ) {
        // Project queries, keys, and values
        query_proj_->forward(input, query_);
        key_proj_->forward(input, key_);
        value_proj_->forward(input, value_);
        
        // Compute attention scores
        computeAttentionScores(
            query_,
            key_,
            attention_scores_,
            attention_mask,
            ctx_.computeStream()
        );
        
        // Apply attention to values
        applyAttention(
            attention_scores_,
            value_,
            attention_output_,
            ctx_.computeStream()
        );
        
        // Project output
        output_proj_->forward(attention_output_, output);
    }

    void backward(
        const Tensor<float>& grad_output,
        const Tensor<float>& input,
        Tensor<float>& grad_input
    ) {
        // Backward through output projection
        output_proj_->backward(grad_output, attention_output_, grad_attention_output_);
        
        // Backward through attention
        backwardAttention(
            grad_attention_output_,
            attention_scores_,
            value_,
            grad_attention_scores_,
            grad_value_,
            ctx_.computeStream()
        );
        
        // Backward through projections
        value_proj_->backward(grad_value_, input, grad_input, false);
        key_proj_->backward(grad_attention_scores_, input, grad_input, false);
        query_proj_->backward(grad_attention_scores_, input, grad_input, true);
    }

private:
    TensorParallelContext& ctx_;
    int hidden_dim_;
    int num_heads_;
    
    std::unique_ptr<TensorParallelLinear> query_proj_;
    std::unique_ptr<TensorParallelLinear> key_proj_;
    std::unique_ptr<TensorParallelLinear> value_proj_;
    std::unique_ptr<TensorParallelLinear> output_proj_;
    
    Tensor<float> query_;
    Tensor<float> key_;
    Tensor<float> value_;
    Tensor<float> attention_scores_;
    Tensor<float> attention_output_;
    Tensor<float> grad_attention_output_;
    Tensor<float> grad_attention_scores_;
    Tensor<float> grad_value_;
};

} // namespace parallel
} // namespace ltm
