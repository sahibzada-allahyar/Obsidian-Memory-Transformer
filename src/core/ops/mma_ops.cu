#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/layout/matrix.h>
#include "core/ops/mma_ops.cuh"
#include "core/utils/cuda_utils.cuh"

namespace ltm {
namespace ops {

// CUTLASS GEMM configurations
using ElementInput = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using ElementAccumulator = float;
using ElementCompute = float;

using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

constexpr int NumStages = 3;
constexpr bool SplitKSerial = false;

using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,
    128 / cutlass::sizeof_bits<ElementOutput>::value,
    ElementAccumulator,
    ElementCompute
>;

using Gemm = cutlass::gemm::device::Gemm<
    ElementInput,
    LayoutInputA,
    ElementInput,
    LayoutInputB,
    ElementOutput,
    LayoutOutput,
    ElementAccumulator,
    MMAOp,
    SmArch,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    NumStages,
    128 / cutlass::sizeof_bits<ElementInput>::value,
    128 / cutlass::sizeof_bits<ElementInput>::value,
    SplitKSerial
>;

template<typename T>
void matmul(
    const Tensor<T>& A,
    const Tensor<T>& B,
    Tensor<T>& C,
    bool transpose_a,
    bool transpose_b,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    // Get dimensions
    int m = A.shape()[0];
    int k = transpose_a ? A.shape()[0] : A.shape()[1];
    int n = transpose_b ? B.shape()[0] : B.shape()[1];
    
    // Create GEMM configuration
    typename Gemm::Arguments args(
        {m, n, k},                                // Problem size
        {reinterpret_cast<ElementInput*>(const_cast<T*>(A.data())), k},  // A
        {reinterpret_cast<ElementInput*>(const_cast<T*>(B.data())), n},  // B
        {reinterpret_cast<ElementOutput*>(C.data()), n},                  // C
        {reinterpret_cast<ElementOutput*>(C.data()), n},                  // D
        {alpha, beta}                             // alpha, beta
    );
    
    // Initialize GEMM object
    Gemm gemm_op;
    
    // Launch kernel
    cutlass::Status status = gemm_op(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS GEMM failed");
    }
}

// Fused MMA + GELU
template<typename T>
void mmaGelu(
    const Tensor<T>& A,
    const Tensor<T>& B,
    Tensor<T>& C,
    bool transpose_a,
    bool transpose_b,
    cudaStream_t stream
) {
    // Custom epilogue with GELU activation
    using GeluEpilogueOp = cutlass::epilogue::thread::LinearCombinationGELU<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementCompute
    >;
    
    using GemmGelu = cutlass::gemm::device::Gemm<
        ElementInput,
        LayoutInputA,
        ElementInput,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        MMAOp,
        SmArch,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        GeluEpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        NumStages,
        128 / cutlass::sizeof_bits<ElementInput>::value,
        128 / cutlass::sizeof_bits<ElementInput>::value,
        SplitKSerial
    >;
    
    // Get dimensions
    int m = A.shape()[0];
    int k = transpose_a ? A.shape()[0] : A.shape()[1];
    int n = transpose_b ? B.shape()[0] : B.shape()[1];
    
    // Create GEMM configuration
    typename GemmGelu::Arguments args(
        {m, n, k},
        {reinterpret_cast<ElementInput*>(const_cast<T*>(A.data())), k},
        {reinterpret_cast<ElementInput*>(const_cast<T*>(B.data())), n},
        {reinterpret_cast<ElementOutput*>(C.data()), n},
        {reinterpret_cast<ElementOutput*>(C.data()), n},
        {1.0f, 0.0f}
    );
    
    // Initialize GEMM object
    GemmGelu gemm_op;
    
    // Launch kernel
    cutlass::Status status = gemm_op(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS GEMM+GELU failed");
    }
}

// Fused MMA + Dropout
template<typename T>
void mmaDropout(
    const Tensor<T>& A,
    const Tensor<T>& B,
    Tensor<T>& C,
    float dropout_prob,
    unsigned long long seed,
    bool transpose_a,
    bool transpose_b,
    cudaStream_t stream
) {
    // Custom epilogue with dropout
    using DropoutEpilogueOp = cutlass::epilogue::thread::LinearCombinationDropout<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementCompute
    >;
    
    using GemmDropout = cutlass::gemm::device::Gemm<
        ElementInput,
        LayoutInputA,
        ElementInput,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        MMAOp,
        SmArch,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        DropoutEpilogueOp,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        NumStages,
        128 / cutlass::sizeof_bits<ElementInput>::value,
        128 / cutlass::sizeof_bits<ElementInput>::value,
        SplitKSerial
    >;
    
    // Get dimensions
    int m = A.shape()[0];
    int k = transpose_a ? A.shape()[0] : A.shape()[1];
    int n = transpose_b ? B.shape()[0] : B.shape()[1];
    
    // Create GEMM configuration
    typename GemmDropout::Arguments args(
        {m, n, k},
        {reinterpret_cast<ElementInput*>(const_cast<T*>(A.data())), k},
        {reinterpret_cast<ElementInput*>(const_cast<T*>(B.data())), n},
        {reinterpret_cast<ElementOutput*>(C.data()), n},
        {reinterpret_cast<ElementOutput*>(C.data()), n},
        {1.0f, 0.0f},
        dropout_prob,
        seed
    );
    
    // Initialize GEMM object
    GemmDropout gemm_op;
    
    // Launch kernel
    cutlass::Status status = gemm_op(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS GEMM+Dropout failed");
    }
}

// Explicit instantiations
template void matmul<float>(
    const Tensor<float>&, const Tensor<float>&, Tensor<float>&,
    bool, bool, float, float, cudaStream_t
);
template void matmul<half>(
    const Tensor<half>&, const Tensor<half>&, Tensor<half>&,
    bool, bool, float, float, cudaStream_t
);

template void mmaGelu<float>(
    const Tensor<float>&, const Tensor<float>&, Tensor<float>&,
    bool, bool, cudaStream_t
);
template void mmaGelu<half>(
    const Tensor<half>&, const Tensor<half>&, Tensor<half>&,
    bool, bool, cudaStream_t
);

template void mmaDropout<float>(
    const Tensor<float>&, const Tensor<float>&, Tensor<float>&,
    float, unsigned long long, bool, bool, cudaStream_t
);
template void mmaDropout<half>(
    const Tensor<half>&, const Tensor<half>&, Tensor<half>&,
    float, unsigned long long, bool, bool, cudaStream_t
);

} // namespace ops
} // namespace ltm
