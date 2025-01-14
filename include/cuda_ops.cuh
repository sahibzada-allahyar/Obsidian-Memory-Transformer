#pragma once

#include <cuda_runtime.h>

namespace cuda_ops {

// CUDA kernel function to add two arrays
__global__ void addArrays(const float* a, const float* b, float* c, int size);

// Host function to allocate memory and launch kernel
void vectorAdd(const float* hostA, const float* hostB, float* hostC, int size);

} // namespace cuda_ops
