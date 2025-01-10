#include "cuda_ops.cuh"
#include <cstdio>

namespace cuda_ops {

__global__ void addArrays(const float* a, const float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

void vectorAdd(const float* hostA, const float* hostB, float* hostC, int size) {
    // Declare device pointers
    float *deviceA, *deviceB, *deviceC;
    
    // Allocate device memory
    cudaMalloc(&deviceA, size * sizeof(float));
    cudaMalloc(&deviceB, size * sizeof(float));
    cudaMalloc(&deviceC, size * sizeof(float));
    
    // Copy inputs to device
    cudaMemcpy(deviceA, hostA, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    addArrays<<<blocksPerGrid, threadsPerBlock>>>(deviceA, deviceB, deviceC, size);
    
    // Copy result back to host
    cudaMemcpy(hostC, deviceC, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
}
