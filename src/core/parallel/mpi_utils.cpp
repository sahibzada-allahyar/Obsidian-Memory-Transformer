#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <vector>
#include <stdexcept>
#include <memory>
#include "core/parallel/mpi_utils.hpp"

namespace ltm {
namespace parallel {

class MPIContext {
public:
    static MPIContext& getInstance() {
        static MPIContext instance;
        return instance;
    }

    void initialize() {
        if (!initialized_) {
            int provided;
            MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
            if (provided < MPI_THREAD_MULTIPLE) {
                throw std::runtime_error("MPI implementation does not support MPI_THREAD_MULTIPLE");
            }
            
            MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
            MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
            
            initializeNCCL();
            initialized_ = true;
        }
    }

    ~MPIContext() {
        if (initialized_) {
            ncclCommDestroy(nccl_comm_);
            MPI_Finalize();
        }
    }

    int getRank() const { return rank_; }
    int getWorldSize() const { return world_size_; }
    ncclComm_t getNCCLComm() const { return nccl_comm_; }

private:
    MPIContext() : initialized_(false), rank_(-1), world_size_(-1) {}
    
    void initializeNCCL() {
        // Get unique ID from rank 0
        ncclUniqueId nccl_id;
        if (rank_ == 0) {
            ncclGetUniqueId(&nccl_id);
        }
        
        // Broadcast NCCL ID to all ranks
        MPI_Bcast(&nccl_id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
        
        // Initialize NCCL communicator
        ncclCommInitRank(&nccl_comm_, world_size_, nccl_id, rank_);
    }

    bool initialized_;
    int rank_;
    int world_size_;
    ncclComm_t nccl_comm_;
};

// Global synchronization
void synchronize() {
    MPI_Barrier(MPI_COMM_WORLD);
    cudaStreamSynchronize(nullptr);
}

// All-reduce operation for gradients
void allReduceGradients(void* data, size_t count, ncclDataType_t dtype, cudaStream_t stream) {
    auto& ctx = MPIContext::getInstance();
    ncclAllReduce(
        data,                  // sendbuff
        data,                  // recvbuff
        count,                // count
        dtype,                // datatype
        ncclSum,             // reduction operation
        ctx.getNCCLComm(),   // communicator
        stream               // CUDA stream
    );
}

// Broadcast parameters from rank 0
void broadcastParameters(void* data, size_t count, ncclDataType_t dtype, cudaStream_t stream) {
    auto& ctx = MPIContext::getInstance();
    ncclBroadcast(
        data,                // sendbuff
        data,                // recvbuff
        count,               // count
        dtype,               // datatype
        0,                   // root rank
        ctx.getNCCLComm(),  // communicator
        stream              // CUDA stream
    );
}

// Scatter data across ranks
void scatterData(const void* send_data, void* recv_data, size_t count_per_rank, 
                ncclDataType_t dtype, cudaStream_t stream) {
    auto& ctx = MPIContext::getInstance();
    if (ctx.getRank() == 0) {
        for (int i = 0; i < ctx.getWorldSize(); ++i) {
            if (i == 0) {
                // Copy local data
                cudaMemcpyAsync(
                    recv_data,
                    send_data,
                    count_per_rank * ncclTypeSize(dtype),
                    cudaMemcpyDeviceToDevice,
                    stream
                );
            } else {
                // Send to other ranks
                MPI_Send(
                    static_cast<const char*>(send_data) + i * count_per_rank * ncclTypeSize(dtype),
                    count_per_rank * ncclTypeSize(dtype),
                    MPI_BYTE,
                    i,
                    0,
                    MPI_COMM_WORLD
                );
            }
        }
    } else {
        // Receive data from rank 0
        MPI_Recv(
            recv_data,
            count_per_rank * ncclTypeSize(dtype),
            MPI_BYTE,
            0,
            0,
            MPI_COMM_WORLD,
            MPI_STATUS_IGNORE
        );
    }
}

// Gather data from all ranks
void gatherData(const void* send_data, void* recv_data, size_t count_per_rank,
               ncclDataType_t dtype, cudaStream_t stream) {
    auto& ctx = MPIContext::getInstance();
    if (ctx.getRank() == 0) {
        // Copy local data
        cudaMemcpyAsync(
            recv_data,
            send_data,
            count_per_rank * ncclTypeSize(dtype),
            cudaMemcpyDeviceToDevice,
            stream
        );
        
        // Receive from other ranks
        for (int i = 1; i < ctx.getWorldSize(); ++i) {
            MPI_Recv(
                static_cast<char*>(recv_data) + i * count_per_rank * ncclTypeSize(dtype),
                count_per_rank * ncclTypeSize(dtype),
                MPI_BYTE,
                i,
                0,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE
            );
        }
    } else {
        // Send to rank 0
        MPI_Send(
            send_data,
            count_per_rank * ncclTypeSize(dtype),
            MPI_BYTE,
            0,
            0,
            MPI_COMM_WORLD
        );
    }
}

// Initialize MPI environment
void initializeMPI() {
    MPIContext::getInstance().initialize();
}

// Get current rank
int getCurrentRank() {
    return MPIContext::getInstance().getRank();
}

// Get world size
int getWorldSize() {
    return MPIContext::getInstance().getWorldSize();
}

// Check if current process is master
bool isMaster() {
    return getCurrentRank() == 0;
}

} // namespace parallel
} // namespace ltm
