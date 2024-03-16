#pragma once

#include <memory>
#include <cuda_runtime_api.h>
namespace cudapp
{
class CudaEventPool;
std::shared_ptr<CudaEventPool> getCudaEventPool();
struct PooledCudaEventDeleter
{
    std::shared_ptr<CudaEventPool> pool;
    void operator()(cudaEvent_t ev) const;
};
using PooledCudaEvent = std::unique_ptr<std::remove_pointer_t<cudaEvent_t>, PooledCudaEventDeleter>;

PooledCudaEvent createPooledCudaEvent();

} // namespace cudapp
