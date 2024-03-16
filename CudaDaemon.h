#pragma once
#include <mutex>
#include <unordered_map>
#include <memory>
#include <cuda_runtime_api.h>
#include <thread>
#include <ConcurrentQueue.h>
#include <CudaEventPool.h>

namespace cudapp
{
// For out-of-stream callbacks, i.e. the callback waits untils finish of previous tasks, but does not block later tasks in the stream.
class ICudaDaemon
{
public:
    virtual ~ICudaDaemon();
    virtual void notifyDestroy(cudaStream_t stream) = 0;
    virtual void postOffStreamCallback(cudaStream_t stream, std::function<void()> callback) = 0;
};

std::shared_ptr<ICudaDaemon> getCudaDaemon();

} // namespace cudapp
