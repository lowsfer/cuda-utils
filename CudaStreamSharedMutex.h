#include "CudaStreamMutex.h"
#include <shared_mutex>
#include <variant>
#include "CudaEventPool.h"

namespace cudapp
{
class CudaStreamSharedMutex;

class CudaStreamSharedLock
{
public:
    CudaStreamSharedLock(CudaStreamSharedMutex& mutex, cudaStream_t stream);
    ~CudaStreamSharedLock();
    void migrate(cudaStream_t dst);
private:
    CudaStreamSharedMutex& mMutex;
    cudaStream_t mStream;
};

class CudaStreamSharedMutex
{
public:
    CudaStreamSharedMutex();
    CudaStreamSharedMutex(cudaStream_t stream);
    CudaStreamSharedMutex(PooledCudaEvent ev);
    CudaStreamLockGuard<CudaStreamSharedMutex> acquire(cudaStream_t stream);
    CudaStreamSharedLock acquireShared(cudaStream_t stream);
    void sync();
private:
    friend CudaStreamSharedLock;
    friend CudaStreamLockGuard<CudaStreamSharedMutex>;
    void lockShared(cudaStream_t stream);
    void unlockShared(cudaStream_t stream);
    void lock(cudaStream_t stream);
    void unlock(cudaStream_t stream);
private:
    std::shared_mutex mMutex;
    PooledCudaEvent mWriteFinished;
    std::unique_ptr<ICudaMultiEvent> mReadFinished;
};

} // namespace cudapp
