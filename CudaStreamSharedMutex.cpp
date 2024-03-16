#include "CudaStreamSharedMutex.h"
#include "cpp_utils.h"
namespace cudapp
{
static void scrub(PooledCudaEvent& ev) {
    if (ev != nullptr && cudaEventQuery(ev.get()) == cudaSuccess) {
        cudaCheck(cudaEventSynchronize(ev.get()));
        ev.reset();
    }
}

CudaStreamSharedMutex::CudaStreamSharedMutex() : mReadFinished {createCudaMultiEvent(true)} {}
CudaStreamSharedMutex::CudaStreamSharedMutex(PooledCudaEvent ev)
    : mWriteFinished{std::move(ev)}
    , mReadFinished {createCudaMultiEvent(true)}
{}

CudaStreamSharedMutex::CudaStreamSharedMutex(cudaStream_t stream)
    : mReadFinished {createCudaMultiEvent(true)}
{
    mWriteFinished = createPooledCudaEvent();
    cudaCheck(cudaEventRecord(mWriteFinished.get(), stream));
}
CudaStreamLockGuard<CudaStreamSharedMutex> CudaStreamSharedMutex::acquire(cudaStream_t stream) {
    return {*this, stream};
}
CudaStreamSharedLock CudaStreamSharedMutex::acquireShared(cudaStream_t stream) {
    return {*this, stream};
}
void CudaStreamSharedMutex::sync() {
    std::lock_guard lk{mMutex};
    if (mReadFinished->empty()) {
        if (mWriteFinished != nullptr) {
            cudaCheck(cudaEventSynchronize(mWriteFinished.get()));
        }
    }
    else {
        mReadFinished->sync();
        mReadFinished->clear();
    }
    mWriteFinished.reset();
}

void CudaStreamSharedMutex::lock(cudaStream_t stream) {
    mMutex.lock();
    if (mReadFinished->empty()) {
        scrub(mWriteFinished);
        if (mWriteFinished != nullptr) {
            cudaCheck(cudaStreamWaitEvent(stream, mWriteFinished.get()));
        }
    }
    else {
        mReadFinished->streamWaitEvent(stream);
        mReadFinished->clear();
    }
    mWriteFinished.reset();
}

void CudaStreamSharedMutex::unlock(cudaStream_t stream) {
    mWriteFinished = createPooledCudaEvent();
    cudaCheck(cudaEventRecord(mWriteFinished.get(), stream));
    mMutex.unlock();
}

void CudaStreamSharedMutex::lockShared(cudaStream_t stream) {
    mMutex.lock_shared();
    scrub(mWriteFinished);
    if (mWriteFinished != nullptr) {
        cudaCheck(cudaStreamWaitEvent(stream, mWriteFinished.get()));
    }
}

void CudaStreamSharedMutex::unlockShared(cudaStream_t stream) {
    mReadFinished->recordEvent(stream);
    mReadFinished->scrub(); // Optional. Remove if CUDA graph capture is required.
    scrub(mWriteFinished);
    mMutex.unlock_shared();
}

CudaStreamSharedLock::CudaStreamSharedLock(CudaStreamSharedMutex& mutex, cudaStream_t stream)
: mMutex {mutex}, mStream {stream}
{
    mMutex.lockShared(stream);
}

CudaStreamSharedLock::~CudaStreamSharedLock() {
    mMutex.unlockShared(mStream);
}

void CudaStreamSharedLock::migrate(cudaStream_t dst) {
    connectStreams(mStream, dst);
    mStream = dst;
}

} // namespace cudapp
