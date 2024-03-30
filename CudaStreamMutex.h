/*
Copyright [2024] [Yao Yao]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

*/

#pragma once
#include <mutex>
#include <condition_variable>
#include "cuda_utils.h"
#include "CudaEventPool.h"

namespace cudapp
{
template <typename StreamMutex>
class CudaStreamLockGuard
{
public:
    CudaStreamLockGuard(StreamMutex& mutex, cudaStream_t stream)
        :mMutex{mutex}, mStream{stream}
    {
        mMutex.lock(mStream);
    }
    ~CudaStreamLockGuard() { mMutex.unlock(mStream); }
    //! No thread-safe. Must be in the same thread as constructor execution thread.
    void migrate(cudaStream_t dst) {
        connectStreams(mStream, dst);
        mStream = dst;
    }
private:
    StreamMutex& mMutex;
    cudaStream_t mStream;
};

template <typename Mutex = std::mutex>
class CudaStreamMutex
{
public:
    //! For resources available now.
    CudaStreamMutex() = default;
    //! Protect resource available in stream
    CudaStreamMutex(cudaStream_t stream)
        : mFinishedEvent {createPooledCudaEvent()} 
    {
        cudaCheck(cudaEventRecord(mFinishedEvent.get(), stream));
    }
    //! Protect resource available after event trigger
    CudaStreamMutex(PooledCudaEvent ev) : mFinishedEvent {std::move(ev)} {}

    CudaStreamLockGuard<CudaStreamMutex> acquire(cudaStream_t stream) {
        return {*this, stream};
    }
    void sync() {
        mMutex.lock();
        if (mFinishedEvent != nullptr) {
            cudaCheck(cudaEventSynchronize(mFinishedEvent.get()));
        }
        mFinishedEvent.reset();
    }
private:
    void lock(cudaStream_t stream) {
        mMutex.lock();
        if (mFinishedEvent != nullptr) {
            cudaCheck(cudaStreamWaitEvent(stream, mFinishedEvent.get()));
            mFinishedEvent.reset();
        }
    }
    //! stream must be the same as the one passed to lock(), or connected to the locking stream
    void unlock(cudaStream_t stream) {
        mFinishedEvent = createPooledCudaEvent();
        cudaCheck(cudaEventRecord(mFinishedEvent.get(), stream));
        mMutex.unlock();
    }
    bool try_lock(cudaStream_t stream) {
        const bool locked = mMutex.try_lock();
        if (locked) {
            if (mFinishedEvent != nullptr) {
                cudaCheck(cudaStreamWaitEvent(stream, mFinishedEvent.get()));
                mFinishedEvent.reset();
            }
        }
        return locked;
    }
private:
    friend CudaStreamLockGuard<CudaStreamMutex>;
private:
    Mutex mMutex;
    PooledCudaEvent mFinishedEvent;
};

} // namespace cudapp
