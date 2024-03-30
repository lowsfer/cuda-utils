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
