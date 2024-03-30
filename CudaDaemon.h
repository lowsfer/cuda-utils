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
#include <unordered_map>
#include <memory>
#include <cuda_runtime_api.h>
#include <thread>
#include "ConcurrentQueue.h"
#include "CudaEventPool.h"

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
