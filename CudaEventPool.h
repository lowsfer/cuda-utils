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
