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

#include "FiberUtils.h"
#include "cuda_utils.h"

namespace cudapp
{
void fiberSyncCudaStream(cudaStream_t stream)
{
#define IMPL_ID 2
#if IMPL_ID == 0
    // fb::cuda::waitfor_all is causing segfault. Using a custom implementation
    fb::promise<void> p;
    fb::future<void> f = p.get_future();
    launchCudaHostFunc(stream, [p{std::move(p)}]()mutable{p.set_value();});
    f.get();
#elif IMPL_ID == 1
    cudaCheck(cudaStreamSynchronize(stream)); // @fixme: add a FiberBlockingService member function to do this with cuda events.
#else
    auto p = std::make_shared<fb::promise<void>>();
    fb::future<void> f = p->get_future();
    getCudaDaemon()->postOffStreamCallback(stream, [p{std::move(p)}]() mutable{
        p->set_value();
    });
    f.get();
#endif
}

FiberBlockingService::IBlockingTask::~IBlockingTask() = default;

} // namespace cudapp
