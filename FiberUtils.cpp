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
