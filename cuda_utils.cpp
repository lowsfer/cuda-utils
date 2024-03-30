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

#include "cuda_utils.h"
#include "CudaEventPool.h"

CudaEvent makeCudaEvent(unsigned flags)
{
    cudaEvent_t ev = nullptr;
    cudaCheck(cudaEventCreateWithFlags(&ev, flags));
    return CudaEvent{ev};
}

CudaStream makeCudaStream(unsigned flags)
{
    cudaStream_t stream = nullptr;
    cudaCheck(cudaStreamCreateWithFlags(&stream, flags));
    return CudaStream{stream};
}

CudaStream makeCudaStreamWithPriority(int priority, unsigned flags)
{
    cudaStream_t stream = nullptr;
    cudaCheck(cudaStreamCreateWithPriority(&stream, flags, priority));
    return CudaStream{stream};
}

CudaGraph makeCudaGraph() {
    cudaGraph_t g = nullptr;
    cudaCheck(cudaGraphCreate(&g, 0));
    return CudaGraph{g};
}

CudaGraphExec instantiateCudaGraph(cudaGraph_t graph) {
    cudaGraphExec_t exec = nullptr;
    cudaCheck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    return CudaGraphExec{exec};
}

void connectStreams(cudaStream_t first, cudaStream_t second)
{
    const cudapp::PooledCudaEvent event = cudapp::createPooledCudaEvent();
    cudaCheck(cudaEventRecord(event.get(), first));
    cudaCheck(cudaStreamWaitEvent(second, event.get(), 0));
}
void connectStreams(cudaStream_t first, cudaStream_t second, cudaEvent_t event, std::mutex* pMutex)
{
    std_optional<std::lock_guard<std::mutex>> lock;
    if (pMutex != nullptr)
    {
        lock.emplace(*pMutex);
    }
    cudaCheck(cudaEventRecord(event, first));
    cudaCheck(cudaStreamWaitEvent(second, event, 0));
}

ICudaMultiEvent::~ICudaMultiEvent() = default;

template <bool isPooled>
class CudaMultiEvent : public ICudaMultiEvent
{
public:
    using Event = std::conditional_t<isPooled, cudapp::PooledCudaEvent, CudaEvent>;
    void clear() override {
        std::lock_guard lk{mMutex};
        mEvents.clear();
    }
    void recordEvent(cudaStream_t stream) override;
    // This stream will wait until all
    void streamWaitEvent(cudaStream_t stream) const override;

    void sync() const override;

    void scrub() override;
    bool query() override;
    bool empty() const override {
        std::lock_guard lk{mMutex};
        return mEvents.empty();
    }
private:
    mutable std::mutex mMutex;
    std::vector<Event> mEvents;
};
template <>
void CudaMultiEvent<false>::recordEvent(cudaStream_t stream){
    CudaEvent event = makeCudaEvent();
    cudaCheck(cudaEventRecord(event.get(), stream));
    std::lock_guard lk{mMutex};
    mEvents.emplace_back(std::move(event));
}
template <>
void CudaMultiEvent<true>::recordEvent(cudaStream_t stream){
    cudapp::PooledCudaEvent event = cudapp::createPooledCudaEvent();
    cudaCheck(cudaEventRecord(event.get(), stream));
    std::lock_guard lk{mMutex};
    mEvents.emplace_back(std::move(event));
}
// This stream will wait until all
template <bool isPooled>
void CudaMultiEvent<isPooled>::streamWaitEvent(cudaStream_t stream) const{
    std::lock_guard lk{mMutex};
    for (const auto& ev : mEvents){
        cudaCheck(cudaStreamWaitEvent(stream, ev.get(), 0));
    }
}
template <bool isPooled>
void CudaMultiEvent<isPooled>::sync() const {
    std::lock_guard lk{mMutex};
    for (const auto& ev : mEvents){
        cudaCheck(cudaEventSynchronize(ev.get()));
    }
}
template <bool isPooled>
void CudaMultiEvent<isPooled>::scrub() {
    std::lock_guard lk{mMutex};
    const auto iterLast = std::remove_if(mEvents.begin(), mEvents.end(), [](const Event& u){
        const cudaError_t state = cudaEventQuery(u.get());
        switch (state)
        {
        case cudaSuccess:
            cudaCheck(cudaEventSynchronize(u.get()));
            return true;
        case cudaErrorNotReady: return false;
        default: cudaCheck(state);
        }
        throw std::logic_error("You should never reach here");
    });
    mEvents.erase(iterLast, mEvents.end());
}
template <bool isPooled>
bool CudaMultiEvent<isPooled>::query() {
    scrub();
    std::lock_guard lk{mMutex};
    return mEvents.empty();
}

std::unique_ptr<ICudaMultiEvent> createCudaMultiEvent(bool isPooled)
{
    if (isPooled) {
        return std::make_unique<CudaMultiEvent<true>>();
    }
    else {
        return std::make_unique<CudaMultiEvent<false>>();
    }
}

namespace cudapp
{
void streamSync(cudaStream_t stream)
{
#if 1
    cudaCheck(cudaStreamSynchronize(stream));
#else
    const auto ev = createPooledCudaEvent();
    cudaCheck(cudaEventRecord(ev.get(), stream));
    cudaCheck(cudaEventSynchronize(ev.get()));
#endif
}
}
