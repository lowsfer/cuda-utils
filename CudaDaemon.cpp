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

#include "CudaDaemon.h"
#include "cuda_utils.h"

namespace cudapp
{
ICudaDaemon::~ICudaDaemon() = default;

namespace
{

class CudaDaemonWorker
{
public:
    CudaDaemonWorker(cudaStream_t stream) : mStream{stream}, mThread{[this]{run();}} {}
    ~CudaDaemonWorker() {
        mCallbacks.close();
        if (mThread.joinable()) {
            mThread.join();
        }
    }
    void postOffStreamCallback(std::function<void()> callback) {
        auto ev = createPooledCudaEvent();
        cudaCheck(cudaEventRecord(ev.get(), mStream));
        mCallbacks.emplace_back(std::move(ev), std::move(callback));
    }
private:
    void run() {
        while (true) {
            auto opt = mCallbacks.pop();
            if (!opt.has_value()) {
                break;
            }
            auto& [ev, cb] = opt.value();
            cudaCheck(cudaEventSynchronize(ev.get()));
            cb();
        }
    }
private:
    cudaStream_t mStream;
    ConcurrentQueue<std::pair<PooledCudaEvent, std::function<void()>>> mCallbacks;
    std::thread mThread;
};

class CudaDaemon : public ICudaDaemon
{
public:
    void notifyDestroy(cudaStream_t stream) override {
        std::lock_guard lk{mMutex};
        mWorkers.erase(stream);
    }
    void postOffStreamCallback(cudaStream_t stream, std::function<void()> callback) override {
        const std::shared_ptr<CudaDaemonWorker> worker = [this, stream] {
            std::lock_guard lk{mMutex};
            auto iter = mWorkers.find(stream);
            if (iter == mWorkers.end()) {
                bool success;
                std::tie(iter, success) = mWorkers.try_emplace(stream, std::make_unique<CudaDaemonWorker>(stream));
                ASSERT(success);
            }
            return iter->second;
        }();
        worker->postOffStreamCallback(std::move(callback));
    }

private:
    mutable std::mutex mMutex; // only protects the container, not the workers.
    std::unordered_map<cudaStream_t, std::shared_ptr<CudaDaemonWorker>> mWorkers;
};
} // unnamed namespace

std::shared_ptr<ICudaDaemon> getCudaDaemon() {
    static std::mutex lock;
    static std::weak_ptr<ICudaDaemon> observer;
    std::lock_guard lk {lock};
    auto p = observer.lock();
    if (p != nullptr) {
        return p;
    }
    p = std::make_shared<CudaDaemon>();
    observer = p;
    return p;
}
} // namespace cudapp
