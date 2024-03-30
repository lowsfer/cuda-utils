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
#include "fiberProxy.h"
#include "ConcurrentQueue.h"
#include <future>
#include <vector>
#include <cstddef>
#include "cpp_utils.h"
#include <chrono>
#include <list>
#include <boost/fiber/cuda/waitfor.hpp>
#include <boost/context/protected_fixedsize_stack.hpp>
#include "cuda_utils.h"
#include "CudaEventPool.h"
#include "StackAllocator.h"
#include <stack>

namespace cudapp
{
constexpr fb::launch fiberLaunchPolicy = fb::launch::post;
constexpr size_t defaultFiberStackSize = 128UL << 10;

class FiberFactory
{
public:
#if 0
    using PackagedFiberCreationTask = std::packaged_task<fb::fiber()>;
    using FiberFuture = std::future<fb::fiber>;
#else
    // This uses 20% more memory, but allow waiting in worker threads without blocking workers.
    using PackagedFiberCreationTask = fb::packaged_task<fb::fiber()>;
    using FiberFuture = fb::future<fb::fiber>;
#endif

    FiberFactory(size_t defaultStackSize = defaultFiberStackSize, size_t nbThreads = 1, size_t maxPendingFiberCreators = 1u<<30)
        : mDefaultStackSize{defaultStackSize}
        , mThreads{nbThreads}
        , mFiberCreators{maxPendingFiberCreators}
    {
        REQUIRE(nbThreads > 0);
        std::generate_n(mThreads.begin(), nbThreads, [this, nbThreads](){return std::thread{&FiberFactory::worker, this, nbThreads};});
    }

    ~FiberFactory(){
        // emplace a invalid task to indicate stop
        mFiberCreators.close();
        for (auto& t : mThreads){
            t.join();
        }
        mThreads.clear();
    }

    // Not returning fb::fiber directly because we want fibers created in the worker thread, to reduce fiber migration and enjoy NUMA locality.
    // args... are the arguments passed to fb::fiber::fiber()
    template <typename... Args>
    FiberFuture create(Args&&... args)
    {
        PackagedFiberCreationTask fiberCreator{std::bind(&fiberCtorBindProxy<Args...>, fiberLaunchPolicy, std::allocator_arg, StackAllocator{mDefaultStackSize}, std::forward<Args>(args)...)};
        auto f = fiberCreator.get_future();
        mFiberCreators.emplace(std::move(fiberCreator));
        return f;
    }
private:
    template <typename... Args>
    static fb::fiber fiberCtorBindProxy(fb::launch policy, const std::allocator_arg_t&, const StackAllocator& alloc, Args&... args)
    {
        return fb::fiber{policy, std::allocator_arg, alloc, std::move(args)...};
    }

    void worker(size_t nbThreads) noexcept
    {
        const bool suspendOnIdle = true; //Otherwise it keeps spinning for work stealing
        fb::use_scheduling_algorithm<fb::algo::work_stealing>(nbThreads, suspendOnIdle);
        size_t nbCreated = 0;
        while(true){
            std_optional<PackagedFiberCreationTask> creator = mFiberCreators.pop();
            if (creator.has_value()){
                creator.value()();
            }
            else {
                break;
            }
            if (++nbCreated % mYieldInterval == 0){
                this_fiber::yield();
            }
        }
    }
private:
    size_t mDefaultStackSize = defaultFiberStackSize;
    std::vector<std::thread> mThreads;
    ConcurrentQueue<PackagedFiberCreationTask, fb::mutex, fb::condition_variable> mFiberCreators; // tasks pending to be launched as new fibers
    size_t mYieldInterval = 1;
};

class FiberPool
{
public:
    FiberPool(size_t nbThreads = 1, size_t defaultStackSize = defaultFiberStackSize, size_t maxPendingFiberCreators = 1u<<30)
        : mDefaultStackSize{defaultStackSize}
        , mFiberCreators(maxPendingFiberCreators)
    {
        REQUIRE(nbThreads > 0);
        mThreads.reserve(nbThreads);
        // mThreads.size() must be nbThreads before calling worker(), as worker() may need nbThreads for work stealing scheduler.
        std::generate_n(std::back_inserter(mThreads), nbThreads, [this, nbThreads](){return std::thread{&FiberPool::worker, this, nbThreads};});
    }

    ~FiberPool() {
        // emplace a invalid task to indicate stop
        mFiberCreators.close();
        for (auto& t : mThreads) {
            t.join();
        }
        mThreads.clear();
    }

    template <typename T, typename... Args>
    auto async(T stackSize, Args&&... args)-> std::enable_if_t<std::is_integral_v<T>, fb::future<fb::future<decltype(fb::async(fiberLaunchPolicy, std::allocator_arg, std::declval<StackAllocator>(), std::forward<Args>(args)...).get())>>>{
        // Result itself is a fb::future
        using Result = decltype(fb::async(fiberLaunchPolicy, std::allocator_arg, std::declval<StackAllocator>(), std::forward<Args>(args)...));

        auto task = std::make_shared<fb::packaged_task<Result()>>(std::bind(&fiberAsyncBindProxy<Args...>, fiberLaunchPolicy, std::allocator_arg, StackAllocator{stackSize}, std::forward<Args>(args)...));
        fb::future<Result> futureResult = task->get_future();
        mFiberCreators.emplace([task{std::move(task)}]{std::invoke(*task);});

        return futureResult;
    }

    template <typename... Args>
    auto async(Args&&... args)-> std::enable_if_t<!std::is_integral_v<decltype(std::get<0>(std::declval<std::tuple<Args...>>()))>, fb::future<fb::future<decltype(fb::async(fiberLaunchPolicy, std::allocator_arg, std::declval<StackAllocator>(), std::forward<Args>(args)...).get())>>>{
        return async(mDefaultStackSize, std::forward<Args>(args)...);
    }

    size_t getNbPendingFibers() const {
        return mFiberCreators.peekSize();
    }

private:
    template <typename... Args>
    static auto fiberAsyncBindProxy(fb::launch policy, const std::allocator_arg_t&, const StackAllocator& alloc, Args&... args)
        -> decltype(fb::async(policy, std::allocator_arg, alloc, std::move(args)...))
    {
        return fb::async(policy, std::allocator_arg, alloc, std::move(args)...);
    }

    void worker(size_t nbWorkers) noexcept
    {
        if (nbWorkers > 1) {
#if 0
            using schedAlgo = fb::algo::round_robin;
            fb::use_scheduling_algorithm<schedAlgo>();
#else
            // Looks like cuda driver has problem with work-stealing.
            const bool suspendOnIdle = true; //Otherwise it keeps spinning for work stealing
            using schedAlgo = fb::algo::work_stealing;
            fb::use_scheduling_algorithm<schedAlgo>(nbWorkers, suspendOnIdle);
#endif
        }
        size_t nbCreated = 0;
        while(true){
            std_optional<std::function<void()>> creator = mFiberCreators.pop();
            if (creator.has_value()){
                creator.value()();
            }
            else {
                break;
            }
            if (++nbCreated % mYieldInterval == 0){
                this_fiber::yield();
            }
        }
    }
private:
    size_t mDefaultStackSize = defaultFiberStackSize;
    std::vector<std::thread> mThreads;
    ConcurrentQueue<std::function<void()>, fb::mutex, fb::condition_variable> mFiberCreators; // tasks pending to be launched as new fibers
    size_t mYieldInterval = 1;
};

void fiberSyncCudaStream(cudaStream_t stream);

class FiberBlockingService
{
private:
    class IBlockingTask
    {
    public:
        virtual ~IBlockingTask();
        virtual bool isReady() const = 0;
        // return true if ready, false if timeout, and throw if deferred.
        virtual bool waitFor(const std::chrono::duration<float>& duration) const = 0;
        virtual void complete() = 0;
        bool tryComplete(const std::chrono::duration<float>& duration = std::chrono::seconds{0}){
            const bool ready = waitFor(duration);
            if (ready){
                complete();
            }
            return ready;
        }
    };
    template<typename T>
    class BlockingTask : public IBlockingTask
    {
    public:
        BlockingTask(std::future<T>&& f) : mFuture{std::move(f)} {}
        ~BlockingTask() override {if(!mCompleted) complete();}
        fb::future<T> getFuture() {return mPromise.get_future();}
        bool isReady() const override { return waitFor(std::chrono::seconds{0}); }
        bool waitFor(const std::chrono::duration<float> &duration) const override {
            switch (mFuture.wait_for(duration))
            {
            case std::future_status::ready: return true;
            case std::future_status::timeout: return false;
            case std::future_status::deferred: throw std::runtime_error("deferred task is not allowed");
            default: throw std::runtime_error("unknown error");
            }
        }
        void complete() override {
            mCompleted = true;
            std_optional<T> t;
            try {
                t = mFuture.get();
            } catch (...) {
                mPromise.set_exception(std::current_exception());
            }
            if (t.has_value()) {
                mPromise.set_value(std::move(t.value()));
            }
        }
    private:
        fb::promise<T> mPromise;
        std::future<T> mFuture;
        bool mCompleted = false;
    };

    class CudaStreamSyncTask : public IBlockingTask
    {
    public:
        CudaStreamSyncTask(cudaStream_t stream) {
            cudaCheck(cudaEventRecord(mEvent.get(), stream));
        }
        ~CudaStreamSyncTask() override {if(!mCompleted) complete();}
        fb::future<void> getFuture() {return mPromise.get_future();}
        bool isReady() const override { return waitFor(std::chrono::seconds{0}); }
        bool waitFor(const std::chrono::duration<float> &duration) const override {
            if (duration == duration.zero()) {
                const cudaError_t err = cudaEventQuery(mEvent.get());
                if (err == cudaErrorNotReady) {
                    return false;
                }
                else if (err == cudaSuccess) {
                    return true;
                }
                else {
                    cudaCheck(err);
                }
            }
            else {
                cudaCheck(cudaEventSynchronize(mEvent.get()));
                return true;
            }
            throw std::runtime_error("fatal error");
        }
        void complete() override {
            mCompleted = true;
            try {
                cudaCheck(cudaEventSynchronize(mEvent.get()));
                mPromise.set_value();
            } catch (...) {
                mPromise.set_exception(std::current_exception());
            }
        }
    private:
        fb::promise<void> mPromise;
        cudapp::PooledCudaEvent mEvent = cudapp::createPooledCudaEvent();
        bool mCompleted = false;
    };
public:
    template<typename Rep, typename Period>
    FiberBlockingService(std::chrono::duration<Rep, Period> checkInterval = std::chrono::milliseconds{100}, size_t windowSize = 1024u, size_t maxPendingTasks = 1u<<20)
        : mCheckInterval{checkInterval}
        , mWindowSize{windowSize}
        , mCapacity{maxPendingTasks}
        , mTasks{}
        , mThread{std::thread{&FiberBlockingService::worker, this}}
    {
    }
    ~FiberBlockingService() {
        {
            std::lock_guard<fb::mutex> lk{mMutex};
            mStop = true;
        }
        mCVarNotEmpty.notify_all();
        mThread->join();
        mThread.reset();
    }

    template <typename T>
    fb::future<T> delegate(std::future<T>&& f){
        auto task = std::make_unique<BlockingTask<T>>(std::move(f));
        fb::future<T> result = task->getFuture();
        enqueue(std::move(task));
        return result;
    }

    void syncCudaStream(cudaStream_t stream){
#define FIBER_BLOCKING_SERVICE_CUDA_SYNC_METHOD 2
#if FIBER_BLOCKING_SERVICE_CUDA_SYNC_METHOD == 0
        cudaCheck(cudaStreamSynchronize(stream));
#elif FIBER_BLOCKING_SERVICE_CUDA_SYNC_METHOD == 1
        fiberSyncCudaStream(stream);
#else
        auto task = std::make_unique<CudaStreamSyncTask>(stream);
        fb::future<void> f = task->getFuture();
        enqueue(std::move(task));
        return f.get();
#endif
    }

    size_t getNbTasks() const {
        std::unique_lock<fb::mutex> lk{mMutex};
        return mTasks.size();
    }
private:
    void enqueue(std::unique_ptr<IBlockingTask> task) {
        if (!task->tryComplete()) {
            {
                std::unique_lock<fb::mutex> lk{mMutex};
                if (mStop) {
                    throw std::runtime_error("Service stopped");
                }
                mCVarNotFull.wait(lk, [this]{return mTasks.size() < mCapacity;});
                mTasks.emplace_back(std::move(task));
            }
            mCVarNotEmpty.notify_all();
        }
    }

private:
    void worker() noexcept{
        std::vector<std::list<std::unique_ptr<IBlockingTask>>::iterator> eraseList; // put this outside to save some alloc/free
        while(true){
            {
                std::unique_lock<fb::mutex> lk{mMutex};
                mCVarNotEmpty.wait(lk, [this]{return !mTasks.empty() || mStop;});
                if (mTasks.empty() && mStop){
                    break;
                }
                auto iter = mTasks.begin();
                for (size_t i = 0; i < mWindowSize && iter != mTasks.end(); i++, iter = std::next(iter)){
                    const bool success = (*iter)->tryComplete(i == 0 ? mCheckInterval : std::chrono::milliseconds{0});
                    if (success) {
                        eraseList.emplace_back(iter);
                    }
                }
                for(auto iter : eraseList) {
                    mTasks.erase(iter);
                }
            }
            switch (eraseList.size())
            {
            case 0: break;
            case 1: mCVarNotFull.notify_one(); break;
            default: mCVarNotFull.notify_all(); break;
            }
            eraseList.clear();
        }
    }
private:
    std::chrono::duration<float> mCheckInterval = std::chrono::milliseconds{100};
    size_t mWindowSize = 1024u;
    size_t mCapacity = 1u<<20;
    bool mStop = false;
    mutable fb::mutex mMutex;
    mutable fb::condition_variable mCVarNotFull;
    mutable fb::condition_variable mCVarNotEmpty;
    std::list<std::unique_ptr<IBlockingTask>> mTasks;
    std_optional<std::thread> mThread;
};

} // cudapp
