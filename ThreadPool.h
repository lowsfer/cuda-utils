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

#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include <unordered_map>

class ThreadPool {
public:
    template <typename T>
    class Future
    {
    public:
        Future(ThreadPool& pool, std::future<T>&& fut) : mPool{&pool}, mStdFuture{std::move(fut)} {}
        T get();
    private:
        bool isReady()
        {
            return mStdFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
        }
    private:
        ThreadPool* mPool;
        std::future<T> mStdFuture;
    };
public:
    explicit ThreadPool(size_t nbThreads = 1);
    ~ThreadPool() {stop();}

    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args) -> Future<typename std::invoke_result<F, Args...>::type>
    {
        using RetType = typename std::invoke_result<F, Args...>::type;
        return Future<RetType>{*this, enqueueStd(std::forward<F>(f), std::forward<Args>(args)...)};
    }
    template<typename F, typename... Args>
    auto enqueueStd(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type>
    {
        using RetType = typename std::invoke_result<F, Args...>::type;

        // use shared_ptr to make it copyable
        auto pTask = std::make_shared<std::packaged_task<RetType()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::future<RetType> result = pTask->get_future();
        {
            std::lock_guard<std::mutex> lk(mMutex);
            if(mStop)
            {
                throw std::runtime_error("the pool has been stopped");
            }
            mTasks.emplace([pTask](){ (*pTask)(); });
        }
        mCondVar.notify_one();
        return result;
    }
private:
    void worker();
    void stop();
    // returns invalid function when empty and stopped
    std::function<void()> consume();
    // returns invalid function when empty
    std::function<void()> tryConsume();
    bool isInWorkerThread() const;
private:
    std::unordered_map<std::thread::id, std::thread> mWorkers;

    mutable std::mutex mMutex;
    std::condition_variable mCondVar;
    bool mStop{false};
    std::queue<std::function<void()>> mTasks;
};

template <typename T>
inline T ThreadPool::Future<T>::get()
{
    if (!mPool->isInWorkerThread())
    {
        return mStdFuture.get();
    }
    while (!isReady())
    {
        auto const task = mPool->tryConsume();
        if (task == nullptr)
        {
            break;
        }
        task();
    }
    return mStdFuture.get();
}
