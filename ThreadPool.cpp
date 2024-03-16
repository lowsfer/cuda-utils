#include "ThreadPool.h"
#include <algorithm>

std::function<void()> ThreadPool::consume()
{
    std::unique_lock<std::mutex> lk(mMutex);
    mCondVar.wait(lk, [this]{ return mStop || !mTasks.empty(); });
    if(mStop && mTasks.empty())
    {
        return {};
    }
    std::function<void()> task = std::move(mTasks.front());
    mTasks.pop();
    return task;
}
std::function<void()> ThreadPool::tryConsume()
{
    std::unique_lock<std::mutex> lk(mMutex);
    if(mTasks.empty())
    {
        return {};
    }
    std::function<void()> task = std::move(mTasks.front());
    mTasks.pop();
    return task;
}

void ThreadPool::worker()
{
    while (true)
    {
        auto const task = consume();
        if (task == nullptr)
        {
            break;
        }
        task();
    }
}

ThreadPool::ThreadPool(size_t nbThreads)
{
    while (mWorkers.size() < nbThreads){
        std::thread thrd{&ThreadPool::worker, this};
        mWorkers.emplace(thrd.get_id(), std::move(thrd));
    }
}

void ThreadPool::stop()
{
    {
        std::lock_guard<std::mutex> lock(mMutex);
        mStop = true;
    }
    mCondVar.notify_all();
    for (auto& w: mWorkers)
    {
        w.second.join();
    }
}

bool ThreadPool::isInWorkerThread() const
{
    const auto tid = std::this_thread::get_id();
    return mWorkers.find(tid) != mWorkers.end();
}

// examples

#if 0
// A normal example. The main thread divide tasks and dispatch them to the thread pool.
#include <numeric>
int main()
{
    int data[2048];
    std::iota(std::begin(data), std::end(data), 0);
    auto func = [data](size_t beg, size_t end){
        return std::accumulate(&data[beg], &data[end], 0, [](int acc, int x){return acc + x;});
    };

    ThreadPool pool{4};
    size_t batchSize = 128;
    std::vector<ThreadPool::Future<int>> futures;
    const size_t totalSize = sizeof(data) / sizeof(data[0]);
    for (size_t beg = 0; beg < totalSize; beg += batchSize)
    {
        futures.emplace_back(pool.enqueue(func, beg, std::min(beg + batchSize, totalSize)));
    }
    int result = std::accumulate(futures.begin(), futures.end(), 0, [](int acc, ThreadPool::Future<int>& f){return acc + f.get();});
    printf("%d\n", result);

    return 0;
}
#endif

#if 0
// A extreme (inefficient) example.
// The purpose is to show that the thread pool will not deadlock even if we wait for results inside the worker threads.
// The enqueued tasks will dispatch new tasks and wait for them.
#include <numeric>

int func (ThreadPool* pool, int const* data, size_t beg, size_t end) {
    const size_t distance = end - beg;
    if (distance == 0)
    {
        return 0;
    }
    else if (distance == 1)
    {
        return data[beg];
    }
    else
    {
        size_t mid = (beg + end) / 2;
        auto a = pool->enqueue(func, pool, data, beg, mid);
        auto b = pool->enqueue(func, pool, data, mid, end);
        return a.get() + b.get();
    }
    return std::accumulate(&data[beg], &data[end], 0, [](int acc, int x){return acc + x;});
}

int main()
{
    int data[2048];
    std::iota(std::begin(data), std::end(data), 0);

    ThreadPool pool{4};
    
    int result = pool.enqueue(func, &pool, data, 0, sizeof(data) / sizeof(data[0])).get();
    printf("%d\n", result);

    return 0;
}
#endif
