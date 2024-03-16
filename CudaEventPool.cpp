#include "CudaEventPool.h"
#include <queue>
#include "cuda_utils.h"
#include "macros.h"

namespace cudapp
{

class CudaEventPool : public std::enable_shared_from_this<CudaEventPool>
{
public:
    ~CudaEventPool() {ASSERT(mNbInUse == 0);}
    using Deleter = PooledCudaEventDeleter;
    using PooledEvent = PooledCudaEvent;
    PooledEvent create();
    void reserve(size_t count);
private:
    friend Deleter;
    void recycle(cudaEvent_t ev);
private:
    std::mutex mMutex;
    std::queue<CudaEvent> mCache;
    size_t mNbInUse {0};
};

void PooledCudaEventDeleter::operator()(cudaEvent_t ev) const
{
    if (ev == nullptr) {
        return;
    }
    assert(pool != nullptr);
    pool->recycle(ev);
}

PooledCudaEvent CudaEventPool::create() {
    std::lock_guard<std::mutex> lk{mMutex};
    if (mCache.empty()) {
        do{
            mCache.push(makeCudaEvent());
        }while (mCache.size() < mNbInUse);
    }
    assert(!mCache.empty());
    CudaEvent ev = std::move(mCache.front());
    mCache.pop();
    mNbInUse++;
    return PooledEvent(ev.release(), Deleter{shared_from_this()});
}

void CudaEventPool::recycle(cudaEvent_t ev) {
    std::lock_guard<std::mutex> lk{mMutex};
    mCache.push(CudaEvent{ev});
    mNbInUse--;
}

void CudaEventPool::reserve(size_t count) {
    while (mCache.size() < count) {
        mCache.push(makeCudaEvent());
    }
}

std::shared_ptr<CudaEventPool> getCudaEventPool() {
    static std::mutex lock;
    static std::weak_ptr<CudaEventPool> sPool;
    std::lock_guard<std::mutex> lk{lock};
    auto pool = sPool.lock();
    if (pool == nullptr) {
        pool = std::make_shared<CudaEventPool>();
        pool->reserve(32);
        sPool = pool;
    }
    return pool;
}

PooledCudaEvent createPooledCudaEvent(){
    return getCudaEventPool()->create();
}

} // namespace cudapp
