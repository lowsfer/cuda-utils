#pragma once
#include "cpp_utils.h"
#include "cuda_utils.h"
#include <map>
#include <unordered_map>
#include <list>
#include <mutex>
#include "CudaEventPool.h"
#include <variant>
#include "CudaArray.h"
#include <atomic>
#include <shared_mutex>

#if CUDAPP_ENABLE_TEST_CODE
#include <cstring>
#endif

#define CUDAPP_CUDA_MEM_POOL_ALLOW_STREAM_MIGRATION 1

namespace cudapp
{
namespace storage
{

template <CudaMemType memType>
class CudaMemPool;

template <CudaMemType memType>
struct CudaMemPoolDeleter
{
    void operator()(void* p);
    CudaMemPool<memType>* pool;
    cudaStream_t stream;
    size_t nbBytes;// Not needed for deleter but useful for user. May be smaller than size of the memory block.
};

template <typename Elem, CudaMemType memType>
class PooledCudaMem : public std::unique_ptr<Elem[], CudaMemPoolDeleter<memType>>{
public:
    using Deleter = CudaMemPoolDeleter<memType>;
    using std::unique_ptr<Elem[], Deleter>::unique_ptr;
    using std::unique_ptr<Elem[], Deleter>::operator=;
    size_t size() const {return this->get_deleter().nbBytes / sizeof(Elem);}
    void migrateToStream(cudaStream_t stream){
        if (stream != this->get_deleter().stream){
            connectStreams(this->get_deleter().stream, stream);
            this->get_deleter().stream = stream;
        }
    }
};

struct CudaMemPoolArrayDeleter
{
    void operator()(cudaArray_t p);
    CudaMemPool<CudaMemType::kDevice>* pool;
    cudaStream_t stream;
};

class PooledCudaArray : public std::unique_ptr<std::remove_pointer_t<cudaArray_t>, CudaMemPoolArrayDeleter>
{
public:
    using Deleter = CudaMemPoolArrayDeleter;
    using std::unique_ptr<std::remove_pointer_t<cudaArray_t>, CudaMemPoolArrayDeleter>::unique_ptr;
    using std::unique_ptr<std::remove_pointer_t<cudaArray_t>, CudaMemPoolArrayDeleter>::operator=;
    cudaExtent extent() const {return getArrayExtent(this->get());}
    void migrateToStream(cudaStream_t stream){
        if (stream != this->get_deleter().stream){
            connectStreams(this->get_deleter().stream, stream);
            this->get_deleter().stream = stream;
        }
    }
};

template <CudaMemType memType>
class CudaMemPool
{
    struct LinearMem
    {
        CudaMem<std_byte, memType> data;
        size_t size;// in bytes
    };
public:
    CudaMemPool() = default;
    CudaMemPool(size_t maxTotalBytes) : mMaxTotalBytes{maxTotalBytes}{}
    ~CudaMemPool(){
        REQUIRE(mInUseBytes == 0 && mInUseBlocks.empty());
        clearCache();
        if (isVerboseEnvSet()) {
            printf("CudaMemPool<%s> hit rate for linear memory: %lu/%lu\n", toStr(memType),
                mStatistics.nbLinearAllocHit.load(std::memory_order_relaxed),
                mStatistics.nbLinearAlloc.load(std::memory_order_relaxed));
            if (memType == CudaMemType::kDevice) {
                printf("CudaMemPool<%s> hit rate for cuda array: %lu/%lu\n", toStr(memType),
                    mStatistics.nbArrayAllocHit.load(std::memory_order_relaxed),
                    mStatistics.nbArrayAlloc.load(std::memory_order_relaxed));
            }
        }
    }

    using Deleter = CudaMemPoolDeleter<memType>;
    using ArrayDeleter = CudaMemPoolArrayDeleter;
    template <typename Elem>
    using PooledCudaMem = PooledCudaMem<Elem, memType>;
    friend Deleter;
    friend ArrayDeleter;

    template <typename Elem, bool allowNonTrivial = false>
    PooledCudaMem<Elem> alloc(size_t nbElems, cudaStream_t stream){
        static_assert(allowNonTrivial || std::is_trivial<Elem>::value);
        static_assert(std::is_standard_layout<Elem>::value);
        const auto nbBytes = sizeof(Elem) * nbElems;
        return PooledCudaMem<Elem>{static_cast<Elem*>(allocImpl<LinearTraits>(nbBytes, stream)), Deleter{this, stream, nbBytes}};
    }

    template <typename Pixel, bool enabler = true>
    std::enable_if_t<enabler && memType == CudaMemType::kDevice, PooledCudaArray>
    allocArray(size_t width, size_t height, unsigned flags/* = cudaArrayDefault*/, cudaStream_t stream){
        const CudaArrayAttributes attr{
            cudaCreateChannelDesc<Pixel>(),
            cudaExtent{width, height, 0},
            flags
        };
        return allocArray(attr, stream);
    }

    template <bool enabler = true>
    std::enable_if_t<enabler && memType == CudaMemType::kDevice, PooledCudaArray>
    allocArray(const CudaArrayAttributes& attr, cudaStream_t stream){
        return PooledCudaArray{allocImpl<ArrayTraits>(attr, stream), ArrayDeleter{this, stream}};
    }

    template <bool enabler = true>
    std::enable_if_t<enabler && memType == CudaMemType::kDevice>
    setOnArrayFreeCallback(std::function<void(cudaArray_t)> callback) {
        std::lock_guard lk{mMutexOnCudaArrayFree};
        mOnCudaArrayFree = std::move(callback);
    }
    

    // Transfer ownership of existing memory block to the pool. Need tests.
    template <typename Elem>
    PooledCudaMem<Elem> registerExternalMem(CudaMem<Elem, memType>&& mem, size_t nbElems, cudaStream_t stream){
        const auto onExit = makeScopeGuard([this](){fitCache();});
        std::lock_guard<std::mutex> lock{mMutex};
        Elem* const p = mem.get();
        const size_t nbBytes = sizeof(Elem) * nbElems;
        registerNewMemUnsafe(CudaMem<std_byte, memType>{mem.release()}, nbBytes, stream);
        return PooledCudaMem<Elem>{p, Deleter{this, stream, nbBytes}};
    }

    void clearCache(){
        std::lock_guard<std::mutex> lock{mMutex};
        for (const auto& block : mCachedBlocks){
            cudaCheck(cudaEventSynchronize(block.readyEvent.get()));
        }
        mCachedBlocks.clear();
        mSortedCachedBlocks.clear();
        mCachedBytes = 0;
        mTotalBytes = mInUseBytes;
    }


private:
    struct ArrayDeleterInternal{
        void operator()(cudaArray_t arr) {
            pool.onCudaArrayFree(arr);
            cudaCheck(cudaFreeArray(arr));
        }
        CudaMemPool<memType>& pool;
    };
    using CudaArrayInternal = std::unique_ptr<cudaArray, ArrayDeleterInternal>;
    struct Block
    {
        std::variant<LinearMem, CudaArrayInternal> memory;
        cudaStream_t stream; // indicates availability
        PooledCudaEvent readyEvent; // indicates finish of use after release
        uint32_t ageOrder; // small means older. reset on alloc/free
        bool isLinear() const {
            return std::holds_alternative<LinearMem>(memory);
        }
        std_byte* data() const {
            return std::get<LinearMem>(memory).data.get();
        }
        cudaArray_t array() const {
            return std::get<CudaArrayInternal>(memory).get();
        }
        //! In bytes
        size_t size() const {
            return std::visit(Overloaded{
                [](const LinearMem& m){return m.size;},
                [](const CudaArrayInternal& a){return getCudaArray2DBytes(a.get());}
            }, memory);
        }
        using Handle = void*;
        Handle handle() const {
            return std::visit(Overloaded{
                [](const LinearMem& m)->Handle{return m.data.get();},
                [](const CudaArrayInternal& a)->Handle{return a.get();}
            }, memory);
        }
    };
private:
    mutable std::mutex mMutex;
    uint32_t mIdxNextAlloc = 0;
    uint32_t mIdxNextFree = 0;
    const int mDeviceId = getCudaDevice();
    size_t mMaxTotalBytes = (memType == CudaMemType::kDevice ? 4ul << 30 : 32ul << 30);
    float mMaxOverAllocRatio = 1.5f;
    size_t mTotalBytes = 0;
    size_t mInUseBytes = 0;
    size_t mCachedBytes = 0;
    std::unordered_map<void*, Block> mInUseBlocks;
    std::list<Block> mCachedBlocks; // order by age old to new
    std::multimap<size_t, typename std::list<Block>::iterator> mSortedCachedBlocks; // order by size small to large
    std::unordered_multimap<cudapp::CudaArrayAttributes, typename std::list<Block>::iterator> mCachedBlockGroups; // grouped by attributes

    struct {
        std::atomic_size_t nbLinearAlloc;
        std::atomic_size_t nbLinearAllocHit;
        std::atomic_size_t nbArrayAlloc;
        std::atomic_size_t nbArrayAllocHit;
    } mStatistics {};

    void onCudaArrayFree(cudaArray_t arr) {
        std::shared_lock lk{mMutexOnCudaArrayFree};
        if (mOnCudaArrayFree != nullptr) {
            mOnCudaArrayFree(arr);
        }
    }
    std::shared_mutex mMutexOnCudaArrayFree;
    std::function<void(cudaArray_t)> mOnCudaArrayFree;

private:
    struct LinearTraits {
        static constexpr bool isLinear = true;
        using Desc = size_t; // nbBytes;
        using Holder = CudaMem<std_byte, memType>;
        using Handle = void*;
        static Holder allocate(CudaMemPool<memType>&, Desc desc) {return allocCudaMem<std_byte, memType>(desc);}
        static size_t getNbBytes(Desc desc) {return desc;}
        using CacheMap = std::multimap<size_t, typename std::list<Block>::iterator>;
        static constexpr CacheMap CudaMemPool<memType>::* cacheMap = &CudaMemPool<memType>::mSortedCachedBlocks;
        static Desc getDesc(const Block& b) {return std::get<LinearMem>(b.memory).size;}
        static std::pair<typename CacheMap::iterator, typename CacheMap::iterator> cacheLookUp(CudaMemPool<memType>& pool, Desc desc) {
            CacheMap& cacheMapRef = pool.*cacheMap;
            return {
                cacheMapRef.lower_bound(desc),
                cacheMapRef.upper_bound(desc * pool.mMaxOverAllocRatio)
            };
        }
        static LinearMem makeBlockMem(Holder&& holder, const Desc& desc) { return LinearMem{std::move(holder), desc}; }
    };
    struct ArrayTraits {
        static constexpr bool isLinear = false;
        using Desc = cudapp::CudaArrayAttributes;
        using Handle = cudaArray_t;
        using Holder = CudaArrayInternal;
        static Holder allocate(CudaMemPool<memType>& pool, const Desc& desc) {return CudaArrayInternal{createCudaArray2D(desc).release(), ArrayDeleterInternal{pool}};}
        static size_t getNbBytes(const Desc& desc) {return getCudaArray2DBytes(desc);}
        using CacheMap = std::unordered_multimap<cudapp::CudaArrayAttributes,
            typename std::list<Block>::iterator>;
        static constexpr CacheMap CudaMemPool<memType>::* cacheMap = &CudaMemPool<memType>::mCachedBlockGroups;
        static Desc getDesc(const Block& b) {return getCudaArrayAttributes(b.array());}
        static std::pair<typename CacheMap::iterator, typename CacheMap::iterator> cacheLookUp(CudaMemPool<memType>& pool, const Desc& desc) {
            return (pool.*cacheMap).equal_range(desc);
        }
        static Holder makeBlockMem(Holder&& holder, const Desc&) { return std::move(holder); }
    };
    // internal use only. not thread-safe
    template <typename Traits>
    void registerNewMemUnsafe(typename Traits::Holder&& newMem, typename Traits::Desc desc, cudaStream_t stream){
        void* const p = newMem.get();
        const auto [iterBlock, success] = mInUseBlocks.emplace(p, Block{Traits::makeBlockMem(std::move(newMem), desc), stream, createPooledCudaEvent(), mIdxNextAlloc++});
        REQUIRE(success);
        REQUIRE(Traits::getDesc(iterBlock->second) == desc);
        const size_t nbBytes = Traits::getNbBytes(desc);
        mTotalBytes += nbBytes;
        mInUseBytes += nbBytes;
        assert(mTotalBytes == mCachedBytes + mInUseBytes);
    }

    template <typename Traits>
    typename Traits::Handle allocImpl(const typename Traits::Desc& desc, cudaStream_t stream){
        using Handle = typename Traits::Handle;
        const auto onExit = makeScopeGuard([this](){fitCache();});
        std::lock_guard<std::mutex> lock{mMutex};
        auto& cacheMap = this->*Traits::cacheMap;
        const auto [iterLower, iterUpper] = Traits::cacheLookUp(*this, desc);
        if constexpr (Traits::isLinear) {
            mStatistics.nbLinearAlloc.fetch_add(1U, std::memory_order_relaxed);
        }
        else {
            mStatistics.nbArrayAlloc.fetch_add(1U, std::memory_order_relaxed);
        }
        if (iterLower == iterUpper) {
            typename Traits::Holder newMem = Traits::allocate(*this, desc);
            void* const p = newMem.get();
            registerNewMemUnsafe<Traits>(std::move(newMem), desc, stream);
            return static_cast<Handle>(p);
        }
        else{
            if constexpr (Traits::isLinear) {
                mStatistics.nbLinearAllocHit.fetch_add(1U, std::memory_order_relaxed);
            }
            else {
                mStatistics.nbArrayAllocHit.fetch_add(1U, std::memory_order_relaxed);
            }
            const auto iterReady = std::find_if(iterLower, iterUpper, [stream](const auto item){
                if (item.second->stream == stream){
                    return true;
                }
                const cudaError_t error = cudaEventQuery(item.second->readyEvent.get());
                if (error != cudaErrorNotReady) cudaCheck(error);
                return error == cudaSuccess;
            });
            auto iterToUse = cacheMap.end();
            if (iterReady != iterUpper){
                iterToUse = iterReady;
            }
            else{
                iterToUse = std::min_element(iterLower, iterUpper, [](const auto& a, const auto& b){return a.second->ageOrder < b.second->ageOrder;});
            }
            const auto iterBlock = iterToUse->second;
            cudaCheck(cudaStreamWaitEvent(stream, iterBlock->readyEvent.get(), 0));
            iterBlock->stream = stream;
            const auto p = iterBlock->handle();
            const auto blockSize = Traits::getNbBytes(iterToUse->first);
            REQUIRE(blockSize == iterBlock->size());
            const auto emplaceResult = mInUseBlocks.emplace(p, std::move(*iterBlock));
            REQUIRE(emplaceResult.second);
            assert(emplaceResult.first->first == p);
            emplaceResult.first->second.ageOrder = mIdxNextAlloc++;
            mInUseBytes += blockSize;
            mCachedBytes -= blockSize;
            assert(mTotalBytes == mCachedBytes + mInUseBytes);
            mCachedBlocks.erase(iterBlock);
            if constexpr (Traits::isLinear) {
                mSortedCachedBlocks.erase(iterToUse);
            }
            else {
                mCachedBlockGroups.erase(iterToUse);
            }
            return static_cast<Handle>(p);
        }
    }
    // for both linear and cuda array
    void freeImpl(void* handle, cudaStream_t stream){
        const auto onExit = makeScopeGuard([this](){fitCache();});
        std::lock_guard<std::mutex> lock{mMutex};
        {
            const auto iterInUseBlock = mInUseBlocks.find(handle);
            REQUIRE(iterInUseBlock != mInUseBlocks.end());
#if CUDAPP_ENABLE_TEST_CODE
            if (iterInUseBlock->second.isLinear()) {
                if (memType == CudaMemType::kDevice || memType == CudaMemType::kManaged){
                    cudaCheck(cudaMemsetAsync(handle, 0xCC, iterInUseBlock->second.size(), stream));
                }
                else {
                    launchCudaHostFunc(stream, [handle, size{iterInUseBlock->second.size()}]{
                        std::memset(handle, 0xCC, size);});
                }
            }
#endif
#if CUDAPP_CUDA_MEM_POOL_ALLOW_STREAM_MIGRATION
            if (iterInUseBlock->second.stream != stream){
                // If you see segfault in libcuda.so inside this, this is likely because one stream is already destroyed.
                // Try to acquire memory in streams that are destroyed late
                // Or manually remove objects from storage manager in destructors.
                connectStreams(stream, iterInUseBlock->second.stream);
            }
#else
            REQUIRE(iterInUseBlock->second.stream == stream);
#endif
            mCachedBlocks.emplace_back(std::move(iterInUseBlock->second));
            mInUseBlocks.erase(iterInUseBlock);
        }
        const auto iterBlock = std::prev(mCachedBlocks.end());
        assert(iterBlock->handle() == handle);
        iterBlock->ageOrder = mIdxNextFree++;
        cudaCheck(cudaEventRecord(iterBlock->readyEvent.get(), iterBlock->stream));
        const size_t nbBytes = iterBlock->size();
        if (iterBlock->isLinear()) {
            mSortedCachedBlocks.emplace(nbBytes, iterBlock);
        }
        else {
            mCachedBlockGroups.emplace(getCudaArrayAttributes(iterBlock->array()), iterBlock);
        }
        mCachedBytes += nbBytes;
        mInUseBytes -= nbBytes;
        assert(mTotalBytes == mCachedBytes + mInUseBytes);
    }
    
    void fitCache(){
        std::lock_guard<std::mutex> lock{mMutex};
        const bool needCleaning = mTotalBytes > mMaxTotalBytes;
        const size_t maxCachedBytes = needCleaning ? mCachedBytes / 2 : mMaxTotalBytes;
#if 0
        if (isVerboseEnvSet() && needCleaning && !mCachedBlocks.empty()) {
            printf("CudaMemPool<%s> cached bytes: %lu -> %lu\n", toStr(memType), mCachedBytes, maxCachedBytes);
        }
#endif
        while (!mCachedBlocks.empty() && (mTotalBytes > mMaxTotalBytes || mCachedBytes > maxCachedBytes)){
            const auto iterBlockRm = mCachedBlocks.begin();
            if (iterBlockRm->isLinear()) {
                removeCacheEntryUnsafe<LinearTraits>(iterBlockRm);
            }
            else {
                removeCacheEntryUnsafe<ArrayTraits>(iterBlockRm);
            }
        }
    }
    
    template <typename Traits>
    void removeCacheEntryUnsafe(typename std::list<Block>::iterator iterBlockRm) {
        ASSERT(iterBlockRm->isLinear() == Traits::isLinear);
        const typename Traits::Desc desc = Traits::getDesc(*iterBlockRm);
        const size_t size = iterBlockRm->size();
        cudaCheck(cudaEventSynchronize(iterBlockRm->readyEvent.get()));
        mCachedBytes -= size;
        mTotalBytes -= size;
        assert(mTotalBytes == mCachedBytes + mInUseBytes);
        auto& cacheMap = this->*Traits::cacheMap;
        const auto [beg, end] = cacheMap.equal_range(desc);
        const auto iterMapRm = std::find_if(beg, end, [iterBlockRm](const auto x){return x.second == iterBlockRm;});
        REQUIRE(iterMapRm != cacheMap.end());
        cacheMap.erase(iterMapRm);
        mCachedBlocks.erase(iterBlockRm);
    }
};

template <CudaMemType memType>
inline void CudaMemPoolDeleter<memType>::operator()(void* p) {pool->freeImpl(p, stream);}
inline void CudaMemPoolArrayDeleter::operator()(cudaArray_t p) { pool->freeImpl(p, stream); }

using CudaDevMemPool = CudaMemPool<CudaMemType::kDevice>;
using CudaPinnedMemPool = CudaMemPool<CudaMemType::kPinned>;
using CudaSysMemPool = CudaMemPool<CudaMemType::kSystem>;

} // namespace storage
} // namespace cudapp
