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
#include <cuda_runtime_api.h>
#include "cuda_utils.h"
#include "macros.h"
#include <memory>
#include <mutex>
#include <list>
#include <unordered_map>
#include "CudaEventPool.h"
#include <cuda_fp16.h>

inline bool operator==(const cudaChannelFormatDesc& a, const cudaChannelFormatDesc& b) {
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w && a.f == b.f;
}
inline bool operator==(const cudaExtent& a, const cudaExtent& b) {
    return a.width == b.width && a.height == b.height && a.depth == b.depth;
}

namespace cudapp
{
struct CudaArrayAttributes
{
    cudaChannelFormatDesc channelFormat;
    cudaExtent extent;
    unsigned int flags;
    bool operator==(const CudaArrayAttributes& r) const {
        return channelFormat == r.channelFormat && extent == r.extent && flags == r.flags;
    }
};
} // namespace cudapp

namespace std
{
template <>
struct hash<cudaChannelFormatDesc>
{
    size_t operator()(const cudaChannelFormatDesc& f) const {
        return combinedHash(f.x, f.y, f.z, f.w, f.f);
    }
};
template <>
struct hash<cudaExtent>
{
    size_t operator()(const cudaExtent& e) const {
        return combinedHash(e.depth, e.width, e.height);
    }
};
template <>
struct hash<cudapp::CudaArrayAttributes>
{
    size_t operator()(const cudapp::CudaArrayAttributes& a) const {
        return combinedHash(a.channelFormat, a.extent, a.flags);
    }
};
} // namespace std

namespace cudapp
{
using CudaArray = CudaRes<cudaArray_t, &cudaFreeArray>;

// @fixme: I don't know what value it is ... need to find out.
static constexpr int32_t cudaArrayBlockSize = 16;

inline CudaArray createCudaArray2D(const cudaChannelFormatDesc& channelFmt, size_t width, size_t height, int flags = cudaArrayDefault){
    cudaArray_t arr = nullptr;
    cudaCheck(cudaMallocArray(&arr, &channelFmt, width, height, flags));
    return CudaArray{arr};
}

inline CudaArray createCudaArray2D(const CudaArrayAttributes& attr){
    ASSERT(attr.extent.depth == 1 || attr.extent.depth == 0);
    return createCudaArray2D(attr.channelFormat, attr.extent.width, attr.extent.height, attr.flags);
}


template <typename T>
inline CudaArray createCudaArray2D(size_t width, size_t height, int flags = cudaArrayDefault){
    return createCudaArray2D(std::is_same_v<std::decay_t<T>, half2> ? cudaCreateChannelDescHalf2() : cudaCreateChannelDesc<T>(), width, height, flags);
}

inline cudaExtent getArrayExtent(cudaArray_t arr){
    cudaExtent extent{};
    cudaCheck(cudaArrayGetInfo(nullptr, &extent, nullptr, arr));
    return extent;
}

inline CudaArrayAttributes getCudaArrayAttributes(cudaArray_t array) {
    CudaArrayAttributes a{};
    cudaCheck(cudaArrayGetInfo(&a.channelFormat, &a.extent, &a.flags, array));
    return a;
}

inline size_t getCudaArray2DBytes(const CudaArrayAttributes& attr) {
    const auto& fmt = attr.channelFormat;
    const auto extent = attr.extent;
    assert(extent.depth == 1 || extent.depth == 0);
    return divUp(fmt.x + fmt.y + fmt.z + fmt.w, 8) * roundUp(extent.width, cudaArrayBlockSize) * roundUp(extent.height, cudaArrayBlockSize);
}

inline size_t getCudaArrayNbPixels(cudaArray_t array) {
    cudaExtent extent{};
    cudaCheck(cudaArrayGetInfo(nullptr, &extent, nullptr, array));
    ASSERT(extent.depth == 1 || extent.depth == 0);
    return extent.width * extent.height * (extent.depth == 0 ? size_t{1} : extent.depth);
}

inline size_t getCudaArray2DBytes(cudaArray_t array) {
    return getCudaArray2DBytes(getCudaArrayAttributes(array));
}

#if 0 // use CudaMemPool<CudaMemType::kDevice> instead
class CudaArrayPool;

struct CudaArrayPoolDeleter
{
    void operator()(cudaArray_t p);
    CudaArrayPool* pool;
    cudaStream_t stream;
};

class PooledCudaArray : public std::unique_ptr<std::remove_pointer_t<cudaArray_t>, CudaArrayPoolDeleter>
{
public:
    using Deleter = CudaArrayPoolDeleter;
    using std::unique_ptr<std::remove_pointer_t<cudaArray_t>, CudaArrayPoolDeleter>::unique_ptr;
    using std::unique_ptr<std::remove_pointer_t<cudaArray_t>, CudaArrayPoolDeleter>::operator=;
    cudaExtent extent() const {return getArrayExtent(this->get());}
    void migrateToStream(cudaStream_t stream){
        if (stream != this->get_deleter().stream){
            connectStreams(this->get_deleter().stream, stream);
            this->get_deleter().stream = stream;
        }
    }
};

class CudaArrayPool
{
public:
    CudaArrayPool() = default;
    CudaArrayPool(size_t maxTotalBytes) : mMaxTotalBytes{maxTotalBytes}{}
    ~CudaArrayPool(){
        REQUIRE(mInUseBytes == 0 && mInUseBlocks.empty());
        clearCache();
    }
    using Deleter = CudaArrayPoolDeleter;
    friend void Deleter::operator()(cudaArray_t);

    template <typename Pixel>
    PooledCudaArray alloc(size_t width, size_t height, int flags, cudaStream_t stream){
        return PooledCudaArray{allocImpl(cudaCreateChannelDesc<Pixel>(), width, height, flags, stream), Deleter{this, stream}};
    }

    void clearCache(){
        std::lock_guard<std::mutex> lock{mMutex};
        for (const auto& block : mCachedBlocks){
            cudaCheck(cudaEventSynchronize(block.readyEvent.get()));
        }
        mCachedBlocks.clear();
        mCachedBlockGroups.clear();
        mCachedBytes = 0;
        mTotalBytes = mInUseBytes;
    }

private:
    void registerNewMemUnsafe(CudaArray&& newArray, cudaStream_t stream)
    {
        const cudaArray_t a = newArray.get();
        mInUseBlocks.try_emplace(a, Block{std::move(newArray), stream, createPooledCudaEvent(), mIdxNextAlloc++});
        const size_t nbBytes = getCudaArray2DBytes(a);
        mTotalBytes += nbBytes;
        mInUseBytes += nbBytes;
    }
    cudaArray_t allocImpl(const cudaChannelFormatDesc& channelFmt, size_t width, size_t height, int flags, cudaStream_t stream)
    {
        const auto onExit = makeScopeGuard([this](){fitCache();});
        std::lock_guard<std::mutex> lock{mMutex};
        const CudaArrayAttributes attributes {
            channelFmt, cudaExtent{width, height, 1}, static_cast<unsigned int>(flags)
        };
        const auto iterRange = mCachedBlockGroups.equal_range(attributes);
        if (iterRange.first == iterRange.second) {
            CudaArray newArray = createCudaArray2D(channelFmt, width, height, flags);
            const cudaArray_t array = newArray.get();
            registerNewMemUnsafe(std::move(newArray), stream);
            return array;
        }
        else {
            const auto iterReady = std::find_if(iterRange.first, iterRange.second, [stream](const auto item){
                if (item.second->stream == stream){
                    return true;
                }
                const cudaError_t error = cudaEventQuery(item.second->readyEvent.get());
                if (error != cudaErrorNotReady) cudaCheck(error);
                return error == cudaSuccess;
            });
            auto iterToUse = mCachedBlockGroups.end();
            if (iterReady != iterRange.second) {
                iterToUse = iterReady;
            }
            else {
                iterToUse = std::min_element(iterRange.first, iterRange.second, [](const auto& a, const auto& b){return a.second->ageOrder < b.second->ageOrder;});
            }
            const auto iterBlock = iterToUse->second;
            cudaCheck(cudaStreamWaitEvent(stream, iterBlock->readyEvent.get(), 0));
            iterBlock->stream = stream;
            const auto array = iterBlock->array.get();
            const auto attributes = iterToUse->first;
            REQUIRE(attributes == getCudaArrayAttributes(iterBlock->array.get()));
            const auto emplaceResult = mInUseBlocks.emplace(array, std::move(*iterBlock));
            REQUIRE(emplaceResult.second);
            assert(getCudaArrayAttributes(emplaceResult.first->first) == attributes);
            emplaceResult.first->second.ageOrder = mIdxNextAlloc++;
            mInUseBytes += getCudaArray2DBytes(attributes);
            mCachedBytes -= getCudaArray2DBytes(attributes);
            assert(mTotalBytes == mCachedBytes + mInUseBytes);
            mCachedBlocks.erase(iterBlock);
            mCachedBlockGroups.erase(iterToUse);
            return array;
        }
    }
    void freeImpl(cudaArray_t array, cudaStream_t stream){
        const auto onExit = makeScopeGuard([this](){fitCache();});
        std::lock_guard<std::mutex> lock{mMutex};
        {
            const auto iterInUseBlock = mInUseBlocks.find(array);
            REQUIRE(iterInUseBlock != mInUseBlocks.end());
#if CUDAPP_CUDA_MEM_POOL_ALLOW_STREAM_MIGRATION
            if (iterInUseBlock->second.stream != stream){
                connectStreams(stream, iterInUseBlock->second.stream);
            }
#else
            REQUIRE(iterInUseBlock->second.stream == stream);
#endif
            mCachedBlocks.emplace_back(std::move(iterInUseBlock->second));
            mInUseBlocks.erase(iterInUseBlock);
        }
        const auto iterBlock = std::prev(mCachedBlocks.end());
        iterBlock->ageOrder = mIdxNextFree++;
        cudaCheck(cudaEventRecord(iterBlock->readyEvent.get(), iterBlock->stream));
        const auto attr = getCudaArrayAttributes(iterBlock->array.get());
        mCachedBlockGroups.emplace(attr, iterBlock);
        mCachedBytes += getCudaArray2DBytes(attr);
        mInUseBytes -= getCudaArray2DBytes(attr);
        assert(mTotalBytes == mCachedBytes + mInUseBytes);
    }
    void fitCache(){
        std::lock_guard<std::mutex> lock{mMutex};
        while ((mTotalBytes > mMaxTotalBytes || mCachedBytes > mMaxCachedBytes) && !mCachedBlocks.empty()){
            const auto iterBlockRm = mCachedBlocks.begin();
            const auto attr = getCudaArrayAttributes(iterBlockRm->array.get());
            const size_t size = getCudaArray2DBytes(attr);
            cudaCheck(cudaEventSynchronize(iterBlockRm->readyEvent.get()));
            mCachedBytes -= size;
            mTotalBytes -= size;
            assert(mTotalBytes == mCachedBytes + mInUseBytes);
            const auto [beg, end] = mCachedBlockGroups.equal_range(attr);
            const auto iterMapRm = std::find_if(beg, end, [iterBlockRm](const auto x){return x.second == iterBlockRm;});
            REQUIRE(iterMapRm != mCachedBlockGroups.end());
            mCachedBlockGroups.erase(iterMapRm);
            mCachedBlocks.erase(iterBlockRm);
        }
    }
private:
    struct Block
    {
        CudaArray array;
        cudaStream_t stream; // indicates availability
        PooledCudaEvent readyEvent; // indicates finish of use after release
        uint32_t ageOrder; // small means older. reset on alloc/free
    };
private:
    mutable std::mutex mMutex;
    uint32_t mIdxNextAlloc = 0;
    uint32_t mIdxNextFree = 0;
    const int mDeviceId = getCudaDevice();
    size_t mMaxTotalBytes = 1ul << 30; // 1GB
    size_t mMaxCachedBytes = mMaxTotalBytes / 4;
    size_t mTotalBytes {0};
    size_t mInUseBytes {0};
    size_t mCachedBytes {0};
    std::unordered_map<cudaArray_t, Block> mInUseBlocks;
    std::list<Block> mCachedBlocks; // order by age old to new
    std::unordered_multimap<cudapp::CudaArrayAttributes, typename std::list<Block>::iterator> mCachedBlockGroups; // grouped by attributes
};

inline void CudaArrayPoolDeleter::operator()(cudaArray_t p) { pool->freeImpl(p, stream); }
#endif

} // namespace cudapp