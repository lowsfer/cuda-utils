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
#include "StorageManager.h"
#include <utility>
#include "StorageObjects.h"
#include "CudaEventPool.h"

namespace cudapp::storage
{
template <typename ElemType, CudaMemType memType = CudaMemType::kDevice, bool isBlockLinear = false>
class CacheableScratchObject : public CacheableObjectBase
{
    static_assert(!isBlockLinear || memType == CudaMemType::kDevice);
public:
    template <bool enabler = true, typename = std::enable_if_t<enabler && !isBlockLinear, void>>
    CacheableScratchObject(CudaMemPool<memType>& pool, size_t scratchSize)
    : mScratchSize{scratchSize}
    , mStorage{pool}
    {
        setStorageLocation(StorageLocation::kDisk);
    }

    template <bool enabler = true, typename = std::enable_if_t<enabler && isBlockLinear, void>>
    CacheableScratchObject(CudaMemPool<memType>& pool, uint32_t width, uint32_t height, uint32_t flags = uint32_t{cudaArrayDefault})
    : mScratchSize{width * height}
    , mStorage{pool, width, height, flags}
    {
        setStorageLocation(StorageLocation::kDisk);
    }

    void migrateTo(StorageLocation dst, cudaStream_t stream) override {
        const StorageLocation src = getCurrentStorageLocation();
        if (src == impl::toStorageLocation(memType)) {
            mStorage.notifyMigratedToStream(stream);
        }
        auto isValid = [](StorageLocation loc){return loc == impl::toStorageLocation(memType);};
        if (isValid(src) && !isValid(dst)) {
            ASSERT(!mStorage.empty());
            mStorage.clear(stream);
        }
        if (!isValid(src) && isValid(dst)) {
            ASSERT(mStorage.empty());
            mStorage.resize(mScratchSize, stream);
        }
        setStorageLocation(dst);
    }
    StorageLocation getCurrentEvictionTarget() override {
        ASSERT(getCurrentStorageLocation() != StorageLocation::kDisk);
        return StorageLocation::kDisk;
    }
    size_t getCurrentStorageBytes(StorageLocation loc) const override {
        if (loc == impl::toStorageLocation(memType)) {
            return sizeof(ElemType) * mStorage.getSize();
        }
        return 0;
    }
    std::variant<ElemType*, cudaArray_t> getData() const {
        return mStorage.data();
    }
    template <bool enabler = true>
    std::enable_if_t<enabler && !isBlockLinear, ElemType*> getMemData() const {
        return std::get<ElemType*>(getData());
    }
    template <bool enabler = true>
    std::enable_if_t<enabler && isBlockLinear, cudaArray_t> getCudaArray() const {
        return std::get<cudaArray_t>(getData());
    }
    StorageLocation getCurrentStorageLocation() const override {
        return mLocation;
    }
private:
    void setStorageLocation(StorageLocation loc) {mLocation = loc;}
    StorageLocation mLocation = StorageLocation::kUnknown;

    size_t mScratchSize;
    std::conditional_t<isBlockLinear, BlockLinearStorage<ElemType>, CudaMemStorage<memType, ElemType>> mStorage;
};


template <typename ElemType, CudaMemType memType = CudaMemType::kDevice, bool isBlockLinear = false>
struct AcquiredScratch
{
    AcquiredObj holder; // RAII resource holder

    CacheableScratchObject<ElemType, memType, isBlockLinear>* obj() const {
        return dynamic_cast<CacheableScratchObject<ElemType, memType, isBlockLinear>*>(holder.get());
    }
    StorageLocation loc() const {
        return holder.get()->getCurrentStorageLocation();
    }
    template <bool enabler = true>
    std::enable_if_t<enabler && !isBlockLinear, ElemType*> data() const {
        return obj()->getMemData();
    }
    template <bool enabler = true>
    std::enable_if_t<enabler && isBlockLinear, cudaArray_t> array() const {
        return obj()->getCudaArray();
    }
    size_t nbElems() const {
        return obj()->getCurrentStorageBytes(loc()) / sizeof(ElemType);
    }
    void reset() {holder.reset();}
};

// unique == true for read and write; false for read-only.
// block means wait on mutex contention.
// !Important: stream must live longer than the cache object. Otherwise we get segfault when destroying the cached object as its resource is still associated with the stream.
template <typename ElemType, CudaMemType memType = CudaMemType::kDevice, bool isBlockLinear = false>
AcquiredScratch<ElemType, memType, isBlockLinear> acquireScratch(StorageManager& manager, const CacheObjKeyType key, const cudaStream_t stream, const bool unique, const bool block = true)
{
    constexpr StorageLocation loc = impl::toStorageLocation(memType);
    AcquiredObj holder = manager.acquire(unique, block, key, loc, stream);
#ifndef NDEBUG
    const auto obj = dynamic_cast<CacheableScratchObject<std::remove_const_t<ElemType>, memType, isBlockLinear>*>(holder.get());
    const auto onExit = makeScopeGuard([obj]{REQUIRE(obj->getCurrentStorageLocation() == loc);});
#endif
    return {std::move(holder)};
}
template <typename ElemType, CudaMemType memType = CudaMemType::kDevice, bool isBlockLinear = false>
void evictScratch(StorageManager& manager, const CacheObjKeyType key, const cudaStream_t stream)
{
    auto obj = dynamic_cast<CacheableScratchObject<ElemType, memType, isBlockLinear>*>(manager.getObj(key));
    const auto dstLoc = obj->getCurrentEvictionTarget();
    AcquiredObj holder = manager.acquire(true, true, key, dstLoc, stream);
}


} // namespace cudapp::storage
