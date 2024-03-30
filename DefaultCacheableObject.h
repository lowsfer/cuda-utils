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
#include <variant>
#include "CudaArray.h"
#include "CudaEventPool.h"
#include <atomic>

namespace cudapp::storage
{
template <typename ElemType>
struct LinearCacheableObjectTraits
{
    using Elem = ElemType;
    using CudaDataType = CudaDevMemStorage<ElemType>;
    using PinnedDataType = CudaPinnedMemStorage<ElemType>;
    using SysDataType = CudaSysMemStorage<ElemType>;
    using DiskDataType = DiskStorage<ElemType>;
    using HandleVariant = std::variant<ElemType*, fs::path>;
};

#if ENABLE_BLOCK_LINEAR_STORAGE
template <typename ElemType>
struct CudaArrayCacheableObjectTraits
{
    using Elem = ElemType;
    using CudaDataType = BlockLinearStorage<ElemType>;
    using PinnedDataType = CudaPinnedMemStorage<ElemType>;
    using SysDataType = CudaSysMemStorage<ElemType>;
    using DiskDataType = DiskStorage<ElemType>;
    using HandleVariant = std::variant<cudaArray_t, ElemType*, fs::path>;
};
#endif

// `DevStorage` may be `CudaDevMemStorage` or `CudaArray`
template <typename Traits>
struct DefaultTraitsWrapper : Traits
{
    using Elem = typename Traits::Elem;
    using CudaDataType = typename Traits::CudaDataType;
    using PinnedDataType = typename Traits::PinnedDataType;
    using SysDataType = typename Traits::SysDataType;
    using DiskDataType = typename Traits::DiskDataType;
    using AllTypes = std::tuple<CudaDataType, PinnedDataType, SysDataType, DiskDataType>;

    template <StorageLocation loc>
    struct StorageTypeSelector{
        using Type =
            std::conditional_t<loc == StorageLocation::kCudaDeviceMem, CudaDataType,
            std::conditional_t<loc == StorageLocation::kPinnedMem, PinnedDataType,
            std::conditional_t<loc == StorageLocation::kSysMem, SysDataType,
            std::conditional_t<loc == StorageLocation::kDisk, DiskDataType,
            void>>>>;
    };

    static_assert(StorageManager::storageHierarchy.size() == 4);
    static_assert(StorageManager::storageHierarchy[0] == StorageLocation::kCudaDeviceMem);
    static_assert(StorageManager::storageHierarchy[1] == StorageLocation::kPinnedMem);
    static_assert(StorageManager::storageHierarchy[2] == StorageLocation::kSysMem);
    static_assert(StorageManager::storageHierarchy[3] == StorageLocation::kDisk);
};

template <typename ElemType>
using DefaultLinearCacheableObjectTraits = DefaultTraitsWrapper<LinearCacheableObjectTraits<ElemType>>;
#if ENABLE_BLOCK_LINEAR_STORAGE
template <typename ElemType>
using DefaultCudaArrayCacheableObjectTraits = DefaultTraitsWrapper<CudaArrayCacheableObjectTraits<ElemType>>;
#endif

template <typename ElemType, bool isBlockLinear = false>
class DefaultCacheableObject
    : public CacheableObjectBase
    , public std::conditional_t<isBlockLinear, DefaultCudaArrayCacheableObjectTraits<ElemType>, DefaultLinearCacheableObjectTraits<ElemType>>
{
    static_assert(StorageManager::storageHierarchy.size() == 4);
    static_assert(!std::is_const_v<ElemType>);
public:
    using Elem = ElemType;
    using Traits = std::conditional_t<isBlockLinear, DefaultCudaArrayCacheableObjectTraits<ElemType>, DefaultLinearCacheableObjectTraits<ElemType>>;
    static_assert(std::is_same<typename Traits::Elem, Elem>::value);
    using CudaDataType = typename Traits::CudaDataType;
    using PinnedDataType = typename Traits::PinnedDataType;
    using SysDataType = typename Traits::SysDataType;
    using DiskDataType = typename Traits::DiskDataType;
    using HandleVariant = typename Traits::HandleVariant;
    template <StorageLocation loc>
    using StorageType = typename Traits::template StorageTypeSelector<loc>::Type;

    using DevMemPool = typename CudaDataType::CudaMemPoolType;
    using PinnedMemPool = typename PinnedDataType::CudaMemPoolType;
    using SysMemPool = typename SysDataType::CudaMemPoolType;

    // stream is for the file, as the file may be created by a callback in the stream asynchronously. nullopt means it's already available.
    // For cuda array, extBlkLinearArgs is:
    //    size_t width, size_t height, int arrayFlags = cudaArrayDefault
    // This ctor may be marked private
    template <typename... ExtBlockLinearArgs>
    DefaultCacheableObject(DevMemPool& devPool, PinnedMemPool& pinnedPool, SysMemPool& sysPool,
                           const fs::path& filePath, std_optional<cudaStream_t> stream,
                           DiskStoragePolicy diskStoragePolicy, bool evictPinnedToSys/* = false*/, ExtBlockLinearArgs... extBlkLinearArgs)
        : mDiskStoragePolicy {diskStoragePolicy}
        , mEvictPinnedToSys {evictPinnedToSys}
        , mData{CudaDataType{devPool, extBlkLinearArgs...}, PinnedDataType{pinnedPool}, SysDataType{sysPool},
                DiskDataType{filePath, fs::exists(filePath) ? fs::file_size(filePath) / sizeof(ElemType) : 0lu}}
    {
        if (!std::get<DiskDataType>(mData).empty()){
            mLocation = StorageLocation::kDisk;
            *mLocationInStream = mLocation;
        }
        cudaCheck(cudaEventRecord(getReadyEvent(), stream.has_value() ? *stream : mHelperStream.get()));
    }

    template <bool enabler = true, typename = std::enable_if_t<enabler && isBlockLinear, void>>
    DefaultCacheableObject(DevMemPool& devPool, uint32_t width, uint32_t height, unsigned flags/* = cudaArrayDefault*/,
                           PinnedMemPool& pinnedPool, SysMemPool& sysPool,
                           const fs::path& filePath, std_optional<cudaStream_t> stream,
                           DiskStoragePolicy diskStoragePolicy, bool evictPinnedToSys = false)
        : DefaultCacheableObject{devPool, pinnedPool, sysPool, filePath, stream, diskStoragePolicy, evictPinnedToSys, width, height, flags}
    {}
    template <bool enabler = true, typename = std::enable_if_t<enabler && !isBlockLinear, void>>
    DefaultCacheableObject(DevMemPool& devPool, PinnedMemPool& pinnedPool, SysMemPool& sysPool,
                           const fs::path& filePath, std_optional<cudaStream_t> stream,
                           DiskStoragePolicy diskStoragePolicy)
        : DefaultCacheableObject{devPool, pinnedPool, sysPool, filePath, stream, diskStoragePolicy, false}
    {}

    // All previously dispatched operation to mem content must be sync'ed with mem.get_deleter().stream
    // size is nbElems.
    // This ctor may be marked private
    template <CudaMemType memType, typename... ExtBlockLinearArgs>
    DefaultCacheableObject(DevMemPool& devPool, PinnedMemPool& pinnedPool, SysMemPool& sysPool, const fs::path& filePath,
                           PooledCudaMem<ElemType, memType>&& mem, size_t size,
                           DiskStoragePolicy diskStoragePolicy, bool evictPinnedToSys/* = false*/, ExtBlockLinearArgs... extBlkLinearArgs)
        : DefaultCacheableObject{devPool, pinnedPool, sysPool, filePath, mem.get_deleter().stream, diskStoragePolicy, evictPinnedToSys, extBlkLinearArgs...}
    {
        REQUIRE(mem.size() == size);
        if (diskStoragePolicy == DiskStoragePolicy::kNormal){
            REQUIRE(mLocation == StorageLocation::kUnknown && "You provided both memory data and disk data");
        }
        const std::tuple<DevMemPool*, PinnedMemPool*, SysMemPool*> memPools{&devPool, &pinnedPool, &sysPool};
        REQUIRE(std::get<CudaMemPool<memType>*>(memPools) == mem.get_deleter().pool);
        constexpr StorageLocation location = impl::toStorageLocation(memType);
        std::get<StorageType<location>>(mData) = StorageType<location>(std::move(mem), size);
        mLocation = location;
        const cudaStream_t stream = mem.get_deleter().stream;
        launchCudaHostFunc(stream, [inStreamLoc{mLocationInStream}](){*inStreamLoc = location;});
        cudaCheck(cudaEventRecord(getReadyEvent(), stream));
    }
    template <CudaMemType memType, typename = std::enable_if_t<(memType == CudaMemType::kPinned || memType == CudaMemType::kSystem) && isBlockLinear, void>>
    DefaultCacheableObject(DevMemPool& devPool, uint32_t width, uint32_t height, unsigned flags/* = cudaArrayDefault*/,
                           PinnedMemPool& pinnedPool, SysMemPool& sysPool, const fs::path& filePath, 
                           PooledCudaMem<ElemType, memType>&& mem, size_t size,
                           DiskStoragePolicy diskStoragePolicy, bool evictPinnedToSys = false)
        : DefaultCacheableObject{devPool, pinnedPool, sysPool, filePath, std::move(mem), size, diskStoragePolicy, evictPinnedToSys, width, height, flags}
    {}
    template <CudaMemType memType, typename = std::enable_if_t<memType != CudaMemType::kManaged && !isBlockLinear, void>>
    DefaultCacheableObject(DevMemPool& devPool, PinnedMemPool& pinnedPool, SysMemPool& sysPool, const fs::path& filePath,
                           PooledCudaMem<ElemType, memType>&& mem, size_t size,
                           DiskStoragePolicy diskStoragePolicy)
        : DefaultCacheableObject{devPool, pinnedPool, sysPool, filePath, std::move(mem), size, diskStoragePolicy, false}
    {}


    ~DefaultCacheableObject() override{
        for (auto& f : mOnDestroy) {
            f(*this, mLocation);
        }
        REQUIRE(getManager() == nullptr);
        if (getDiskStoragePolicy() != DiskStoragePolicy::kConstPersistent && !std::get<DiskDataType>(mData).empty()){
            auto& diskData = std::get<DiskDataType>(mData);
            const cudaStream_t stream = mHelperStream.get(); // used for disk file deletion. Any stream would do.
            cudaCheck(cudaStreamWaitEvent(stream, getReadyEvent()));
            getReaderFinishEvents()->streamWaitEvent(stream);
            migrateTo(StorageLocation::kDisk, stream);
            cudaCheck(cudaEventRecord(getReadyEvent(), stream));
            diskData.clear(stream);
        }
        // Is this needed? probably not. We have guarantee that even if the storage objects are destroyed, the previous dispatched async task can still use its resource.
//        getReaderFinishEvents()->sync();
    }

    void addOnMigrationCallback(std::function<void(const DefaultCacheableObject<ElemType, isBlockLinear>&, StorageLocation, StorageLocation)> callback) {
        mOnMigration.emplace_back(std::move(callback));
    }
    void addOnDestroyCallback(std::function<void(const DefaultCacheableObject<ElemType, isBlockLinear>&, StorageLocation)> callback) {
        mOnDestroy.emplace_back(std::move(callback));
    }

    // Only kNormal -> kImmutable -> kConstPersistent
    void updateDiskStoragePolicy(DiskStoragePolicy newPolicy) {
        ASSERT(newPolicy > getDiskStoragePolicy());
        mDiskStoragePolicy.store(newPolicy, std::memory_order_relaxed);
    }
    DiskStoragePolicy getDiskStoragePolicy() { return mDiskStoragePolicy.load(std::memory_order_relaxed); }

    void migrateTo(StorageLocation dst, cudaStream_t stream) override;
    StorageLocation getCurrentEvictionTarget() override {
        switch (mLocation)
        {
        case StorageLocation::kDisk: throw std::runtime_error("Cannot evict");
        case StorageLocation::kSysMem: return StorageLocation::kDisk;
        case StorageLocation::kPinnedMem: return mEvictPinnedToSys ? StorageLocation::kSysMem : StorageLocation::kDisk;
        case StorageLocation::kCudaDeviceMem: return StorageLocation::kPinnedMem;
        default: throw std::runtime_error("The current location is invalid");
        }
    }
    StorageLocation getCurrentStorageLocation() const override {return mLocation;}
    // in bytes
    size_t getCurrentStorageBytes(StorageLocation loc) const override{
        switch (loc)
        {
        case StorageLocation::kDisk: return sizeof(ElemType) * std::get<StorageType<StorageLocation::kDisk>>(mData).getSize();
        case StorageLocation::kSysMem: return sizeof(ElemType) * std::get<StorageType<StorageLocation::kSysMem>>(mData).getSize();
        case StorageLocation::kPinnedMem: return sizeof(ElemType) * std::get<StorageType<StorageLocation::kPinnedMem>>(mData).getSize();
        case StorageLocation::kCudaDeviceMem: return sizeof(ElemType) * std::get<StorageType<StorageLocation::kCudaDeviceMem>>(mData).getSize();
        default: throw std::runtime_error("invalid location");
        }
    }
    uint32_t getIdx() const {return mIdx;}
    HandleVariant getData() const {
        switch (mLocation) {
        case StorageLocation::kDisk: return std::get<StorageType<StorageLocation::kDisk>>(mData).data();
        case StorageLocation::kSysMem: return std::get<StorageType<StorageLocation::kSysMem>>(mData).data();
        case StorageLocation::kPinnedMem: return std::get<StorageType<StorageLocation::kPinnedMem>>(mData).data();
        case StorageLocation::kCudaDeviceMem: return std::get<StorageType<StorageLocation::kCudaDeviceMem>>(mData).data();
        default: throw std::runtime_error("fatal error");
        }
    }
    ElemType* getMemData() const {
        return std::get<ElemType*>(getData());
    }
    template <bool enabler = true>
    std::enable_if_t<enabler && isBlockLinear, cudaArray_t> getCudaArray() const {
        return std::get<cudaArray_t>(getData());
    }

    static constexpr auto storageHierarchy = StorageManager::storageHierarchy;
private:
    template <StorageLocation loc> struct LocWrapper{
        static constexpr StorageLocation location = loc;
    };
    virtual void migrateImpl(LocWrapper<StorageLocation::kDisk> src, LocWrapper<StorageLocation::kDisk> dst, cudaStream_t stream);
    virtual void migrateImpl(LocWrapper<StorageLocation::kDisk> src, LocWrapper<StorageLocation::kSysMem> dst, cudaStream_t stream);
    virtual void migrateImpl(LocWrapper<StorageLocation::kDisk> src, LocWrapper<StorageLocation::kPinnedMem> dst, cudaStream_t stream);
    virtual void migrateImpl(LocWrapper<StorageLocation::kDisk> src, LocWrapper<StorageLocation::kCudaDeviceMem> dst, cudaStream_t stream);
    virtual void migrateImpl(LocWrapper<StorageLocation::kSysMem> src, LocWrapper<StorageLocation::kDisk> dst, cudaStream_t stream);
    virtual void migrateImpl(LocWrapper<StorageLocation::kSysMem> src, LocWrapper<StorageLocation::kSysMem> dst, cudaStream_t stream);
    virtual void migrateImpl(LocWrapper<StorageLocation::kSysMem> src, LocWrapper<StorageLocation::kPinnedMem> dst, cudaStream_t stream);
    virtual void migrateImpl(LocWrapper<StorageLocation::kSysMem> src, LocWrapper<StorageLocation::kCudaDeviceMem> dst, cudaStream_t stream);
    virtual void migrateImpl(LocWrapper<StorageLocation::kPinnedMem> src, LocWrapper<StorageLocation::kDisk> dst, cudaStream_t stream);
    virtual void migrateImpl(LocWrapper<StorageLocation::kPinnedMem> src, LocWrapper<StorageLocation::kSysMem> dst, cudaStream_t stream);
    virtual void migrateImpl(LocWrapper<StorageLocation::kPinnedMem> src, LocWrapper<StorageLocation::kPinnedMem> dst, cudaStream_t stream);
    virtual void migrateImpl(LocWrapper<StorageLocation::kPinnedMem> src, LocWrapper<StorageLocation::kCudaDeviceMem> dst, cudaStream_t stream);
    virtual void migrateImpl(LocWrapper<StorageLocation::kCudaDeviceMem> src, LocWrapper<StorageLocation::kDisk> dst, cudaStream_t stream);
    virtual void migrateImpl(LocWrapper<StorageLocation::kCudaDeviceMem> src, LocWrapper<StorageLocation::kSysMem> dst, cudaStream_t stream);
    virtual void migrateImpl(LocWrapper<StorageLocation::kCudaDeviceMem> src, LocWrapper<StorageLocation::kPinnedMem> dst, cudaStream_t stream);
    virtual void migrateImpl(LocWrapper<StorageLocation::kCudaDeviceMem> src, LocWrapper<StorageLocation::kCudaDeviceMem> dst, cudaStream_t stream);

private:
    template <StorageLocation src, StorageLocation dst>
    void migrate(cudaStream_t stream){ // never override this
        REQUIRE(mLocation == src);
#ifndef NDEBUG
        launchCudaHostFunc(stream, [inStreamLoc{mLocationInStream}, this](){
            if (*inStreamLoc != src) {
                printf("bad obj at %p\n", this);
            }
            REQUIRE(*inStreamLoc == src);
        });
#endif
        migrateImpl(LocWrapper<src>{}, LocWrapper<dst>{}, stream);
        REQUIRE(mLocation == dst);
#ifndef NDEBUG
        launchCudaHostFunc(stream, [inStreamLoc{mLocationInStream}](){
            REQUIRE(*inStreamLoc == dst);
        });
#endif
    }
    // called by the virtual migrateImpl()
    template <StorageLocation src, StorageLocation dst>
    void migrateImpl(cudaStream_t stream){
        REQUIRE(mLocation == src);
        // mayOmit = true: omit if dst is not empty
        const bool mayOmit = (dst == StorageLocation::kDisk && getDiskStoragePolicy() != DiskStoragePolicy::kNormal) || src == dst;
        const bool keepSrc = (src == StorageLocation::kDisk && getDiskStoragePolicy() != DiskStoragePolicy::kNormal) || src == dst;
        migrateStorage<StorageType<src>, StorageType<dst>>(mayOmit, keepSrc, std::get<StorageType<src>>(mData), std::get<StorageType<dst>>(mData), stream);
        mLocation = dst;
        launchCudaHostFunc(stream, [inStreamLoc{mLocationInStream}](){*inStreamLoc = dst;});
    }

private:
    static WeakHelperSharedCudaStream sSharedHelperStream;
    std::shared_ptr<CUstream_st> mHelperStream = sSharedHelperStream.getStream(); // used when a temp stream is needed, e.g. for disk file deletion. Any stream would do.
private:
    std::atomic<DiskStoragePolicy> mDiskStoragePolicy {DiskStoragePolicy::kNormal};
    bool mEvictPinnedToSys {false};
    static inline std::atomic_uint32_t mIdxNext{0u};
    const uint32_t mIdx = mIdxNext.fetch_add(1u, std::memory_order_relaxed);
    StorageLocation mLocation = StorageLocation::kUnknown;
    std::tuple<CudaDataType, PinnedDataType, SysDataType, DiskDataType> mData;
    std::shared_ptr<StorageLocation> mLocationInStream = std::make_shared<StorageLocation>(StorageLocation::kUnknown); // updated and used by cuda stream, not on host.

    std::vector<std::function<void(const DefaultCacheableObject<ElemType, isBlockLinear>&, StorageLocation, StorageLocation)>> mOnMigration; // mOnMigration(src, dst)
    std::vector<std::function<void(const DefaultCacheableObject<ElemType, isBlockLinear>&, StorageLocation)>> mOnDestroy; // mOnDestroy(loc)
};

template <typename ElemType, bool isBlockLinear>
WeakHelperSharedCudaStream DefaultCacheableObject<ElemType, isBlockLinear>::sSharedHelperStream;

template <typename ElemType, bool isBlockLinear>
void DefaultCacheableObject<ElemType, isBlockLinear>::migrateTo(const StorageLocation dst, const cudaStream_t stream){
    for (auto& f : mOnMigration) {
        f(*this, mLocation, dst);
    }
    switch (getCurrentStorageLocation()) {
        case StorageLocation::kCudaDeviceMem: std::get<CudaDataType>(mData).notifyMigratedToStream(stream); break;
        case StorageLocation::kPinnedMem: std::get<PinnedDataType>(mData).notifyMigratedToStream(stream); break;
        case StorageLocation::kSysMem: std::get<SysDataType>(mData).notifyMigratedToStream(stream); break;
        case StorageLocation::kDisk:
        case StorageLocation::kUnknown: break;
    }
    using ThisType = DefaultCacheableObject<ElemType, isBlockLinear>;
    using Func = void (ThisType::*)(cudaStream_t);
#define MFUNC(src, dst) &ThisType::migrate<static_cast<StorageLocation>(src), static_cast<StorageLocation>(dst)>
    static const Func functions[4][4] = {
        {MFUNC(0, 0), MFUNC(0, 1), MFUNC(0, 2), MFUNC(0, 3)},
        {MFUNC(1, 0), MFUNC(1, 1), MFUNC(1, 2), MFUNC(1, 3)},
        {MFUNC(2, 0), MFUNC(2, 1), MFUNC(2, 2), MFUNC(2, 3)},
        {MFUNC(3, 0), MFUNC(3, 1), MFUNC(3, 2), MFUNC(3, 3)}
    };
#undef MFUNC
//    const auto tid = std::this_thread::get_id();
//    printf("host obj%u: %u%u in thread %lu\n", getIdx(), static_cast<uint32_t>(mLocation), static_cast<uint32_t>(dst), reinterpret_cast<const unsigned long&>(tid));
    (this->*functions[static_cast<uint32_t>(mLocation)][static_cast<uint32_t>(dst)])(stream);
    REQUIRE(mLocation == dst);
}

template <typename ElemType, bool isBlockLinear>
void DefaultCacheableObject<ElemType, isBlockLinear>::migrateImpl(LocWrapper<StorageLocation::kDisk> src, LocWrapper<StorageLocation::kDisk> dst, cudaStream_t stream){
    migrateImpl<decltype(src)::location, decltype(dst)::location>(stream);
}
template <typename ElemType, bool isBlockLinear>
void DefaultCacheableObject<ElemType, isBlockLinear>::migrateImpl(LocWrapper<StorageLocation::kDisk> src, LocWrapper<StorageLocation::kSysMem> dst, cudaStream_t stream){
    migrateImpl<decltype(src)::location, decltype(dst)::location>(stream);
}
template <typename ElemType, bool isBlockLinear>
void DefaultCacheableObject<ElemType, isBlockLinear>::migrateImpl(LocWrapper<StorageLocation::kDisk> src, LocWrapper<StorageLocation::kPinnedMem> dst, cudaStream_t stream){
    migrateImpl<decltype(src)::location, decltype(dst)::location>(stream);
}
template <typename ElemType, bool isBlockLinear>
void DefaultCacheableObject<ElemType, isBlockLinear>::migrateImpl(LocWrapper<StorageLocation::kDisk> src, LocWrapper<StorageLocation::kCudaDeviceMem> dst, cudaStream_t stream){
    const LocWrapper<StorageLocation::kPinnedMem> transit;
    migrateImpl(src, transit, stream);
    migrateImpl(transit, dst, stream);
}
template <typename ElemType, bool isBlockLinear>
void DefaultCacheableObject<ElemType, isBlockLinear>::migrateImpl(LocWrapper<StorageLocation::kSysMem> src, LocWrapper<StorageLocation::kDisk> dst, cudaStream_t stream){
    migrateImpl<decltype(src)::location, decltype(dst)::location>(stream);
}
template <typename ElemType, bool isBlockLinear>
void DefaultCacheableObject<ElemType, isBlockLinear>::migrateImpl(LocWrapper<StorageLocation::kSysMem> src, LocWrapper<StorageLocation::kSysMem> dst, cudaStream_t stream){
    migrateImpl<decltype(src)::location, decltype(dst)::location>(stream);
}
template <typename ElemType, bool isBlockLinear>
void DefaultCacheableObject<ElemType, isBlockLinear>::migrateImpl(LocWrapper<StorageLocation::kSysMem> src, LocWrapper<StorageLocation::kPinnedMem> dst, cudaStream_t stream){
    migrateImpl<decltype(src)::location, decltype(dst)::location>(stream);
}
template <typename ElemType, bool isBlockLinear>
void DefaultCacheableObject<ElemType, isBlockLinear>::migrateImpl(LocWrapper<StorageLocation::kSysMem> src, LocWrapper<StorageLocation::kCudaDeviceMem> dst, cudaStream_t stream){
    const LocWrapper<StorageLocation::kPinnedMem> transit;
    migrateImpl(src, transit, stream);
    migrateImpl(transit, dst, stream);
}
template <typename ElemType, bool isBlockLinear>
void DefaultCacheableObject<ElemType, isBlockLinear>::migrateImpl(LocWrapper<StorageLocation::kPinnedMem> src, LocWrapper<StorageLocation::kDisk> dst, cudaStream_t stream){
    migrateImpl<decltype(src)::location, decltype(dst)::location>(stream);
}
template <typename ElemType, bool isBlockLinear>
void DefaultCacheableObject<ElemType, isBlockLinear>::migrateImpl(LocWrapper<StorageLocation::kPinnedMem> src, LocWrapper<StorageLocation::kSysMem> dst, cudaStream_t stream){
    migrateImpl<decltype(src)::location, decltype(dst)::location>(stream);
}
template <typename ElemType, bool isBlockLinear>
void DefaultCacheableObject<ElemType, isBlockLinear>::migrateImpl(LocWrapper<StorageLocation::kPinnedMem> src, LocWrapper<StorageLocation::kPinnedMem> dst, cudaStream_t stream){
    migrateImpl<decltype(src)::location, decltype(dst)::location>(stream);
}
template <typename ElemType, bool isBlockLinear>
void DefaultCacheableObject<ElemType, isBlockLinear>::migrateImpl(LocWrapper<StorageLocation::kPinnedMem> src, LocWrapper<StorageLocation::kCudaDeviceMem> dst, cudaStream_t stream){
    migrateImpl<decltype(src)::location, decltype(dst)::location>(stream);
}
template <typename ElemType, bool isBlockLinear>
void DefaultCacheableObject<ElemType, isBlockLinear>::migrateImpl(LocWrapper<StorageLocation::kCudaDeviceMem> src, LocWrapper<StorageLocation::kDisk> dst, cudaStream_t stream){
    const LocWrapper<StorageLocation::kPinnedMem> transit;
    migrateImpl(src, transit, stream);
    migrateImpl(transit, dst, stream);
}
template <typename ElemType, bool isBlockLinear>
void DefaultCacheableObject<ElemType, isBlockLinear>::migrateImpl(LocWrapper<StorageLocation::kCudaDeviceMem> src, LocWrapper<StorageLocation::kSysMem> dst, cudaStream_t stream){
    const LocWrapper<StorageLocation::kPinnedMem> transit;
    migrateImpl(src, transit, stream);
    migrateImpl(transit, dst, stream);
}
template <typename ElemType, bool isBlockLinear>
void DefaultCacheableObject<ElemType, isBlockLinear>::migrateImpl(LocWrapper<StorageLocation::kCudaDeviceMem> src, LocWrapper<StorageLocation::kPinnedMem> dst, cudaStream_t stream){
    migrateImpl<decltype(src)::location, decltype(dst)::location>(stream);
}
template <typename ElemType, bool isBlockLinear>
void DefaultCacheableObject<ElemType, isBlockLinear>::migrateImpl(LocWrapper<StorageLocation::kCudaDeviceMem> src, LocWrapper<StorageLocation::kCudaDeviceMem> dst, cudaStream_t stream){
    migrateImpl<decltype(src)::location, decltype(dst)::location>(stream);
}

template <typename ElemType, bool isBlockLinear = false>
struct AcquiredMemory
{
    AcquiredObj holder; // RAII resource holder

    DefaultCacheableObject<std::remove_const_t<ElemType>, isBlockLinear>* obj() const {
        return dynamic_cast<DefaultCacheableObject<std::remove_const_t<ElemType>, isBlockLinear>*>(holder.get());
    }
    StorageLocation loc() const {
        return holder.get()->getCurrentStorageLocation();
    }
    ElemType* data() const {
        return obj()->getMemData();
    }
    template <bool enabler = true>
    std::enable_if_t<enabler && isBlockLinear, cudaArray_t> array() const {
        return obj()->getCudaArray();
    }
    size_t nbElems() const {
        return obj()->getCurrentStorageBytes(loc()) / sizeof(ElemType);
    }
    void reset() {
        holder.reset();
    }
};

// unique == true for read and write; false for read-only.
// block means wait on mutex contention.
// !Important: stream must live longer than the cache object. Otherwise we get segfault when destroying the cached object as its resource is still associated with the stream.
template <typename ElemType, bool isBlockLinear = false>
AcquiredMemory<ElemType, isBlockLinear> acquireMemory(StorageManager& manager, const CacheObjKeyType key, const StorageLocation loc, const cudaStream_t stream, const bool unique, const bool block = true)
{
    REQUIRE((loc == StorageLocation::kSysMem) || (loc == StorageLocation::kPinnedMem) || (loc == StorageLocation::kCudaDeviceMem));
    REQUIRE(unique || std::is_const_v<ElemType>);
    AcquiredObj holder = manager.acquire(unique, block, key, loc, stream);
#ifndef NDEBUG
    const auto obj = dynamic_cast<DefaultCacheableObject<std::remove_const_t<ElemType>, isBlockLinear>*>(holder.get());
    const auto onExit = makeScopeGuard([loc, obj]{REQUIRE(obj->getCurrentStorageLocation() == loc);});
#endif
    return {std::move(holder)};
}

} //namespace cudapp::storage
