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
#include "IRuntime.h"
#include "cuda_utils.h"
#include "StorageFwd.h"
#include "ArbitraryPool.h"
#include "LRUCache.h"
#include "CudaTexture.h"
#include "CudaEventPool.h"

namespace cudapp
{

class Runtime : public IRuntime
{
public:
    Runtime(const fs::path& cacheFolder, size_t nbRandStream,
        size_t devPoolMaxBytes, size_t pinnedPoolMaxBytes, size_t sysPoolMaxBytes,
        size_t deviceSoftLimit, size_t pinnedSoftLimit, size_t sysSoftLimit);
    ~Runtime() override;
    cudaStream_t anyStream() const {return mRandStreams.get().get();}
    template <CudaMemType memType>
    cudapp::storage::CudaMemPool<memType>& cudaMemPool() {return *std::get<std::unique_ptr<cudapp::storage::CudaMemPool<memType>>>(mCudaMemPools);}
    cudapp::storage::StorageManager& storageManager() {return *mStorageManager;}

    using DiskStoragePolicy = cudapp::storage::DiskStoragePolicy;
    using CacheObjKeyType = cudapp::storage::CacheObjKeyType;
    static constexpr CacheObjKeyType kInvalidKey = cudapp::storage::kInvalidKey;
    // before registerCacheableData, make sure it is synchronized.
    template<typename T> // data will be copied
    /*[[deprecated]]*/ CacheObjKeyType registerCacheableData(const std::vector<T>& src, const std::string& filename,
                                          DiskStoragePolicy diskStoragePolicy, bool evictPinnedToSys = false);
    
    // srcGetter.operator()() should return T*
    template <CudaMemType dstType, typename SrcGetter> // srcGetter is moved from
    CacheObjKeyType registerCacheableData(SrcGetter srcGetter, size_t srcSize, const std::string& filename,
        DiskStoragePolicy diskStoragePolicy, bool evictPinnedToSys = false);

    template<typename T, CudaMemType memType> // src is moved from
    CacheObjKeyType registerCacheableData(CudaMem<T, memType>&& src, size_t nbElems, const std::string& filename,
        DiskStoragePolicy diskStoragePolicy, bool evictPinnedToSys = false);
    template<typename T, CudaMemType memType> // src is moved from
    CacheObjKeyType registerCacheableData(cudapp::storage::PooledCudaMem<T, memType>&& src, const std::string& filename,
        DiskStoragePolicy diskStoragePolicy, bool evictPinnedToSys = false);
    
    // For cudaArray_t
    template<typename T, CudaMemType memType> // src is moved from
    std::enable_if_t<memType == CudaMemType::kPinned || memType == CudaMemType::kSystem, CacheObjKeyType>
        registerCacheableDataForCudaArray(cudapp::storage::PooledCudaMem<T, memType>&& src,
                                          uint32_t width, uint32_t height, uint32_t flags, const std::string& filename,
                                          DiskStoragePolicy diskStoragePolicy, bool evictPinnedToSys = false);
    // srcGetter.operator()() should return T*
    template <CudaMemType dstType, typename SrcGetter> // srcGetter is moved from
    CacheObjKeyType registerCacheableDataForCudaArray(SrcGetter srcGetter,
        uint32_t width, uint32_t height, uint32_t flags, const std::string& filename,
        DiskStoragePolicy diskStoragePolicy, bool evictPinnedToSys = false);

    // Better use DiskStoragePolicy::kNormal and then update to kImmutable or kConstPersistent later.
    // Otherwise if it's evicted to disk before setting up the data, the uninitialized data will persist on disk.
    template <typename T, CudaMemType memType>
    CacheObjKeyType allocCacheableData(size_t nbElems, const std::string& filename,
        DiskStoragePolicy diskStoragePolicy, bool evictPinnedToSys = false,
        std::optional<cudaStream_t> optStream = std::nullopt);

    // Better use DiskStoragePolicy::kNormal and then update to kImmutable or kConstPersistent later.
    // Otherwise if it's evicted to disk before setting up the data, the uninitialized data will persist on disk.
    template <typename T, CudaMemType memType>
    CacheObjKeyType allocCacheableDataForCudaArray(uint32_t width, uint32_t height, uint32_t flags,
        const std::string& filename,
        DiskStoragePolicy diskStoragePolicy, bool evictPinnedToSys = false,
        std::optional<cudaStream_t> optStream = std::nullopt);
    
    template <typename T, bool isBlockLinear = false>
    void updateDiskStoragePolicy(CacheObjKeyType key, DiskStoragePolicy newPolicy);
    
    template<typename T, CudaMemType memType>
    CacheObjKeyType allocCacheableScratch(size_t nbElems);

    template<typename T>
    CacheObjKeyType allocCacheableArrayScratch(uint32_t width, uint32_t height, uint32_t flags);

    const fs::path& cachePath() const {return mCachePath;}

    // returned texture object is owned by Runtime and automatically destroyed when the cuda array is invalidated.
    // The cuda array must be managed by storageManager().
    cudaTextureObject_t createTexObjCached(cudaArray_t array, const cudaTextureDesc& texDesc);
private:
    fs::path mCachePath;
    cudapp::ArbitraryPool<CudaStream> mRandStreams;

    // holder to avoid duplicate create destroy of the pool
    // to create a pooled event, you don't need this. Just use cudapp::createPooledCudaEvent();
    std::shared_ptr<cudapp::CudaEventPool> mEventPool;

    // holder to avoid repeated creation/destroy of the daemon
    std::shared_ptr<cudapp::ICudaDaemon> mCudaDaemon;
    
    void invalidateTexObjCacheEntry(cudaArray_t array);
    mutable std::mutex mTexObjCacheLock;
    std::unordered_map<cudaArray_t, std::vector<std::pair<cudaTextureDesc, cudapp::TexObj>>> mTexObjCache;

    std::tuple< std::unique_ptr<cudapp::storage::CudaMemPool<CudaMemType::kDevice>>,
                std::unique_ptr<cudapp::storage::CudaMemPool<CudaMemType::kPinned>>,
                std::unique_ptr<cudapp::storage::CudaMemPool<CudaMemType::kSystem>>> mCudaMemPools;
    std::unique_ptr<cudapp::storage::StorageManager> mStorageManager;
}; // class Runtime
} // namespace cudapp
