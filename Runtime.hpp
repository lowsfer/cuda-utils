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
#include "Runtime.h"
#include "CudaMemPool.h"
#include "DefaultCacheableObject.h"
#include "CacheableScratchObject.h"
#include "StorageManager.h"
#include <cuda_fp16.h>

namespace cudapp
{
template <typename T>
constexpr bool allowNonTrivial() {
    return std::is_same_v<T, half> || std::is_same_v<T, half2>;
}

template <typename T>
Runtime::CacheObjKeyType Runtime::registerCacheableData(const std::vector<T>& src, const std::string& filename,
    DiskStoragePolicy diskStoragePolicy, bool evictPinnedToSys)
{
    const cudaStream_t stream = anyStream();
    auto mem = cudaMemPool<CudaMemType::kPinned>().alloc<T>(src.size(), stream);
    cudaCheck(cudaStreamSynchronize(stream));
    assert(src.size() == mem.size());
    std::copy(src.begin(), src.end(), mem.get());
    return registerCacheableData(std::move(mem), filename, diskStoragePolicy, evictPinnedToSys);
}

template <CudaMemType dstType, typename SrcGetter> // srcGetter is moved from
Runtime::CacheObjKeyType Runtime::registerCacheableData(SrcGetter srcGetter, size_t srcSize, const std::string& filename,
    DiskStoragePolicy diskStoragePolicy, bool evictPinnedToSys)
{
    static_assert(dstType == CudaMemType::kPinned || dstType == CudaMemType::kSystem);
    using T = std::invoke_result_t<SrcGetter>;
    const cudaStream_t stream = anyStream();
    auto mem = cudaMemPool<dstType>().template alloc<T, allowNonTrivial<T>()>(srcSize, stream);
    assert(srcSize == mem.size());
    launchCudaHostFunc(stream, [srcSize, getter{std::move(srcGetter)}, dst{mem.get()}](){
        std::copy_n(getter(), srcSize, dst);
    });
    return registerCacheableData(std::move(mem), filename, diskStoragePolicy, evictPinnedToSys);
}

template <typename T, CudaMemType memType>
Runtime::CacheObjKeyType Runtime::registerCacheableData(CudaMem<T, memType>&& src, size_t nbElems, const std::string& filename,
    DiskStoragePolicy diskStoragePolicy, bool evictPinnedToSys)
{
    const cudaStream_t stream = anyStream();
    auto mem = cudaMemPool<memType>().registerExternalMem(std::move(src), nbElems, stream);
    assert(nbElems == src.size());
    return registerCacheableData(std::move(mem), filename, diskStoragePolicy, evictPinnedToSys);
}

template <typename T, CudaMemType memType> // src is moved from
Runtime::CacheObjKeyType Runtime::registerCacheableData(cudapp::storage::PooledCudaMem<T, memType>&& src, const std::string& filename,
    DiskStoragePolicy diskStoragePolicy, bool evictPinnedToSys)
{
    const size_t nbElems = src.size();
    auto obj = std::make_unique<cudapp::storage::DefaultCacheableObject<T>>(
                cudaMemPool<CudaMemType::kDevice>(), cudaMemPool<CudaMemType::kPinned>(), cudaMemPool<CudaMemType::kSystem>(),
                cachePath() / filename, std::move(src), nbElems,
                diskStoragePolicy, evictPinnedToSys);
    const auto key = storageManager().addItem(std::move(obj));
    return key;
}

template <typename T, CudaMemType memType> // src is moved from
std::enable_if_t<memType == CudaMemType::kPinned || memType == CudaMemType::kSystem, Runtime::CacheObjKeyType>
    Runtime::registerCacheableDataForCudaArray(cudapp::storage::PooledCudaMem<T, memType>&& src,
                                        uint32_t width, uint32_t height, uint32_t flags,
                                        const std::string& filename,
                                        DiskStoragePolicy diskStoragePolicy, bool evictPinnedToSys)
{
    using namespace cudapp::storage;
    const size_t size = width * height;
    ASSERT(src.size() == size);
    auto obj = std::make_unique<DefaultCacheableObject<T, true>>(
        cudaMemPool<CudaMemType::kDevice>(), width, height, flags,
        cudaMemPool<CudaMemType::kPinned>(), cudaMemPool<CudaMemType::kSystem>(),
        cachePath() / filename, std::move(src), size, diskStoragePolicy, evictPinnedToSys);
    const auto key = storageManager().addItem(std::move(obj));
    return key;
}

template <CudaMemType dstType, typename SrcGetter> // srcGetter is moved from
cudapp::storage::CacheObjKeyType Runtime::registerCacheableDataForCudaArray(SrcGetter srcGetter,
    uint32_t width, uint32_t height, uint32_t flags, const std::string& filename,
    DiskStoragePolicy diskStoragePolicy, bool evictPinnedToSys)
{
    static_assert(dstType == CudaMemType::kPinned || dstType == CudaMemType::kSystem);
    using Src = std::invoke_result_t<SrcGetter>;
    static_assert(std::is_pointer_v<Src>);
    using T = std::decay_t<std::remove_pointer_t<Src>>;
    const cudaStream_t stream = anyStream();
    const size_t srcSize = width * height;
    auto mem = cudaMemPool<dstType>().template alloc<T, allowNonTrivial<T>()>(srcSize, stream);
    assert(srcSize == mem.size());
    launchCudaHostFunc(stream, [srcSize, getter{std::move(srcGetter)}, dst{mem.get()}](){
        std::copy_n(getter(), srcSize, dst);
    });
    return registerCacheableDataForCudaArray<T, dstType>(std::move(mem), width, height, flags, filename, diskStoragePolicy, evictPinnedToSys);
}

template <typename T, CudaMemType memType>
Runtime::CacheObjKeyType Runtime::allocCacheableData(size_t nbElems, const std::string& filename,
    DiskStoragePolicy diskStoragePolicy, bool evictPinnedToSys, std::optional<cudaStream_t> optStream)
{
    const cudaStream_t stream = optStream.has_value() ? optStream.value() : anyStream();
    return registerCacheableData(cudaMemPool<memType>().template alloc<T, allowNonTrivial<T>()>(nbElems, stream), filename, diskStoragePolicy, evictPinnedToSys);
}

template <typename T, CudaMemType memType>
Runtime::CacheObjKeyType Runtime::allocCacheableDataForCudaArray(uint32_t width, uint32_t height, uint32_t flags,
    const std::string& filename,
    DiskStoragePolicy diskStoragePolicy, bool evictPinnedToSys,
    std::optional<cudaStream_t> optStream)
{
    const cudaStream_t stream = optStream.has_value() ? optStream.value() : anyStream();
    const size_t nbElems = width * height;
    return registerCacheableDataForCudaArray(cudaMemPool<memType>().template alloc<T, allowNonTrivial<T>()>(nbElems, stream), width, height, flags, filename, diskStoragePolicy, evictPinnedToSys);
}

template <typename T, CudaMemType memType>
Runtime::CacheObjKeyType Runtime::allocCacheableScratch(size_t nbElems)
{
    auto obj = std::make_unique<cudapp::storage::CacheableScratchObject<T, memType>>(
                cudaMemPool<memType>(), nbElems);
    const auto key = storageManager().addItem(std::move(obj));
    return key;
}

template<typename T>
Runtime::CacheObjKeyType Runtime::allocCacheableArrayScratch(uint32_t width, uint32_t height, uint32_t flags)
{
    auto obj = std::make_unique<cudapp::storage::CacheableScratchObject<T, CudaMemType::kDevice, true>>(cudaMemPool<CudaMemType::kDevice>(), width, height, flags);
    const auto key = storageManager().addItem(std::move(obj));
    return key;
}

template <typename T, bool isBlockLinear>
void Runtime::updateDiskStoragePolicy(CacheObjKeyType key, DiskStoragePolicy newPolicy)
{
    dynamic_cast<cudapp::storage::DefaultCacheableObject<T, isBlockLinear>*>(storageManager().getObj(key))->updateDiskStoragePolicy(newPolicy);
}

} // namespace cudapp
