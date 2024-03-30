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
#include <limits>
#include <cstdint>
#include "cuda_utils.h"

namespace cudapp {
class IRuntime;
class Runtime;
namespace storage {

template <CudaMemType memType>
class CudaMemPool;

template <typename Elem, CudaMemType memType>
class PooledCudaMem;

enum class StorageLocation : std::uint32_t
{
    kCudaDeviceMem,
    kPinnedMem,
    kSysMem,
    kDisk,
    kUnknown = std::numeric_limits<std::uint32_t>::max()
};

constexpr inline const char* toStr(StorageLocation loc)
{
    switch (loc) {
    case StorageLocation::kCudaDeviceMem: return "kCudaDeviceMem";
    case StorageLocation::kPinnedMem: return "kPinnedMem";
    case StorageLocation::kSysMem: return "kSysMem";
    case StorageLocation::kDisk: return "kDisk";
    case StorageLocation::kUnknown: return "kUnknown";
    }
    return nullptr;
}

namespace impl{
constexpr inline StorageLocation toStorageLocation(CudaMemType memType)
{
    switch (memType){
    case CudaMemType::kDevice: return StorageLocation::kCudaDeviceMem;
    case CudaMemType::kPinned: return StorageLocation::kPinnedMem;
    case CudaMemType::kSystem: return StorageLocation::kSysMem;
    default: throw std::runtime_error("invalid memory type");
    }
}
}

class StorageManager;
class AcquiredObj;

using CacheObjKeyType = const void*;
constexpr CacheObjKeyType kInvalidKey = nullptr;

enum class DiskStoragePolicy : uint8_t
{
    kNormal,            // Removed on migration to another level or CacheableObject destruction
    kImmutable,         // Kept on migration but removed on CacheableObject destruction
    kConstPersistent    // Kept on migration and CacheableObject destruction
};

} // namespace storage


template <typename State, typename Mutex, typename ConditionVariable>
class Semaphore;

template <typename State, typename Mutex, typename ConditionVariable>
class ConcurrentQueue;

class FiberPool;
class FiberBlockingService;

template <typename Input, typename Product>
class IPipeLine;

} // namespace cudapp

