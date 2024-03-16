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

