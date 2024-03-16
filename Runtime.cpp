#include "Runtime.h"
#include "cuda_utils.h"
#include "CudaMemPool.h"
#include "StorageManager.h"
#include <cstring>

inline bool operator==(const cudaTextureDesc& a, const cudaTextureDesc& b) {
    return std::memcmp(&a, &b, sizeof(cudaTextureDesc)) == 0;
}

namespace cudapp
{
Runtime::Runtime(const fs::path& cacheFolder, size_t nbRandStream,
    size_t devPoolMaxBytes, size_t pinnedPoolMaxBytes, size_t sysPoolMaxBytes,
    size_t deviceSoftLimit, size_t pinnedSoftLimit, size_t sysSoftLimit)
: mCachePath{cacheFolder}
, mRandStreams{nbRandStream, makeCudaStream, cudaStreamNonBlocking}
, mEventPool{cudapp::getCudaEventPool()}
, mCudaDaemon{cudapp::getCudaDaemon()}
, mCudaMemPools{
    std::make_unique<cudapp::storage::CudaMemPool<CudaMemType::kDevice>>(devPoolMaxBytes),
    std::make_unique<cudapp::storage::CudaMemPool<CudaMemType::kPinned>>(pinnedPoolMaxBytes),
    std::make_unique<cudapp::storage::CudaMemPool<CudaMemType::kSystem>>(sysPoolMaxBytes)}
, mStorageManager{std::make_unique<cudapp::storage::StorageManager>(deviceSoftLimit, pinnedSoftLimit, sysSoftLimit)}
{
    cudaMemPool<CudaMemType::kDevice>().setOnArrayFreeCallback([this](cudaArray_t array){
        invalidateTexObjCacheEntry(array);
    });
    if (fs::exists(cacheFolder) && !fs::is_directory(cacheFolder)) {
        fs::remove_all(cacheFolder);
    }
    if (!fs::exists(cacheFolder)) {
        fs::create_directory(cacheFolder);
    }
    for (auto& f : fs::directory_iterator(cacheFolder)) {
        fs::remove_all(f);
    }
    ASSERT(fs::is_empty(cacheFolder));
}

cudaTextureObject_t Runtime::createTexObjCached(cudaArray_t array, const cudaTextureDesc& texDesc) {
    std::lock_guard lk{mTexObjCacheLock};
    const auto iterArr = mTexObjCache.find(array);
    if (iterArr != mTexObjCache.end()) {
        const auto iterVec = std::find_if(iterArr->second.begin(), iterArr->second.end(), [&texDesc](const std::pair<cudaTextureDesc, cudapp::TexObj>& entry){
            return entry.first == texDesc;
        });
        if (iterVec != iterArr->second.end()) {
            return iterVec->second.get();
        }
    }
    return mTexObjCache[array].emplace_back(texDesc, cudapp::createTexObj(array, texDesc)).second.get();
}

void Runtime::invalidateTexObjCacheEntry(cudaArray_t array) {
    std::lock_guard lk{mTexObjCacheLock};
    mTexObjCache.erase(array);
}

Runtime::~Runtime() = default;

IRuntime::~IRuntime() = default;

extern "C" IRuntime* createRuntimeCudappImpl(const char* cacheFolder, size_t nbRandStream,
    size_t devPoolMaxBytes, size_t pinnedPoolMaxBytes, size_t sysPoolMaxBytes,
    size_t deviceSoftLimit, size_t pinnedSoftLimit, size_t sysSoftLimit)
{
    return new Runtime(cacheFolder, nbRandStream, devPoolMaxBytes,
        pinnedPoolMaxBytes, sysPoolMaxBytes,
        deviceSoftLimit, pinnedSoftLimit, sysSoftLimit);
}
} // namespace cudapp
