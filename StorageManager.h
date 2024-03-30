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
#include <cstddef>
#include <stdexcept>
#include <mutex>
#include <shared_mutex>
#include <unordered_set>
#include <unordered_map>
#include <set>
#include <map>
#include <atomic>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_utils.h"
#include <list>
#include <array>
#include <algorithm>
#include "cpp_utils.h"
#include "StorageFwd.h"
#include "CudaEventPool.h"

// fixed in since r445
#define NVBUG_200569951_WAR 0

namespace cudapp
{
namespace storage
{

class ICacheableObject
{
public:
    enum class State
    {
        kIdle, // not acquired
        kUnique, // acquired exclusively. Usually allows read+write access.
        kShared, // acquired as shared. Usually allows read access
    };

    virtual ~ICacheableObject() = default;
//private:
    // Called by the manager to notify user about ownership change.
    // returns:
    //     1. event for availability in acquired stream, i.e. finish of migration, AND finished read+write usage
    //     2. events for finished read-only usage (release)
    virtual std::pair<cudaEvent_t, ICudaMultiEvent*> notifyTaken(StorageManager& manager) = 0;// manager is taking control
    virtual void notifyReturned() = 0; // manager releases control
    // Called by the manager to notify state change, so user can implemente sanity check when getting the resource inside
    //! \return the old state
    virtual State notifyStateChange(State state) = 0;

    // These functions below are called by StorageManager. Do not directly call them when the object is owned/managed by a StorageManager.

    // The transfer is async in the provided stream. StorageManager should have let stream wait on finished events before migrateTo() and record available event after migrateTo()
    virtual void migrateTo(StorageLocation dst, cudaStream_t stream) = 0;
    // Get the eviction target from the current location in case eviction is needed.
    virtual StorageLocation getCurrentEvictionTarget() = 0;

    // Get asynchronous location, i.e. the destination of the last dispatched migration.
    virtual StorageLocation getCurrentStorageLocation() const = 0;
    // The object may take space in multiple levels of cache. This is used to query for each level.
    // It should also be async size, i.e. the state when the dispatched migrations are complete.
    virtual size_t getCurrentStorageBytes(StorageLocation loc) const = 0;

    template <size_t nbQueries>
    std::array<size_t, nbQueries> getCurrentStorageBytes(const std::array<StorageLocation, nbQueries>& locations){
        std::array<size_t, nbQueries> result;
        std::transform(locations.begin(), locations.end(), result.begin(), [this](StorageLocation loc){return getCurrentStorageBytes(loc);});
        return result;
    }

    friend StorageManager;
    friend AcquiredObj;
};

class CacheableObjectBase : public ICacheableObject
{
public:
    std::pair<cudaEvent_t, ICudaMultiEvent*> notifyTaken(StorageManager &manager) override{
        if (mManager != nullptr){
            throw std::runtime_error("already occupied");
        }
        mManager = &manager;
        return {mAvailEvent.get(), mFinishReaders.get()};
    }
    void notifyReturned() override{
        mManager = nullptr;
    }
    State notifyStateChange(State state) override {
        std::swap(mState, state);
        return state;
    }
protected:
    cudaEvent_t getReadyEvent() const {return mAvailEvent.get();}
    ICudaMultiEvent* getReaderFinishEvents() const {return mFinishReaders.get();}
    StorageManager* getManager() const {return mManager;}
private:
    State mState = State::kIdle;
    StorageManager* mManager = nullptr;
    cudapp::PooledCudaEvent mAvailEvent = createPooledCudaEvent();
    std::unique_ptr<ICudaMultiEvent> mFinishReaders = createCudaMultiEvent(true);
};


class StorageManagerException : public std::runtime_error{
public:
    template <typename... Args>
    StorageManagerException(Args... args) : std::runtime_error{std::forward<Args>(args)...}{}
    ~StorageManagerException() override;
};

#define DEFINE_StorageManagerException(NAME, WHAT) \
    class NAME : public StorageManagerException\
    {\
    public:\
        NAME() : StorageManagerException{WHAT} {}\
        ~NAME() override;\
    }
DEFINE_StorageManagerException(DuplicateObject, "duplicate object");
DEFINE_StorageManagerException(InUseRemove, "trying to remove an in-use object");
DEFINE_StorageManagerException(KeyNotFound, "the key is not found in the StorageManager");
DEFINE_StorageManagerException(InvalidMigration, "such migration is not implemented");
#undef DEFINE_StorageManagerException

struct StorageManagerTraits
{
    using KeyType = CacheObjKeyType;
    static constexpr KeyType kInvalidKey = cudapp::storage::kInvalidKey;
    // In eviction order, i.e. from fastest to slowest
    static constexpr std::array<StorageLocation, 4> storageHierarchy = {
        StorageLocation::kCudaDeviceMem,
        StorageLocation::kPinnedMem,
        StorageLocation::kSysMem,
        StorageLocation::kDisk
    };
};

class AcquiredObj : public StorageManagerTraits
{
public:
    AcquiredObj() = default;

    AcquiredObj(StorageManager* manager, KeyType key, ICacheableObject* obj,
                StorageLocation location, cudaStream_t stream, cudaEvent_t eventAvailable, bool isUnique)
        : mManager{manager}, mKey{key}, mObj{obj}, mLocation{location}, mStream{stream}
        , mEventAvailable{eventAvailable}, mIsUnique{isUnique}
    {}

    AcquiredObj(const AcquiredObj&) = delete;
    AcquiredObj& operator=(const AcquiredObj&) = delete;

    void swap(AcquiredObj& other){
        std::swap(mManager, other.mManager);
        std::swap(mKey, other.mKey);
        std::swap(mObj, other.mObj);
        std::swap(mLocation, other.mLocation);
        std::swap(mStream, other.mStream);
        std::swap(mEventAvailable, other.mEventAvailable);
    }
    
    AcquiredObj(AcquiredObj&& src) : AcquiredObj() {swap(src);}
    AcquiredObj& operator=(AcquiredObj&& src) { reset(); swap(src); return *this; }
    ~AcquiredObj(){ reset(); }

    bool isValid() const {return mLocation != StorageLocation::kUnknown;}
    bool isUnique() const {return mIsUnique;}

    ICacheableObject* get() const {return mObj;}

    void sync() { cudaCheck(cudaEventSynchronize(mEventAvailable.value())); }

    void reset();
private:
    StorageManager* mManager = nullptr;
    KeyType mKey = kInvalidKey;
    ICacheableObject* mObj = nullptr;
    StorageLocation mLocation = StorageLocation::kUnknown;
    std_optional<cudaStream_t> mStream = std_nullopt; // Streams in which it is in use.
    std_optional<cudaEvent_t> mEventAvailable = std_nullopt; // Event for availability in the original stream.
    bool mIsUnique = false;
};

// For locking, we follow the order: manager -> item -> residency to avoid dead lock.
class StorageManager : public StorageManagerTraits
{
public:
    StorageManager() = default;
    StorageManager(size_t deviceSoftLimit, size_t pinnedSoftLimit, size_t sysSoftLimit);
    ~StorageManager();
    void setLimits(StorageLocation location, size_t soft, size_t hard = std::numeric_limits<size_t>::max());

    KeyType addItem(std::unique_ptr<ICacheableObject>&& obj);
    bool hasItem(KeyType key) const;

    std::unique_ptr<ICacheableObject> removeItem(KeyType key);

    ICacheableObject* getObj(KeyType key) const;

    StorageLocation peekStorageLocation(KeyType key) const;

    size_t peekStorageUsage(StorageLocation location) const;

    // The API is asynchronous. If the data is currently acquired in different location, a (default-constructed) invalid result is returned.
    //! \param blocking: if this item is acquired in another storage level and cannot migrate, we may block until it is released, or return immediately.
    // Note if blocking == true, the internal condition variable requires that item->mLock is locked only once. So any function calling this
    // API shall not lock a item outside of this API.
    AcquiredObj acquire(bool unique, bool blocking, KeyType key, StorageLocation loc, cudaStream_t stream);
    
    // Asynchronously release the resource. The stream must be same as the one used for acquire
    void release(KeyType key, cudaStream_t stream);

    void fitCache();

protected:
#if NVBUG_200569951_WAR
    mutable std::mutex mBugWarLockContainerLock;
    mutable std::unordered_map<cudaStream_t, std::unique_ptr<std::recursive_mutex>> mBugWarLocks; // @fixme: remove when Nvidia 445 driver is available.
    std::recursive_mutex& bugWarLock(cudaStream_t stream) const {
        std::lock_guard<std::mutex> lk{mBugWarLockContainerLock};
        if (mBugWarLocks.count(stream) == 0) {
            mBugWarLocks.try_emplace(stream, std::make_unique<std::recursive_mutex>());
        }
        return *mBugWarLocks.at(stream).get();
    }
#endif
    mutable std::shared_mutex mLock;
    struct Residency
    {
        mutable std::recursive_mutex lock; // unique_ptr to make it movable
        std::list<KeyType> acquired; // acquired and resident in this level
        std::list<KeyType> cached; // resident but not acquired.
        std::atomic<size_t> storageSize {0}; // Async size, i.e. size when all dispatched async tasks are finished
        std::atomic<size_t> softLimit{0}; // By default, cache is always cleaned immediately.
        std::atomic<size_t> hardLimit{std::numeric_limits<size_t>::max()}; // By default, hard limit is never reached.
    };

    // This map itself does not need protection, as we never add or remove elements, though the values do need protection (with their own mutex).
    const std::unordered_map<StorageLocation, std::unique_ptr<Residency>> mResidency = []() {
        std::unordered_map<StorageLocation, std::unique_ptr<Residency>> result;
        for (auto loc : storageHierarchy){
            result.emplace(loc, std::make_unique<Residency>());
        }
        return result;
    }();

    class Item;
    std::unordered_map<KeyType, std::shared_ptr<Item>> mItems;
    CudaStream mStream = makeCudaStream();
};

} // namespace storage
} // namespace cudapp
