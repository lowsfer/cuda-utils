#include "StorageManager.h"
#include <mutex>
#include "CudaEventPool.h"
#include "Profiler.h"
#include "macros.h"
namespace
{
template <typename Container>
bool contains(const Container& container, const typename Container::value_type& val){
    return std::find(container.begin(), container.end(), val) != container.end();
}

template <typename T>
bool containsIter(const std::list<T>& container, const typename std::list<T>::iterator& p){
    for (auto iter = container.begin(); iter != container.end(); iter = std::next(iter)){
        if (p == iter){
            return true;
        }
    }
    return false;
}
}

namespace cudapp
{
namespace storage
{
StorageManagerException::~StorageManagerException() = default;
DuplicateObject::~DuplicateObject() = default;
InUseRemove::~InUseRemove() = default;
KeyNotFound::~KeyNotFound() = default;

void AcquiredObj::reset(){
    if (isValid()){
        mManager->release(mKey, mStream.value());
        mManager = nullptr;
        mKey = kInvalidKey;
        mObj = nullptr;
        mLocation = StorageLocation::kUnknown;
        mStream = std_nullopt;
        mEventAvailable = std_nullopt;
    }
    assert(!isValid());
}

class StorageManager::Item
{
public:
    Item(StorageManager& manager_,
         KeyType key, std::unique_ptr<ICacheableObject>&& instance_,
         StorageLocation initLoc,
         cudaEvent_t availEvent,
         ICudaMultiEvent* finishReadEvents) // Only after sync these events, the resource is available in initLoc.
        : mObject{std::move(instance_)}
        , mLocation{initLoc}
        , mEventAvailable{availEvent}
        , mFinishedReaders{finishReadEvents}
        , mManager{manager_}
        , mResidencyIter{
            [&](){
                auto& residents = *mManager.mResidency.at(initLoc);
                std::lock_guard<std::recursive_mutex> residentsLock{residents.lock};
                residents.cached.emplace_back(key);
                return std::prev(residents.cached.end());
            }()}
    {
        for (auto loc : storageHierarchy){
            auto& residents = *mManager.mResidency.at(loc);
            residents.storageSize.fetch_add(mObject->getCurrentStorageBytes(loc));
        }
    }
    ~Item() {
        std::lock_guard<std::recursive_mutex> itemLock{mLock};
        if (isValid()){
            clear();
        }
    }
    ICacheableObject* getObj() const {return mObject.get();}
    std::unique_ptr<ICacheableObject> clear(){
        std::lock_guard<std::recursive_mutex> itemLock{mLock};
        if (!isValid()){
            throw std::runtime_error("Already cleared previously");
        }
        scrubFinishedUsers();
        if(isAcquired()){
            throw InUseRemove{};
        }
        {
            auto& residents = *mManager.mResidency.at(mLocation);
            std::lock_guard<std::recursive_mutex> residentsLock{residents.lock};
            assert(containsIter(residents.cached, mResidencyIter));
            residents.cached.erase(mResidencyIter);
        }
        for (auto loc : storageHierarchy){
            auto& residents = *mManager.mResidency.at(loc);
            residents.storageSize.fetch_sub(mObject->getCurrentStorageBytes(loc));
        }

        assert(!isAcquired());
        mLocation = StorageLocation::kUnknown;
        std::unique_ptr<ICacheableObject> obj = std::move(mObject);
        obj->notifyReturned();

        assert(!isValid());
        return obj;
    }
    bool isAcquired() const {
        std::lock_guard<std::recursive_mutex> itemLock{mLock};
        return getNbActiveUsers() != 0u;
    }
    KeyType key() const { return *mResidencyIter; } // ResidencyIter never changes. No need for locking
    StorageLocation getLocation() const {
        std::lock_guard<std::recursive_mutex> itemLock{mLock};
        return mLocation;
    }
    bool isValid() const {
        std::lock_guard<std::recursive_mutex> itemLock{mLock};
        return mObject != nullptr;
    }
    size_t getNbActiveUsers() const{
        std::lock_guard<std::recursive_mutex> itemLock{mLock};
        return mActiveUsers.size();
    }
    // Remove users that have finished synchronously, i.e. the finish event has already triggered. We don't need to keep such events.
    void scrubFinishedUsers(){
        std::lock_guard<std::recursive_mutex> itemLock{mLock};
        mFinishedReaders->scrub();
    }
    std::recursive_mutex& getMutex() const {return mLock;}
    std::condition_variable_any& getCondVar() const {return mCondVar;}

    AcquiredObj acquire(bool isUnique, const StorageLocation dst, cudaStream_t stream){
        const auto trace = cudapp::Profiler::instance().mark(__func__);
        std::lock_guard<std::recursive_mutex> itemLock{mLock};
        assert(isValid());
        assert(mLocation == mObject->getCurrentStorageLocation());
        if (mIsUniquelyAcquired){
            assert(mActiveUsers.size() == 1);
            return AcquiredObj{};
        }

        scrubFinishedUsers();

        auto lockAndCheckContainsIter = [](std::recursive_mutex& mutex, const auto& container, auto iter){
            std::lock_guard<std::recursive_mutex> residentsLock(mutex);
            return containsIter(container, iter);
        };
        unused(lockAndCheckContainsIter);

        if (isAcquired()) {
            if (dst == mLocation){
                if (isUnique){
                    return AcquiredObj{};
                }
                assert(!mActiveUsers.empty());
                const auto& residents = *mManager.mResidency.at(dst); unused(residents);
                assert(lockAndCheckContainsIter(residents.lock, residents.acquired, mResidencyIter));
                // Before migration, make sure previous users have finished.
                assert(!isUnique);
                cudaCheck(cudaStreamWaitEvent(stream, mEventAvailable, cudaEventWaitDefault));
                mObject->migrateTo(dst, stream);
            }
            else{
                if (isUnique){
                    return AcquiredObj{};
                }
                assert(!mActiveUsers.empty());
                return AcquiredObj{};
            }
        }
        else {
            if (dst == mLocation){
                assert(mActiveUsers.empty());
                auto& residents = *mManager.mResidency.at(dst); unused(residents);
                std::lock_guard<std::recursive_mutex> residentsLock(residents.lock);
                assert(containsIter(residents.cached, mResidencyIter));
                residents.acquired.splice(residents.acquired.end(), residents.cached, mResidencyIter);
                // Before migration, make sure previous users have finished.
                cudaCheck(cudaStreamWaitEvent(stream, mEventAvailable, cudaEventWaitDefault));
                if (isUnique)
                {
                    mFinishedReaders->streamWaitEvent(stream);
                    mFinishedReaders->clear();
                }
                mObject->migrateTo(dst, stream);
                if (isUnique) {
                    cudaCheck(cudaEventRecord(mEventAvailable, stream));
                }
            }
            else{
                const StorageLocation src = mLocation;
                assert(mActiveUsers.empty());
                auto& residentsSrc = *mManager.mResidency.at(src);
                auto& residentsDst = *mManager.mResidency.at(dst);
                assert(lockAndCheckContainsIter(residentsSrc.lock, residentsSrc.cached, mResidencyIter));
                // Before migration, make sure previous users have finished.
                cudaCheck(cudaStreamWaitEvent(stream, mEventAvailable, cudaEventWaitDefault));
                mFinishedReaders->streamWaitEvent(stream);
                mFinishedReaders->clear();
                const auto storageSizeOld = mObject->getCurrentStorageBytes(storageHierarchy);
                mObject->migrateTo(dst, stream);
                cudaCheck(cudaEventRecord(mEventAvailable, stream));
                {
                    std::scoped_lock residentsLockBoth(residentsSrc.lock, residentsDst.lock);
                    residentsDst.acquired.splice(
                                residentsDst.acquired.end(),
                                residentsSrc.cached, mResidencyIter);
                }
                mLocation = dst;
                const auto storageSizeNew = mObject->getCurrentStorageBytes(storageHierarchy);
                for (size_t i = 0; i < storageHierarchy.size(); i++){
                    const auto loc = storageHierarchy[i];
                    auto& residents = *mManager.mResidency.at(loc);
                    const auto delta = storageSizeNew[i] - storageSizeOld[i];
                    static_assert(std::is_unsigned_v<decltype(delta)> && std::is_integral_v<decltype(delta)>, "expected unsigned integral type for modular math");
                    static_assert(std::is_same_v<decltype(residents.storageSize.load()), std::decay_t<decltype(delta)>>);
                    residents.storageSize.fetch_add(delta);
                }
            }

            if (isUnique){
                mIsUniquelyAcquired = true;
            }
            const auto oldState = mObject->notifyStateChange(isUnique ? ICacheableObject::State::kUnique : ICacheableObject::State::kShared);
            REQUIRE(oldState == ICacheableObject::State::kIdle);
        }

        mActiveUsers.emplace_back(stream);
        AcquiredObj result{&mManager, key(), mObject.get(), dst, stream, mEventAvailable, isUnique};
        return result;
    }

    void release(cudaStream_t stream){
        std::unique_lock<std::recursive_mutex> itemLock{mLock};
        assert(isValid());
        assert(mLocation == mObject->getCurrentStorageLocation());
        assert(contains(mActiveUsers, stream));

        if (mIsUniquelyAcquired) {
            cudaCheck(cudaEventRecord(mEventAvailable, stream));
        }
        else {
            mFinishedReaders->recordEvent(stream);
        }
        const auto iterActive = std::find(mActiveUsers.begin(), mActiveUsers.end(), stream);
        assert(iterActive != mActiveUsers.end());
        mActiveUsers.erase(iterActive);

        if (mActiveUsers.empty())
        {
            const auto oldState = getObj()->notifyStateChange(ICacheableObject::State::kIdle);
            REQUIRE(oldState == ICacheableObject::State::kShared || oldState == ICacheableObject::State::kUnique);
            auto& residents = *mManager.mResidency.at(mLocation); // const and does not need protection
            std::lock_guard<std::recursive_mutex> residentsLock(residents.lock);
            assert(containsIter(residents.acquired, mResidencyIter));
            assert(!containsIter(residents.cached, mResidencyIter));
            // always insert at end
            residents.cached.splice(residents.cached.end(), residents.acquired, mResidencyIter);
        }

        if (mIsUniquelyAcquired){
            assert(mActiveUsers.empty());
            mIsUniquelyAcquired = false;
        }

        scrubFinishedUsers();

        if (mActiveUsers.empty()){
            itemLock.unlock();
            mCondVar.notify_all();
        }
    }

private:
    mutable std::recursive_mutex mLock; // Also needs to be locked while moving resident list node (referred by residencyIter) from one list to another.
    mutable std::condition_variable_any mCondVar;
    std::unique_ptr<ICacheableObject> mObject;
    StorageLocation mLocation;
    bool mIsUniquelyAcquired{false};
    // When acquired, a stream is stored. When a user finish using, a event is stored to indicate finish.
    // std::set and unordered_set have better complexity, but not necessarily faster, as we don't expect many elements here.
    std::vector<cudaStream_t> mActiveUsers;
    const cudaEvent_t mEventAvailable; // for migration and release of uniquely acquire objects
    ICudaMultiEvent* const mFinishedReaders; // for non-unique acquire

    StorageManager& mManager;
    // iterator to mResidency.at(location).acquired or .cached, depending on if streams.empty() or not.
    // must be always valid, residing in this->location. Use std::list::splice to move around
    const typename std::list<KeyType>::iterator mResidencyIter;
};

cudapp::storage::StorageManager::StorageManager(size_t deviceSoftLimit, size_t pinnedSoftLimit, size_t sysSoftLimit)
{
    setLimits(StorageLocation::kCudaDeviceMem, deviceSoftLimit);
    setLimits(StorageLocation::kPinnedMem, pinnedSoftLimit);
    setLimits(StorageLocation::kSysMem, sysSoftLimit);
    setLimits(StorageLocation::kDisk, std::numeric_limits<size_t>::max());
}

StorageManager::~StorageManager(){
    std::lock_guard<std::shared_mutex> managerLock{mLock};
    assert(std::all_of(mResidency.begin(), mResidency.end(), [](const auto& e){
        std::lock_guard<std::recursive_mutex> residentsLock(e.second->lock);
        return e.second->acquired.empty();
    }));
    mItems.clear();
}

void StorageManager::setLimits(StorageLocation location, size_t soft, size_t hard){
    auto& residents = *mResidency.at(location);
    residents.softLimit.store(soft);
    residents.hardLimit.store(hard);
}

StorageManager::KeyType StorageManager::addItem(std::unique_ptr<ICacheableObject> &&obj){
    // std::lock_guard<std::recursive_mutex> bugWarLk{bugWarLock()};
    std::lock_guard<std::shared_mutex> managerLock{mLock};
    const StorageLocation initLoc = obj->getCurrentStorageLocation();
    const auto [availEvent, finishReadEvents] = obj->notifyTaken(*this);
    const KeyType key = obj.get();
    assert(initLoc == obj->getCurrentStorageLocation());
    ASSERT(mItems.emplace(key, std::make_shared<Item>(*this, key, std::move(obj), initLoc, availEvent, finishReadEvents)).second);
    ASSERT(key != kInvalidKey);
    return key;
}

bool StorageManager::hasItem(KeyType key) const{
    std::shared_lock<std::shared_mutex> managerLock{mLock};
    return mItems.find(key) != mItems.end();
}

std::unique_ptr<ICacheableObject> StorageManager::removeItem(KeyType key){
    // std::lock_guard<std::recursive_mutex> bugWarLk{bugWarLock()};
    std::unique_ptr<ICacheableObject> obj;
    {
        std::lock_guard<std::shared_mutex> managerLock{mLock};
        const auto iter = mItems.find(key);
        if (iter == mItems.end()){
            throw KeyNotFound{};
        }
        const auto item = iter->second;
        {
            std::lock_guard<std::recursive_mutex> lk{item->getMutex()};
            obj = item->clear();
            mItems.erase(key);
        }
    }
    return obj;
}

ICacheableObject *StorageManager::getObj(StorageManagerTraits::KeyType key) const {
    std::lock_guard<std::shared_mutex> managerLock{mLock};
    const auto item = mItems.at(key);
    std::lock_guard<std::recursive_mutex> itemLock{item->getMutex()};
    return item->getObj();
}

StorageLocation StorageManager::peekStorageLocation(StorageManagerTraits::KeyType key) const{
    const auto item = [this, key](){
        std::shared_lock<std::shared_mutex> managerLock(mLock);
        return mItems.at(key);
    }();
    std::lock_guard<std::recursive_mutex> itemLock(item->getMutex());
    if (!item->isValid()){
        throw std::runtime_error("This item was just removed from the manager while you try to acquire lock");
    }
    assert(item->getLocation() == item->getObj()->getCurrentStorageLocation());
    return item->getLocation();
}

size_t StorageManager::peekStorageUsage(StorageLocation location) const {
    return mResidency.at(location)->storageSize.load();
}

// unique: unique acquire for write
// blocking: block until acquire succeed.
// !Important: stream must live longer than the cache object. Otherwise we get segfault when destroying the cached object as its resource is still associated with the stream.
AcquiredObj StorageManager::acquire(bool unique, bool blocking, KeyType key, StorageLocation dst, cudaStream_t stream)
{
#if NVBUG_200569951_WAR
    std::lock_guard<std::recursive_mutex> bugWarLk{bugWarLock(stream)};
#endif
    auto scopeGuard = makeScopeGuard([this](){fitCache();});
    // Get a shared_ptr copy so we don't need to hold the global manager lock. This approach is only beneficial
    // when we have a lot of addItem/removeItem calls, which is usually not the case. May consider simplify,
    // then mItems.at(key) does not need to be a shared_ptr.
    const auto item = [this, key](){
        std::shared_lock<std::shared_mutex> managerLock(mLock);
        const auto& items = mItems;
        const auto iter = items.find(key);
        if (iter == items.end()){
            throw KeyNotFound{};
        }
        return iter->second;
    }();
    std::unique_lock<std::recursive_mutex> itemLock(item->getMutex());
    if (!item->isValid()){
        throw KeyNotFound{};
    }
    if (blocking){
        item->getCondVar().wait(itemLock, [&item, unique, dst](){
            if (unique){
                return !item->isAcquired();
            }
            else{
                return item->getLocation() == dst || !item->isAcquired();
            }
        });
    }
    AcquiredObj result = item->acquire(unique, dst, stream);
    if (blocking){
        assert(result.isValid());
    }
    return result;
}

void StorageManager::release(KeyType key, cudaStream_t stream)
{
#if NVBUG_200569951_WAR
    std::lock_guard<std::recursive_mutex> bugWarLk{bugWarLock(stream)};
#endif
    // No need to get a copy like acquire() does, as the item must be acquired when calling release, and acquired items cannot be removed.
    std::shared_lock<std::shared_mutex> managerLock(mLock);
    {
        const auto& items = mItems;
        auto& item = *items.at(key);
        std::lock_guard<std::recursive_mutex> itemLock(item.getMutex());
        managerLock.unlock();

        item.release(stream);
    }
    // release itself never causes overloading, but it's a good chance to clear some cache if it's overloaded previously.
    fitCache();
}

void StorageManager::fitCache()
{
    //@fixme: change to calling functions (e.g. acquire()) to func() and funcImpl() and only func() calls fitCache(). Internally we always use funcImpl(), then there will be no recursion.
    thread_local static bool isInFitCache = false;
    if (isInFitCache){
        // Prevent recursion. Otherwise we run into deal-lock situation, where two (or more) threads each hold an item lock and try to lock the other item lock of the other thread.
        return;
    }
    else{
        isInFitCache = true;
    }
    // We just set isInFitCache to true. Revert on exit.
    auto resetIsInFitCache = makeScopeGuard([](){isInFitCache = false;});
    // We never evict from the last-level storage, because there is no way to evict.
    for (uint32_t i = 0; i < storageHierarchy.size() - 1; i++){
        const auto location = storageHierarchy.at(i);
        auto& residents = *mResidency.at(location);
        size_t occupiedStorageSize = residents.storageSize.load();
        while (occupiedStorageSize > residents.softLimit){
            // All other functions lock item first and residents later, so also follow this order here to avoid dead lock.
            std::shared_ptr<Item> item;
            std::shared_lock<std::shared_mutex> managerLock;
            std::unique_lock<std::recursive_mutex> itemLock;
            while(true)
            {
                KeyType key = kInvalidKey;
                const auto& items = mItems;

                {
                    std::shared_lock<std::shared_mutex> managerLock(mLock);
                    std::lock_guard<std::recursive_mutex> residentsLock(residents.lock);
                    if (residents.cached.empty()){
                        break;
                    }
                    key = residents.cached.front();
                    assert(key != kInvalidKey);
                    item = items.at(key);
                }
                {
                    std::shared_lock<std::shared_mutex> managerLockTrial(mLock);
                    std::unique_lock<std::recursive_mutex> itemLockTrial{item->getMutex()};
                    std::lock_guard<std::recursive_mutex> residentsLock{residents.lock};
                    if (residents.cached.front() == key) {
                        assert(items.at(key) == item);
                        assert(item->isValid());
                        managerLock = std::move(managerLockTrial);
                        itemLock = std::move(itemLockTrial);
                        break;
                    }
                    else{
                        key = kInvalidKey;
                        item = nullptr;
                    }
                }
            }
            // Cache is empty, cannot find anything to evict.
            if (item == nullptr){
                break;
            }

            assert(itemLock.owns_lock());
            // let the returned AcquiredObj destruct immediately, so it goes from acquired to cache in the target level
            unused(acquire(false, false, item->key(), item->getObj()->getCurrentEvictionTarget(), mStream.get()));

            occupiedStorageSize = residents.storageSize.load();
        }
        if (occupiedStorageSize > residents.hardLimit){
            throw std::runtime_error("failure to meet hard limit");
        }
    }
}



} // namespace cudapp
} // namespace cudapp
