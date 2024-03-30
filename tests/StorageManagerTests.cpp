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

#include <gtest/gtest.h>
#include "../StorageManager.h"
#include <tuple>
#define USE_STD_THREAD 1
#if USE_STD_THREAD
#include <thread>
#else
#include <future>
#endif
#include <memory>
#include <random>
#include "../CudaEventPool.h"

using namespace cudapp::storage;
using namespace cudapp;

#define DEBUG_PRINT 0

class TestCacheableObject : public ICacheableObject
{
public:
    struct DataType{uint32_t value;};
    struct CudaDataType : DataType{};
    struct PinnedDataType : DataType{};
    struct SysDataType : DataType{};
    struct DiskDataType : DataType{};

    TestCacheableObject() {
    }

    ~TestCacheableObject() override{
        REQUIRE(mManager == nullptr);
        mFinishEvents->sync();
    }

    std::pair<cudaEvent_t, ICudaMultiEvent*> notifyTaken(StorageManager &manager) override{
        if (mManager != nullptr){
            throw std::runtime_error("already occupied");
        }
        mManager = &manager;
        return {mAvailEvent.get(), mFinishEvents.get()};
    }
    void notifyReturned() override{
        mManager = nullptr;
    }

    State notifyStateChange(State state) override{
        std::swap(mState, state);
        return state;
    }

    void migrateTo(StorageLocation dst, cudaStream_t stream) override;
    StorageLocation getCurrentEvictionTarget() override{
        if (mLocation == StorageLocation::kDisk){
            throw std::runtime_error("Cannot evict from the last level storage");
        }
        return static_cast<StorageLocation>(static_cast<uint32_t>(mLocation) + 1);
    }
    StorageLocation getCurrentStorageLocation() const override {return mLocation;}
    size_t getCurrentStorageBytes(StorageLocation loc) const override{
        return (loc == mLocation) ? 1u : 0u;
    }
    uint32_t getIdx() const {return mIdx;}
    uint32_t getVal() const {
        switch(mLocation){
        case StorageLocation::kCudaDeviceMem: return std::get<CudaDataType>(mData).value;
        case StorageLocation::kPinnedMem: return std::get<PinnedDataType>(mData).value;
        case StorageLocation::kSysMem: return std::get<SysDataType>(mData).value;
        case StorageLocation::kDisk: return std::get<DiskDataType>(mData).value;
        default: break;
        }
        throw std::runtime_error("invalid location");
    }
    uint32_t& value() {
        switch(mLocation){
        case StorageLocation::kCudaDeviceMem: return std::get<CudaDataType>(mData).value;
        case StorageLocation::kPinnedMem: return std::get<PinnedDataType>(mData).value;
        case StorageLocation::kSysMem: return std::get<SysDataType>(mData).value;
        case StorageLocation::kDisk: return std::get<DiskDataType>(mData).value;
        default: break;
        }
        throw std::runtime_error("invalid location");
    }

    static constexpr auto storageHierarchy = StorageManager::storageHierarchy;
    static constexpr uint32_t badVal = ~0u/2;
protected:
    template <StorageLocation src, StorageLocation dst>
    void migrate(cudaStream_t stream){
//        const auto hostFunc = [](void* p){
        const auto callback = [](cudaStream_t stream, cudaError status, void* p){
            cudaCheck(status);
            TestCacheableObject* obj = static_cast<TestCacheableObject*>(p);
            unused(stream);
#if DEBUG_PRINT
            printf("cuda obj%u: %u%u in stream %p\n", obj->getIdx(), static_cast<uint32_t>(src), static_cast<uint32_t>(dst), stream);
#endif
            EXPECT_EQ(obj->mRealLocation.load(), src);
            auto& data = obj->mData;
//            auto& data = *static_cast<std::tuple<CudaDataType, PinnedDataType, SysDataType, DiskDataType>*>(p);
            if (src != dst) {
                EXPECT_EQ(std::get<static_cast<uint32_t>(dst)>(data).value, badVal);
            }
            EXPECT_NE(std::get<static_cast<uint32_t>(src)>(data).value, badVal);
            std::swap(std::get<static_cast<uint32_t>(src)>(data).value,
                      std::get<static_cast<uint32_t>(dst)>(data).value);
            obj->mRealLocation.store(dst);
        };
//        cudaCheck(cudaLaunchHostFunc(stream, hostFunc, this));
        cudaCheck(cudaStreamAddCallback(stream, callback, this, 0));
        mLocation = dst;
#undef checkBadVal
    }
protected:
    static inline std::atomic_uint32_t mIdxNext{0u};
    const uint32_t mIdx = mIdxNext.fetch_add(1u, std::memory_order_relaxed);
    State mState = State::kIdle;
    StorageManager* mManager = nullptr;
    PooledCudaEvent mAvailEvent = createPooledCudaEvent();
    std::unique_ptr<ICudaMultiEvent> mFinishEvents = createCudaMultiEvent(true);
    StorageLocation mLocation = StorageLocation::kDisk;
    std::tuple<CudaDataType, PinnedDataType, SysDataType, DiskDataType> mData = {{{badVal}}, {{badVal}}, {{badVal}}, {{mIdx}}};
    std::atomic<StorageLocation> mRealLocation {mLocation}; // updated and used by cuda stream, not on host.
};

void TestCacheableObject::migrateTo(const StorageLocation dst, const cudaStream_t stream){
    using Func = void (TestCacheableObject::*)(cudaStream_t);
#define MFUNC(src, dst) &TestCacheableObject::migrate<static_cast<StorageLocation>(src), static_cast<StorageLocation>(dst)>
    const Func functions[4][4] = {
        {MFUNC(0, 0), MFUNC(0, 1), MFUNC(0, 2), MFUNC(0, 3)},
        {MFUNC(1, 0), MFUNC(1, 1), MFUNC(1, 2), MFUNC(1, 3)},
        {MFUNC(2, 0), MFUNC(2, 1), MFUNC(2, 2), MFUNC(2, 3)},
        {MFUNC(3, 0), MFUNC(3, 1), MFUNC(3, 2), MFUNC(3, 3)}
    };
#undef MFUNC
#if DEBUG_PRINT
    const auto tid = std::this_thread::get_id();
    printf("host obj%u: %u%u in thread %lu\n", getIdx(), static_cast<uint32_t>(mLocation), static_cast<uint32_t>(dst), reinterpret_cast<const unsigned long&>(tid));
#endif
    (this->*functions[static_cast<uint32_t>(mLocation)][static_cast<uint32_t>(dst)])(stream);
    assert (mLocation == dst);
}

class StorageManagerTest : public testing::Test, public StorageManager
{
public:
    void SetUp() override{
        setLimits(StorageLocation::kCudaDeviceMem, 4, 256);
        setLimits(StorageLocation::kPinnedMem, 8, 256);
        setLimits(StorageLocation::kSysMem, 16, 256);
        setLimits(StorageLocation::kDisk, 256, 256);

        // setLimits(StorageLocation::kCudaDeviceMem, 10, 256);
        // setLimits(StorageLocation::kPinnedMem, 10, 256);
        // setLimits(StorageLocation::kSysMem, 10, 256);
        // setLimits(StorageLocation::kDisk, 10, 256);

        for (size_t i = 0; i < nbObj; i++){
            auto obj = std::make_unique<TestCacheableObject>();
            auto ptr = obj.get();
            const auto initVal = obj->getVal();
            const auto key = addItem(std::move(obj));
            mManagedKeys.push_back(key);
            mRefs.emplace(std::piecewise_construct, std::tuple(ptr), std::tuple(initVal));
        }
        mThreads.resize(nbThrds);
        mRandEngine = decltype(mRandEngine){nbThrds};
        mStreams.clear();
        std::generate_n(std::back_inserter(mStreams), nbStreams, []{return makeCudaStream();});
    }
    void TearDown() override{
        mThreads.clear();
        mStreams.clear();
        while(!mManagedKeys.empty()){
            auto key = mManagedKeys.back();
            auto obj = removeItem(key);
            mFreeObj.emplace_back(std::move(obj));
            mManagedKeys.erase(std::prev(mManagedKeys.end()));
        }
        EXPECT_EQ(mFreeObj.size(), nbObj);
        mRefs.clear();
        mFreeObj.clear();
    }

    void runSingleThrd(uint32_t idxThrd){
        for (size_t n = 0; n < mMaxIters; n++) {
//            const uint32_t idxStream = idxThrd % mStreams.size();
            const uint32_t idxStream = randVal(idxThrd) % mStreams.size();
            const cudaStream_t stream = mStreams.at(idxStream).get();
            uint32_t randChoice = randVal(idxThrd);
            if (randChoice < std::numeric_limits<uint32_t>::max() * 0.6f){
                acquireAndModify(idxThrd, stream);
            }
            else if(randChoice < std::numeric_limits<uint32_t>::max() * 0.8f){
                removeOne(idxThrd, stream);
            }
            else {
                addOne(idxThrd, stream);
            }
        }
    }

    void syncAndCheck(){
        for (auto& t : mThreads) { // join all threads
#if USE_STD_THREAD
            t.join();
#else
            t.get();
#endif
        }
        cudaCheck(cudaDeviceSynchronize());
        for (const auto& obj : mFreeObj) {
            EXPECT_EQ(static_cast<TestCacheableObject*>(obj.get())->getVal(), mRefs.at(obj.get()));
        }
        for (const auto& key : mManagedKeys) {
            const auto obj = getObj(key);
            EXPECT_EQ(static_cast<TestCacheableObject*>(obj)->getVal(), mRefs.at(obj));
        }
        size_t nbCachedTotal = 0u;
        for (const auto loc : storageHierarchy){
            const auto& residents = mResidency.at(loc);
            EXPECT_EQ(residents->acquired.size(), 0);
            EXPECT_EQ(residents->cached.size(), residents->storageSize.load());
            nbCachedTotal += residents->cached.size();
        }
        EXPECT_EQ(nbCachedTotal + mFreeObj.size(), nbObj);
    }

    void acquireAndModify(uint32_t idxThrd, cudaStream_t stream) {
        const StorageLocation location = static_cast<StorageLocation>(randVal(idxThrd) % static_cast<uint32_t>(storageHierarchy.size()));
        KeyType key = kInvalidKey;
        AcquiredObj acquiredObj;
        while (!acquiredObj.isValid())
        {
            key = [this, idxThrd](){
                std::shared_lock<std::shared_mutex> lk{mTestMutex};
                const auto& managedKeys = mManagedKeys;
                if (managedKeys.empty()){
                    return kInvalidKey;
                }
                return managedKeys.at(randVal(idxThrd) % managedKeys.size());
//                return managedKeys.at(idxThrd % managedKeys.size());
            }();
            if (key == kInvalidKey){
                break;
            }
            try {
                acquiredObj = acquire(true, false, key, location, stream);
            } catch (const KeyNotFound&) {
                // OK. This may happen because we are not locking between mManagedKeys and the StorageManager.
            }
        }
        if (key == kInvalidKey){
            return;
        }
        EXPECT_TRUE(acquiredObj.isValid());

        const uint32_t delta = randVal(idxThrd) % 16u;
        struct CallbackData{
            uint32_t* pValue;
            uint32_t delta;
            StorageLocation location;
            uint32_t id;
        };
        EXPECT_EQ(peekStorageLocation(key), location);
        EXPECT_EQ(acquiredObj.get()->getCurrentStorageLocation(), location);
        CallbackData* callbackData = new CallbackData{
                &static_cast<TestCacheableObject*>(acquiredObj.get())->value(),
                delta, location,
                static_cast<TestCacheableObject*>(acquiredObj.get())->getIdx()};
#if DEBUG_PRINT
        printf("host obj%u: update at %u\n", callbackData->id, static_cast<uint32_t>(location));
#endif
        cudaCheck(cudaLaunchHostFunc(stream, [](void* p){
//            std::cout << "\t\t" << std::this_thread::get_id() << std::endl;
            auto ptr = static_cast<const CallbackData*>(p);
#if DEBUG_PRINT
            printf("cuda obj%u: update at\t%u\n", ptr->id, static_cast<uint32_t>(ptr->location));
#endif
            EXPECT_NE(*ptr->pValue, TestCacheableObject::badVal);
            *ptr->pValue += ptr->delta;
            delete ptr;
        }, callbackData));
        mRefs.at(acquiredObj.get()).fetch_add(delta);
//        printf("acquireAndModify\n");
    }

    void removeOne(uint32_t idxThrd, cudaStream_t stream) {
        bool retry = false;
        do{
            try {
                const KeyType key = [this, idxThrd](){
                    std::shared_lock<std::shared_mutex> lk{mTestMutex};
                    const auto& managedKeys = mManagedKeys;
                    if (managedKeys.empty()){
                        return kInvalidKey;
                    }
                    uint32_t idx = randVal(idxThrd) % managedKeys.size();
                    KeyType key = managedKeys.at(idx);
                    return key;
                }();
                if (key == kInvalidKey){
                    break;
                }
                {
                    std::lock_guard<std::shared_mutex> lk{mTestMutex};
                    const auto iterKey = std::find(mManagedKeys.begin(), mManagedKeys.end(), key);
                    if (iterKey != mManagedKeys.end()){
                        auto obj = removeItem(key);
                        if (obj != nullptr){
#if DEBUG_PRINT
                            printf("host: removed %u\n", static_cast<uint32_t>(obj->getCurrentStorageLocation()));
#endif
                            mManagedKeys.erase(iterKey);
                            mFreeObj.emplace_back(std::move(obj));
                            retry = false;
                        }
                    }
                }
            } catch (const KeyNotFound&) {
                retry = true;
            } catch (const InUseRemove&) {
                retry = true;
            }
        }while (retry);
        unused(stream);
    }

    void addOne(uint32_t idxThrd, cudaStream_t stream) {
        std::unique_ptr<ICacheableObject> obj = [this, idxThrd](){
            std::lock_guard<std::shared_mutex> lk{mTestMutex};
            if (mFreeObj.empty()){
                return std::unique_ptr<ICacheableObject>{};
            }
            const uint32_t idx = randVal(idxThrd) % mFreeObj.size();
            auto obj = std::move(mFreeObj.at(idx));
            mFreeObj.erase(mFreeObj.begin() + idx);
            return obj;
        }();
        if (obj == nullptr){
            return;
        }
#if DEBUG_PRINT
        printf("host: add %u\n", static_cast<uint32_t>(obj->getCurrentStorageLocation()));
#endif
        auto key = addItem(std::move(obj));
        {
            std::lock_guard<std::shared_mutex> lk{mTestMutex};
            mManagedKeys.push_back(key);
        }
        unused(stream);
    }

protected:
    std::shared_mutex mTestMutex;
    //config
    size_t nbObj = 256u;
    size_t nbThrds = 4u;
    size_t nbStreams = 16u;

    std::vector<std::unique_ptr<ICacheableObject>> mFreeObj;
    std::vector<KeyType> mManagedKeys;
    std::map<ICacheableObject*, std::atomic<uint32_t>> mRefs;
#if USE_STD_THREAD
    std::vector<std::thread> mThreads;
#else
    std::vector<std::future<void>> mThreads;
#endif
    std::vector<CudaStream> mStreams;
    std::size_t mMaxIters = 100000;

    std::vector<std::tuple<std::once_flag, std::mt19937_64, std::uniform_int_distribution<uint32_t>>> mRandEngine;
    uint32_t randVal(uint32_t idxThrd) {
        auto& engine = mRandEngine.at(idxThrd);
        std::call_once(std::get<0>(engine), [&engine, idxThrd](){
            std::get<1>(engine) = std::mt19937_64{idxThrd};
        });
        return std::get<2>(engine)(std::get<1>(engine));
    }
};

TEST_F(StorageManagerTest, multiThread)
{
    for (uint32_t idxThrd = 0; idxThrd < mThreads.size(); idxThrd++){
#if USE_STD_THREAD
        mThreads.at(idxThrd) = std::thread(&StorageManagerTest::runSingleThrd, this, idxThrd);
#else
        mThreads.at(idxThrd) = std::async(std::launch::async, &StorageManagerTest::runSingleThrd, this, idxThrd);
#endif
    }
#if !USE_STD_THREAD
    while(mThreads.at(0).wait_for(std::chrono::milliseconds(250)) == std::future_status::timeout){
        printf("\rStorage usage: %lu, %lu, %lu, %lu                     ",
               peekStorageUsage(StorageLocation::kCudaDeviceMem), peekStorageUsage(StorageLocation::kPinnedMem),
               peekStorageUsage(StorageLocation::kSysMem), peekStorageUsage(StorageLocation::kDisk));
    }
    printf("\n");
#endif
    syncAndCheck();
}
