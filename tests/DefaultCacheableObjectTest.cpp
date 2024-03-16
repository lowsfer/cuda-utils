#include "DefaultCacheableObject.h"
#include <gtest/gtest.h>
#include "StorageManager.h"
#include <thread>
#include <tuple>
#define USE_STD_THREAD 1
#if USE_STD_THREAD
#include <thread>
#else
#include <future>
#endif
#include <memory>
#include <random>

#define DEBUG_PRINT 0

using namespace cudapp::storage;

void launchDeviceCheckAndSetVal(cudaStream_t stream, bool checkOnly, volatile uint32_t* data, uint32_t expectedOldVal, uint32_t newVal = ~0u, uint32_t nbCycles = 0);

void launchHostCheckAndSetVal(cudaStream_t stream, bool checkOnly, volatile uint32_t* data, uint32_t expectedOldVal, uint32_t newVal, uint32_t nbCycles)
{
    launchCudaHostFunc(stream, [checkOnly, data, expectedOldVal, newVal, nbCycles]{
        const clock_t start = clock();
#if DEBUG_PRINT
        if (*data != expectedOldVal) {
            printf("%d != %d\n", *data, expectedOldVal);
        }
#endif
        REQUIRE(*data == expectedOldVal);
        if (checkOnly) return;
        *data = newVal;
        while (clock() - start < nbCycles) {
            REQUIRE(*data == newVal);
        }
    });
}

void launchDiskCheckAndSetVal(cudaStream_t stream, bool checkOnly, const fs::path& filePath, uint32_t expectedOldVal, uint32_t newVal, uint32_t nbCycles)
{
    launchCudaHostFunc(stream, [checkOnly, filePath, expectedOldVal, newVal, nbCycles]{
        const clock_t start = clock();

        auto readVal = [&]()->uint32_t{
            std::ifstream fin;
            fin.exceptions(std::ios::badbit | std::ios::failbit | std::ios::eofbit);
            fin.open(filePath, std::ios::binary);
            uint32_t val;
            fin.read(reinterpret_cast<char*>(&val), sizeof(int));
            return val;
        };
        auto writeVal = [&](uint32_t val){
            std::ofstream fout;
            fout.exceptions(std::ios::badbit | std::ios::failbit);
            fout.open(filePath, std::ios::binary | std::ios::trunc);
            fout.write(reinterpret_cast<const char*>(&val), sizeof(uint32_t));
        };

        REQUIRE(readVal() == expectedOldVal);
        if (checkOnly) return;

        writeVal(newVal);
        while (clock() - start < nbCycles) {
            REQUIRE(readVal() == newVal);
        }
    });
}

//template <typename CacheableObject>
using CacheableObject = DefaultCacheableObject<uint32_t>;
class DefaultCacheableObjectTest : public testing::Test, public StorageManager
{
public:
    void SetUp() override{
        mThreads.resize(nbThrds);
        mRandEngine = decltype(mRandEngine){nbThrds};
        mStreams.clear();
        std::generate_n(std::back_inserter(mStreams), nbStreams, []{return makeCudaStream();});

        setLimits(StorageLocation::kCudaDeviceMem, 4, 256);
        setLimits(StorageLocation::kPinnedMem, 8, 256);
        setLimits(StorageLocation::kSysMem, 16, 256);
        setLimits(StorageLocation::kDisk, 256, 256);

        // setLimits(StorageLocation::kCudaDeviceMem, 10, 256);
        // setLimits(StorageLocation::kPinnedMem, 10, 256);
        // setLimits(StorageLocation::kSysMem, 10, 256);
        // setLimits(StorageLocation::kDisk, 10, 256);

        const size_t nbElems = 1;
        const int initVal = 0;

        for (size_t i = 0; i < nbObj; i++){
            const cudaStream_t stream = mStreams.at(i % nbStreams).get();
            auto mem = std::get<2>(mMemPools).alloc<uint32_t>(nbElems, stream);
            launchCudaHostFunc(stream, [p{mem.get()}](){*p = initVal;});

            const fs::path file = fs::path{"obj" + std::to_string(i) + ".dat"};

            if (fs::exists(file)) {
                fs::remove(file);
            }

            auto obj = std::make_unique<CacheableObject>(
                        std::get<0>(mMemPools),
                        std::get<1>(mMemPools),
                        std::get<2>(mMemPools),
                        file,
                        std::move(mem), nbElems,
                        DiskStoragePolicy::kNormal,
                        false);
            auto ptr = obj.get();
            const auto key = addItem(std::move(obj));
            mManagedKeys.push_back(key);
            mRefs.emplace(std::piecewise_construct, std::tuple(ptr), std::tuple(initVal));
            mFiles.emplace(ptr, file);
        }
    }
    void TearDown() override{
        while(!mManagedKeys.empty()){
            auto key = mManagedKeys.back();
            auto obj = removeItem(key);
            mFreeObj.emplace_back(std::move(obj));
            mManagedKeys.erase(std::prev(mManagedKeys.end()));
        }
        EXPECT_EQ(mFreeObj.size(), nbObj);
        mRefs.clear();
        mFreeObj.clear();
        mStreams.clear();
        mThreads.clear();
    }

    void runSingleThrd(uint32_t idxThrd){
        for (size_t n = 0; n < mMaxIters; n++) {
//            const uint32_t idxStream = idxThrd % mStreams.size();
            const uint32_t idxStream = randVal(idxThrd) % mStreams.size();
            const cudaStream_t stream = mStreams.at(idxStream).get();
            uint32_t randChoice = randVal(idxThrd);
            if (randChoice < std::numeric_limits<uint32_t>::max() * 0.45f){
                acquireAndModify(idxThrd, stream, false);
            }
            else if (randChoice < std::numeric_limits<uint32_t>::max() * 0.9f){
                acquireAndModify(idxThrd, stream, true);
            }
            else if(randChoice < std::numeric_limits<uint32_t>::max() * 0.95f){
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
    }

    void acquireAndModify(uint32_t idxThrd, cudaStream_t stream, bool doModify = true) {
        const StorageLocation location = static_cast<StorageLocation>(randVal(idxThrd) % static_cast<uint32_t>(storageHierarchy.size()));
        KeyType key = kInvalidKey;
        AcquiredObj acquiredObj;
#if DEBUG_PRINT
        StorageLocation unreliableOldLoc;
#endif
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
#if DEBUG_PRINT
                unreliableOldLoc = peekStorageLocation(key);
#endif
                acquiredObj = acquire(doModify, false, key, location, stream);
            } catch (const KeyNotFound&) {
                // OK. This may happen because we are not locking between mManagedKeys and the StorageManager.
            }
        }
        if (key == kInvalidKey){
            return;
        }
        EXPECT_TRUE(acquiredObj.isValid());

        CacheableObject* const obj = dynamic_cast<CacheableObject*>(acquiredObj.get());
        const uint32_t expectedOldVal = mRefs.at(acquiredObj.get()).load();
        const uint32_t newVal = doModify ? expectedOldVal + 1 : expectedOldVal;
        const bool checkOnly = !doModify;
#if DEBUG_PRINT
        printf("Loc: %15s -> %15s, stream: %p, expectedVal: %u -> %u\n", toStr(oldLoc), toStr(location), stream, expectedOldVal, newVal);
#endif
        EXPECT_EQ(peekStorageLocation(key), location);
        EXPECT_EQ(acquiredObj.get()->getCurrentStorageLocation(), location);
        switch (location)
        {
        case StorageLocation::kDisk: launchDiskCheckAndSetVal(stream, checkOnly, mFiles.at(obj), expectedOldVal, newVal, nbCycles); break;
        case StorageLocation::kSysMem:
        case StorageLocation::kPinnedMem: launchHostCheckAndSetVal(stream, checkOnly, std::get<uint32_t*>(obj->getData()), expectedOldVal, newVal, nbCycles); break;
        case StorageLocation::kCudaDeviceMem: launchDeviceCheckAndSetVal(stream, checkOnly, std::get<uint32_t*>(obj->getData()), expectedOldVal, newVal, nbCycles); break;
        default: throw std::runtime_error("fatal error");
        }
        if (doModify) {
            mRefs.at(obj).store(newVal);
        }
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
    std::tuple<CudaDevMemPool, CudaPinnedMemPool, CudaSysMemPool> mMemPools;

    std::shared_mutex mTestMutex;
    //config
    size_t nbObj = 64u;
    size_t nbThrds = 4u;
    size_t nbStreams = 16u;

    std::vector<CudaStream> mStreams;
#if USE_STD_THREAD
    std::vector<std::thread> mThreads;
#else
    std::vector<std::future<void>> mThreads;
#endif

    std::vector<std::unique_ptr<ICacheableObject>> mFreeObj;
    std::vector<KeyType> mManagedKeys;
    std::map<ICacheableObject*, std::atomic<uint32_t>> mRefs;
    std::map<ICacheableObject*, fs::path> mFiles;
    std::size_t mMaxIters = 10000;
    uint32_t nbCycles = 0;

    std::vector<std::tuple<std::once_flag, std::mt19937_64, std::uniform_int_distribution<uint32_t>>> mRandEngine;
    uint32_t randVal(uint32_t idxThrd) {
        auto& engine = mRandEngine.at(idxThrd);
        std::call_once(std::get<0>(engine), [&engine, idxThrd](){
            std::get<1>(engine) = std::mt19937_64{idxThrd};
        });
        return std::get<2>(engine)(std::get<1>(engine));
    }
};

TEST_F(DefaultCacheableObjectTest, multiThread)
{
    for (uint32_t idxThrd = 0; idxThrd < mThreads.size(); idxThrd++){
#if USE_STD_THREAD
        mThreads.at(idxThrd) = std::thread(&DefaultCacheableObjectTest::runSingleThrd, this, idxThrd);
#else
        mThreads.at(idxThrd) = std::async(std::launch::async, &DefaultCacheableObjectTest::runSingleThrd, this, idxThrd);
#endif
    }
    syncAndCheck();
}
