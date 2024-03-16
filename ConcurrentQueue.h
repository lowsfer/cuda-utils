#pragma once
#include <mutex>
#include <condition_variable>
#include <deque>
#include <vector>
#include "cpp_utils.h"
#include <chrono>
#include "macros.h"

namespace cudapp
{
class ConcurrentQueueIsClosed : std::runtime_error
{
public:
    ConcurrentQueueIsClosed() : std::runtime_error{"queue is closed"} {}
};

template <typename T, typename Mutex = std::mutex, typename ConditionVariable = std::condition_variable>
class ConcurrentQueue
{
public:
    using Closed = ConcurrentQueueIsClosed;
    ConcurrentQueue(size_t capacity = std::numeric_limits<size_t>::max()) : mCapacity(capacity){}

    ~ConcurrentQueue(){
//        waitEmpty();
        close();
    }

    size_t getCapacity() const {
        std::unique_lock<Mutex> lock{mLock};
        return mCapacity;
    }

    void setCapacity(size_t capacity){
        ASSERT(capacity > 0);
        {
            std::unique_lock<Mutex> lock{mLock};
            mCapacity = capacity;
        }
        mCVarNotFull.notify_all();
        mCVarNotEmpty.notify_all();
    }

    template <typename... Args>
    // [[deprecated]]
    void emplace(Args&&... args){
        return emplace_back(std::forward<Args>(args)...);
    }

    template <typename... Args>
    void emplace_back(Args&&... args){
        return emplaceAt(false, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void emplace_front(Args&&... args){
        return emplaceAt(true, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void emplaceAt(bool front, Args&&... args){
        {
            std::unique_lock<Mutex> lock{mLock};
            if (mIsClosed) {
                throw Closed{};
            }
            mCVarNotFull.wait(lock, [this]{return mData.size() < mCapacity;});
            if (front) {
                mData.emplace_front(std::forward<Args>(args)...);
            }
            else {
                mData.emplace_back(std::forward<Args>(args)...);
            }
        }
        mCVarNotEmpty.notify_one();
    }

    void pushN(std::vector<T> items){
        {
            std::unique_lock<Mutex> lock{mLock};
            if (mIsClosed) {
                throw Closed{};
            }
            mCVarNotFull.wait(lock, [this]{return mData.size() < mCapacity;});
            for(T& item : items){
                mData.emplace_back(std::move(item));
            }
        }
        switch (items.size())
        {
        case 0: break; // queue is closed
        case 1: mCVarNotEmpty.notify_one(); break;
        default: mCVarNotEmpty.notify_all();
        }
    }

    void close() {
        {
            std::unique_lock<Mutex> lock{mLock};
            mIsClosed = true;
        }
        mCVarNotEmpty.notify_all();
    }

    // if empty, wait until close
    [[nodiscard]] std_optional<T> pop(){
        std_optional<T> result;
        {
            std::unique_lock<Mutex> lock{mLock};
            mCVarNotEmpty.wait(lock, [this]{return !mData.empty() || mIsClosed;});
            if (!mData.empty()){
                result.emplace(std::move(mData.front()));
                mData.pop_front();
            }
            else {
                assert(mIsClosed);
            }
        }
        if (result.has_value()) {
            mCVarNotFull.notify_one();
        }
        return result;
    }

    enum class ResultStatus{kOK, kTimeOut, kClosed};
    template <class Rep, class Period>
    [[nodiscard]] std::pair<ResultStatus, std_optional<T>> popFor(const std::chrono::duration<Rep, Period>& duration) {
        std_optional<T> result;
        {
            std::unique_lock<Mutex> lock{mLock};
            const bool success = mCVarNotEmpty.wait_for(lock, duration, [this]{return !mData.empty() || mIsClosed;});
            if (!success) {
                return {ResultStatus::kTimeOut, result};
            }
            if (mData.empty()) {
                assert(mIsClosed);
                return {ResultStatus::kClosed, result};
            }
            result.emplace(std::move(mData.front()));
            mData.pop_front();
        }
        assert(result.has_value());
        mCVarNotFull.notify_one();
        return {ResultStatus::kOK, result};
    }

    template <class Clock, class Duration>
    [[nodiscard]] std::pair<ResultStatus, std_optional<T>> popUntil(const std::chrono::time_point<Clock, Duration>& deadline) {
        const auto duration = deadline - Clock::now();
        return popFor(duration);
    }

    [[nodiscard]] std::vector<T> popN(size_t nbItems){
        ASSERT(nbItems > 0);
        std::vector<T> result;
        {
            std::unique_lock<Mutex> lock{mLock};
            result.reserve(std::min(nbItems, mData.size()));
            mCVarNotEmpty.wait(lock, [this]{return !mData.empty() || mIsClosed;});
            while (!mData.empty() && result.size() < nbItems){
                result.emplace_back(std::move(mData.front()));
                mData.pop_front();
            }
            if (result.empty()){
                assert(mIsClosed);
            }
        }
        switch (result.size())
        {
        case 0: break; // queue is closed
        case 1: mCVarNotFull.notify_one(); break;
        default: mCVarNotFull.notify_all();
        }
        return result;
    }

    bool isClosed() const {
        std::unique_lock<Mutex> lock{mLock};
        return mIsClosed;
    }

    bool isEmptyAndClosed() const {
        std::unique_lock<Mutex> lock{mLock};
        return mData.empty() && mIsClosed;
    }

    // Just check and pop if not empty. Never wait.
    [[nodiscard]] std_optional<T> tryPop()
    {
        std_optional<T> result;
        {
            std::unique_lock<Mutex> lock{mLock};
            if (!mData.empty()){
                result.emplace(std::move(mData.front()));
                mData.pop_front();
            }
        }
        if (result.has_value()){
            mCVarNotFull.notify_one();
        }
        return result;
    }

    [[nodiscard]] std::vector<T> tryPopN(size_t nbItems){
        std::vector<T> result;
        {
            std::unique_lock<Mutex> lock{mLock};
            result.reserve(std::min(mData.size(), nbItems));
            while (!mData.empty() && result.size() < nbItems){
                result.emplace_back(std::move(mData.front()));
                mData.pop_front();
            }
        }
        switch (result.size())
        {
        case 0: break;
        case 1: mCVarNotFull.notify_one(); break;
        default: mCVarNotFull.notify_all();
        }
        return result;
    }

    void waitEmpty() {
        std::unique_lock<Mutex> lock{mLock};
        mCVarNotFull.wait(lock, [this]{return mData.empty();});
    }

    void waitSize(size_t size) {
        std::unique_lock<Mutex> lock{mLock};
        if (mData.size() >= size) {
            mCVarNotFull.wait(lock, [this, size]{return mData.size() <= size;});
        }
        else {
            mCVarNotEmpty.wait(lock, [this, size]{return mData.size() >= size;});
        }
    }

    void waitClosed() {
        std::unique_lock<Mutex> lock{mLock};
        mCVarNotEmpty.wait(lock, [this]{return mIsClosed;});
    }

    void waitEmptyAndClosed() {
        std::unique_lock<Mutex> lock{mLock};
        mCVarNotEmpty.wait(lock, [this]{return mIsClosed && mData.empty();});
    }

    size_t peekSize() const {
        std::lock_guard<Mutex> lock{mLock};
        return mData.size();
    }
private:
    size_t mCapacity;
    std::deque<T> mData;
    mutable Mutex mLock;
    mutable ConditionVariable mCVarNotEmpty; // also used to notify closed
    mutable ConditionVariable mCVarNotFull;
    bool mIsClosed = {false};
};

} //namespace cudapp
