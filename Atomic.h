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

// The purpose of this is to provide a atomic class that share lock among many objects.
// The most useful case is that when there are many objects, the chance for lock contention
// is low. It would be nice to make each mutex protect multiple objects. When contention is
// low, this should not impact performance and reduce the amount of mutex we need.

#pragma once
#include <shared_mutex>
#include <vector>

class SharedMutexPool
{
public:
    static SharedMutexPool& instance() {
        static SharedMutexPool pool{8192};
        return pool;
    }
    template <typename Key>
    std::shared_mutex& getLock(Key key) {
        return std::hash<Key>{}(key) % mLocks.size();
    }
private:
    explicit SharedMutexPool(size_t nbLocks) :mLocks(nbLocks){};
private:
    std::vector<std::shared_mutex> mLocks;
};

template <typename T>
class Atomic
{
public:
    template <typename Args...>
    Atomic(Args&&... args) : mData{std::forward<Args>(args)...} {}
    Atomic(T&& src) : mData{[]()->T{
        std::lock_guard<std::shared_mutex> lk{src.getLock()};
        return T{std::move(src.mData)};
    }()}{}
    Atomic(const T& src) : mData{src.load()} {}
    T load() const {
        std::shared_lock<std::shared_mutex> lk{getLock()};
        return mData;
    }
    void store(T src) {
        std::lock_guard<std::shared_mutex> lk{getLock()};
        mData = std::move(src);
    }
private:
    std::uintptr_t key() const {
        return reinterpret_cast<std::uintptr_t>(&mData) % alignof(mData);
    }
    std::shared_mutex& getLock() const {
        return SharedMutexPool::instance().getLock(key());
    }
private:
    T mData;
};

