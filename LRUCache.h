#pragma once
#include <unordered_map>
#include <list>
#include <memory>
#include <cassert>
#include <optional>

namespace cudapp {
template <typename Key, typename Value>
class LRUCache
{
public:
    LRUCache(size_t capacity) : mCapacity{capacity} {}
    std::optional<Value> get(const Key& key) {
        std::lock_guard<std::mutex> lk{mMutex};
        const auto iterMap = mMap.find(key);
        if (iterMap == mMap.end()) {
            return std::nullopt;
        }
        assert(iterMap->first == key);
        const auto iterEntry = iterMap->second;
        assert(iterEntry->first == key);
        mEntries.splice(mEntries.end(), mEntries, iterEntry);
        return iterEntry->second;
    }
    void put(const Key& key, Value value) {
        std::lock_guard<std::mutex> lk{mMutex};
        mEntries.emplace_back(key, std::move(value));
        mMap.try_emplace(key, std::prev(mEntries.end()));
        while (mEntries.size() > mCapacity) {
            const auto& key = mEntries.front().first;
            assert(mMap.at(key) == mEntries.begin());
            invalidateImpl(key);
        }
    }
    void invalidate(const Key& key) {
        std::lock_guard<std::mutex> lk{mMutex};
        invalidateImpl(key);
    }
    void clear() {
        std::lock_guard<std::mutex> lk{mMutex};
        mEntries.clear();
        mMap.clear();
    }
private:
    void invalidateImpl(const Key& key) {
        const typename std::list<Entry>::iterator iterEntry = mMap.at(key);
        assert(iterEntry->first == key);
        mMap.erase(key);
        mEntries.erase(iterEntry);
    }
private:
    mutable std::mutex mMutex;
    size_t mCapacity;
    using Entry = std::pair<Key, Value>;
    // old to new
    std::list<Entry> mEntries;
    std::unordered_map<Key, typename std::list<Entry>::iterator> mMap;
};
} // namespace cudapp
