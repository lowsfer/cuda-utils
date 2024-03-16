#pragma once
#include <vector>
#include <stdexcept>
#include "cpp_utils.h"

//@fixme: use deque and add offset for cases where keys does not start from 0
template <typename Key_, typename Value_>
class VectorMap
{
public:
    using Key = Key_;
    using Value = Value_;
    using Index = std::conditional_t<std::is_enum_v<Key>, UnderlyingType<Key>, Key>;
    static Index key2Index(Key key){
        const Index idx = static_cast<Index>(key);
        if constexpr (std::is_signed_v<Index>) {
            REQUIRE(idx >= 0);
        }
        return idx;
    }

    static_assert(std::is_integral_v<Index>);
    using key_type = Key;
    using value_type = Value;
    template <bool isConst>
    class IteratorImpl
    {
    public:
        using Container = std::conditional_t<isConst, const VectorMap<Key, Value>, VectorMap<Key, Value>>;
        IteratorImpl(Container* container, Index key): mContainer{container}, mStorageIndex{key}{
            if constexpr (std::is_signed_v<Index>) {
                REQUIRE(key >= 0);
            }
        }

        IteratorImpl<isConst>& operator++(){ // prefix
            if (static_cast<size_t>(mStorageIndex) >= mContainer->getStorageSize()){
                return *this;
            }
            const auto findBeg = mContainer->mMask.begin() + mStorageIndex + 1;
            const auto iterNext = std::find(findBeg, mContainer->mMask.end(), true);
            mStorageIndex = iterNext - mContainer->mMask.begin();
            return *this;
        }
        IteratorImpl<isConst> operator++(int){ // postfix
            IteratorImpl<isConst> ret = *this;
            ++(*this);
            return ret;
        }
        bool operator==(const IteratorImpl<isConst>& other) const {
            return mContainer == other.mContainer && key() == other.key();
        }
        bool operator!=(const IteratorImpl<isConst>& other) const {
            return !(*this == other);
        }
        std::conditional_t<isConst, const Value&, Value&> operator*() const{
            assert(mContainer->mMask.at(mStorageIndex));
            return mContainer->at(key());
        }
        Key key() const {return static_cast<Key>(mStorageIndex);}
        friend class VectorMap<Key, Value>;
    private:
        Container* mContainer;
        Index mStorageIndex; // the key
    };

    using iterator = IteratorImpl<false>;
    using const_iterator = IteratorImpl<true>;

    VectorMap() = default;
    VectorMap(const VectorMap<Key, Value>&) = delete;
    VectorMap<Key, Value>& operator=(const VectorMap<Key, Value>&) = delete;
    void swap(VectorMap& other){
        std::swap(mData, other.mData);
        std::swap(mMask, other.mMask);
        std::swap(mSize, other.mSize);
        std::swap(mStorageSize, other.mStorageSize);
    }
    VectorMap(VectorMap<Key, Value>&& other) {
        this->swap(other);
    }
    VectorMap<Key, Value>& operator=(VectorMap<Key, Value>&& other){
        this->swap(other);
        other.clear();
        return *this;
    }
    ~VectorMap(){
        clear();
    }
    const Value& at(Key key) const {
        const Index idx = key2Index(key);
        if (idx >= static_cast<Index>(getStorageSize()) || !mMask.at(idx)){
            throw std::out_of_range(FILELINE);
        }
        assert(mMask.size() == mData.size() && getStorageSize() == mData.size());
        return reinterpret_cast<const Value&>(mData[idx]);
    }
    Value& at(Key key) {
        return const_cast<Value&>(static_cast<const VectorMap<Key, Value>*>(this)->at(key));
    }
    Key getKey(const Value& v) const {
        if (empty()) {
            throw std::out_of_range("empty container");
        }
        const auto idx = &v - &reinterpret_cast<const Value&>(mData[0]);
        if (idx < 0 || static_cast<size_t>(idx) >= getStorageSize() || !mMask.at(idx)) {
            throw std::out_of_range("No such item in the container");
        }
        return static_cast<Key>(static_cast<Index>(idx));
    }
    template <typename... Args>
    std::pair<iterator, bool> try_emplace(Key key, Args&&... args) {
        const size_t idx = static_cast<size_t>(key2Index(key));
        if (idx < getStorageSize() && mMask.at(idx)){
            return std::make_pair(iterator{this, static_cast<Index>(idx)}, false);
        }
        if (idx >= getStorageSize()){
            if (idx < mData.capacity()){
                mData.insert(mData.end(), idx + 1 - getStorageSize(), StorageType{});
            }
            else {
                std::vector<StorageType> newData;
                newData.reserve(mData.capacity() * 2);
                newData.resize(idx + 1);
                for (Index i = 0; i < static_cast<Index>(getStorageSize()); i++){
                    if (mMask[i]){
                        new(&newData[i]) Value{std::move(at(static_cast<Key>(i)))};
                    }
                }
                mData = std::move(newData);
            }
            assert(mStorageSize == mMask.size());
            mMask.insert(mMask.end(), idx + 1 - mMask.size(), false);
            mStorageSize = idx + 1;
            assert(mData.size() == idx + 1);
        }
        new (&mData.at(idx)) Value{std::forward<Args>(args)...};
        mMask.at(idx) = true;
        mSize++;
        return std::pair(iterator{this, static_cast<Index>(idx)}, true);
    }
    Value& operator[](Key key){
        try_emplace(key);
        return at(key);
    }

    size_t size() const {return mSize;}
    bool empty() const {return size() == 0u;}

    inline size_t getStorageSize() const {
        assert(mData.size() == mMask.size() && mData.size() == mStorageSize);
        return mStorageSize;
    }

    const_iterator begin() const {
        const auto iterMask = std::find(mMask.begin(), mMask.end(), true);
        return const_iterator{this, static_cast<Index>(iterMask - mMask.begin())};
    }
    iterator begin() {
        return iterator{this, static_cast<const VectorMap<Key, Value>*>(this)->begin().mStorageIndex};
    }
    const_iterator end() const {
        return const_iterator{this, static_cast<Index>(getStorageSize())};
    }
    iterator end() {
        return iterator{this, static_cast<const VectorMap<Key, Value>*>(this)->end().mStorageIndex};
    }

    size_t erase(Key key) {
        const Index idx = key2Index(key);
        if (static_cast<size_t>(idx) < getStorageSize() && mMask[idx]) {
            at(key).~Value();
            mData[idx] = StorageType{};
            mMask[idx] = false;
            mSize--;
            return 1;
        }
        return 0;
    }

    void clear() {
        if (!std::is_trivially_destructible<Value>::value) {
            const auto storageSize = static_cast<Index>(getStorageSize());
            for (Index i = 0; i < storageSize; i++){
                if (mMask[i]) {
                    this->at(static_cast<Key>(i)).~Value();
                }
            }
        }
        mData.clear();
        mMask.clear();
        mSize = 0;
        mStorageSize = 0;
        assert(size() == 0);
    }

    const_iterator find(const Key& key) const {
        const Index idx = key2Index(key);
        if (idx >= getStorageSize() || !mMask.at(idx)){
            return end();
        }
        return const_iterator{this, idx};
    }

    iterator find(const Key& key) {
        const Index idx = key2Index(key);
        if (idx >= getStorageSize() || !mMask.at(idx)){
            return end();
        }
        return iterator{this, idx};
    }

    bool has(const Key& key) const {
        const Index idx = key2Index(key);
        return idx < getStorageSize() && mMask.at(idx);
    }

    size_t count(const Key& key) const {
        return has(key) ? 1UL : 0UL;
    }

    void reserve(size_t space) {
        mData.reserve(space);
        mMask.reserve(space);
    }
private:
    static constexpr int mOffset = 0;
    struct alignas(alignof(Value)) StorageType{std_byte data[sizeof(Value)];};
    std::vector<StorageType> mData;
    std::vector<bool> mMask;
    size_t mSize{0};
    size_t mStorageSize{0};
};
