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

#include <cuda_runtime_api.h>
#include <cstdint>

namespace cudapp
{
template <typename T, typename OrigType = T>
class Ptr
{
public:
    using Offset = uint32_t;
    using PtrDiff = int32_t;
    static_assert(sizeof(Offset) == sizeof(PtrDiff));
    __host__ __device__ __forceinline__ Ptr(OrigType* base, Offset size) : mBase{base}, mOrigSize{size}{assert(size < std::numeric_limits<PtrDiff>::max());}
    __host__ __device__ __forceinline__ Ptr(OrigType* base, Offset size, Offset offset) : Ptr(base, size){mOffset = offset;}
    __host__ __device__ __forceinline__ Ptr(OrigType* base, Offset size, int offset) : Ptr(base, size){mOffset = static_cast<Offset>(offset);}
    __host__ __device__ __forceinline__ bool isInBound() const {
        return sizeof(OrigType) < sizeof(T)
            ? mOffset < mOrigSize / (sizeof(T) / sizeof(OrigType))
            : mOffset / (sizeof(OrigType) / sizeof(T)) < mOrigSize;
    }
    __host__ __device__ __forceinline__ T* get() const {return base<T>() + mOffset;}
    __host__ __device__ __forceinline__ uint32_t getOffset() const {return mOffset;}
    __host__ __device__ __forceinline__ uint32_t getSize() const {return mOrigSize * sizeof(OrigType) / sizeof(T);}
    __host__ __device__ __forceinline__ operator T*() const {return get();}
    template <bool enabler = true, typename = typename std::enable_if<enabler && !std::is_const<T>::value>::type>
    __host__ __device__ __forceinline__ operator Ptr<const T, OrigType>() const {return Ptr<const T, OrigType>{mBase, mOrigSize, mOffset};}
    __host__ __device__ __forceinline__ T& operator*() const {assert(isInBound()); return *get();}
    __host__ __device__ __forceinline__ T& operator[](uint32_t x) const {
        const auto p = *this + x;
        assert(p.isInBound());
        return *p;
    }
    __host__ __device__ __forceinline__ Ptr operator+(uint32_t x) const {
        return Ptr(mBase, mOrigSize, mOffset+x);
    }
    __host__ __device__ __forceinline__ Ptr operator+(int32_t x) const {
        assert(x >= 0);
        return (*this) + static_cast<uint32_t>(x);
    }
    __host__ __device__ __forceinline__ Ptr operator-(uint32_t x) const {
        assert(mOffset >= x);
        return Ptr(mBase, mOrigSize, mOffset-x);
    }
    __host__ __device__ __forceinline__ PtrDiff operator-(const Ptr& r) const {
        assert(mBase == r.mBase);
        const Offset x = mOffset - r.mOffset;
        return reinterpret_cast<const PtrDiff&>(x);
    }
    __host__ __device__ __forceinline__ bool operator<(const Ptr& r) const {assert(mBase == r.mBase); return mOffset < r.mOffset;}
    __host__ __device__ __forceinline__ bool operator<=(const Ptr& r) const {assert(mBase == r.mBase); return mOffset <= r.mOffset;}
    __host__ __device__ __forceinline__ bool operator>(const Ptr& r) const {assert(mBase == r.mBase); return mOffset > r.mOffset;}
    __host__ __device__ __forceinline__ bool operator>=(const Ptr& r) const {assert(mBase == r.mBase); return mOffset >= r.mOffset;}
    __host__ __device__ __forceinline__ bool operator==(const Ptr& r) const {assert(mBase == r.mBase); return mOffset == r.mOffset;}
    __host__ __device__ __forceinline__ bool operator!=(const Ptr& r) const {assert(mBase == r.mBase); return mOffset != r.mOffset;}

    template <typename Dst = T>
    __host__ __device__ __forceinline__ Dst* base() const {return reinterpret_cast<Dst*>(mBase);}
    
    template <typename Dst>
    __host__ __device__ __forceinline__ Ptr<Dst, OrigType> cast() const {
        if (sizeof(Dst) > sizeof(T)) {
            assert(mOffset % (sizeof(Dst) / sizeof(T)) == 0);
        }
        return Ptr<Dst, OrigType>(mBase, mOrigSize,
            sizeof(T) < sizeof(Dst) ? mOffset / (sizeof(Dst) / sizeof(T)) : mOffset * (sizeof(T) / sizeof(Dst)));
    }

private:
    OrigType* const mBase = nullptr;
    const Offset mOrigSize = 0u; // in #OrigType
    Offset mOffset = 0u; // in #T
};
} // namespace cudapp
