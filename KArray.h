//
// Created by yao on 10/18/19.
//

#pragma once
#include <cstdint>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <cassert>
#include <algorithm>

namespace cudapp
{

template <typename T, size_t... size>
struct KArray;

template <typename T, size_t size>
struct alignas(std::min<uint32_t>(alignof(T) * (1u<<__builtin_ctz((uint32_t)size)), 16)) KArray<T, size>
{
    static constexpr uint32_t dimension = size;
    using Elem = T;
    __host__ __device__ __forceinline__
    T& operator[](uint32_t idx) {assert(idx < dimension); return data[idx];}
    __host__ __device__ __forceinline__
    const T& operator[](uint32_t idx) const {assert(idx < dimension); return data[idx];}
    __host__ __device__ __forceinline__
    T* begin() {return &data[0];}
    __host__ __device__ __forceinline__
    T* end() {return &data[dimension];}
    __host__ __device__ __forceinline__
    const T* begin() const {return &data[0];}
    __host__ __device__ __forceinline__
    const T* end() const {return &data[dimension];}

    T data[size];
};

template <typename T>
struct KArray<T, 0>
{
    static constexpr uint32_t dimension = 0;
    using Elem = T;
    __host__ __device__ __forceinline__
    T& operator[](uint32_t idx) {return *reinterpret_cast<T*>(alignof(T));}
    __host__ __device__ __forceinline__
    const T& operator[](uint32_t idx) const {return *reinterpret_cast<const T*>(alignof(T));}
    __host__ __device__ __forceinline__
    T* begin() {return nullptr;}
    __host__ __device__ __forceinline__
    T* end() {return nullptr;}
    __host__ __device__ __forceinline__
    const T* begin() const {return nullptr;}
    __host__ __device__ __forceinline__
    const T* end() const {return nullptr;}
};

template <typename T, size_t dim0, size_t... dims>
struct KArray<T, dim0, dims...> : public KArray<KArray<T, dims...>, dim0>{};

template <typename T, size_t... size>
bool operator!=(const KArray<T, size...>& a, const KArray<T, size...>& b){
    for (uint32_t i = 0; i < a.dimension; i++){
        if (a[i] != b[i]){
            return true;
        }
    }
    return false;
}

template <typename T, size_t... size>
bool operator==(const KArray<T, size...>& a, const KArray<T, size...>& b){
    return !(a != b);
}

} // namespace cudapp
