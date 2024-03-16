//
// Created by yao on 9/7/19.
//

#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include "ptr.h"

namespace cudapp
{

template <size_t size>
struct LdgType;

template <>
struct LdgType<1>
{
    using Type = uint8_t;
};
template <>
struct LdgType<2>
{
    using Type = uint16_t;
};
template <>
struct LdgType<4>
{
    using Type = uint32_t;
};
template <>
struct LdgType<8>
{
    using Type = float2;
};
template <>
struct LdgType<16>
{
    using Type = float4;
};

// Note that ldg() may cause usage of stack memory. Use it with care.
template <typename T>
__device__ __forceinline__
T ldg(const T* __restrict__ ptr)
{
    static_assert(sizeof(T) == alignof(T));
    constexpr uint32_t loadSize = unsigned(sizeof(T));
    typename LdgType<loadSize>::Type buffer = __ldg(reinterpret_cast<const typename LdgType<loadSize>::Type*>(ptr));
    static_assert(sizeof(T) == sizeof(buffer), "fatal error");
    const T result = reinterpret_cast<const T&>(buffer);
    return result;
}

template <typename T, typename OrigType>
__device__ __forceinline__
T ldg(const Ptr<T, OrigType>& ptr)
{
    return ldg(ptr.get());
}

} // namespace cudapp
