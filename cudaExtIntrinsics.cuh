#pragma once

#include "cuda_hint.cuh"

#include <cuda_runtime.h>
#include <cstdint>

namespace cudapp
{
__device__ __forceinline__ uint8_t atomicIncOne(uint8_t* p) {
    const std::uintptr_t uintp = reinterpret_cast<const std::uintptr_t&>(p);
    const uint offset = uintp % 4;
    const std::uintptr_t base = uintp - offset;
    static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__);
    const uint32_t old = atomicAdd(reinterpret_cast<uint32_t*>(base), 1u << (8*offset));
    const uint32_t oldU8 = (old >> 8*offset) & 0xFF;
    assert(oldU8 < 255u); // when overflow we get wrong result
    return static_cast<uint8_t>(oldU8);
}
__device__ __forceinline__ uint8_t atomicDecOne(uint8_t* p) {
    const std::uintptr_t uintp = reinterpret_cast<const std::uintptr_t&>(p);
    const uint offset = uintp % 4;
    const std::uintptr_t base = uintp - offset;
    static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__);
    const uint32_t old = atomicSub(reinterpret_cast<uint32_t*>(base), 1u << (8*offset));
    const uint32_t oldU8 = (old >> 8*offset) & 0xFF;
    assert(oldU8 > 0); // when overflow we get wrong result
    return static_cast<uint8_t>(oldU8);
}

}