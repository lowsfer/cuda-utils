#pragma once
#include <array>
#include <cstdint>
#include <cstddef>

namespace cudapp{
std::array<uint32_t, 4> md5sum(const void* data, size_t size);
std::array<uint32_t, 4> md5sum(const char* filename);

std::array<uint32_t, 5> sha1sum(const void* data, size_t size);
std::array<uint32_t, 5> sha1sum(const char* filename);
}
