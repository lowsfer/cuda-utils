#include "checksum.h"
#include <boost/uuid/detail/md5.hpp>
#include <boost/uuid/detail/sha1.hpp>
#include <fstream>
#include <filesystem>

namespace {
// for C++20 std::to_array
template <typename T, size_t size>
constexpr std::array<T, size> toArray(const T(&src)[size]) {
    std::array<T, size> dst{};
    std::copy_n(src, size, dst.begin());
    return dst;
}

template <typename Algo>
auto checksum(const void* data, size_t size)
{
    Algo hash{};
    hash.process_bytes(data, size);
    typename Algo::digest_type digest;
    hash.get_digest(digest);
    return toArray(digest);
}

template <typename Algo>
auto checksum(const char* filename)
{
    Algo hash{};
    std::ifstream fin;
    fin.exceptions(std::ios::badbit);
    fin.open(filename, std::ios::binary);
    std::vector<char> buffer(256<<10);
    while (fin.good()) {
        const auto nbBytes = fin.read(buffer.data(), buffer.size()).gcount();
        hash.process_bytes(buffer.data(), nbBytes);
    }
    typename Algo::digest_type digest;
    hash.get_digest(digest);
    return toArray(digest);
}
}

namespace cudapp {

std::array<uint32_t, 4> md5sum(const void* data, size_t size)
{
    return checksum<boost::uuids::detail::md5>(data, size);
}

std::array<uint32_t, 4> md5sum(const char* filename)
{
    return checksum<boost::uuids::detail::md5>(filename);
}

std::array<uint32_t, 5> sha1sum(const void* data, size_t size)
{
    return checksum<boost::uuids::detail::sha1>(data, size);
}
std::array<uint32_t, 5> sha1sum(const char* filename)
{
    return checksum<boost::uuids::detail::sha1>(filename);
}

} // namespace cudapp
