#include "cpp_utils.h"
#include <fstream>
#include <ctime>
#include <chrono>
#include <iostream>

template <typename T>
std::vector<T> loadBinaryFile(const fs::path& filename)
{
    return loadBinaryFile<T>(filename.string().c_str());
}
template <typename T>
std::vector<T> loadBinaryFile(const char* filename) {
    std::ifstream fin;
    fin.exceptions(std::ios::badbit | std::ios::failbit | std::ios::eofbit);
    fin.open(filename, std::ios::binary);
    fin.seekg (0, fin.end);
    const size_t nbBytes = fin.tellg();
    fin.seekg (0, fin.beg);
    std::vector<T> content(divUp(nbBytes, sizeof(T)));
    char* byteData = reinterpret_cast<char*>(content.data());
    fin.read(byteData, static_cast<std::streamsize>(nbBytes));
    REQUIRE(fin.good());
    std::fill(byteData + nbBytes, reinterpret_cast<char*>(content.data() + content.size()), char{0});
    return content;
}
#if __cplusplus >= 201703L
template std::vector<std::byte> loadBinaryFile(const fs::path& filename);
#endif
template std::vector<char> loadBinaryFile(const fs::path& filename);
template std::vector<uint8_t> loadBinaryFile(const fs::path& filename);
template std::vector<int8_t> loadBinaryFile(const fs::path& filename);

void saveBinaryFile(const fs::path& filename, const void* data, size_t size) {
    saveBinaryFile(filename.string().c_str(), data, size);
}
void saveBinaryFile(const char* filename, const void* data, size_t size)
{
    std::ofstream fout;
    fout.exceptions(std::ios::badbit | std::ios::failbit);
    fout.open(filename, std::ios::binary);
    fout.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    REQUIRE(fout.good());
}

std::string loadTextFile(const fs::path &filename) {
    return loadTextFile(filename.string().c_str());
}
std::string loadTextFile(const char* filename)
{
    std::ifstream fin;
    fin.exceptions(std::ios::badbit | std::ios::failbit | std::ios::eofbit);
    fin.open(filename);
    return std::string{std::istreambuf_iterator<char>{fin}, std::istreambuf_iterator<char>{}};
}

NotImplemented::~NotImplemented() = default;

bool isVerboseEnvSet()
{
    static const bool verbose = [](){
        const char* verbose = std::getenv("VERBOSE");
        return verbose != nullptr && std::stoi(verbose) == 1;
    }();
    return verbose;
}

void printCurrentTime(const char* tag)
{
    using namespace std::chrono;

    // get current time
    auto now = system_clock::now();

    // get number of milliseconds for the current second
    // (remainder after division into seconds)
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

    // convert to std::time_t in order to convert to std::tm (broken time)
    auto timer = system_clock::to_time_t(now);

    // convert to broken time
    std::tm bt = *std::localtime(&timer);

    std::cout << std::put_time(&bt, "%H:%M:%S"); // HH:MM:SS
    std::cout << '.' << std::setfill('0') << std::setw(3) << ms.count() << '\t' << (tag != nullptr ? tag : "") << std::endl;
}
