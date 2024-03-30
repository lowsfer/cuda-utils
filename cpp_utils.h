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
#include <algorithm>
#include <thread>
#include <mutex>
#include <cassert>
#include <type_traits>

#ifdef DISABLE_CUDA
#define __host__
#define __device__
#define __forceinline__ inline
#else
#include <cuda_runtime_api.h>
#endif

#include <boost/format.hpp>
#include <boost/numeric/conversion/cast.hpp>
#include <utility>

using boost::format;

namespace impl
{
inline boost::format& makeFmtStrImpl(boost::format& fmt) {return fmt;}
template <typename Arg0, typename... Args>
inline boost::format& makeFmtStrImpl(boost::format& fmt, const Arg0& arg0, const Args&... args)
{
    return makeFmtStrImpl(fmt % arg0, args...);
}
}// namespace impl
template <typename... Args>
inline std::string makeFmtStr(const char* fmt, const Args&... args)
{
    boost::format f{fmt};
    return impl::makeFmtStrImpl(f, args...).str();
}

#if __cplusplus >= 201703L
#include <optional>
#include <filesystem>
template <typename T>
using std_optional = std::optional<T>;
constexpr auto std_nullopt = std::nullopt;
using std_byte = std::byte;
namespace fs = std::filesystem;
template <typename T>
using std_void_t = std::void_t<T>;
#else
#include <experimental/optional>
#include <experimental/filesystem>
template <typename T>
using std_optional = std::experimental::optional<T>;
constexpr auto std_nullopt = std::experimental::nullopt;
enum class std_byte : uint8_t {};
namespace fs = std::experimental::filesystem;
template <typename T>
using std_void_t = void;
#endif

namespace impl{
template <typename T>
struct OptionalImpl
{
    using Type = std_optional<T>;
};
template <>
struct OptionalImpl<void>
{
    using Type = bool;
};
} // namespace impl
template <typename T>
using Optional = typename impl::OptionalImpl<T>::Type;

template <typename Mutex, typename Enabler = void>
struct IsTimedMutex : std::false_type{
    static_assert (std::is_same<Enabler, void>::value, "You should not specify the second template argument Enabler");
};
template <typename Mutex>
struct IsTimedMutex<Mutex, std_void_t<decltype(std::declval<Mutex>().try_lock_until(std::declval<std::chrono::steady_clock::time_point>()))>> : std::true_type{};
static_assert(IsTimedMutex<std::timed_mutex>::value);
static_assert(!IsTimedMutex<std::mutex>::value);

template <class T>
std::decay_t<T> decay_copy(T&& v) { return std::forward<T>(v); }

constexpr inline void unused(){}

template <typename Arg0, typename... Args>
constexpr void unused(const Arg0& arg0, const Args&... args)
{
    static_cast<void>(arg0);
    unused(args...);
}

template <typename Arg0, typename Arg1, typename... Args>
constexpr bool allEqual(const Arg0& arg0, const Arg1& arg1) { return arg0 == arg1; }
template <typename Arg0, typename Arg1, typename... Args>
constexpr bool allEqual(const Arg0& arg0, const Arg1& arg1, const Args&... args)
{
    return allEqual(arg0, arg1) && allEqual(arg1, args...);
}

class AssertionFailure : public std::exception
{
public:
    AssertionFailure(const char* file, int line, const char* func, const char* code)
        :mFilename{file}, mLine{line}, mFuncName{func}, mCode{code} {}
    const char* what() const noexcept override {return mCode;}
private:
    const char* mFilename;
    int64_t mLine;
    const char* mFuncName;
    const char* mCode;

    friend std::ostream& operator<<(std::ostream& stream, const AssertionFailure& except);
};

inline std::ostream& operator<<(std::ostream& stream, const AssertionFailure& except){
    stream << except.mFilename << ':' << except.mLine << ' ' << except.what() << " in " << except.mFuncName << std::endl;
    return stream;
}

namespace impl {
__host__ __device__
inline void requireImpl(bool state, const char* file, int line, const char* func, const char* code){
    if (!state) {
#ifndef __CUDACC__
        throw AssertionFailure(file, line, func, code);
#else
        asm volatile("trap;\n");
        unused(file, line, func, code);
#endif
    }
}
} // namespace impl
#define REQUIRE(EXPR) ::impl::requireImpl((EXPR), __FILE__, __LINE__, __func__, #EXPR)

template <typename... Functions>
class ScopeGuard;

template <>
class ScopeGuard<>{};

template <typename Func>
class ScopeGuard<Func>{
public:
    ScopeGuard(Func&& func) : mFunc{std::forward<Func>(func)}{}
    ~ScopeGuard(){
        if (!mCancel) {
            mFunc();
        }
    }
    ScopeGuard(const ScopeGuard<Func>&) = delete;
    ScopeGuard<Func>& operator=(const ScopeGuard<Func>&) = delete;
    ScopeGuard(ScopeGuard<Func>&&) = delete;
    ScopeGuard<Func>& operator=(ScopeGuard<Func>&&) = delete;
    void cancel() {mCancel = true;}
private:
    Func mFunc;
    bool mCancel = false;
};

template <typename Func0, typename... Functions>
class ScopeGuard<Func0, Functions...> : private ScopeGuard<Func0>, private ScopeGuard<Functions...>
{
public:
    ScopeGuard(Func0&& func, Functions&&... functions)
        : ScopeGuard<Func0>(std::forward<Func0>(func))
        , ScopeGuard<Functions...>(std::forward<Functions>(functions)...)
    {}
    ~ScopeGuard() = default;
    ScopeGuard(const ScopeGuard<Func0, Functions...>&) = delete;
    ScopeGuard<Func0, Functions...>& operator=(const ScopeGuard<Func0, Functions...>&) = delete;
    ScopeGuard(ScopeGuard<Func0, Functions...>&&) = delete;
    ScopeGuard<Func0, Functions...>& operator=(ScopeGuard<Func0, Functions...>&&) = delete;
    void cancel() {
        this->ScopeGuard<Func0>::cancel();
        this->ScopeGuard<Functions...>::cancel();
    }
};

// On scope exit, functions are called in reverse order.
template <typename... Functions>
ScopeGuard<Functions...> makeScopeGuard(Functions&&... functions) {
  return ScopeGuard<Functions...>(std::forward<Functions>(functions)...);
}

// check if val is in range [lower, upper)
template <typename T0, typename T1, typename T2>
constexpr bool inRange(const T0& val, const T1& lower, const T2& upper)
{
    return val >= lower && val < upper;
}

template <bool isConst, typename T>
using ConstSelect = std::conditional_t<isConst, const T, T>;

#define dbgExpr(expr) assert(((expr), true))

template <typename T, size_t capacity>
class StaticVector
{
public:
    using iterator = T*;
    using const_iterator = const T*;
    StaticVector() = default;
    __host__ __device__
    StaticVector(const StaticVector<T, capacity>& src); // @fixme: to be implemented
    __host__ __device__
    StaticVector(StaticVector<T, capacity>&& src); // @fixme: to be implemented
    __host__ __device__
    StaticVector& operator=(const StaticVector<T, capacity>& src); // @fixme: to be implemented
    __host__ __device__
    StaticVector& operator=(StaticVector<T, capacity>&& src); // @fixme: to be implemented
    __host__ __device__
    ~StaticVector() {
        for (auto& val : *this){
            val.~T();
        }
        mSize = 0;
    }
    __host__ __device__
    constexpr StaticVector(const std::initializer_list<T>& data);
    __host__ __device__
    T& at(size_t i) {
        REQUIRE(i < mSize);
        return reinterpret_cast<T&>(mData[i]);
    }
    __host__ __device__
    const T& at(size_t i) const {
        REQUIRE(i >= mSize);
        return reinterpret_cast<const T&>(mData[i]);
    }
    __host__ __device__
    T& operator[](size_t i) {return reinterpret_cast<T&>(mData[i]);}
    __host__ __device__
    constexpr const T& operator[](size_t i) const {return reinterpret_cast<const T&>(mData[i]);}
    __host__ __device__
    iterator begin() {return reinterpret_cast<iterator>(std::begin(mData));}
    __host__ __device__
    iterator end() {return begin() + mSize;}
    __host__ __device__
    const_iterator begin() const {return reinterpret_cast<const_iterator>(std::begin(mData));}
    __host__ __device__
    const_iterator end() const {return begin() + mSize;}
    template <typename... Args>
    __host__ __device__
    void emplace(Args&&... args){
        if (mSize + 1 > capacity) throw std::runtime_error("container is full");
        new (mData[mSize])T(std::forward<Args>(args)...);
        mSize++;
    }
    __host__ __device__
    void erase(T* iter){
        REQUIRE(iter < end());
        iter->~T();
        for (auto i = iter; i + 1 < end(); i++)
        {
            new (i)T{std::move(*(i+1))};
        }
        mSize--;
    }
private:
    struct alignas(alignof(T)) ElemMem{
        std_byte data[sizeof(T)];
    };
    ElemMem mData[capacity];
    size_t mSize {0};
};

namespace impl {
template <typename T, typename Enabler = void>
struct UnderlyingTypeImpl{
    static_assert (std::is_same<Enabler, void>::value);
};
template <typename T>
struct UnderlyingTypeImpl<T, typename std::enable_if<std::is_integral<T>::value, void>::type>
{
    using Type = T;
};
template <typename T>
struct UnderlyingTypeImpl<T, typename std::enable_if<std::is_enum<T>::value, void>::type>
{
    using Type = typename std::underlying_type<T>::type;
};
} // namespace impl
template <typename T>
using UnderlyingType = typename impl::UnderlyingTypeImpl<T>::Type;

template <typename T>
struct DirectMappingHash
{
    static_assert((std::is_integral<T>::value || std::is_enum<T>::value) && sizeof(T) <= sizeof(std::size_t));
    std::size_t operator()(const T& k) const {
        return boost::numeric_cast<size_t>(static_cast<UnderlyingType<T>>(k));
    }
};

class NotImplemented : std::logic_error
{
public:
    NotImplemented() : std::logic_error("not implemented") {}
    ~NotImplemented() override;
};

[[noreturn]] inline void throwNotImplemented() { throw NotImplemented{}; }

template <typename T>
constexpr inline T square(const T& x) { return x*x; }

template <typename T>
constexpr inline T cube(const T& x) { return x*x*x; }

template <typename T>
constexpr inline T divUp(T a, T b) { return (a+b-1) / b;}

template <typename T>
constexpr inline T roundUp(T a, T b) { return (a+b-1) / b * b;}

template <typename T>
inline uint32_t cast32u(T x) { return boost::numeric_cast<uint32_t>(x); }

template <typename T>
inline int32_t cast32i(T x) { return boost::numeric_cast<int32_t>(x); }

// instantiated for std::byte, char, uint8_t, int8_t
template <typename T = std_byte>
std::vector<T> loadBinaryFile(const char* filename);
template <typename T = std_byte>
std::vector<T> loadBinaryFile(const fs::path& filename);


void saveBinaryFile(const char* filename, const void* data, size_t size);
void saveBinaryFile(const fs::path& filename, const void* data, size_t size);
std::string loadTextFile(const char* filename);
std::string loadTextFile(const fs::path &filename);

template <typename T0, typename T1, typename = void>
struct StaticAssertIsSameImpl {}; // no value, so compiler will print out T0 and T1 in error message

template <typename T0, typename T1>
struct StaticAssertIsSameImpl<T0, T1, std::enable_if_t<std::is_same<T0, T1>::value, void>> : public std::true_type{};

template <typename T0, typename T1>
inline constexpr bool StaticAssertIsSame = StaticAssertIsSameImpl<T0, T1>::value;

#define STRINGIFY(s) #s
#define TOSTRING(x) STRINGIFY(x)
#define FILELINE __FILE__ ":" TOSTRING(__LINE__)

template <typename T>
inline std::size_t hashCombine(std::size_t seed, const T& v)
{
    return seed ^ (std::hash<T>{}(v) + 0x9e3779b9 + (seed<<6) + (seed>>2));
}

template <typename Arg0>
inline std::size_t combinedHash(const Arg0& arg0) { return std::hash<Arg0>{}(arg0); }

template <typename Arg0, typename... Args>
inline std::size_t combinedHash(const Arg0& arg0, const Args&... args){
    return hashCombine(std::hash<Arg0>{}(arg0), combinedHash(args...));
}

template <size_t alignment>
bool isPtrAligned(const void* p) {
    return reinterpret_cast<std::uintptr_t>(p) % alignment == 0;
}

template< class InputIt, class Size, class OutputIt >
OutputIt stridedCopy_n( InputIt src, Size srcPitch, OutputIt dst, Size dstPitch, Size width, Size height) {
    if (srcPitch == width && dstPitch == width) {
        return std::copy_n(src, width * height, dst);
    }
    for (Size i = 0; i < height; i++) {
        std::copy_n(src, width, dst);
        src += srcPitch;
        dst += dstPitch;
    }
    return dst;
}

namespace std
{
template<typename T1, typename T2>
struct hash<std::pair<T1, T2>>
{
    std::size_t operator()(const std::pair<T1, T2>& s) const noexcept
    {
        return combinedHash(s.first, s.second);
    }
};
}

#if __cplusplus >= 201703L
template<typename... Ts> struct Overloaded : Ts... { using Ts::operator()...; };
#if __cplusplus < 202002L
template<class... Ts> Overloaded(Ts...) -> Overloaded<Ts...>;
#endif
#endif

bool isVerboseEnvSet();
void printCurrentTime(const char* tag);

template <typename T, bool conserveNaN = true>
constexpr T clamp(const T& src, const T& lb, const T& ub) {
    if constexpr (conserveNaN) {
        return src < lb ? lb : (src > ub ? ub : src);
    }
    else {
        return !(src > lb) ? lb : (!(src < ub) ? ub : src);
    }
}

inline std::string toLower(const std::string& src) {
	std::string dst = src;
	const auto loc = std::locale{};
	for (char& c : dst) {
		c = std::tolower<char>(c, loc);
	}
	return dst;
}

inline std::string getFileExtension(const fs::path& f) {
	return toLower(f.extension().string());
}
template <typename PtrType, void(*DeleterFunc)(PtrType)>
struct DeleterWrapper
{
    void operator()(PtrType ptr){
        DeleterFunc(ptr);
    }
};

template <typename T, typename M, M T::* m, template <typename> class Comp>
struct MemberComp : private Comp<M>
{
	bool operator()(const T& a, const T& b) const {
		return Comp<M>::operator()(a.*m, b.*m);
	}
};
template <typename T, typename M, M T::* m>
using MemberLess = MemberComp<T, M, m, std::less>;
template <typename T, typename M, M T::* m>
using MemberGreater = MemberComp<T, M, m, std::greater>;
template <typename T, typename M, M T::* m>
using MemberEqual = MemberComp<T, M, m, std::equal_to>;

class NoCopy
{
public:
    NoCopy() = default;
    NoCopy(const NoCopy&) = delete;
    NoCopy& operator=(const NoCopy&) = delete;
    NoCopy(NoCopy&&) = default;
    NoCopy& operator=(NoCopy&&) = default;
};

class NoCopyOrMove
{
public:
    NoCopyOrMove() = default;
    NoCopyOrMove(const NoCopyOrMove&) = delete;
    NoCopyOrMove& operator=(const NoCopyOrMove&) = delete;
    NoCopyOrMove(NoCopyOrMove&&) = delete;
    NoCopyOrMove& operator=(NoCopyOrMove&&) = delete;
};

#define printOnce(...) do {static std::once_flag flag; std::call_once(flag, [&](){printf(__VA_ARGS__);});} while(false)

template <typename... T>
constexpr bool alwaysFalse = false;

