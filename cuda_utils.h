#pragma once

#if defined(ENABLE_CUDA_DRIVER_UTILS) && ENABLE_CUDA_DRIVER_UTILS
#include <cuda.h>
#endif
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <iostream>
#include <cstddef>
#include <cassert>
#include "cpp_utils.h"
#include <type_traits>
#include <CudaDaemon.h>

//for IDE parser
#if defined(Q_CREATOR_RUN) || defined(__CLION_IDE__) || defined (__INTELLISENSE__) || defined(IN_KDEVELOP_PARSER) || defined(__JETBRAINS_IDE__)
#define IS_IN_IDE_PARSER 1
#else
#define IS_IN_IDE_PARSER 0
#endif

class CudaExceptionBase : public std::exception
{
public:
    CudaExceptionBase(const char* file, int line, const char* func)
    :mFilename{file}, mLine{line}, mFuncName{func} {}
    const char* what() const noexcept override = 0;
private:
    const char* mFilename;
    int64_t mLine;
    const char* mFuncName;

    friend std::ostream& operator<<(std::ostream& stream, const CudaExceptionBase& except);
};

inline std::ostream& operator<<(std::ostream& stream, const CudaExceptionBase& except){
    stream << except.mFilename << ':' << except.mLine << ' ' << except.what() << " in " << except.mFuncName << std::endl;
    return stream;
}

class CudaException : public CudaExceptionBase
{
public:
    CudaException(cudaError_t error, const char* file, int line, const char* func)
        : CudaExceptionBase{file, line, func}, mError{error} {}
    const char* what() const noexcept override { return cudaGetErrorName(mError); }
private:
    cudaError_t mError;
};

namespace impl{
inline void cudaCheckImpl(cudaError_t err, const char* file, int line, const char* func){
    if (err != cudaSuccess) {
        throw CudaException(err, file, line, func);
    }
}
} // namespace impl
#define cudaCheck(EXPR) ::impl::cudaCheckImpl((EXPR), __FILE__, __LINE__, __func__)

#if defined(ENABLE_CUDA_DRIVER_UTILS) && ENABLE_CUDA_DRIVER_UTILS
class CuException : public CudaExceptionBase
{
public:
    CuException(CUresult error, const char* file, int line, const char* func)
            : CudaExceptionBase{file, line, func}, mError{error} {}
    const char* what() const noexcept override {
        const char* p{nullptr};
        return cuGetErrorName(mError, &p) == CUDA_SUCCESS ? p : "Failed to retrieve cuda driver error name, the driver has probably shutdown";
    }
private:
    CUresult mError;
};

inline void cuCheckImpl(CUresult err, const char* file, int line, const char* func){
    if (err != CUDA_SUCCESS) { throw CuException(err, file, line, func); }
}
#define cuCheck(EXPR) cuCheckImpl((EXPR), __FILE__, __LINE__, __func__)
#endif

template <typename T1, typename T2>
inline constexpr auto divUp(T1 x, T2 y) -> decltype((x + y - 1) / y)
{
    return (x + y - 1) / y;
}

template <typename T1, typename T2>
inline constexpr auto roundUp(T1 x, T2 y) -> decltype(divUp(x, y) * y)
{
    return divUp(x, y) * y;
}

inline int getCudaDevice() {
    int id;
    cudaCheck(cudaGetDevice(&id));
    return id;
};

template <typename PtrType, cudaError_t(*CudaDeleterAPI)(PtrType)>
struct CudaDeleter
{
    void operator()(PtrType ptr){
        cudaCheck(CudaDeleterAPI(ptr));
    }
};

template <typename PtrType>
using RmPtrType = typename std::remove_pointer<PtrType>::type;

template <typename PtrType, cudaError_t(*CudaDeleterAPI)(PtrType)>
using CudaRes = std::unique_ptr<RmPtrType<PtrType>, CudaDeleter<PtrType, CudaDeleterAPI>>;

// Did some tests. Cost of one cudaEventCreate + one cudaEventDestroy is about 0.3 - 0.4us and does not cause synchronization
// Probably no need for a event recycler. Recycler is faster but this is usually fast enough.
using CudaEvent = CudaRes<cudaEvent_t, &cudaEventDestroy>;
CudaEvent makeCudaEvent(unsigned flags = cudaEventBlockingSync | cudaEventDisableTiming);

struct CudaStreamDeleter
{
    void operator()(cudaStream_t stream) {
        daemon->notifyDestroy(stream);
        cudaCheck(cudaStreamDestroy(stream));
    }
    const std::shared_ptr<cudapp::ICudaDaemon> daemon = cudapp::getCudaDaemon();
};
using CudaStream = CudaRes<cudaStream_t, &cudaStreamDestroy>;
CudaStream makeCudaStream(unsigned flags = cudaStreamDefault /*cudaStreamNonBlocking*/);
// 0 is default, lower value is higher priority.
CudaStream makeCudaStreamWithPriority(int priority = 0, unsigned flags = cudaStreamDefault);

using CudaGraph = CudaRes<cudaGraph_t, &cudaGraphDestroy>;
CudaGraph makeCudaGraph();
using CudaGraphExec = CudaRes<cudaGraphExec_t, &cudaGraphExecDestroy>;
CudaGraphExec instantiateCudaGraph(cudaGraph_t graph);

// Usually used as static member object. Every object gets a strong ref. Use to replace streams that are created, used and destroyed locally.
class WeakHelperSharedCudaStream
{
public:
     std::shared_ptr<CUstream_st> getStream(){
        std::lock_guard<std::mutex> lk(mLock);
        std::shared_ptr<CUstream_st> stream = mStream.lock();
        if (stream == nullptr){
            stream = makeCudaStream();
            mStream = stream;
        }
        return stream;
    }
private:
    mutable std::mutex mLock;
    std::weak_ptr<CUstream_st> mStream;
};

enum class CudaMemType
{
    kPinned,
    kHost [[deprecated]] = kPinned, // deprecated
    kDevice,
    kManaged,
    kSystem // system memory (non-cuda)
};

inline constexpr const char* toStr(CudaMemType memType) {
    switch (memType) {
    case CudaMemType::kPinned: return "kPinned";
    case CudaMemType::kDevice: return "kDevice";
    case CudaMemType::kManaged: return "kManaged";
    case CudaMemType::kSystem: return "kSystem";
    }
    return nullptr;
}

#if __cplusplus >= 201703L
using CudaMemAllocApiType = cudaError_t(*)(void**, size_t);
namespace impl {
struct CudaManagedMemAllocFree
{
    static cudaError_t alloc(void** p, size_t bytes){return cudaMallocManaged(p, bytes);}
};
struct CudaSysMemAllocFree
{
    static cudaError_t alloc(void**p, size_t bytes){
        cudaCheck(cudaDeviceSynchronize());
        *p = std::malloc(bytes);
        return cudaSuccess;
    }
    static cudaError_t free(void* p){
        // This sync is required, as the memory may be used by an async task.
        // cudaFree/cudaFreeHost does this implicitly.
        cudaCheck(cudaDeviceSynchronize());
        std::free(p);
        return cudaSuccess;
    }
};
}
template <CudaMemType memType>
constexpr CudaMemAllocApiType getCudaMemAllocApi(){
    switch(memType)
    {
    case CudaMemType::kPinned: return &cudaMallocHost;
    case CudaMemType::kDevice: return &cudaMalloc;
    case CudaMemType::kManaged: return &impl::CudaManagedMemAllocFree::alloc;
    case CudaMemType::kSystem: return &impl::CudaSysMemAllocFree::alloc;
    }
    return nullptr;
}

using CudaMemDelApiType = cudaError_t(*)(void*);
constexpr CudaMemDelApiType getCudaMemDelApi(CudaMemType memType){
    switch(memType)
    {
    case CudaMemType::kPinned: return &cudaFreeHost;
    case CudaMemType::kDevice:
    case CudaMemType::kManaged: return &cudaFree;
    case CudaMemType::kSystem: return &impl::CudaSysMemAllocFree::free;
    }
    return nullptr;
}
template <typename ElemType, CudaMemType memType>
using CudaMem = std::unique_ptr<ElemType[], CudaDeleter<void*, getCudaMemDelApi(memType)>>;
#else
template <typename ElemType, CudaMemType memType>
class CudaMem : public std::unique_ptr<ElemType[], CudaDeleter<void*, &cudaFree>>{
    using std::unique_ptr<ElemType[], CudaDeleter<void*, &cudaFree>>::unique_ptr;
};
template <typename ElemType>
class CudaMem<ElemType, CudaMemType::kPinned> : public std::unique_ptr<ElemType[], CudaDeleter<void*, &cudaFreeHost>>{
    using std::unique_ptr<ElemType[], CudaDeleter<void*, &cudaFreeHost>>::unique_ptr;
};
#endif

template <typename T> using CudaHostMem = CudaMem<T, CudaMemType::kPinned>;
template <typename T> using CudaDevMem = CudaMem<T, CudaMemType::kDevice>;
template <typename T> using CudaMngMem = CudaMem<T, CudaMemType::kManaged>;


template <typename ElemType = std_byte, CudaMemType memType = CudaMemType::kDevice, bool forcedNonTrivial = false>
CudaMem<ElemType, memType> allocCudaMem(std::size_t nbElems, std_optional<unsigned int> flags = std_nullopt){
    static_assert(forcedNonTrivial || std::is_void<ElemType>::value || (std::is_trivially_constructible<ElemType>::value && std::is_trivially_destructible<ElemType>::value),
        "ElemType must be void or both trivially constructable and destructable");
    void* ptr = nullptr;
    if (nbElems == 0) {
        return CudaMem<ElemType, memType>{nullptr};
    }
    constexpr size_t elemSize = (!std::is_void<ElemType>::value ? sizeof(ElemType) : 1U);
    const size_t nbBytes = elemSize * nbElems;
    switch (memType)
    {
    case CudaMemType::kPinned:
    {
        if (flags == std_nullopt) {
            cudaCheck(cudaMallocHost(&ptr, nbBytes));
        }
        else{
            cudaCheck(cudaMallocHost(&ptr, nbBytes, *flags));
        }
        break;
    }
    case CudaMemType::kDevice:
    {
        assert(flags == std_nullopt);
#if 1
        cudaCheck(cudaMalloc(&ptr, nbBytes));
#else
        cudaCheck(cudaMallocManaged(&ptr, nbBytes));
#endif
        break;
    }
    case CudaMemType::kManaged:
    {
        if (flags == std_nullopt) {
            cudaCheck(cudaMallocManaged(&ptr, nbBytes));
        }
        else {
            cudaCheck(cudaMallocManaged(&ptr, nbBytes, *flags));
        }
        break;
    }
    case CudaMemType::kSystem:
    {
        ptr = malloc(nbBytes);
        break;
    }
    default:
        throw std::logic_error("Invalid memType");
    }
    return CudaMem<ElemType, memType>{static_cast<ElemType*>(ptr)};
}

template <typename T, CudaMemType memType>
struct CudaAllocator {
    static_assert(memType != CudaMemType::kDevice, "Device memory cannot be access from host");
    using value_type    = T;
    template <typename U>
    struct rebind {using other = CudaAllocator<U, memType>;};

    CudaAllocator() noexcept {}
    template <typename U, CudaMemType memType_> CudaAllocator (const CudaAllocator<U, memType_>&) noexcept {}
    T* allocate (std::size_t nbElems) {
        return allocCudaMem<T, memType>(nbElems).release();
    }
    void deallocate (T* p, std::size_t) {
        CudaMem<T, memType>{p}.reset(nullptr);
    }
};
template <typename T, typename U, CudaMemType memType>
constexpr bool operator== (const CudaAllocator<T, memType>&, const CudaAllocator<U, memType>&) noexcept {return true;}
template <typename T, typename U, CudaMemType memType>
constexpr bool operator!= (const CudaAllocator<T, memType>&, const CudaAllocator<U, memType>&) noexcept {return false;}

template <typename T>
using CudaHostAllocator = CudaAllocator<T, CudaMemType::kPinned>;
template <typename T>
using CudaManagedAllocator = CudaAllocator<T, CudaMemType::kManaged>;

void connectStreams(cudaStream_t first, cudaStream_t second);
// pMutex helps make cudaEventRecord+cudaStreamWaitEvent atomic
void connectStreams(cudaStream_t first, cudaStream_t second, cudaEvent_t event, std::mutex* pMutex);

// This event can be recorded multiple times in multiple streams and you can use this event to wait for all previous recording in the same session.
class ICudaMultiEvent
{
public:
    virtual ~ICudaMultiEvent();
    virtual void clear() = 0;
    virtual void recordEvent(cudaStream_t stream) = 0;
    // This stream will wait until all
    virtual void streamWaitEvent(cudaStream_t stream) const = 0;

    virtual void sync() const = 0;

    virtual void scrub() = 0;
    virtual bool query() = 0;
    virtual bool empty() const = 0;
};

std::unique_ptr<ICudaMultiEvent> createCudaMultiEvent(bool isPooled);

template <typename Func>
void launchCudaHostFunc(cudaStream_t stream, Func&& func){
    std::remove_reference_t<Func>* const pFunc = new std::remove_reference_t<Func>{std::forward<Func>(func)};
    cudaCheck(cudaLaunchHostFunc(stream, [](void* p){
        const std::unique_ptr<std::remove_reference_t<Func>> pFunc{static_cast<std::remove_reference_t<Func>*>(p)};
        (*pFunc)();
    }, pFunc));
}

#ifdef __CUDACC__
__device__ __forceinline__ uint32_t lane_id()
{
    uint32_t laneid;
    asm("mov.u32 %0, %%laneid;\n" : "=r"(laneid));
    return laneid;
}
__device__ __forceinline__ void kassert(bool cond) {
    if (!cond) {
        asm volatile("trap;\n");
    }
}
__device__ __forceinline__ float fast_rcp(float x) {return 1.f / x;}
__device__ __forceinline__ float fast_sqrt(float x) {return std::sqrt(x);}
__device__ __forceinline__ float fast_rsqrt(float x) {return rsqrtf(x);}

template <typename T>
__device__ __forceinline__ void prefetchL1(T* p) {
    asm volatile("prefetch.global.L1 [%0];\n" : : "l"(p));
}
template <typename T>
__device__ __forceinline__ void prefetchL1Uniform(T* p) {
    asm volatile("prefetchu.global.L1 [%0];\n" : : "l"(p));
}
#endif

#ifdef __CUDACC__
template <typename... Args>
void launchKernel(void(*kernel)(Args...), dim3 grid, dim3 cta, size_t smem, cudaStream_t stream, const Args&... args)
{
#if IS_IN_IDE_PARSER
    kernel(args...);
#else
    if (grid.x != 0 && grid.y != 0 && grid.z != 0) {
        kernel<<<grid, cta, smem, stream>>>(args...);
    }
#endif
    cudaCheck(cudaGetLastError());
}

template <typename... Args>
void launchKernel(void(*kernel)(Args...), dim3 grid, uint32_t cta, size_t smem, cudaStream_t stream, const Args&... args)
{
    launchKernel(kernel, grid, dim3{cta}, smem, stream, args...);
}

template <typename... Args>
void launchKernel(void(*kernel)(Args...), uint32_t grid, dim3 cta, size_t smem, cudaStream_t stream, const Args&... args)
{
    launchKernel(kernel, dim3{grid}, cta, smem, stream, args...);
}

template <typename... Args>
void launchKernel(void(*kernel)(Args...), uint32_t grid, uint32_t cta, size_t smem, cudaStream_t stream, const Args&... args)
{
    launchKernel(kernel, dim3{grid}, dim3{cta}, smem, stream, args...);
}

#endif

template <typename T, bool isRestrict = true>
struct PitchedPtr
{
    using Elem = T;
    std::conditional_t<isRestrict, T* __restrict__, T*> ptr;
    uint32_t pitch; // in elements

    __host__ __device__ __forceinline__
    T* operator[](uint32_t i) const {
        return ptr + pitch * i;
    }

    __host__ __device__ __forceinline__
    T& operator()(uint32_t i, uint32_t j) const {
        return (*this)[i][j];
    }

    __host__ __device__ __forceinline__
    PitchedPtr<const T> toConst() const {
        return {ptr, pitch};
    }
    __host__ __device__ __forceinline__
    PitchedPtr<volatile T> toVolatile() const {
        return {ptr, pitch};
    }
    __host__ __device__ __forceinline__
    PitchedPtr<const volatile T> toConstVolatile() const {
        return {ptr, pitch};
    }
};

template <typename T, bool isRestrict = true>
struct PitchedPtr3d
{
    using Elem = T;
    std::conditional_t<isRestrict, T* __restrict__, T*>ptr;
    uint32_t pitches[2]; // in elements, little endian

    __host__ __device__ __forceinline__
    PitchedPtr<T> operator[](uint32_t i) const {
        return PitchedPtr<T>{ptr + pitches[1] * i, pitches[0]};
    }

    __host__ __device__ __forceinline__
    T& operator()(uint32_t i, uint32_t j, uint32_t k) const {
        return (*this)[i][j][k];
    }

    __host__ __device__ __forceinline__
    PitchedPtr3d<const T> toConst() const {
        return {ptr, {pitches[0], pitches[1]}};
    }
    __host__ __device__ __forceinline__
    PitchedPtr3d<volatile T> toVolatile() const {
        return {ptr, {pitches[0], pitches[1]}};
    }
    __host__ __device__ __forceinline__
    PitchedPtr3d<const volatile T> toConstVolatile() const {
        return {ptr, {pitches[0], pitches[1]}};
    }
};

namespace cudapp
{
void streamSync(cudaStream_t stream);

inline constexpr int32_t warp_size = 32;

struct HW
{
    int h;
    int w;
    __host__ __device__ __forceinline__
    constexpr HW operator*(const HW other) const {
        return {h * other.h, w * other.w};
    }
    __host__ __device__ __forceinline__
    constexpr HW operator/(const HW other) const {
        return {h / other.h, w / other.w};
    }
    __host__ __device__ __forceinline__
    constexpr HW operator%(const HW other) const {
        return {h % other.h, w % other.w};
    }
    __host__ __device__ __forceinline__
    constexpr HW operator+(const HW other) const {
        return {h + other.h, w + other.w};
    }
    __host__ __device__ __forceinline__
    constexpr HW operator-(const HW other) const {
        return {h - other.h, w - other.w};
    }
    __host__ __device__ __forceinline__
    constexpr int operator[](int idx) const {
        return idx == 0 ? h : w;
    }
};
}


