cmake_minimum_required(VERSION 3.5)

project(cuda-utils LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 75-real 80-real 86-real 90-real 75-virtual)

find_package(Boost REQUIRED COMPONENTS fiber)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CMAKE_SOURCE_DIR})
link_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfatal-errors -Wreturn-type -Wall -Wextra -Wno-unknown-pragmas -Werror -Wparentheses")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -pthread -Wfatal-errors -Wreturn-type -Wall -Wextra -Wno-unknown-pragmas")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -keep --expt-relaxed-constexpr") #-res-usage
set(CUDA_PTXAS_FLAGS_RELEASE "${CUDA_PTXAS_FLAGS_RELEASE} -warn-lmem-usage -warn-spills -Werror") # -v
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} --use_fast_math -DNDEBUG -Xptxas='${CUDA_PTXAS_FLAGS_RELEASE}'")

# Defined in StorageObjects.h
# add_definitions("-DENABLE_BLOCK_LINEAR_STORAGE=1")

# enable some test code
if(NOT CMAKE_BUILD_TYPE MATCHES DEBUG)
    add_definitions("-DCUDAPP_ENABLE_TEST_CODE=1")
else()
    add_definitions("-DCUDAPP_ENABLE_TEST_CODE=0")
endif()

add_library(cuda-utils
    cpp_utils.h cpp_utils.cpp
    macros.h exceptions.h
    GenericSemaphore.h
    ArbitraryPool.h
    Recycler.h
    StackAllocator.h StackAllocator.cpp
    FiberUtils.h FiberUtils.cpp fiberProxy.h
    FiberPrioritySchedAlgo.h FiberPrioritySchedAlgo.cpp
    PriorityFiberPool.h PriorityFiberPool.cpp
    PipeLine.h
    ConcurrentQueue.h
    cuda_utils.h cuda_utils.cpp
    StorageFwd.h
    StorageManager.h StorageManager.cpp
    DefaultCacheableObject.h DefaultCacheableObject.cpp
    CacheableScratchObject.h
    CudaMemPool.h
    VectorMap.h
    StorageObjects.h StorageObjects.cpp
    Atomic.h
    KArray.h
    platform.h cuda_hint.cuh
    CudaEventPool.h CudaEventPool.cpp
    kmat.h
    CudaArray.h CudaTexture.h
    WatchDog.h
    LRUCache.h
    CudaStreamMutex.h CudaStreamSharedMutex.h CudaStreamSharedMutex.cpp
    Generator.h
    checksum.h checksum.cpp
    Profiler.h Profiler.cpp
    ThreadPool.h ThreadPool.cpp
    cudaExtIntrinsics.cuh
    IRuntime.h Runtime.h Runtime.cpp Runtime.hpp
    CudaDaemon.h CudaDaemon.cpp
    ImageReader.h ImageReader.cpp
    )

add_executable(tests_cuda-utils
    tests/StorageManagerTests.cpp
    tests/CudaMemPoolTests.cpp
    tests/CudaMultiEventTests.cpp
    tests/DefaultCacheableObjectTest.cpp
    tests/DefaultCacheableObjectTest.cu
    tests/VectorMapTests.cpp
    tests/nvbug200569951.cpp
    tests/CudaStreamMutexTest.cu
    tests/CudaStreamSharedMutexTest.cu
    tests/kmatTest.cpp)

target_include_directories(tests_cuda-utils PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(tests_cuda-utils PRIVATE cuda-utils gtest_main gtest cudart pthread turbojpeg jpeg heif exiv2)

