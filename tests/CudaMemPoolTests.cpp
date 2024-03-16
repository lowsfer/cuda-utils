#include "../CudaMemPool.h"
#include <gtest/gtest.h>
#include <thread>
#include <chrono>

using namespace cudapp::storage;

TEST(CudaMemPoolTest, dependency)
{
    CudaMemPool<CudaMemType::kPinned> pool;
    const auto stream0 = makeCudaStream();
    const auto stream1 = makeCudaStream();
    void* ptr = nullptr;
    {
        auto mem = pool.alloc<int>(1, stream0.get());
        ptr = mem.get();
        static constexpr int val = 0x345a3ed8;
        cudaCheck(cudaLaunchHostFunc(stream0.get(), [](void* p){
            *static_cast<int*>(p) = val;
        }, mem.get()));
        cudaCheck(cudaLaunchHostFunc(stream0.get(), [](void*){
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }, nullptr));
        cudaCheck(cudaLaunchHostFunc(stream0.get(), [](void* p){
            EXPECT_EQ(*static_cast<int*>(p), val);
        }, mem.get()));
    }
    {
        auto mem = pool.alloc<int>(1, stream1.get());
        EXPECT_EQ(ptr, mem.get());

        cudaCheck(cudaLaunchHostFunc(stream1.get(), [](void* p){
            *static_cast<int*>(p) = -1;
        }, mem.get()));
    }
    cudaCheck(cudaStreamSynchronize(stream0.get()));
    cudaCheck(cudaStreamSynchronize(stream1.get()));
}
