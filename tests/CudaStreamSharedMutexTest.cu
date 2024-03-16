#include "../cuda_utils.h"
#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include "CudaStreamSharedMutex.h"
#include <random>

__global__ static void kernelWrite(volatile uint32_t* data)
{
    uint32_t ref = *data;
    for (uint32_t i = 0; i < 5000U; i++) {
        ref = i;
        *data = ref;
        kassert(ref == *data);
    }
}
__global__ static void kernelRead(const volatile uint32_t* data)
{
    const uint32_t ref = *data;
    for (uint32_t i = 0; i < 5000U; i++) {
        kassert(ref == *data);
    }
}

TEST(CudaStreamSharedMutexTest, random)
{
    using namespace cudapp;
    const auto evPoolHolder = createPooledCudaEvent();

    CudaStreamSharedMutex sharedMutex;
    const auto mem = allocCudaMem<uint32_t, CudaMemType::kDevice>(1);
    cudaCheck(cudaMemset(mem.get(), 0, sizeof(*mem.get())));

    auto task = [&mem, &sharedMutex](){
        const CudaStream stream = makeCudaStream();
        std::default_random_engine rng{std::random_device{}()};
        std::bernoulli_distribution dist{0.1f};
        for (int i = 0; i < 1000; i++) {
            if (dist(rng)) {
                auto lk = sharedMutex.acquire(stream.get());
                kernelWrite<<<1, 1, 0, stream.get()>>>(mem.get());
                cudaCheck(cudaGetLastError());
            }
            else {
                auto lk = sharedMutex.acquireShared(stream.get());
                kernelRead<<<1, 1, 0, stream.get()>>>(mem.get());
                cudaCheck(cudaGetLastError());
            }
        }
        cudaCheck(cudaStreamSynchronize(stream.get()));
    };
    std::vector<std::thread> threads;
    const int nbThrds = 4;

    threads.reserve(nbThrds);
    for (int i = 0; i < nbThrds; i++) {
        threads.emplace_back(task);
    }
    for (int i = 0; i < nbThrds; i++) {
        EXPECT_TRUE(threads.at(i).joinable());
        threads.at(i).join();
    }
}
