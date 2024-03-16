#include "../cuda_utils.h"
#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include "CudaStreamMutex.h"

__global__ static void kernel(volatile uint32_t* data)
{
    uint32_t ref = *data;
    for (uint32_t i = 0; i < 10000U; i++) {
        ref = i;
        *data = ref;
        kassert(ref == *data);
    }
}

TEST(CudaStreamMutexTest, random)
{
    using namespace cudapp;
    const auto evPoolHolder = createPooledCudaEvent();

    CudaStreamMutex mutex;
    const auto mem = allocCudaMem<uint32_t, CudaMemType::kDevice>(1);
    cudaCheck(cudaMemset(mem.get(), 0, sizeof(*mem.get())));

    auto task = [&mem, &mutex](){
        const CudaStream stream = makeCudaStream();
        for (int i = 0; i < 100; i++) {
            auto lk = mutex.acquire(stream.get());
            kernel<<<1, 1, 0, stream.get()>>>(mem.get());
            cudaCheck(cudaGetLastError());
        }
        cudaCheck(cudaStreamSynchronize(stream.get()));
    };
    std::vector<std::thread> threads;
    const int nbThrds = 8;

    threads.reserve(nbThrds);
    for (int i = 0; i < nbThrds; i++) {
        threads.emplace_back(task);
    }
    for (int i = 0; i < nbThrds; i++) {
        EXPECT_TRUE(threads.at(i).joinable());
        threads.at(i).join();
    }
}
