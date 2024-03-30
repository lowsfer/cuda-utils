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
