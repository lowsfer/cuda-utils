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

#include "../cuda_utils.h"
#include <gtest/gtest.h>
#include <thread>
#include <chrono>

TEST(CudaMultiEventTest, dependency)
{
    const auto stream0 = makeCudaStream();
    const auto stream1 = makeCudaStream();
    const auto stream2 = makeCudaStream();
    std::unique_ptr<ICudaMultiEvent> multiEvent = createCudaMultiEvent(true);

    int val = 0;
    {
        cudaCheck(cudaLaunchHostFunc(stream0.get(), [](void*){
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        }, nullptr));
        cudaCheck(cudaLaunchHostFunc(stream0.get(), [](void* p){
            *static_cast<int*>(p) += 1;
        }, &val));
        multiEvent->recordEvent(stream0.get());

    }
    {
        cudaCheck(cudaLaunchHostFunc(stream1.get(), [](void* p){
            *static_cast<int*>(p) += 2;
        }, &val));
        multiEvent->recordEvent(stream1.get());
    }
    {
        multiEvent->streamWaitEvent(stream2.get());
        cudaCheck(cudaLaunchHostFunc(stream1.get(), [](void* p){
            EXPECT_EQ(*static_cast<int*>(p), 3);
        }, &val));
    }

    cudaCheck(cudaStreamSynchronize(stream0.get()));
    cudaCheck(cudaStreamSynchronize(stream1.get()));
    cudaCheck(cudaStreamSynchronize(stream2.get()));
}
