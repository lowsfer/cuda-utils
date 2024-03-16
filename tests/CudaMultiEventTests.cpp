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
