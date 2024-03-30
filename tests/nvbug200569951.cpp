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

// NVBUG-200569951 reported as
// http://nvbugs/200569951
// Fixed in r445

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <mutex>
#include <thread>
#include <vector>
#include <gtest/gtest.h>

constexpr int nbThreads = 2;
constexpr int nbStreams = 2;
constexpr int maxIters = 100000;

inline void cudaCheck(cudaError_t err){
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorName(err));
}

struct CallbackData
{
    const int32_t idxOfThisCallback;
    int32_t* pIdxOfLastFinishedCallback;
    bool checkOrder;
    bool dbgPrint;
    int32_t idxThread; // thread from which this callback was dispatched
};

void callback(void* p)
{
    const CallbackData* data = static_cast<CallbackData*>(p);
    if (data->dbgPrint){
        printf("Executing callback %d dispatch from thread %d\n", data->idxOfThisCallback, data->idxThread);
    }

    if (data->checkOrder){
        if (data->idxOfThisCallback - 1 != *data->pIdxOfLastFinishedCallback){
            printf("Wrong order detected: last was %d and this is %d (dispatched from thread %d)\n", *data->pIdxOfLastFinishedCallback, data->idxOfThisCallback, data->idxThread);
            throw std::runtime_error("Wrong callback execution order");
        }
    }
    *data->pIdxOfLastFinishedCallback = data->idxOfThisCallback;
    delete data;
};

std::mutex streamMutexes[nbStreams];

void run(const cudaStream_t streams[nbStreams], bool checkOrder, bool dbgPrint, int32_t idxThread)
{
    // Before each cuda host func dispatch, this event is waited for
    // After each cuda host func dispatch, this event is recorded.
    // This guarantees ordered serialization of callbacks dispatched from this thread.
    cudaEvent_t event = nullptr;
    cudaCheck(cudaEventCreate(&event));

    // Index of the last finished callback, used inside callbacks.
    int32_t idxLastFinishedCallback = -1; // Access only in the cuda driver thread by callbacks

    // i is the index of the current callback to be dispatched
    for(int i = idxLastFinishedCallback + 1; i < maxIters; i++){
        const auto idxStream = i % nbStreams;
        const cudaStream_t stream = streams[idxStream];
        std::mutex& streamMutex = streamMutexes[idxStream];

        CallbackData* data = new CallbackData{i, &idxLastFinishedCallback, checkOrder, dbgPrint, idxThread};

        // Wait for finish of previous update
        {
            std::lock_guard<std::mutex> lock{streamMutex};
            cudaCheck(cudaStreamWaitEvent(stream, event, 0));
        }
        {
            std::lock_guard<std::mutex> lock{streamMutex};
            cudaCheck(cudaLaunchHostFunc(stream, callback, data));
        }
        // Record finish of this update, so the next update will wait
        {
            std::lock_guard<std::mutex> lock{streamMutex};
            cudaCheck(cudaEventRecord(event, stream));
        }
    }
    // Sync is required, otherwise idxLastFinishedCallback goes out of scope but callbacks may still be using it.
    for (int i = 0; i < nbStreams; i++){
        std::lock_guard<std::mutex> lock{streamMutexes[i]};
        cudaCheck(cudaStreamSynchronize(streams[i]));
    }
    cudaCheck(cudaEventDestroy(event));
}

TEST(DriverTest, nvbug200569951)
{
    cudaStream_t streams[nbStreams];
    for (int i = 0; i < nbStreams; i++)
        cudaCheck(cudaStreamCreateWithFlags(&streams[i], cudaStreamDefault));

    std::vector<std::thread> threads;
    for (int i = 0; i < nbThreads; i++){
        const bool checkOrder = true; // (i == 0)
        const bool dbgPrint = false; // (i == 0)
        const int idxThread = i;
        threads.emplace_back(&run, streams, checkOrder, dbgPrint, idxThread);
    }

    for (int i = 0; i < nbThreads; i++)
        threads.at(i).join();

    cudaCheck(cudaDeviceSynchronize());
    for (int i = 0; i < nbStreams; i++){
        cudaCheck(cudaStreamDestroy(streams[i]));
    }
}

