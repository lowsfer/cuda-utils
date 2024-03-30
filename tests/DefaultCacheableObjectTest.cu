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

#include <cuda_runtime.h>
#include <cstdint>
#include "cuda_utils.h"

namespace{
__global__ void kernelCheckAndSetVal(bool checkOnly, volatile uint32_t* data, uint32_t expectedOldVal, uint32_t newVal = ~0u, uint32_t nbCycles = 0)
{
    const clock_t start = clock();
    REQUIRE(*data == expectedOldVal);
    if (checkOnly) return;
    *data = newVal;
    while (clock() - start < nbCycles) {
        REQUIRE(*data == newVal);
    }
}
}

void launchDeviceCheckAndSetVal(cudaStream_t stream, bool checkOnly, volatile uint32_t* data, uint32_t expectedOldVal, uint32_t newVal, uint32_t nbCycles)
{
    kernelCheckAndSetVal<<<1,1,0, stream>>>(checkOnly, data, expectedOldVal, newVal, nbCycles);
    cudaCheck(cudaGetLastError());
}
