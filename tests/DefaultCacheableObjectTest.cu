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
