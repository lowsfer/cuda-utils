#include <cuda_runtime.h>
#include "CudaArray.h"
#include "CudaTexture.h"
#include <thread>
#include <chrono>
#include <iostream>

cudaError_t launchKernel(cudaTextureObject_t tex, cudaStream_t stream);

int main()
{
    cudaStream_t stream = nullptr;

    const auto data = allocCudaMem<uint8_t, CudaMemType::kPinned>(256*256);
    std::fill_n(data.get(), 256*256, 4);
    const auto arr = cudapp::createCudaArray2D<uint8_t>(256, 256);
    cudaCheck(cudaMemcpy2DToArrayAsync(arr.get(), 0, 0, data.get(), 256, 256, 256, cudaMemcpyHostToDevice, stream));
    std::cout << "Created cuda array" << std::endl;
    launchCudaHostFunc(stream, [](){
        std::this_thread::sleep_for(std::chrono::seconds(2));
        std::cout << "Finished callback 0" << std::endl;
    });
    std::cout << "Launched callback 0" << std::endl;
    {
        const auto tex = cudapp::createTexObj(arr.get(), cudapp::createTexDesc(cudaFilterModeLinear, cudaReadModeNormalizedFloat, cudaAddressModeBorder));
        std::cout << "Created texture object" << std::endl;
        launchCudaHostFunc(stream, [](){
            std::this_thread::sleep_for(std::chrono::seconds(1));
            std::cout << "Finished callback 1" << std::endl;
        });
        std::cout << "Launched callback 1" << std::endl;
        cudaCheck(launchKernel(tex.get(), stream));
        std::cout << "Launched kernel" << std::endl;
    }
    std::cout << "Destroyed texture object" << std::endl;

    cudaCheck(cudaDeviceSynchronize());

    return 0;
}
