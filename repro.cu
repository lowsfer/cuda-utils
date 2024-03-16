#include <cuda_runtime.h>
#include <cstdio>

__global__ void kernel(cudaTextureObject_t texture)
{
    const float val = tex2D<float>(texture, 2.1f, 3.2f);
    printf("val = %f\n", val);
}

cudaError_t launchKernel(cudaTextureObject_t tex, cudaStream_t stream)
{
    kernel<<<1, 1, 0, stream>>>(tex);
    return cudaGetLastError();
}
