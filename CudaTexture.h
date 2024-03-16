#pragma once
#include <cuda_runtime_api.h>
#include "cuda_utils.h"
#include "macros.h"
#include "CudaArray.h"

namespace cudapp
{

template <typename PixelType, int arrayFlags = cudaArrayDefault>
class TypedCudaArray
{
    static_assert(std::is_same_v<std::decay_t<PixelType>, PixelType>);
public:
    using Pixel = PixelType;
    static constexpr int flags = arrayFlags;

    TypedCudaArray() = default;
    explicit TypedCudaArray(cudaArray_t arr) : mArray{arr} {}
    TypedCudaArray(size_t width, size_t height) : mArray{createCudaArray2D<Pixel>(width, height, flags)}{
        sanityCheck();
    }
    TypedCudaArray(TypedCudaArray&&) = default;
    TypedCudaArray& operator=(TypedCudaArray&&) = default;
    cudaArray_t get() const {return mArray.get();}
    void reset(cudaArray_t arr) {
        mArray.reset(arr);
        sanityCheck();
    }
    cudaArray_t release() {return mArray.release();}
    cudaExtent getExtent() const {
        return getArrayExtent(mArray.get());
    }
    bool operator==(std::nullptr_t) const {return mArray.get() == nullptr;}
private:
    void sanityCheck() const {
        if (mArray != nullptr) {
            cudaChannelFormatDesc desc{};
            cudaCheck(cudaArrayGetInfo(&desc, nullptr, nullptr, get()));
            const cudaChannelFormatDesc ref = std::is_same_v<Pixel, half2> ? cudaCreateChannelDescHalf2() : cudaCreateChannelDesc<PixelType>();
            ASSERT(ref.x == desc.x && ref.y == desc.y && ref.z == desc.z
                && ref.w == desc.w && ref.f == desc.f);
        }
    }
    CudaRes<cudaArray_t, &cudaFreeArray> mArray;
};

class TexObj{
public:
    static constexpr cudaTextureObject_t kInvalidTexObj = 0;

    TexObj() = default;
    explicit TexObj(cudaTextureObject_t texture) : mTex(texture){
        ASSERT(texture != kInvalidTexObj);
    }
    ~TexObj(){
        NOEXCEPT(reset());
    }
    TexObj(const TexObj&) = delete;
    TexObj& operator=(const TexObj&) = delete;
    TexObj(TexObj&& other) noexcept : mTex{std::move(other.mTex)}{
        other.mTex = kInvalidTexObj;
    }
    TexObj& operator=(TexObj&& other) noexcept {
        mTex = other.mTex;
        other.mTex = kInvalidTexObj;
        return *this;
    }
    cudaTextureObject_t get() const {
        ASSERT(mTex != kInvalidTexObj);
        return mTex;
    }
    void reset(){
        if(mTex != kInvalidTexObj) {
            cudaCheck(cudaDestroyTextureObject(mTex));
            mTex = kInvalidTexObj;
        }
    }
    operator bool() const {return mTex != kInvalidTexObj;}
private:
    cudaTextureObject_t mTex {kInvalidTexObj};
};

inline TexObj createTexObj(cudaArray_t data, const cudaTextureDesc& texDesc)
{
    const cudaResourceDesc resDesc {
        .resType = cudaResourceType::cudaResourceTypeArray,
        .res = {.array = {.array = data}}
    };
    cudaTextureObject_t tex = TexObj::kInvalidTexObj;
    cudaCheck(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr));
    ASSERT(tex != TexObj::kInvalidTexObj && "We assume a valid texture object handle is never zero. If it can be, we need to change kInvalidTexObj");
    return TexObj{tex};
}

inline constexpr cudaTextureDesc createTexDesc(
    cudaTextureFilterMode filterMode = cudaTextureFilterMode::cudaFilterModePoint,
    cudaTextureReadMode readMode = cudaTextureReadMode::cudaReadModeElementType,
    cudaTextureAddressMode addressMode = cudaTextureAddressMode::cudaAddressModeWrap,
    const std::array<float, 4>& borderColor = {0, 0, 0, 0},
    bool normalizedCoords = false,
    bool sRGB = false)
{
    return cudaTextureDesc{
        {addressMode, addressMode, addressMode},
        filterMode,
        readMode,
        sRGB,
        {borderColor[0], borderColor[1], borderColor[2], borderColor[3]},
        normalizedCoords,
        {}, {}, {}, {}, {}, {}
#if CUDART_VERSION >= 11060 && CUDART_VERSION < 11080
        , {}
#endif
    };
}

template <typename PixelType, int arrayFlags = cudaArrayDefault>
class BitmapTexture
{
public:
    using Pixel = PixelType;
    static constexpr int flags = arrayFlags;
    BitmapTexture() = default;
    BitmapTexture(size_t width, size_t height, const cudaTextureDesc& texDesc)
        : mArray{width, height}
        , mTexObj{createTexObj(mArray.get(), texDesc)}
    {}
    BitmapTexture(const Pixel* hostData, size_t pitchInPixels, size_t width, size_t height, const cudaTextureDesc& texDesc, cudaStream_t stream)
        : BitmapTexture{width, height, texDesc}
    {
        cudaCheck(cudaMemcpy2DToArrayAsync(mArray.get(), 0, 0, hostData, sizeof(Pixel) * pitchInPixels, sizeof(Pixel) * width, height, cudaMemcpyHostToDevice, stream));
    }

    cudaExtent getExtent() const { return mArray.getExtent(); }

    cudaTextureObject_t getTexObj() const {
        return mTexObj.get();
    }

    // Rough estimation, not exact
    size_t getStorageSize() const {
        const auto extent = getExtent();
        return sizeof(Pixel) * extent.width * extent.height;
    }

private:
    TypedCudaArray<Pixel, flags> mArray;
    TexObj mTexObj;
};

class SurfObj{
public:
    static constexpr cudaSurfaceObject_t kInvalidSurfObj = 0;

    SurfObj() = default;
    explicit SurfObj(cudaSurfaceObject_t texture) : mSurf(texture){
        ASSERT(texture != kInvalidSurfObj);
    }
    ~SurfObj(){
        NOEXCEPT(reset());
    }
    SurfObj(const SurfObj&) = delete;
    SurfObj& operator=(const SurfObj&) = delete;
    SurfObj(SurfObj&& other) noexcept : mSurf{std::move(other.mSurf)}{
        other.mSurf = kInvalidSurfObj;
    }
    SurfObj& operator=(SurfObj&& other) noexcept {
        mSurf = other.mSurf;
        other.mSurf = kInvalidSurfObj;
        return *this;
    }
    cudaSurfaceObject_t get() const {
        ASSERT(mSurf != kInvalidSurfObj);
        return mSurf;
    }
    void reset(){
        if(mSurf != kInvalidSurfObj) {
            cudaCheck(cudaDestroyTextureObject(mSurf));
            mSurf = kInvalidSurfObj;
        }
    }
    operator bool() const {return mSurf != kInvalidSurfObj;}
private:
    cudaSurfaceObject_t mSurf {kInvalidSurfObj};
};

inline SurfObj createSurfObj(cudaArray_t data)
{
    const cudaResourceDesc resDesc {
        .resType = cudaResourceType::cudaResourceTypeArray,
        .res = {.array = {.array = data}}
    };
    cudaSurfaceObject_t surf = SurfObj::kInvalidSurfObj;
    cudaCheck(cudaCreateSurfaceObject(&surf, &resDesc));
    ASSERT(surf != SurfObj::kInvalidSurfObj && "We assume a valid surface object handle is never zero. If it can be, we need to change kInvalidSurfObj");
    return SurfObj{surf};
}

} // namespace cudapp
