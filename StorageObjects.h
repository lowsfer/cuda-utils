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

#pragma once
#include <memory>
#include "cpp_utils.h"
#include "CudaMemPool.h"
#include <fstream>
#include "CudaArray.h"
#include "CudaEventPool.h"
#include "StorageFwd.h"

#define ENABLE_BLOCK_LINEAR_STORAGE 1

namespace cudapp
{
namespace storage
{

class ObjStorageBase
{
public:
    // Do not call clear() in this destructor, because clear calls virtual functions
    virtual ~ObjStorageBase();

    virtual size_t getElemBytes() const = 0;
    // in number of elements
    size_t getSize() const {return mSize;}
    bool empty() const { return getSize() == 0; }

    void resize(size_t size, cudaStream_t stream);

    void clear(cudaStream_t stream);

    // not necessary, just for sanity check
    std::shared_ptr<size_t> sizeInStream() const {return mSizeInStream;}
protected:
    ObjStorageBase(size_t size);

    ObjStorageBase(const ObjStorageBase&) = delete;
    ObjStorageBase& operator=(const ObjStorageBase&) = delete;
    ObjStorageBase(ObjStorageBase&& other);
    ObjStorageBase& operator=(ObjStorageBase&& other);

    void setSize(size_t size, cudaStream_t stream);

private:
    // Caller must call cudaStreamWaitEvent(stream, getReadyEvent(), 0) / cudaEventRecord(getReadyEvent(), stream) before/after this two APIs.
    virtual void allocInStreamImpl(size_t size, cudaStream_t stream) = 0;
    virtual void freeInStreamImpl(cudaStream_t stream) = 0;

private:
    size_t mSize = 0; // in number of elements
    std::shared_ptr<size_t> mSizeInStream = std::make_shared<size_t>(0u); // not necessary, just for assertions
};

template <CudaMemType memType, typename ElemType>
class CudaMemStorage : public ObjStorageBase
{
public:
    using CudaMemPoolType = CudaMemPool<memType>;
public:
    CudaMemStorage(CudaMemPoolType& pool)
        : ObjStorageBase(0)
        , mPool(&pool)
        , mData{nullptr, {}}
    {}
    CudaMemStorage(typename CudaMemPool<memType>::template PooledCudaMem<ElemType>&& mem, size_t size)
        : ObjStorageBase(size)
        , mPool {mem.get_deleter().pool}
        , mData {std::move(mem)}
    {}

    size_t getElemBytes() const override {return sizeof(ElemType);}
    const CudaMemPoolType& getMemPool() const {return mPool;}

    void notifyMigratedToStream(cudaStream_t stream) {
        mData.get_deleter().stream = stream;
    }

    // do not call data() from a callback in stream. It should be called in host
    ElemType* data() const {return mData.get();}

private:
    void allocInStreamImpl(size_t size, cudaStream_t stream) final{
        mData = mPool->template alloc<ElemType, true>(size, stream);// data pointer is available immediately in host, but the memory is not until previous work in stream is finished
    }
    void freeInStreamImpl(cudaStream_t stream) final {
        REQUIRE(stream == mData.get_deleter().stream);
        mData.reset();
    }
private:
    CudaMemPoolType* mPool = nullptr;
    typename CudaMemPoolType::template PooledCudaMem<ElemType> mData;
};

template <typename ElemType>
using CudaDevMemStorage = CudaMemStorage<CudaMemType::kDevice, ElemType>;
template <typename ElemType>
using CudaPinnedMemStorage = CudaMemStorage<CudaMemType::kPinned, ElemType>;
template <typename ElemType>
using CudaSysMemStorage = CudaMemStorage<CudaMemType::kSystem, ElemType>;

template <typename ElemType>
class DiskStorage : public ObjStorageBase
{
public:
    using ObjStorageBase::getSize;
public:
    DiskStorage(const fs::path& path, size_t size)
        : ObjStorageBase(size)
        , mPath(std::make_shared<const fs::path>(path))
    {
    }

    size_t getElemBytes() const override {return sizeof(ElemType);}

    fs::path data() const {return *mPath;}

    // memory referred by data must be valid until finish of the async callback.
    void write(const ElemType* data, size_t size, cudaStream_t stream){
        REQUIRE(getSize() == size);
        launchCudaHostFunc(stream, [inStreamSize{sizeInStream()}, path{mPath}, data, size]{
            REQUIRE(*inStreamSize == size);
            std::ofstream fout;
            fout.exceptions(std::ios::badbit | std::ios::failbit);
            fout.open(*path, std::ios::binary | std::ios::trunc);
            fout.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(sizeof(ElemType) * size));
            fout.close();
        });
    }
    // memory referred by data must be valid until finish of the async callback.
    void read(ElemType* data, size_t size, cudaStream_t stream) const{
        REQUIRE(getSize() == size);
        launchCudaHostFunc(stream, [inStreamSize{sizeInStream()}, path{mPath}, data, size]{
            REQUIRE(size == *inStreamSize);
            REQUIRE(fs::exists(*path) && fs::file_size(*path) == sizeof(ElemType) * size);
            std::ifstream fin;
            fin.exceptions(std::ios::badbit | std::ios::failbit | std::ios::eofbit);
            fin.open(*path, std::ios::binary);
            fin.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(sizeof(ElemType) * size));
        });
    }
private:
    void allocInStreamImpl(size_t size, cudaStream_t stream) override{
        unused(size, stream);
    }
    void freeInStreamImpl(cudaStream_t stream) override {
        launchCudaHostFunc(stream, [inStreamSize{sizeInStream()}, path{mPath}](){
            REQUIRE(*inStreamSize != 0);
            fs::remove(*path);
        });
    }
private:
    std::shared_ptr<const fs::path> mPath;
};

#if ENABLE_BLOCK_LINEAR_STORAGE
template <typename ElemType>
class BlockLinearStorage : public ObjStorageBase
{
public:
    using CudaMemPoolType = CudaMemPool<CudaMemType::kDevice>;
    BlockLinearStorage(CudaMemPoolType& pool, size_t width, size_t height, unsigned arrayFlags = cudaArrayDefault)
        : ObjStorageBase{0}
        , mPool{pool}
        , mAttributes{cudaCreateChannelDesc<ElemType>(), {width, height, 0}, static_cast<unsigned int>(arrayFlags)}
    {}
    BlockLinearStorage(PooledCudaArray&& array)
        : ObjStorageBase(getCudaArrayNbPixels(array.get())) // This is the number of elements used to compute size for pinned/sys mem and disk file)
        , mPool {*array.get_deleter().pool}
        , mAttributes{getCudaArrayAttributes(array.get())}
        , mArray {std::move(array)}
    {}

    void notifyMigratedToStream(cudaStream_t stream) {
        mArray.get_deleter().stream = stream;
    }

    size_t getElemBytes() const override {return sizeof(ElemType);}
    const CudaMemPoolType& getMemPool() const {return mPool;}

    // do not call data() from a callback in stream. It should be called in host
    cudaArray_t data() const {return mArray.get();}

    CudaArrayAttributes getAttributes() const {return mAttributes;}

private:
    void allocInStreamImpl(size_t size, cudaStream_t stream) final{
        ASSERT(size == mAttributes.extent.width * mAttributes.extent.height && mAttributes.extent.depth == 0);
        mArray = mPool.allocArray(mAttributes, stream); // cuda array is available immediately in host, but the memory is not until previous work in stream is finished
    }
    void freeInStreamImpl(cudaStream_t stream) final {
        ASSERT(stream == mArray.get_deleter().stream);
        mArray.reset();
    }
private:
    CudaMemPoolType& mPool;
    CudaArrayAttributes mAttributes{};
    PooledCudaArray mArray;
};

#endif

// caller must guarantee that src data and dst space is available in the specified stream
template <typename ElemType, CudaMemType srcType, CudaMemType dstType>
std::enable_if_t<(srcType == CudaMemType::kDevice && dstType == CudaMemType::kPinned) || (srcType == CudaMemType::kPinned && dstType == CudaMemType::kDevice), void>
storageMemcpyAsync(const CudaMemStorage<srcType, ElemType>& src, CudaMemStorage<dstType, ElemType>& dst, cudaStream_t stream)
{
    cudaCheck(cudaMemcpyAsync(dst.data(), src.data(), sizeof(ElemType)*src.getSize(), dstType == CudaMemType::kPinned ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice, stream));
}
// caller must guarantee that src data and dst space is available in the specified stream
template <typename ElemType, CudaMemType srcType, CudaMemType dstType>
std::enable_if_t<(srcType == CudaMemType::kPinned && dstType == CudaMemType::kSystem) || (srcType == CudaMemType::kSystem && dstType == CudaMemType::kPinned), void>
storageMemcpyAsync(const CudaMemStorage<srcType, ElemType>& src, CudaMemStorage<dstType, ElemType>& dst, cudaStream_t stream){
    const size_t size = src.getSize();
    REQUIRE(dst.getSize() == size);
    const auto srcPtr = src.data();
    const auto dstPtr = dst.data();
    launchCudaHostFunc(stream, [srcPtr, dstPtr, size, inStreamSize{dst.sizeInStream()}](){
        REQUIRE(*inStreamSize == size);
        std::copy_n(srcPtr, size, dstPtr);
    });
}

template <typename ElemType, CudaMemType srcType>
std::enable_if_t<srcType == CudaMemType::kPinned || srcType == CudaMemType::kSystem, void>
storageMemcpyAsync(const CudaMemStorage<srcType, ElemType>& src, DiskStorage<ElemType>& dst, cudaStream_t stream) {
    REQUIRE(src.getSize() == dst.getSize());
    dst.write(src.data(), src.getSize(), stream);
}
template <typename ElemType, CudaMemType dstType>
std::enable_if_t<dstType == CudaMemType::kPinned || dstType == CudaMemType::kSystem, void>
storageMemcpyAsync(const DiskStorage<ElemType>& src, CudaMemStorage<dstType, ElemType>& dst, cudaStream_t stream) {
    REQUIRE(src.getSize() == dst.getSize());
    src.read(dst.data(), dst.getSize(), stream);
}

#if ENABLE_BLOCK_LINEAR_STORAGE

// caller must guarantee that src data and dst space is available in the specified stream
template <typename ElemType>
void storageMemcpyAsync(const BlockLinearStorage<ElemType>& src, CudaMemStorage<CudaMemType::kPinned, ElemType>& dst, cudaStream_t stream)
{
    const auto srcAttr = src.getAttributes();
    cudaCheck(cudaMemcpy2DFromArrayAsync(dst.data(), sizeof(ElemType) * srcAttr.extent.width, src.data(), 0, 0, sizeof(ElemType) * srcAttr.extent.width, srcAttr.extent.height, cudaMemcpyDeviceToHost, stream));
}
template <typename ElemType>
void storageMemcpyAsync(const CudaMemStorage<CudaMemType::kPinned, ElemType>& src, BlockLinearStorage<ElemType>& dst, cudaStream_t stream)
{
    const auto dstAttr = dst.getAttributes();
    cudaCheck(cudaMemcpy2DToArrayAsync(dst.data(), 0, 0, src.data(), sizeof(ElemType) * dstAttr.extent.width, sizeof(ElemType) * dstAttr.extent.width, dstAttr.extent.height, cudaMemcpyHostToDevice, stream));
}

#endif

// mayOmit = true: omit if dst is not empty
template <typename SrcStorageType, typename DstStorageType>
void migrateStorage(bool mayOmit, bool keepSrc, SrcStorageType& src, DstStorageType& dst, cudaStream_t stream)
{
    REQUIRE(!src.empty());
    if constexpr (std::is_same_v<SrcStorageType, DstStorageType>){
        unused(stream);
        REQUIRE(mayOmit && keepSrc);
        REQUIRE(&src == &dst);
        return;
    }
    else {
        if (!mayOmit)
        {
            REQUIRE(dst.empty());
        }

        if (dst.empty()){ // (!mayOmit || (mayOmit && dst.empty())), optimized with if(!mayOmit){REQUIRE(dst.empty();}
            dst.resize(src.getSize(), stream);
            storageMemcpyAsync(src, dst, stream);
        }
        if (!keepSrc){
            src.clear(stream);
        }
    }
}

} // namespace storage
} // namespace cudapp
