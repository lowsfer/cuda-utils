#include "StorageObjects.h"

namespace cudapp::storage
{

ObjStorageBase::~ObjStorageBase() = default;

void ObjStorageBase::resize(size_t size, cudaStream_t stream){
    REQUIRE(empty());
    dbgExpr(launchCudaHostFunc(stream, [inStreamSize{sizeInStream()}]{
                REQUIRE(*inStreamSize == 0);
            }));
    allocInStreamImpl(size, stream);
    setSize(size, stream);
}

void ObjStorageBase::clear(cudaStream_t stream) {
    REQUIRE(!empty());
    dbgExpr(launchCudaHostFunc(stream, [inStreamSize{sizeInStream()}, size{getSize()}]{
                REQUIRE(*inStreamSize == size);
            }));
    freeInStreamImpl(stream);
    setSize(0, stream);
}

ObjStorageBase::ObjStorageBase(size_t size)
    : mSize(size)
    , mSizeInStream{std::make_shared<size_t>(size)}
{}

ObjStorageBase::ObjStorageBase(ObjStorageBase &&other){
    std::swap(mSize, other.mSize);
    std::swap(mSizeInStream, other.mSizeInStream);
}

ObjStorageBase &ObjStorageBase::operator=(ObjStorageBase &&other){
    REQUIRE(mSize == 0);
    REQUIRE(*mSizeInStream == 0);
    std::swap(mSize, other.mSize);
    std::swap(mSizeInStream, other.mSizeInStream);
    return *this;
}

void ObjStorageBase::setSize(size_t size, cudaStream_t stream) {
    mSize = size;
    // the lambda may out-live the object, so do not capture this pointer
    launchCudaHostFunc(stream, [inStreamSize{sizeInStream()}, size]{
        *inStreamSize = size;
    });
}

}
