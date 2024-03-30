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
