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
#include <cstddef>
#include <memory>

namespace cudapp
{
class IRuntime
{
public:
    virtual ~IRuntime();
};

extern "C" IRuntime* createRuntimeCudappImpl(const char* cacheFolder, size_t nbRandStream,
    size_t devPoolMaxBytes, size_t pinnedPoolMaxBytes, size_t sysPoolMaxBytes,
    size_t deviceSoftLimit, size_t pinnedSoftLimit, size_t sysSoftLimit);

inline std::unique_ptr<IRuntime> createRuntime(const char* cacheFolder, size_t nbRandStream,
    size_t devPoolMaxBytes, size_t pinnedPoolMaxBytes, size_t sysPoolMaxBytes,
    size_t deviceSoftLimit, size_t pinnedSoftLimit, size_t sysSoftLimit)
{
    return std::unique_ptr<IRuntime>{createRuntimeCudappImpl(cacheFolder, nbRandStream,
        devPoolMaxBytes, pinnedPoolMaxBytes, sysPoolMaxBytes,
        deviceSoftLimit, pinnedSoftLimit, sysSoftLimit)};
}
}