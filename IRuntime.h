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