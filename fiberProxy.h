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
#if 1
#include <boost/fiber/fiber.hpp>
#include <boost/fiber/mutex.hpp>
#include <boost/fiber/recursive_mutex.hpp>
#include <boost/fiber/condition_variable.hpp>
#include <boost/fiber/scheduler.hpp>
#include <boost/fiber/algo/work_stealing.hpp>
#include <boost/fiber/algo/round_robin.hpp>
#include <boost/fiber/future.hpp>
#include <cuda_runtime_api.h>
#include <boost/fiber/cuda/waitfor.hpp>
#include <boost/fiber/policy.hpp>
namespace fb = boost::fibers;
namespace this_fiber = boost::this_fiber;
#else
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <cuda_runtime_api.h>

namespace fb
{
namespace algo
{
class work_stealing{};
}
template< typename SchedAlgo, typename ... Args >
void use_scheduling_algorithm( Args && ...) noexcept {}

using std::future;
using fiber = std::thread;
using std::mutex;
using std::recursive_mutex;
using std::condition_variable;
using std::condition_variable_any;
using std::packaged_task;

using std::async;
using std::promise;

namespace cuda
{
inline
std::tuple< cudaStream_t, cudaError_t > waitfor_all( cudaStream_t st) {
    return {st, cudaStreamSynchronize(st)};
}
}
}


namespace this_fiber
{
constexpr auto yield = std::this_thread::yield;
}

#endif