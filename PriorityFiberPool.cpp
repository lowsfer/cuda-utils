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

#include "PriorityFiberPool.h"

namespace cudapp
{

PriorityFiberPool::PriorityFiberPool(size_t nbThreads, size_t defaultStackSize)
    : mDefaultStackSize{defaultStackSize}
    , mSchedAlgoGroup{std::make_shared<SchedAlgoGroup>()}
    , mWorkersReady{nbThreads + 1}
{
    REQUIRE(nbThreads > 0);
    mThreads.reserve(nbThreads);
    std::generate_n(std::back_inserter(mThreads), nbThreads, [this](){return std::thread{&PriorityFiberPool::worker, this, mSchedAlgoGroup};});
    // prevent users from posting tasks before works are correctly set up.
    mWorkersReady.wait();
}

PriorityFiberPool::~PriorityFiberPool() {
    mStop.trigger();
    for (auto& t : mThreads) {
        t.join();
    }
    mThreads.clear();
}

void PriorityFiberPool::worker(std::shared_ptr<SchedAlgoGroup> group) noexcept
{
    boost::fibers::context::active()->set_properties(new FiberPriorityProp(std::numeric_limits<int32_t>::max()));
    boost::fibers::use_scheduling_algorithm<WorkStealingPrioritySchedAlgo>(std::move(group));
    mWorkersReady.wait();
    mStop.wait();
}

} // namespace cudapp
