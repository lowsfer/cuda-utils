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
