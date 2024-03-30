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

#include "FiberPrioritySchedAlgo.h"
namespace cudapp {

void ReadyQueue::push(boost::fibers::context* context) {
    std::lock_guard lk{mMutex};
    if (isPinned(context)) {
        mPinned.emplace(context);
    }
    else {
        mWorkers.emplace(context);
    }
}
boost::fibers::context* ReadyQueue::pop() {
    std::lock_guard lk{mMutex};
    if (!mPinned.empty()) {
        context* ctx = mPinned.front();
        mPinned.pop();
        return ctx;
    }
    if (!mWorkers.empty()) {
        context* ctx = mWorkers.top();
        mWorkers.pop();
        return ctx;
    }
    return nullptr;
}
boost::fibers::context* ReadyQueue::steal() {
    std::lock_guard lk{mMutex};
    if (!mWorkers.empty()) {
        context* ctx = mWorkers.top();
        mWorkers.pop();
        return ctx;
    }
    return nullptr;
}

bool ReadyQueue::empty() const {
    std::lock_guard lk{mMutex};
    return mPinned.empty() && mWorkers.empty();
}


uint32_t SchedAlgoGroup::append(WorkStealingPrioritySchedAlgo* scheduler) {
    std::lock_guard lk{mMutex};
    const auto idx = static_cast<uint32_t>(mSchedulers.size());
    mSchedulers.emplace_back(scheduler);
    return idx;
}
uint32_t SchedAlgoGroup::size() const {
    std::shared_lock lk{mMutex};
    return static_cast<uint32_t>(mSchedulers.size());
}
WorkStealingPrioritySchedAlgo* SchedAlgoGroup::at(size_t idx) const {
    std::shared_lock lk{mMutex};
    return mSchedulers.at(idx);
}

// can be called from any thread
void SchedAlgoGroup::post(boost::fibers::context* ctx) {
    BOOST_ASSERT(!ctx->is_context(boost::fibers::type::pinned_context));
    WorkStealingPrioritySchedAlgo* dst = WorkStealingPrioritySchedAlgo::algoOfThisThread();
    const bool postToThisThread = false && (dst != nullptr);
    const auto nbSched = size();
    if (!postToThisThread) {
        const uint32_t idx = mIdxNextForPost.fetch_add(1, std::memory_order_relaxed) % nbSched;
        dst = at(idx);
    }
    assert(dst->getGroup() == this);
    const bool shouldYield = postToThisThread &&
        (FiberPriorityProp::getPriority(ctx) > FiberPriorityProp::getPriority(boost::fibers::context::active()));
    dst->post(ctx);
    if (shouldYield) {
        boost::this_fiber::yield();
    }
    // Notify all threads to work on it, or try to steal.
    // @fixme: use a shared semaphore per group?
    static thread_local std::minstd_rand generator{ std::random_device{}() };
    static thread_local std::vector<uint32_t> order;
    static thread_local uint32_t lastSize = 0;
    if (lastSize != nbSched) {
        order.reserve(nbSched);
        for (uint32_t i = lastSize; i < nbSched; i++) {
            order.push_back(i);
        }
        lastSize = nbSched;
    }
    std::shuffle(order.begin(), order.end(), generator);
    for (auto& i : order) {
        at(i)->notify();
    }
}


thread_local WorkStealingPrioritySchedAlgo* WorkStealingPrioritySchedAlgo::instanceOfThisThread = nullptr;

WorkStealingPrioritySchedAlgo::WorkStealingPrioritySchedAlgo(std::shared_ptr<SchedAlgoGroup> group)
    : mGroup{std::move(group)}
{
    mId = mGroup->append(this);
    assert(instanceOfThisThread == nullptr);
    instanceOfThisThread = this;
}

// may be called from other threads
void WorkStealingPrioritySchedAlgo::post(boost::fibers::context* ctx) noexcept {
    BOOST_ASSERT(!ctx->is_context(boost::fibers::type::pinned_context));
    BOOST_ASSERT(algorithm_with_properties_base::get_properties(ctx) != nullptr);
    mRQueue.push(ctx);
}

// For a subclass of algorithm_with_properties<>, it's important to
// override the correct awakened() overload.
void WorkStealingPrioritySchedAlgo::awakened(boost::fibers::context* ctx, FiberPriorityProp& props) noexcept {
    if ( !ctx->is_context( boost::fibers::type::pinned_context) ) {
        ctx->detach();
    }
    assert(ctx->get_properties() == &props); static_cast<void>(props);
    mRQueue.push(ctx);
}

boost::fibers::context* WorkStealingPrioritySchedAlgo::pick_next() noexcept {
    context* victim = mRQueue.pop();
    if (victim != nullptr) {
        boost::context::detail::prefetch_range( victim, sizeof( context) );
        if ( ! victim->is_context( boost::fibers::type::pinned_context) ) {
            context::active()->attach( victim);
        }
    }
    else {
        const uint32_t size = mGroup->size();
        static thread_local std::minstd_rand generator{ std::random_device{}() };
        static thread_local std::vector<uint32_t> others;
        static thread_local uint32_t lastSize = 0;
        if (lastSize != size) {
            for (uint32_t i = lastSize; i < size; i++) {
                if (i != getId()) {
                    others.push_back(i);
                }
            }
            lastSize = size;
        }
        std::shuffle(others.begin(), others.end(), generator);
        assert(others.size() + 1 == size);

        for (const uint32_t id : others) {
            victim = mGroup->at(id)->steal();
            if (victim != nullptr) {
                break;
            }
        }
        if ( nullptr != victim) {
            boost::context::detail::prefetch_range( victim, sizeof( context) );
            BOOST_ASSERT( ! victim->is_context( boost::fibers::type::pinned_context) );
            context::active()->attach( victim);
        }
    }
    return victim;
}

bool WorkStealingPrioritySchedAlgo::has_ready_fibers() const noexcept {
    return !mRQueue.empty();
}

void WorkStealingPrioritySchedAlgo::suspend_until(std::chrono::steady_clock::time_point const& time_point) noexcept {
    std::optional<std::chrono::steady_clock::time_point> deadline;
    if ((std::chrono::steady_clock::time_point::max)() != time_point) {
        deadline = time_point;
    }
    // wait until true or timeout, and set to false
    mEventShouldResume.updateStateUntil(false, deadline, SemaphoreBase::NotifyStyle::kNone, [](bool x){return x;});
}

void WorkStealingPrioritySchedAlgo::notify() noexcept {
    mEventShouldResume.updateState(true, SemaphoreBase::NotifyStyle::kAll);
}
} // namespace cudapp

