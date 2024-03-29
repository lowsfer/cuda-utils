#pragma once

#include <boost/config.hpp>
#include <boost/intrusive_ptr.hpp>

#include <boost/fiber/algo/algorithm.hpp>
#include <boost/fiber/context.hpp>
#include <boost/fiber/detail/config.hpp>
#include <boost/fiber/scheduler.hpp>
#include <boost/fiber/future.hpp>
#include <boost/context/detail/prefetch.hpp>
#include <queue>
#include <cassert>
#include "GenericSemaphore.h"
#include <vector>
#include <shared_mutex>
#include <boost/version.hpp>

#define BOOST_FIBER_SUPPORTS_CONST_PROP (BOOST_VERSION >= 107800)

namespace cudapp
{
class FiberPriorityProp : public boost::fibers::fiber_properties {
public:
    FiberPriorityProp(boost::fibers::context* ctx): fiber_properties{ctx}, priority{0} {
        assert(ctx->get_properties() == nullptr);
        if (!ctx->is_context(boost::fibers::type::pinned_context)) {
            printf("Warning: creating properties for worker context at 0x%p, which is missing properties.\n", ctx);
        }
    }
    FiberPriorityProp(int32_t priority): fiber_properties{nullptr}, priority{priority} {}
    
    const int32_t priority;

    static int32_t getPriority(boost::fibers::context* ctx) {
#if BOOST_FIBER_SUPPORTS_CONST_PROP
#ifdef NDEBUG
        return static_cast<FiberPriorityProp*>(ctx->get_properties())->priority;
#else
        return dynamic_cast<FiberPriorityProp*>(ctx->get_properties())->priority;
#endif
#else
        unused(ctx);
        return 0;
#endif
    }
};

class ReadyQueue {
public:
    using context = boost::fibers::context;
    void push(boost::fibers::context* context);
    boost::fibers::context* pop();
    boost::fibers::context* steal();
    bool empty() const;
private:
    static bool isPinned(context* ctx) {
        return ctx->is_context(boost::fibers::type::pinned_context);
    }
private:
    mutable std::mutex mMutex; // use spin-lock
    // priority_queue outputs last element first. So must use operator<
    struct PriorityLess {
        bool operator()(context* a, context* b) const{
            return FiberPriorityProp::getPriority(a) < FiberPriorityProp::getPriority(b);
        }
    };
    std::queue<context*> mPinned{};
    std::priority_queue<context*, std::vector<context*>, PriorityLess> mWorkers;
};

class WorkStealingPrioritySchedAlgo;

// Single instance for UMA, and multiple instances for NUMA.
class SchedAlgoGroup
{
public:
    using context = boost::fibers::context;
    uint32_t append(WorkStealingPrioritySchedAlgo* scheduler);
    uint32_t size() const;
    WorkStealingPrioritySchedAlgo* at(size_t idx) const;
    // can be called from any thread
    // only use newly created detached context
    void post(boost::fibers::context* ctx);
private:
    mutable std::shared_mutex mMutex;
    // not using boost::intrusive_ptr because WorkStealingPrioritySchedAlgo has strong ref to SchedAlgoGroup.
    // we need to avoid circular strong ref.
    std::vector<WorkStealingPrioritySchedAlgo*> mSchedulers;
    std::atomic<uint32_t> mIdxNextForPost{0};
};

// @fixme: move implementation to cpp file
class WorkStealingPrioritySchedAlgo : public boost::fibers::algo::algorithm_with_properties<FiberPriorityProp> {
private:
    using context = boost::fibers::context;
private:
    std::shared_ptr<SchedAlgoGroup> mGroup;
    uint32_t mId;
    ReadyQueue mRQueue;
    Semaphore<bool, std::mutex, std::condition_variable> mEventShouldResume{false};
    
    static thread_local WorkStealingPrioritySchedAlgo* instanceOfThisThread;
public:
    static WorkStealingPrioritySchedAlgo* algoOfThisThread() noexcept {
        // assert(boost::fibers::context::active()->get_scheduler()->algo_.get() == instanceOfThisThread);
        return instanceOfThisThread;
    }
    static SchedAlgoGroup* algoGroupOfThisThread() noexcept {
        return algoOfThisThread()->getGroup();
    }

    WorkStealingPrioritySchedAlgo(std::shared_ptr<SchedAlgoGroup> group);

    SchedAlgoGroup* getGroup() const noexcept { return mGroup.get(); }

    uint32_t getId() const noexcept { return mId; }

    boost::fibers::context* steal() noexcept { return mRQueue.steal(); }

    // may be called from other threads
    void post(boost::fibers::context* ctx) noexcept;

    // For a subclass of algorithm_with_properties<>, it's important to
    // override the correct awakened() overload.
    void awakened(boost::fibers::context* ctx, FiberPriorityProp& props) noexcept override;

    boost::fibers::context* pick_next() noexcept override;

    bool has_ready_fibers() const noexcept override;

    void suspend_until(std::chrono::steady_clock::time_point const& time_point) noexcept override;

    void notify() noexcept override;
};

}