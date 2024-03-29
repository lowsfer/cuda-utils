#pragma once
#include "FiberPrioritySchedAlgo.h"
#include <memory>
#include "GenericSemaphore.h"
#include <boost/fiber/context.hpp>
#include <boost/fiber/future.hpp>
#include <boost/fiber/mutex.hpp>
#include <boost/fiber/condition_variable.hpp>
#include <boost/fiber/operations.hpp>
#include "StackAllocator.h"

namespace cudapp
{

class PriorityFiberPool
{
public:
    PriorityFiberPool(size_t nbThreads = 1, size_t defaultStackSize = 128UL << 10);

    ~PriorityFiberPool();

    // can be called from any thread
    template<typename Fn, typename ... Args >
    boost::fibers::future<std::result_of_t<std::decay_t<Fn>(std::decay_t<Args>...)>> post(int32_t priority, Fn&& fn, Args&&... args) {
        using Result = std::result_of_t<std::decay_t<Fn>(std::decay_t<Args>...)>;
        boost::fibers::packaged_task<Result(std::decay_t<Args>...)> pt{std::forward<Fn>(fn)};
        boost::fibers::future<Result> f = pt.get_future();
#if BOOST_FIBER_SUPPORTS_CONST_PROP
        boost::intrusive_ptr<boost::fibers::context> ctx = boost::fibers::make_worker_context_with_properties(boost::fibers::launch::post, new FiberPriorityProp(priority), StackAllocator{mDefaultStackSize}, std::move(pt), std::forward<Args>(args)...);
        assert(ctx->get_properties() != nullptr);
#else
        unused(priority);
        boost::intrusive_ptr<boost::fibers::context> ctx = boost::fibers::make_worker_context(boost::fibers::launch::post, StackAllocator{mDefaultStackSize}, std::move(pt), std::forward<Args>(args)...);
#endif
        mSchedAlgoGroup->post(ctx.get());
        return f;
    }
    template<typename Fn, typename ... Args >
    boost::fibers::future<std::result_of_t<std::decay_t<Fn>(std::decay_t<Args>...)>> post(Fn&& fn, Args&&... args) {
        boost::fibers::fiber_properties* props = boost::fibers::context::active()->get_properties();
        const int32_t priority = (props == nullptr ? 0 : static_cast<FiberPriorityProp*>(props)->priority);
        return post(priority + 1, std::forward<Fn>(fn), std::forward<Args>(args)...);
    }

private:
    void worker(std::shared_ptr<SchedAlgoGroup> group) noexcept;
private:
    size_t mDefaultStackSize;
    std::shared_ptr<SchedAlgoGroup> mSchedAlgoGroup;
    Barrier<std::mutex, std::condition_variable> mWorkersReady;
    std::vector<std::thread> mThreads;
    Event<boost::fibers::mutex, boost::fibers::condition_variable> mStop{false};
};
}
