#pragma once
#include <boost/context/stack_context.hpp>
#include <cstddef>
#include <memory>

namespace cudapp
{
//! A pooled and protected stack allocator for boost.fiber/boost.context
//! It depends on the fact boost::context::protected_fixedsize_stack implementation has no internal states 
class StackAllocator
{
public:
    using stack_context = boost::context::stack_context;
    class StackCtxPool;

    StackAllocator(std::size_t allocSize = 128UL<<10) : mAllocSize{allocSize} {}
    ~StackAllocator();

    stack_context allocate();

    void deallocate(stack_context &);
private:
    static std::unique_ptr<StackCtxPool> mPool;
    std::size_t mAllocSize;
};
} // namespace cudapp
