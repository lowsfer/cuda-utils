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
