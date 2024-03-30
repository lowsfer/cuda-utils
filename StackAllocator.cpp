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

#include "StackAllocator.h"
#include <memory>
#include <unordered_map>
#include <mutex>
#include <stack>
#include <boost/context/protected_fixedsize_stack.hpp>
#include "macros.h"
#include "cpp_utils.h"

namespace cudapp {

class StackAllocator::StackCtxPool {
public:
    using Allocator = boost::context::protected_fixedsize_stack;
    static_assert(sizeof(Allocator) == sizeof(std::size_t), "The pool relies on the fact that boost::context::protected_fixedsize_stack has no internal states");
    ~StackCtxPool() {
        std::lock_guard<std::mutex> lk{mLock};
        for (auto& p : mPools) {
            while (true) {
                auto s = p.second.top();
                Allocator{p.first}.deallocate(s);
                p.second.pop();
                if (p.second.empty()) {
                    break;
                }
            }
        }
    }
    void push(stack_context& ctx) {
        std::lock_guard<std::mutex> lk{mLock};
        mPools[ctx.size].emplace(ctx);
        ctx = {};
    }
    stack_context get(std::size_t reqSize) {
        std::lock_guard<std::mutex> lk{mLock};
        const auto iterSize = mFwdSizeMap.find(reqSize);
        if (iterSize == mFwdSizeMap.end()) {
            const stack_context ctx = Allocator{reqSize}.allocate();
            ASSERT(mFwdSizeMap.try_emplace(reqSize, ctx.size).second);
            return ctx;
        }
        const size_t paddedSize = iterSize->second;
        const auto iter = mPools.find(paddedSize);
        if (iter != mPools.end() && !iter->second.empty()) {
            const auto guard = makeScopeGuard([iter](){
                iter->second.pop();
            });
            return iter->second.top();
        }
        {
            const stack_context ctx = Allocator{reqSize}.allocate();
            ASSERT(paddedSize == ctx.size);
            return ctx;
        }
    }
private:
    std::mutex mLock;
    std::unordered_map<std::size_t, std::size_t> mFwdSizeMap; // requested size -> padded size
    std::unordered_map<std::size_t, std::stack<stack_context>> mPools; // key is padded size
};

std::unique_ptr<StackAllocator::StackCtxPool> StackAllocator::mPool = std::make_unique<StackAllocator::StackCtxPool>();

typename StackAllocator::stack_context StackAllocator::allocate() {
    return mPool->get(mAllocSize);
}
void StackAllocator::deallocate(stack_context& s) {
    mPool->push(s);
}
StackAllocator::~StackAllocator() = default;

} // namespace cudapp
