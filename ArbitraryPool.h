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
#include <atomic>
#include <vector>
#include <type_traits>
#include <algorithm>

namespace cudapp
{

// Do not use this with cuda events. Otherwise you may get race condition if mutiple threads do event record/wait at the same time, because record+wait is not atomic. For cuda events, use Recycler instead.
// cuda stream is OK.
template <typename T>
class ArbitraryPool
{
public:
    template<typename Factory, typename... Args>
    ArbitraryPool(size_t nbCandidates, const Factory& factory, const Args&... args){
        mCandidates.reserve(nbCandidates);
        std::generate_n(std::back_inserter(mCandidates), nbCandidates, [&]{return factory(args...);});
    }
    const T& get() const { return mCandidates[mIdxNext.fetch_add(1u) % mCandidates.size()]; }
    size_t size() const {return mCandidates.size();}
private:
    std::vector<T> mCandidates;
    mutable std::atomic_uint32_t mIdxNext {0};
};

} // namespace cudapp
