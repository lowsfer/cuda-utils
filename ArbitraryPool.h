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
