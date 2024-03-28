#include "Profiler.h"
#include <vector>
#include <iosfwd>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <algorithm>

namespace cudapp
{

Profiler& Profiler::instance() {
    static Profiler profiler;
    return profiler;
}

Profiler::Trace::Trace(const std::string_view& name) : mName{name}, mStart{std::chrono::steady_clock::now()} {}
Profiler::Trace::Trace(Trace&& src) : mName{std::move(src.mName)}, mStart{src.mStart}{
    src.mName = std::string_view{};
}
Profiler::Trace& Profiler::Trace::operator=(Trace&& src){
    this->~Trace();
    mName = std::move(src.mName);
    mStart = std::move(src.mStart);
    src.mName = std::string_view{};
    return *this;
}
Profiler::Trace::~Trace() {reset();}
void Profiler::Trace::reset()
{
    if (!mName.empty()) {
        Profiler::instance().record(mName, std::chrono::steady_clock::now() - mStart);
        mName = std::string_view{};
    }
}

void Profiler::record(const std::string_view& name, const std::chrono::nanoseconds& duration) {
    {
        const std::shared_lock<std::shared_mutex> readLock{mLock};
        const auto& stat = mStatistics;
        const auto iter = stat.find(name);
        if (iter != stat.end()) {
            while (true) {
                auto oldVal = iter->second->load(std::memory_order_relaxed);
                const auto newVal = oldVal + duration;
                if (iter->second->compare_exchange_weak(oldVal, newVal, std::memory_order_relaxed, std::memory_order_relaxed)) {
                    break;
                }
            }
            return;
        }
    }
    {
        std::unique_lock<std::shared_mutex> writeLock{mLock};
        const auto iter = mStatistics.find(name);
        // note another thread may have added this entry.
        if (iter == mStatistics.end()) {
            mStatistics.try_emplace(name, std::make_unique<std::atomic<std::chrono::nanoseconds>>(duration));
        }
        else {
            writeLock.unlock();
            record(name, duration);
        }
    }
}

void Profiler::printSummary(std::ostream& os) const {
    std::vector<std::string_view> keys(mStatistics.size());
    std::transform(mStatistics.begin(), mStatistics.end(), keys.begin(), [](const auto& x){return x.first;});
    std::sort(keys.begin(), keys.end());
    os << "Performace summary:\n";
    for (const auto& k: keys) {
        os << "  " << std::setw(24) << k << ":\t" << mStatistics.at(k)->load(std::memory_order_relaxed).count() * 1E-9f << " sec\n";
    }
    os << std::flush;
}

std::chrono::nanoseconds Profiler::query(const std::string_view& name) const {
    const std::shared_lock<std::shared_mutex> readLock{mLock};
    return mStatistics.at(name)->load(std::memory_order_relaxed);
}

} // namespace cudapp
