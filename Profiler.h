#pragma once
#include <shared_mutex>
#include <unordered_map>
#include <chrono>
#include <string_view>
#include <atomic>
#include <memory>

namespace cudapp
{
class Profiler
{
public:
    static Profiler& instance();
private:
    class Trace
    {
    public:
        Trace(const std::string_view& name);
        Trace(const Trace&) = delete;
        Trace(Trace&& src);
        Trace& operator=(Trace&& src);
        ~Trace();
        void reset();
    private:
        std::string_view mName;
        std::chrono::steady_clock::time_point mStart;
    };

    void record(const std::string_view& name, const std::chrono::nanoseconds& duration);
public:
    // pass __func__ or __PRETTY_FUNCTION__
    Trace mark(const std::string_view& name) const {
        return Trace{name};
    }

    void printSummary(std::ostream& os) const;

    std::chrono::nanoseconds query(const std::string_view& name) const;

private:
    mutable std::shared_mutex mLock;
    std::unordered_map<std::string_view, std::unique_ptr<std::atomic<std::chrono::nanoseconds>>> mStatistics;
};

} // namespace cudapp
