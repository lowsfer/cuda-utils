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
