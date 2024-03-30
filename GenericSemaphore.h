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
#include <type_traits>
#include "cpp_utils.h"

namespace cudapp
{

class SemaphoreBase
{
public:
    enum class NotifyStyle
    {
        kOne,
        kAll,
        kNone
    };
    virtual ~SemaphoreBase() = default;
};

template <typename State, typename Mutex/* = std::mutex*/, typename ConditionVariable/* = std::condition_variable*/>
class Semaphore : public SemaphoreBase
{
public:
    Semaphore(State initVal = State{})
        : mState{std::move(initVal)}
    {}
    // in case the wait times out, skip updater and return std_nullopt
    template <typename Updater, typename Clock = std::chrono::steady_clock, typename Duration = std::chrono::nanoseconds, typename Precondition = bool(*)(const State&)>
    std::enable_if_t<std::is_invocable_v<Updater, State&> && std::is_invocable_r_v<bool, Precondition, const State&>,
        Optional<std::invoke_result_t<Updater, State&>>>
    updateStateUntil(Updater&& updater,
        const std_optional<std::chrono::time_point<Clock, Duration>>& timepoint = std_nullopt,
        NotifyStyle notifyStyle = NotifyStyle::kAll,
        Precondition&& precond = [](const State&){return true;})
    {
        auto onExit = makeScopeGuard([this, notifyStyle](){
            switch(notifyStyle)
            {
            case NotifyStyle::kOne: mCVar.notify_one(); break;
            case NotifyStyle::kAll: mCVar.notify_all(); break;
            case NotifyStyle::kNone: break;
            }
        });
        const auto [success, lock] = waitUntil([this, &precond](const State& state){return precond(state);}, timepoint);
        if constexpr (std::is_same_v<std::invoke_result_t<Updater, State&>, void>) {
            updater(mState);
            return success;
        }
        else {
            return success ? std_optional<std::invoke_result_t<Updater, State&>>(updater(mState)) : std_nullopt;
        }
    }

    template <typename Updater, typename Rep = int64_t, typename Period = std::ratio<1>, typename Precondition = bool(*)(const State&)>
    std::enable_if_t<std::is_invocable_v<Updater, State&> && std::is_invocable_r_v<bool, Precondition, const State&>,
        Optional<std::invoke_result_t<Updater, State&>>>
    updateStateFor(Updater&& updater,
        const std_optional<std::chrono::duration<Rep, Period>>& duration = std_nullopt,
        NotifyStyle notifyStyle = NotifyStyle::kAll,
        Precondition&& precond = [](const State&){return true;})
    {
        return updateStateUntil(std::forward<Updater>(updater),
            duration.has_value() ? std::make_optional(std::chrono::steady_clock::now() + duration.value()) : std_nullopt,
            notifyStyle, std::forward<Precondition>(precond));
    }

    template <typename Updater, typename Precondition = bool(*)(const State&)>
    std::enable_if_t<std::is_invocable_v<Updater, State&> && std::is_invocable_r_v<bool, Precondition, const State&>,
        std::invoke_result_t<Updater, State&>>
    updateState(Updater&& updater,
        NotifyStyle notifyStyle = NotifyStyle::kAll,
        Precondition&& precond = [](const State&){return true;})
    {
        auto result = updateStateUntil(std::forward<Updater>(updater),
            std::optional<std::chrono::steady_clock::time_point>{std_nullopt},
            notifyStyle, std::forward<Precondition>(precond));
        assert(bool(result)); static_cast<void>(result);
        if constexpr (std::is_same_v<std::invoke_result_t<Updater, State&>, void>){
            return;
        }
        else {
            return std::move(result.value());
        }
    }

    template <typename NewState, typename Clock = std::chrono::steady_clock, typename Duration = std::chrono::nanoseconds, typename Precondition = bool(*)(const State&)>
    std::enable_if_t<std::is_assignable_v<State&, NewState> && std::is_invocable_r_v<bool, Precondition, const State&>, Optional<State>>
    updateStateUntil(NewState&& newState,
        const std_optional<std::chrono::time_point<Clock, Duration>>& timepoint = std_nullopt,
        NotifyStyle notifyStyle = NotifyStyle::kAll,
        Precondition&& precond = [](const State&){return true;})
    {
        return updateStateUntil([&newState](State& state)->State{
                State oldState = std::move(state);
                state = std::forward<NewState>(newState);
                return oldState;
            }, timepoint, notifyStyle, precond);
    }

    template <typename NewState, typename Precondition = bool(*)(const State&)>
    std::enable_if_t<std::is_assignable_v<State&, NewState> && std::is_invocable_r_v<bool, Precondition, const State&>, State>
    updateState(NewState&& newState,
        NotifyStyle notifyStyle = NotifyStyle::kAll,
        Precondition&& precond = [](const State&){return true;})
    {
        return updateStateUntil(std::forward<NewState>(newState), Optional<std::chrono::steady_clock::time_point>{std_nullopt}, notifyStyle, precond).value();
    }

    // returns: pair of the checker result and the acquired lock
    // If you need to wait & update, just use updateState() with pre-condition instead.
    template <typename Checker, typename Clock = std::chrono::steady_clock, typename Duration = std::chrono::nanoseconds>
    [[nodiscard]] std::pair<bool, std::enable_if_t<std::is_invocable_r_v<bool, Checker, const State&>, std::unique_lock<Mutex>>>
    waitUntil(Checker&& checker, const std_optional<std::chrono::time_point<Clock, Duration>>& timepoint = std_nullopt) const {
        auto predicate = [&checker, this]{ return checker(mState); };
        std::unique_lock<Mutex> lk{mLock, std::defer_lock};
        bool success;
        if (timepoint.has_value()){
            if constexpr (IsTimedMutex<Mutex>::value) {
                success = lk.try_lock_until(timepoint.value()) && mCVar.wait_until(lk, timepoint.value(), predicate);
            }
            else {
                success = lk.try_lock() && mCVar.wait_until(lk, timepoint.value(), predicate);
            }
        }
        else {
            lk.lock();
            mCVar.wait(lk, predicate);
            success = true;
        }
        return std::make_pair(success, std::move(lk));
    }

    //returns: pair of the lock and checker result
    template <typename Checker, typename Rep = int64_t, typename Period = std::ratio<1>>
    [[nodiscard]] std::pair<bool, std::enable_if_t<std::is_invocable_r_v<bool, Checker, const State&>, std::unique_lock<Mutex>>>
    waitFor(Checker&& checker, const std_optional<std::chrono::duration<Rep, Period>>& duration = std_nullopt) const {
        return waitUntil(std::forward<Checker>(checker), duration == std_nullopt ? std_nullopt : std::chrono::steady_clock::now() + duration.value());
    }

    //returns: pair of the lock and checker result
    template <typename RefState>
    [[nodiscard]] std::enable_if_t<std::is_same_v<bool, std::decay_t<decltype(std::declval<const State&>() == std::declval<const RefState&>())>>, std::unique_lock<Mutex>>
    waitState(const RefState& refState) const {
        return waitUntil([&refState](const State& state){return state == refState;}).second;
    }
private:
    State mState;
    mutable Mutex mLock;
    mutable ConditionVariable mCVar;
};

template <typename Mutex = std::mutex, typename ConditionVariable = std::condition_variable>
class Event
{
public:
    Event(bool triggered) : mSemaphore{triggered}{}
    void trigger() {
        mSemaphore.updateState(true);
    }
    // wait until triggered and set it to false
    void consume() {
        mSemaphore.updateState(false, SemaphoreBase::NotifyStyle::kAll, [](bool x){return x;});
    }
    void wait() {
        static_cast<void>(mSemaphore.template waitState<bool>(true));
    }
private:
    Semaphore<bool, Mutex, ConditionVariable> mSemaphore;
};

template <typename Mutex = std::mutex, typename ConditionVariable = std::condition_variable>
class Barrier
{
public:
    Barrier(size_t nbThreads) : mNbThreads{nbThreads} {}
    void wait() {
        mSemaphore.updateState([this](size_t& x){x = (x + 1) % mNbThreads;}, SemaphoreBase::NotifyStyle::kAll);
        static_cast<void>(mSemaphore.waitState(size_t{0}));
    }
    size_t getNbThreads() const {return mNbThreads;}
private:
    size_t mNbThreads;
    Semaphore<size_t, Mutex, ConditionVariable> mSemaphore{0};
};

} // namespace cudapp
