#pragma once

#include <coroutine>
#include <exception>
#include <new>

namespace cudapp
{
struct MonoState{};
template <template<typename, size_t> typename G, typename T = MonoState, size_t alignment = 32>
struct PromiseType {
    auto initial_suspend() { return std::suspend_always{}; }
    auto final_suspend() noexcept { return std::suspend_always{}; }
    auto get_return_object() {
        return G<T, alignment>{G<T, alignment>::handle_type::from_promise(*this)};
    }
    void unhandled_exception() {
        std::terminate();
    }
    void return_void() {}
    auto yield_value(T value) {
        mCurrentValue = std::move(value);
        return std::suspend_always{};
    }
    T getCurrentValue() {
        return std::move(mCurrentValue);
    }
    static void* operator new(size_t size) {
        return operator new[](size, static_cast<std::align_val_t>(alignment));
    }
private:
    T mCurrentValue{};
};

template <typename T = MonoState, size_t alignment = 32>
struct Generator {
    using promise_type = PromiseType<Generator, T, alignment>;
    using handle_type =
        std::coroutine_handle<promise_type>;
    T current_value() {
        return coro.promise().getCurrentValue();
    }
    bool move_next() {
        coro.resume();
        return !isDone();
    }
    bool isDone() const {return coro.done();}
    Generator(handle_type h) : coro(h) {}
private:
    handle_type coro;
};
} // namespace cudapp

