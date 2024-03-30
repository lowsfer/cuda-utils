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
#include <memory>
#include <mutex>
#include <condition_variable>
#include <thread>

namespace cudapp
{
//! If there is no progress (notifyAlive() call) in maxSec, WatchDog terminates the program.
class WatchDog
{
public:
    WatchDog(const float maxSec) :mMaxSeconds{maxSec}, mThrd{&WatchDog::killer, this} {}
    WatchDog(const WatchDog&) = delete;
    WatchDog(WatchDog&&) = delete;
    WatchDog operator=(const WatchDog&) = delete;
    WatchDog operator=(WatchDog&&) = delete;
    ~WatchDog() {
        {
            std::lock_guard<std::mutex> lk{mLock};
            mFinished = true;
        }
        mCVar.notify_all();
        mThrd.join();
    }

    void notifyAlive() {
        {
            std::unique_lock<std::mutex> lk{mLock};
            mRefreshed = true;
        }
        mCVar.notify_all();
    }
private:
    void killer() {
        const char* envFlag = std::getenv("DISABLE_WATCHDOG");
        if (envFlag != nullptr && std::stoi(envFlag) != 0) {
            return;
        }
        std::unique_lock<std::mutex> lk{mLock};
        while (mRefreshed && !mFinished) {
            mRefreshed = false;
            mCVar.wait_for(lk, std::chrono::duration<float>{mMaxSeconds}, [this](){ return mRefreshed || mFinished; });
        }
        if (!mFinished) {
            fprintf(stderr, "No progress in the past %f seconds. Terminating process ...\n", mMaxSeconds);
            std::terminate();
        }
    }
    mutable std::condition_variable mCVar;
    mutable std::mutex mLock;
    bool mRefreshed {true};
    bool mFinished {false};
    float mMaxSeconds;
    std::thread mThrd;
};
}
