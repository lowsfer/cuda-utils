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

#include <stack>
#include <mutex>
#include <memory>

namespace cudapp {

template<typename T, typename Creator, typename Mutex = std::mutex>
class Recycler : public std::enable_shared_from_this<Recycler<T, Creator, Mutex>>{
public:
    Recycler(Creator&& creator = [](){return T{};})
            :mCreator{std::forward<Creator>(creator)}{}

    void recycle(T&& obj){
        std::lock_guard<Mutex> lk(mMutex);
        mRecycler.emplace(std::move(obj));
    }

    class Holder {
    public:
        Holder(std::shared_ptr<Recycler<T, Creator, Mutex>> owner, T&& obj) : mRecycler{std::move(owner)}, mObj{std::move(obj)} {}
        Holder(const Holder&) = delete;
        Holder& operator=(const Holder&) = delete;
        Holder(Holder&&) = default;
        Holder& operator=(Holder&&) = delete;
        ~Holder() {reset();}
        T& get() {return mObj;}
        const T& get() const {return mObj;}
    private:
        void reset() {if (mRecycler) mRecycler->recycle(std::move(mObj));}

        std::shared_ptr<Recycler<T, Creator, Mutex>> mRecycler;
        T mObj;
    };

    Holder get(){
        auto owner = std::enable_shared_from_this<Recycler<T, Creator, Mutex>>::shared_from_this();
        std::lock_guard<Mutex> lk(mMutex);
        if(!mRecycler.empty()){
            auto result = std::move(mRecycler.top());
            mRecycler.pop();
            return Holder{std::move(owner), std::move(result)};
        }
        else{
            return Holder{std::move(owner), mCreator()};
        }
    }
    size_t size() const{
        std::lock_guard<Mutex> lk(mMutex);
        return mRecycler.size();
    }


private:
    mutable Mutex mMutex;
    std::stack<T> mRecycler;
    Creator mCreator;
};
} // namespace cudapp

