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

