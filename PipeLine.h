#pragma once
#include "FiberUtils.h"
#include <mutex>
#include <boost/fiber/condition_variable.hpp>
#include <boost/fiber/protected_fixedsize_stack.hpp>
#include "PriorityFiberPool.h"

namespace cudapp {

template <typename T>
using PipeLineChannel = cudapp::ConcurrentQueue<T, fb::mutex, fb::condition_variable_any>;


namespace pipelineImpl {
template <typename Func, typename = void>
struct FuncParser;
template<typename Ret, typename Arg>
struct FuncParser<Ret(Arg)>
{
    static constexpr bool isCustomizedStage = false;
    using Input = Arg;
    using Output = Ret;
};

template<typename Ret, typename Arg>
struct FuncParser<Ret(*)(Arg)>
{
    static constexpr bool isCustomizedStage = false;
    using Input = Arg;
    using Output = Ret;
};

template<typename Class, typename Ret, typename Arg>
struct FuncParser<Ret(Class::*)(Arg)>
{
    static constexpr bool isCustomizedStage = false;
    using ClassType = Class;
    using Input = Arg;
    using Output = Ret;
};

template<typename Class, typename Ret, typename Arg>
struct FuncParser<Ret(Class::*)(Arg) const>
{
    static constexpr bool isCustomizedStage = false;
    using ClassType = Class;
    using Input = Arg;
    using Output = Ret;
};
template<typename In, typename Out>
struct FuncParser<void(PipeLineChannel<In>&, PipeLineChannel<Out>&), void>{
    static constexpr bool isCustomizedStage = true;
    using Input = In;
    using Output = Out;
};
template<typename In, typename Out>
struct FuncParser<void(*)(PipeLineChannel<In>&, PipeLineChannel<Out>&), void>{
    static constexpr bool isCustomizedStage = true;
    using Input = In;
    using Output = Out;
};
template<typename Class, typename In, typename Out>
struct FuncParser<void(Class::*)(PipeLineChannel<In>&, PipeLineChannel<Out>&), void>{
    static constexpr bool isCustomizedStage = true;
    using Input = In;
    using Output = Out;
};
template<typename Class, typename In, typename Out>
struct FuncParser<void(Class::*)(PipeLineChannel<In>&, PipeLineChannel<Out>&) const, void>{
    static constexpr bool isCustomizedStage = true;
    using Input = In;
    using Output = Out;
};
template<typename T>
struct FuncParser<T, std::void_t<decltype(&T::operator())>> : FuncParser<decltype(&T::operator())> {};

template <typename Func> using FuncInput = typename FuncParser<std::decay_t<Func>>::Input;
template <typename Func> using FuncOutput = typename FuncParser<std::decay_t<Func>>::Output;
template <typename Func> inline constexpr bool isCustomizedStage = FuncParser<std::decay_t<Func>>::isCustomizedStage;
// tests
static_assert(std::is_same_v<int, FuncInput<float(*)(int)>>);
static_assert(std::is_same_v<int, FuncInput<float(*)(int)>>);

void test() {
    int x = 0;
    auto f = [x](int y){return float(x+y);};
    static_assert(std::is_same_v<int, typename FuncParser<decltype(f)>::Input>);
}
} // pipelineImpl
template <typename Input, typename Product>
class IPipeLine
{
public:
    template <typename T>
    using Channel = PipeLineChannel<T>;

    virtual ~IPipeLine() = default;
    virtual void close() = 0; // close and synchronize
    virtual Channel<Input>* getInputChannel() const = 0;
    virtual Channel<Product>* getProductChannel() const = 0;
    virtual void setChannelCapacity(size_t idxChannel, size_t capacity) = 0;
    virtual size_t getNbStages() const = 0;
    virtual size_t getCurrentChannelSize(int idxStage) const = 0;
    virtual void enqueue(Input input) = 0;
};

//! A Pipeline with each stage being a fiber.
template <typename ProductType, typename... Funcs>
class PipeLine;

template <typename ProductType>
class PipeLine<ProductType> final : public IPipeLine <ProductType, ProductType>
{
public:
    using Input = ProductType;
    using Product = ProductType;
    static constexpr size_t nbStages = 0;

    template <typename T>
    using Channel = typename IPipeLine<Input, ProductType>::template Channel<T>;

    PipeLine(std::function<fb::future<void>(int32_t /*priority*/, std::function<void()> /*task*/)>&,
        uint32_t idxStage, Channel<Product>& productChannel)
        : mIdxStage {idxStage}
        , mProductChannel{&productChannel}
    {}

    ~PipeLine() override{
        close();
    }

    void close() override{
        mProductChannel->close();
    }

    Channel<Input>* getInputChannel() const override {return mProductChannel;}
    Channel<Product>* getProductChannel() const override {return mProductChannel;}

    size_t getNbStages() const override {return 0ul;}

    void setChannelCapacity(size_t idxChannel, size_t capacity) override {
        REQUIRE(idxChannel == 0);
        mProductChannel->setCapacity(capacity);
    }

    size_t getCurrentChannelSize(int idxStage) const override {
        REQUIRE(idxStage == 0);
        return mProductChannel->peekSize();
    }

    void enqueue(Input input) override{
        mProductChannel->emplace(std::move(input));
    }
private:
    uint32_t mIdxStage;
    Channel<Input>* mProductChannel = nullptr;
};

template <typename ProductType, typename Func0, typename... Funcs>
class PipeLine<ProductType, Func0, Funcs...> final : public IPipeLine<std::decay_t<pipelineImpl::FuncInput<Func0>>, ProductType>
{
public:
    static constexpr bool isCustomizedStage = pipelineImpl::isCustomizedStage<Func0>;
    using Input = std::decay_t<typename pipelineImpl::FuncInput<Func0>>;
    using Output = std::decay_t<typename pipelineImpl::FuncOutput<Func0>>;
    using Product = ProductType;
    using DownStream = PipeLine<ProductType, Funcs...>;
    static constexpr size_t nbStages = 1 + sizeof...(Funcs);

    template <typename T>
    using Channel = typename IPipeLine<Input, ProductType>::template Channel<T>;

    static_assert (StaticAssertIsSame<Output, typename DownStream::Input>, "type mismatch of connected pipeline stages");

    PipeLine(std::function<fb::future<void>(int32_t /*priority*/, std::function<void()> /*task*/)>& fiberCreator,
        uint32_t idxStage, Channel<Product>& productChannel, Func0 func0, Funcs... funcs)
        : mIdxStage {idxStage}
        , mFunc{std::move(func0)}
        , mDownStream{std::make_unique<DownStream>(fiberCreator, idxStage + 1, productChannel, std::move(funcs)...)}
        , mFiber{fiberCreator(int32_t(idxStage << 10), [this]{process();})} // later stage takes higher priority
    {}

    ~PipeLine(){
        close();
    }

    // also waits until all tasks are finished
    void close() override {
        mInputChannel->close();
        if (mFiber.valid()) {
            mFiber.get();
        }
        mDownStream->close();
    }

    Channel<Input>* getInputChannel() const override {return mInputChannel.get();}
    Channel<Output>* getStageOutputChannel() const {return mDownStream->getInputChannel();}
    Channel<Product>* getProductChannel() const override {return mDownStream->getProductChannel();}

    // idxChannel belongs to range [0, nbStages], but idxChannel = nbStages is for the final channel
    void setChannelCapacity(size_t idxChannel, size_t capacity) override {
        if (idxChannel == 0) {
            mInputChannel->setCapacity(capacity);
        }
        else {
            mDownStream->setChannelCapacity(idxChannel - 1u, capacity);
        }
    }

    size_t getNbStages() const override {return 1ul + sizeof... (Funcs);}

    void enqueue(Input input) override {
        mInputChannel->emplace(std::move(input));
    }

    size_t getCurrentChannelSize(int idxStage) const override {
        REQUIRE(inRange(idxStage, 0, static_cast<int>(getNbStages()) + 1));
        if (idxStage == 0){
            return mInputChannel->peekSize();
        }
        else{
            return mDownStream->getCurrentChannelSize(idxStage - 1);
        }
    }

private:
    void process() noexcept{
        if constexpr (isCustomizedStage) {
            mFunc(*getInputChannel(), *getStageOutputChannel());
        }
        else {
            while (true){
                std_optional<Input> input = mInputChannel->pop();
                if (!input.has_value()) {
                    mDownStream->close();
                    break;// closed
                }
                Output output = mFunc(std::move(input.value()));
                mDownStream->enqueue(std::move(output));
            }
        }
    }
private:
    uint32_t mIdxStage;
    Func0 mFunc;
    std::unique_ptr<DownStream> mDownStream;
    std::unique_ptr<Channel<Input>> mInputChannel = std::make_unique<Channel<Input>>((isCustomizedStage || mIdxStage == 0) ? std::numeric_limits<size_t>::max() : 16U);

    fb::future<void> mFiber;
};
// @fixme: add stack allocator or fiber creator! otherwise we get corrupted memory.
template <typename Product, typename Func0, typename... Funcs>
std::unique_ptr<PipeLine<Product, Func0, Funcs...>> makePipeLine(
    std::function<fb::future<void>(int32_t /*priority*/, std::function<void()> /*task*/)> fiberCreator,
    PipeLineChannel<Product>& productChannel, Func0&& func0, Funcs&&... funcs)
{
    return std::make_unique<PipeLine<Product, Func0, Funcs...>>(fiberCreator, 0u, productChannel, std::forward<Func0>(func0), std::forward<Funcs>(funcs)...);
}

} // namespace cudapp
