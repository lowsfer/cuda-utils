#include <gtest/gtest.h>
#include "../VectorMap.h"
#include <random>

class VectorMapTests : public testing::Test
{
public:
    void TearDown() override {
        mMap.clear();
        mRef.clear();
    }
    void checkRef() const {
        EXPECT_EQ(mMap.size(), mRef.size());
        for (auto& item : mRef) {
            EXPECT_EQ(item.second, mMap.at(item.first));
        }
    }
protected:
    std::mt19937_64 mRandEng{0};
    std::uniform_real_distribution<float> mRealDist{0.f, 1.f};
    VectorMap<int, uint64_t> mMap{};
    std::map<size_t, uint64_t> mRef;
};

TEST_F(VectorMapTests, emplaceAndErase)
{
    auto tmp = mMap.try_emplace(1, 2u);
    EXPECT_EQ(*tmp.first, 2ul);
    EXPECT_TRUE(tmp.second);
    mRef.try_emplace(1, 2u);
    checkRef();

    mMap.try_emplace(2, 3u);
    EXPECT_EQ(mMap.size(), 2ul);
    EXPECT_EQ(mMap.at(2), 3ul);
    mRef.try_emplace(2, 3u);
    checkRef();

    tmp = mMap.try_emplace(1, 4u);
    EXPECT_FALSE(tmp.second);
    EXPECT_EQ(*tmp.first, 2ul);
    mRef.try_emplace(1, 4u);
    checkRef();

    EXPECT_EQ(mMap.erase(1), 1ul);
    mRef.erase(1);
    checkRef();
}

TEST_F(VectorMapTests, iterator)
{
    mMap.try_emplace(1, 2ul);
    mMap.try_emplace(3, 4ul);
    mMap.try_emplace(6, 1ul);

    mRef.try_emplace(1, 2ul);
    mRef.try_emplace(3, 4ul);
    mRef.try_emplace(6, 1ul);

    auto iter = mMap.begin();
    auto iterRef = mRef.begin();
    do {
        EXPECT_EQ(iter.key(), iterRef->first);
        EXPECT_EQ(*iter, iterRef->second);
        iter++;
        iterRef++;
    }while (iter != mMap.end());
    EXPECT_EQ(iterRef, mRef.end());
}


