#include "kmat.h"
#include <gtest/gtest.h>
#include <eigen3/Eigen/Dense>

TEST(kmat, solveGaussElim)
{
    kmat<float, 3, 3> A;
    kmat<float, 3, 2> b;
    auto mapA = toEigenMap(A);
    auto mapB = toEigenMap(b);
    mapA.setRandom();
    mapB.setRandom();

    const auto x = solveGaussElim(A, b);
    const auto refX = mapA.fullPivHouseholderQr().solve(mapB).eval();

    EXPECT_LE((toEigenMap(x) - refX).cwiseAbs().maxCoeff(), 1E-4f);
}
