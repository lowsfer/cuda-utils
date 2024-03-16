//
// Created by yao on 15/09/18.
//

#pragma once
#include <cstdint>
#include <device_launch_parameters.h>
#include <initializer_list>
#include <type_traits>
#include <cassert>
#include <cmath>
#include <algorithm>

#ifndef __CUDACC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#endif

template <typename T>
__device__ __host__ __forceinline__
T* badPtr(){ return reinterpret_cast<T*>(alignof(T)); }

template <typename T>
struct NullArray1D
{
    template<typename Index> __device__ __host__ __forceinline__
    T& operator[](Index) { return *badPtr<T>(); }
    template<typename Index> __device__ __host__ __forceinline__
    const T& operator[](Index) const { return *badPtr<T>(); }
};
template <typename T>
struct NullArray2D
{
    template<typename Index> __device__ __host__ __forceinline__
    NullArray1D<T>& operator[](Index) { return *(NullArray1D<T>*)1u;}
    template<typename Index> __device__ __host__ __forceinline__
    const NullArray1D<T>& operator[](Index) const { return *(NullArray1D<T>*)1u;}
};
template <typename T, std::uint32_t Rows, std::uint32_t Cols>
struct KMatDataType{
    using Type = T[Rows][Cols];
};
template <typename T, std::uint32_t Rows>
struct KMatDataType<T, Rows, 0u>{
    using Type = NullArray2D<T>;
};
template <typename T, std::uint32_t Cols>
struct KMatDataType<T, 0u, Cols>{
    using Type = NullArray2D<T>;
};
template <typename T>
struct KMatDataType<T, 0u, 0u>{
    using Type = NullArray2D<T>;
};

#ifdef __CUDACC__
#pragma nv_diag_suppress unsigned_compare_with_zero
#endif
template <typename T, std::uint32_t Rows, std::uint32_t Cols = 1>
struct alignas(Rows*Cols == 0 ? 1 : (sizeof(T)*Rows*Cols%16==0?16:(sizeof(T)*Rows*Cols%8==0?8:(sizeof(T)*Rows*Cols%4==0?4:sizeof(T))))) kmat
{
//    T _data[Rows][Cols];
    typename KMatDataType<T, Rows, Cols>::Type _data;

    using ValType = T;

    __device__ __host__ __forceinline__
    static constexpr std::uint32_t size() {return Rows*Cols;}

    kmat() = default;
    __device__ __host__ __forceinline__
    kmat(const T* src){
        for (unsigned i = 0; i < Rows; i++) {
            for (unsigned j = 0; j < Cols; j++) {
                (*this)(i, j) = src[i*Cols+j];
            }
        }
    }
    __device__ __host__ __forceinline__
    kmat(std::initializer_list<T>&& src):_data{}{
        assert(src.size() == Rows * Cols);
        for (unsigned i = 0; i < Rows; i++)
            for (unsigned j = 0; j < Cols; j++)
                _data[i][j] = src.begin()[i*Cols+j];
    }
    template <typename DType>
    __device__ __host__ __forceinline__
    kmat<DType, Rows, Cols> cast() const{
        kmat<DType, Rows, Cols> dst;
        for (unsigned i = 0; i < size(); i++)
            dst.data()[i] = DType(data()[i]);
        return dst;
    }
    __device__ __host__ __forceinline__
    T* data() {return &_data[0][0];}
    __device__ __host__ __forceinline__
    const T* data() const {return &_data[0][0];}

//    template <bool enabler = (Rows == 1 || Cols == 1), typename std::enable_if<enabler, int>::type = 1>
    __device__ __host__ __forceinline__
    auto array1d() const -> const T(&)[Rows*Cols] {return reinterpret_cast<const T(&)[Rows*Cols]>(_data);}
    template<typename Index>
    __device__ __host__ __forceinline__
    T& operator()(Index i, Index j) { assert(int(i) >= 0 && i < Index(Rows) && int(j) >= 0 && j < Index(Cols)); return _data[i][j]; }
    template<typename Index>
    __device__ __host__ __forceinline__
    const T& operator()(Index i, Index j) const{assert(int(i) >= 0 && i < Index(Rows) && int(j) >= 0 && j < Index(Cols)); return _data[i][j];}

    template<typename Index, typename Dummy = T&>
    __device__ __host__ __forceinline__
    std::enable_if_t<Rows == 1 || Cols ==1, Dummy>
            operator[](Index i) {return Rows==1 ? (*this)(Index(0), i) : (*this)(i, Index(0));}
    template<typename Index, typename Dummy = const T&>
    __device__ __host__ __forceinline__
    std::enable_if_t<Rows == 1 || Cols ==1, Dummy>
            operator[](Index i) const {return Rows==1 ? (*this)(Index(0), i) : (*this)(i, Index(0));}

    __device__ __host__ __forceinline__
    void assignScalar(T val){
        for (unsigned i = 0; i < Rows; i++)
            for (unsigned j = 0; j < Cols; j++)
                _data[i][j] = val;
    }
    template <std::size_t BRows, std::size_t BCols, typename Index>
    __device__ __host__ __forceinline__
    const kmat<T, BRows, BCols> block(Index i, Index j) const{
        kmat<T, BRows, BCols> result;
        for (unsigned m = 0; m < BRows; m++)
            for (unsigned n = 0; n < BCols; n++)
                result(m,n) = (*this)(m+i, n+j);
        return result;
    }
    template <typename Index, typename SrcMatType>
    __device__ __host__ __forceinline__
    void assignBlock(Index i, Index j, const SrcMatType& src){
//        static_assert((SrcMatType::rows() <= rows() && SrcMatType::cols() <= cols()), "fatal error"); // @fixme: enable this when if constexpr is available in cuda
        for (unsigned m = 0; m < src.rows(); m++)
            for (unsigned n = 0; n < src.cols(); n++)
                (*this)(m+i, n+j) = src(m, n);
    }
    template <typename Index>
    __device__ __host__ __forceinline__
    const kmat<T, 1, Cols> row(Index i) const{
        return block<1, Cols>(i, Index{0});
    }
    template <typename Index>
    __device__ __host__ __forceinline__
    const kmat<T, Rows, 1> col(Index i) const{
        return block<Rows, 1>(Index{0}, i);
    }

    __device__ __host__ __forceinline__
    static kmat<T, Rows, Cols> zeros(){
        kmat<T, Rows, Cols> ret;
        for (unsigned i = 0; i < Rows; i++)
            for (unsigned j = 0; j < Cols; j++)
                ret(i, j) = 0;
        return ret;
    }
    __device__ __host__ __forceinline__
    static kmat<T, Rows, Cols> ones(){
        kmat<T, Rows, Cols> ret;
        for (unsigned i = 0; i < Rows; i++)
            for (unsigned j = 0; j < Cols; j++)
                ret(i, j) = 1;
        return ret;
    }
    __device__ __host__ __forceinline__
    static kmat<T, Rows, Cols> eye(){
        auto ret = kmat<T, Rows, Cols>::zeros();
        for (unsigned i = 0; i < (Rows < Cols ? Rows : Cols); i++)
            ret(i,i) = 1;
        return ret;
    }

    template <std::uint32_t Rows2, std::uint32_t Cols2>
    __device__ __host__ __forceinline__ kmat<T, Rows, Cols2> operator*(const kmat<T, Rows2, Cols2>& other) const{
        static_assert(Cols == Rows2, "matrices cannot multiply");
        auto result = kmat<T, Rows, Cols2>::zeros();
        for (unsigned i = 0; i < Rows; i++){
            for (unsigned k = 0; k < Cols; k++){
                for (unsigned j = 0; j < Cols2; j++){
                    result(i, j) += (*this)(i, k) * other(k, j);
                }
            }
        }
        return result;
    }

    __device__ __host__ __forceinline__ kmat<T, Rows, Cols> operator*(T scale) const{
        kmat<T, Rows, Cols> result;
        for (unsigned i = 0; i < Rows; i++){
            for (unsigned j = 0; j < Cols; j++){
                result(i, j) = (*this)(i, j) * scale;
            }
        }
        return result;
    }

    __device__ __host__ __forceinline__ T sqrNorm() const{
        T result = 0;
        for (unsigned i = 0; i < Rows; i++){
            for (unsigned j = 0; j < Cols; j++){
                result += (*this)(i, j) * (*this)(i, j);
            }
        }
        return result;
    }

    __device__ __host__ __forceinline__
    bool allFinite() const {
#pragma unroll
        for (unsigned i = 0; i < Rows; i++){
#pragma unroll
            for (unsigned j = 0; j < Cols; j++){
                if (!std::isfinite((*this)(i, j))) {
                    return false;
                }
            }
        }
        return true;
    }

    __device__ __host__ __forceinline__ static constexpr uint32_t rows() {return Rows;}
    __device__ __host__ __forceinline__ static constexpr uint32_t cols() {return Cols;}

    __device__ __host__ __forceinline__ kmat<T, Cols, Rows> transpose() const{
        kmat<T, Cols, Rows> result;
        for (unsigned i = 0; i < Rows; i++) {
            for (unsigned j = 0; j < Cols; j++) {
                result(j, i) = (*this)(i, j);
            }
        }
        return result;
    }

    template <typename T1>
    __device__ __host__ __forceinline__
    kmat<std::decay_t<decltype(T{}+T1{})>, Rows, Cols> operator+(const kmat<T1, Rows, Cols>& other) const{
        kmat<std::decay_t<decltype(T{}+T1{})>, Rows, Cols> result;
        for (unsigned i = 0; i < Rows; i++)
            for (unsigned j = 0; j < Cols; j++)
                result(i, j) = (*this)(i, j) + other(i, j);
        return result;
    }

    __device__ __host__ __forceinline__
    kmat<T, Rows, Cols> operator-() const {
        kmat<T, Rows, Cols> result;
#pragma unroll
        for (unsigned i = 0; i < Rows; i++)
#pragma unroll
            for (unsigned j = 0; j < Cols; j++)
                result(i, j) = -(*this)(i, j);
        return result;
    }

    __device__ __host__ __forceinline__
    kmat<std::decay_t<decltype(T{} - T{})>, Rows, Cols> operator-(const kmat<T, Rows, Cols>& other) const{
        kmat<std::decay_t<decltype(T{} - T{})>, Rows, Cols> result;
#pragma unroll
        for (unsigned i = 0; i < Rows; i++)
#pragma unroll
            for (unsigned j = 0; j < Cols; j++)
                result(i, j) = (*this)(i, j) - other(i, j);
        return result;
    }

    __device__ __host__ __forceinline__
    bool operator==(const kmat<T, Rows, Cols>& other) const{
        for (unsigned i = 0; i < Rows; i++) {
            for (unsigned j = 0; j < Cols; j++)
                if ((*this)(i, j) != other(i, j))
                    return false;
        }
        return true;
    }
};

template <typename T, std::uint32_t Rows>
__device__ __host__ __forceinline__
kmat<T, Rows> toKmat(const T(&arr)[Rows]){return kmat<T, Rows>{arr};}

template <typename T, std::uint32_t Rows, std::uint32_t Cols = 1>
__device__ __host__ __forceinline__
kmat<T,Rows,Cols> operator+(T scalar, const kmat<T,Rows,Cols>& mat){
    return mat+scalar;
}
template <typename T, std::uint32_t Rows, std::uint32_t Cols = 1>
__device__ __host__ __forceinline__
kmat<T,Rows,Cols> operator*(T scalar, const kmat<T,Rows,Cols>& mat){
    return mat*scalar;
}

template <typename T, uint32_t M, uint32_t N, uint32_t K, typename AccType = T>
__device__ __host__ __forceinline__
kmat<AccType, M, N> mma(const kmat<T, M, K>& A, const kmat<T, K, N>& B, const kmat<AccType, M, N>& C){
    auto result = C;
    for (unsigned i = 0; i < M; i++){
        for (unsigned k = 0; k < K; k++){
            for (unsigned j = 0; j < N; j++){
                result(i, j) += A(i, k) * B(k, j);
            }
        }
    }
    return result;
}

template <typename T>
struct is_kmat : std::false_type{};
template <typename DataType, std::uint32_t Rows, std::uint32_t Cols>
struct is_kmat<kmat<DataType, Rows, Cols>> : std::true_type{};
#ifndef __CUDACC__
#include <eigen3/Eigen/Dense>
template <typename KMatType>
typename Eigen::Matrix<typename KMatType::ValType, KMatType::rows(), KMatType::cols(), KMatType::cols() == 1 ? Eigen::ColMajor : Eigen::RowMajor>
toEigenMap(const KMatType& mat){
    return typename Eigen::Matrix<typename KMatType::ValType, KMatType::rows(), KMatType::cols(), KMatType::cols() == 1 ? Eigen::ColMajor : Eigen::RowMajor>::ConstMapType(mat.data());
}
template <typename KMatType>
typename Eigen::Matrix<typename KMatType::ValType, KMatType::rows(), KMatType::cols(), KMatType::cols() == 1 ? Eigen::ColMajor : Eigen::RowMajor>::MapType
toEigenMap(KMatType& mat){
    return typename Eigen::Matrix<typename KMatType::ValType, KMatType::rows(), KMatType::cols(), KMatType::cols() == 1 ? Eigen::ColMajor : Eigen::RowMajor>::MapType(mat.data());
}
#endif

template <typename T, std::uint32_t Rows>
struct SymKMatDataType{
    using Type = T[Rows];
};
template <typename T>
struct SymKMatDataType<T, 0u> {
    using Type = NullArray1D<T>;
};
#ifdef __CUDACC__
#pragma nv_diag_suppress unsigned_compare_with_zero
#endif
template <typename T, std::uint32_t Rows>
struct alignas((sizeof(T)*Rows*(Rows+1)/2)%16==0?16:((sizeof(T)*Rows*(Rows+1)/2)%8==0?8:((sizeof(T)*Rows*(Rows+1)/2)%4==0?4:sizeof(T)))) symkmat{
    using ValType = T;
    __device__ __host__ __forceinline__
    static constexpr std::uint32_t rows() {return Rows;}
    __device__ __host__ __forceinline__
    static constexpr std::uint32_t cols() {return Rows;}
    __device__ __host__ __forceinline__
    static constexpr std::uint32_t size() {return Rows*(Rows+1)/2;}
    //store the upper triangle matrix
//    T _data[Rows*(Rows+1)/2];
    typename SymKMatDataType<T, size()>::Type _data;

    __device__ __host__ __forceinline__
    symkmat(){};
    //@fixme: add explicit
    __device__ __host__ __forceinline__
    symkmat(const kmat<T, Rows, Rows>& src){
        for (unsigned i = 0; i < Rows; i++){
            for (unsigned j = i; j < Rows; j++){
                assert(i == j || std::abs(src(i, j) - src(j, i)) < 0.01f || std::abs(src(i, j) - src(j, i)) / std::abs(src(i, j)) < 0.01f || (!std::isfinite(src(i, j)) && !std::isfinite(src(j, i))));
                (*this)(i, j) = src(i, j);
            }
        }
    };

    template<typename Index>
    __device__ __host__ __forceinline__
    constexpr T operator()(Index i, Index j) const {
        assert(int(i) >= 0 && i < Index(Rows) && int(j) >= 0 && j < Index(Rows));
        if (i > j){
            Index tmp = i;
            i = j;
            j = tmp;
        };
        const auto idx = (Rows + Rows - (i - 1)) * i / 2 + (j - i);
        assert(int(idx) >= 0 && idx < this->size());
        return _data[idx];
    }
    template<typename Index>
    __device__ __host__ __forceinline__
    T& operator()(Index i, Index j) {
        assert(int(i) >= 0 && i < Index(Rows) && int(j) >= 0 && j < Index(Rows));
        if (i > j){
            Index tmp = i;
            i = j;
            j = tmp;
        };
        const auto idx = (Rows + Rows - (i - 1)) * i / 2 + (j - i);
        assert(int(idx) >= 0 && idx < this->size());
        return _data[idx];
    }
    __device__ __host__ __forceinline__
    kmat<T, Rows, Rows> toKMat() const{
        return static_cast<kmat<T, Rows, Rows>>(*this);
    }
    __device__ __host__ __forceinline__
    operator kmat<T, Rows, Rows>() const{
        kmat<T, Rows, Rows> mat;
        for (unsigned i = 0; i < Rows; i++){
            mat(i, i) = (*this)(i, i);
            for (unsigned j = i+1; j < Rows; j++){
                mat(i, j) = mat(j, i) = (*this)(i, j);
            }
        }
        return mat;
    }
    __device__ __host__ __forceinline__
    const T* data() const { return &_data[0]; }
    __device__ __host__ __forceinline__
    T* data() { return &_data[0]; }
    __device__ __host__ __forceinline__
    symkmat<T, Rows> operator+(const symkmat<T, Rows>& other) const{
        symkmat<T, Rows> result;
        for (uint32_t i = 0; i < this->size(); i++)
            result._data[i] += _data[i] + other._data[i];
        return result;
    }
    template <typename DstType> __device__ __host__ __forceinline__
    symkmat<DstType, Rows> cast() const
    {
        symkmat<DstType, Rows> result;
        for (uint32_t i = 0; i < size(); i++){
            result._data[i] = static_cast<DstType>(_data[i]);
        }
        return result;
    }

    __device__ __host__ __forceinline__
    bool allFinite() const {
#pragma unroll
        for (uint32_t i = 0; i < size(); i++){
            if (!std::isfinite(data()[i])) {
                return false;
            }
        }
        return true;
    }
};

//returns a lower triangle matrix, i.e. L in LLT Cholesky decomposition
template <typename ScalarType, std::uint32_t Rows>
__device__ __host__ inline
kmat<ScalarType, Rows, Rows> cholesky(const symkmat<ScalarType, Rows>& src){
    auto L = kmat<ScalarType, Rows, Rows>::zeros();
#pragma unroll
    for (uint32_t j = 0; j < Rows; j++){
        {
            ScalarType acc = src(j, j);
#pragma unroll
            for (uint32_t k = 0; k < j; k++)
                acc -= L(j, k) * L(j, k);
            L(j, j) = std::sqrt(acc);
        }
        const ScalarType rcpLjj = 1 / L(j, j);
#pragma unroll
        for (uint32_t i = j + 1; i < Rows; i++){
            ScalarType acc = src(i, j);
#pragma unroll
            for (uint32_t k = 0; k < j; k++){
                acc -= L(i, k) * L(j, k);
            }
            L(i, j) = rcpLjj * acc;
        }
    }
#if 0
    assert(([&]()->bool{
        kmat<ScalarType, Rows, Rows> diff = src.toKMat() - L*L.transpose();
        for (uint32_t i = 0; i < diff.size(); i++) {
            if (std::abs(diff.data()[i]) > 1E-2f && std::abs(diff.data()[i]) > std::abs(diff.data()[i]) * 1E-2f)
                return false;
        }
        return true;
    }()));
#endif
    return L;
}

template <typename DataType, std::uint32_t Rows>
__device__ __host__ inline
symkmat<DataType, Rows> inverseByCholesky(const symkmat<DataType, Rows>& src){
    const auto L = cholesky(src); // L is a lower triangle matrix
    //inverse L
    kmat<DataType, Rows, Rows> invL = kmat<DataType, Rows, Rows>::zeros();
#pragma unroll
    for (uint32_t i = 0; i < src.rows(); i++){
        invL(i, i) = 1 / L(i, i);
#pragma unroll
#if 0
        // @fixme: File a bug. This causes spill (cuda-10.1). The compiler is dumb.
        for (uint32_t j = 0; j < i; j++){
#else
        for (uint32_t j = 0; j < src.rows(); j++){
            if (j >= i){ break; }
#endif
            DataType acc = 0;
#pragma unroll
            for (uint32_t k = j; k < i; k++){
                acc += L(i, k) * invL(k, j);
            }
            //invL(i, j) = -acc / L(i, i);
            invL(i, j) = acc * -invL(i, i);
        }
    }
    //calculate inversed src
    const kmat<DataType, Rows, Rows> invLT = invL.transpose();
    symkmat<DataType, Rows> dst = symkmat<DataType, Rows>(kmat<DataType, Rows, Rows>::zeros());
#pragma unroll
    for(uint32_t i = 0; i < Rows; i++){
#pragma unroll
        for (uint32_t j = 0; j <= i; j++){
#pragma unroll
            for (uint32_t k = std::max(i, j); k < Rows; k++) {
                dst(i, j) += invLT(i, k) * invL(k, j);
            }
        }
    }
    return dst;
}

template <typename ScalarType, std::uint32_t Rows>
__device__ __host__ inline
kmat<ScalarType, Rows> solveLowerTriangle(const kmat<ScalarType, Rows, Rows>& A, const kmat<ScalarType, Rows>& b);

template <typename ScalarType, std::uint32_t Rows, std::uint32_t Cols>
__device__ __host__ inline
kmat<ScalarType, Rows, Cols> solveGaussElim(const kmat<ScalarType, Rows, Rows>& A, const kmat<ScalarType, Rows, Cols>& b){
    assert(A.rows() == b.rows() && A.rows() == A.cols());
    kmat<ScalarType, Rows, Rows + Cols> Ab{};
    #pragma unroll
    for(uint32_t r = 0; r < A.rows(); r++){
        Ab.assignBlock(r, 0u, A.row(r));
        Ab.assignBlock(r, A.cols(), b.row(r));
    }
    #pragma unroll
    for (uint32_t i = 0; i < A.rows(); i++) {
        const ScalarType inv = 1.f / Ab(i, i);
        #pragma unroll
        for (uint32_t j = i + 1; j < A.rows(); j++) {
            const ScalarType factor = Ab(j, i) * inv;
            #pragma unroll
            for (uint32_t k = i + 1; k < A.cols() + b.cols(); k++) {
                Ab(j, k) -= factor * Ab(i, k);
            }
        }
    }
    kmat<ScalarType, Rows, Cols> x = Ab.template block<Rows, Cols>(0u, Rows);
    #pragma unroll
    for (int32_t i = A.rows() - 1; i >= 0; i--) {
        const ScalarType inv = 1.f / Ab(i, i);
        x.assignBlock(i, 0, x.row(i) * inv);
        #pragma unroll
        for (int32_t j = i - 1; j >= 0; j--){
            const ScalarType factor = Ab(j, i);
            x.assignBlock(j, 0, x.row(j) - factor * x.row(i));
        }
    }
    return x;
}

#ifndef __CUDACC__
#pragma GCC diagnostic pop
#endif
