/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file make_layout_impl.h
 * \brief
 */
#ifndef IMPL_TENSOR_API_TENSOR_MAKE_LAYOUT_IMPL_H
#define IMPL_TENSOR_API_TENSOR_MAKE_LAYOUT_IMPL_H

#include "include/experimental/tensor_api/utils/utils.h"
#include "impl/experimental/tensor_api/tensor/layout_struct.h"

namespace AscendC {
namespace Te {

template <typename... Ts>
__aicore__ inline constexpr Shape<Ts...> MakeShapeImpl(const Ts&... t)
{
    return {t...};
}

template <typename... Ts>
__aicore__ inline constexpr Stride<Ts...> MakeStrideImpl(const Ts&... t)
{
    return {t...};
}

template <typename... Ts>
__aicore__ inline constexpr Tile<Ts...> MakeTileImpl(const Ts&... t)
{
    return {t...};
}

template <typename... Ts>
__aicore__ inline constexpr Coord<Ts...> MakeCoordImpl(const Ts&... t)
{
    return {t...};
}

template <typename T, typename U>
__aicore__ inline constexpr auto MakeLayoutImpl(const T& shape, const U& stride)
{
    static_assert(Std::is_tuple_v<T> && Std::is_tuple_v<U>, "Shape or Stride is not tuple!");
    return Layout<T, U>(shape, stride);
}

// shape = ((x1, x2, ..., xn), (y1, y2, ..., yn))
// stride[0][0] = 1; stride[0][i] = shape[0][i-1]*shape[1][i-1]*stride[0][i-1]; stride[1][i] = shape[0][i]*stride[0][i]
template <size_t I, typename Row, typename Col>
struct StrideRowElem {
    __aicore__ static inline constexpr auto value(const Row& row, const Col& col) {
        if constexpr (I == 0) {
            return Std::Int<1>{};
        } else {
            return Std::get<I - 1>(row) * Std::get<I - 1>(col) *
                StrideRowElem<I - 1, Row, Col>::value(row, col);
        }
    }
};

template <size_t I, typename Row, typename Col>
struct StrideColElem {
    __aicore__ static inline constexpr auto value(const Row& row, const Col& col) {
        return Std::get<I>(row) * StrideRowElem<I, Row, Col>::value(row, col);
    }
};

template <typename Row, typename Col, size_t... Is>
__aicore__ inline constexpr auto BuildStrideRowImpl(const Row& row, const Col& col,
    Std::index_sequence<Is...>) {
    return MakeStrideImpl(StrideRowElem<Is, Row, Col>::value(row, col)...);
}

template <typename Row, typename Col, size_t... Is>
__aicore__ inline constexpr auto BuildStrideColImpl(const Row& row, const Col& col,
    Std::index_sequence<Is...>) {
    return MakeStrideImpl(StrideColElem<Is, Row, Col>::value(row, col)...);
}

template <typename ShapeType>
__aicore__ inline constexpr auto ComputeStride(const ShapeType& shape) {
    static_assert(Std::is_tuple_v<ShapeType> && Std::tuple_size_v<ShapeType> == 2,
        "ShapeType must be tuple of two tuples");
    const auto& row = Std::get<0>(shape);
    const auto& col = Std::get<1>(shape);
    static_assert(Std::tuple_size_v<Std::remove_cvref_t<decltype(row)>> ==
        Std::tuple_size_v<Std::remove_cvref_t<decltype(col)>>,
        "ShapeType rows must have same length");
    constexpr size_t N = Std::tuple_size_v<Std::remove_cvref_t<decltype(row)>>;
    using Row = Std::remove_cvref_t<decltype(row)>;
    using Col = Std::remove_cvref_t<decltype(col)>;
    auto stride0 = BuildStrideRowImpl(row, col, Std::make_index_sequence<N>{});
    auto stride1 = BuildStrideColImpl(row, col, Std::make_index_sequence<N>{});
    return MakeStrideImpl(stride0, stride1);
}

// shape = (x1, x2, x3, ..., xn) -> stride = (x2*x3*...*xn, ..., x_{n-1}*xn, xn, 1)
template <size_t I, typename ShapeType>
struct FlatStrideElem {
    __aicore__ static inline constexpr auto value(const ShapeType& shape) {
        constexpr size_t N = Std::tuple_size_v<ShapeType>;
        static_assert(N > 0, "ShapeType must not be empty");
        if constexpr (I == N - 1) {
            return Std::Int<1>{};
        } else {
            return FlatStrideElem<I + 1, ShapeType>::value(shape) * Std::get<I + 1>(shape);
        }
    }
};

template <typename ShapeType, size_t... Is>
__aicore__ inline constexpr auto BuildFlatStrideImpl(const ShapeType& shape,
    Std::index_sequence<Is...>) {
    return MakeStrideImpl(FlatStrideElem<Is, ShapeType>::value(shape)...);
}

template <typename ShapeType>
__aicore__ inline constexpr auto ComputeFlatStride(const ShapeType& shape) {
    static_assert(Std::is_tuple_v<ShapeType>, "ShapeType must be tuple");
    constexpr size_t N = Std::tuple_size_v<ShapeType>;
    return BuildFlatStrideImpl(shape, Std::make_index_sequence<N>{});
}

template <typename ShapeType>
__aicore__ inline constexpr auto MakeLayoutImpl(const ShapeType& shape) {
    static_assert(Std::is_tuple_v<ShapeType>, "ShapeType is not tuple!");
    using ElemT = Std::remove_cvref_t<decltype(Std::get<0>(shape))>;
    if constexpr (Std::is_tuple_v<ElemT>) {
        return MakeLayoutImpl(shape, ComputeStride(shape));
    } else {
        return MakeLayoutImpl(shape, ComputeFlatStride(shape));
    }
}

template <size_t... Is, typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto RankImpl(const Layout<ShapeType, StrideType>& layout)
{
    static_assert(Std::tuple_size_v<ShapeType> == Std::tuple_size_v<StrideType>, "The dimensions of the ShapeType and StrideType are not the same.");
    return layout.template Rank<Is...>();
}

template <size_t... Is, typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto GetShapeImpl(const Layout<ShapeType, StrideType>& layout)
{
    static_assert(Std::is_tuple_v<ShapeType> && Std::is_tuple_v<StrideType>, "ShapeType or StrideType is not tuple!");
    return layout.template Shape<Is...>();
}

template<typename T>
__aicore__ inline constexpr auto GetShapeImpl(const T& shape)
{
    static_assert(Std::is_tuple_v<T> || Std::is_integral_v<T>, "shape is not a tuple or integer");
    return shape;
}

template<size_t I, size_t... Is, typename Tuple>
__aicore__ inline constexpr auto GetShapeImpl(const Tuple& shape)
{
    if constexpr (Std::is_tuple_v<Tuple>) {
        return GetShapeImpl<Is...>(Std::get<I>(shape));
    } else {
        return GetTuple<I,Is...>(shape);
    }
}

template <size_t... Is, typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto GetStrideImpl(const Layout<ShapeType, StrideType>& layout)
{
    static_assert(Std::is_tuple_v<ShapeType> && Std::is_tuple_v<StrideType>, "ShapeType or StrideType is not tuple!");
    return layout.template Stride<Is...>();
}

template <size_t... Is, typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto GetImpl(const Layout<ShapeType, StrideType>& layout)
{
    static_assert(Std::is_tuple_v<ShapeType> && Std::is_tuple_v<StrideType>, "ShapeType or StrideType is not tuple!");
    return MakeLayoutImpl(GetTuple<Is...>(layout.Shape()),
                            GetTuple<Is...>(layout.Stride()));
}

template <size_t... Is, typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto SelectImpl(const Layout<ShapeType, StrideType>& layout)
{
    static_assert(Std::is_tuple_v<ShapeType> && Std::is_tuple_v<StrideType>, "ShapeType or StrideType is not tuple!");
    return MakeLayoutImpl(SelectTuple<Is...>(layout.Shape()),
                            SelectTuple<Is...>(layout.Stride()));
}

template <size_t... Is, typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto ShapeSizeImpl(const Layout<ShapeType, StrideType>& layout)
{
    static_assert(Std::is_tuple_v<ShapeType> && Std::is_tuple_v<StrideType>, "ShapeType or StrideType is not tuple!");
    return layout.template Size<Is...>();
}

template <typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto CapacityImpl(const Layout<ShapeType, StrideType>& layout)
{
    static_assert(Std::is_tuple_v<ShapeType> && Std::is_tuple_v<StrideType>, "ShapeType or StrideType is not tuple!");
    return layout.Capacity();
}

struct CoshapeSum {
    template <typename... Args>
    __aicore__ inline constexpr auto operator()(const Args&... args) const {
        return (Std::Int<0>{} + ... + args);
    }
};

struct CoshapeCompute {
    template <typename T, typename U>
    __aicore__ inline constexpr auto operator()(const T& shape, const U& stride) const {
        if constexpr (Std::is_tuple_v<T> && Std::is_tuple_v<U>) {
            static_assert(Std::tuple_size_v<T> == Std::tuple_size_v<U>, "Mismatched ranks");
            return TransformApply(shape, stride, CoshapeCompute{}, CoshapeSum{});
        } else {
            auto m1Shape = shape - Std::Int<1>{};
            auto absStride = stride < 0 ? -stride : stride;
            return m1Shape * absStride;
        }
    }
};

template <size_t... Is, typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto CoshapeImpl(const Layout<ShapeType, StrideType>& layout)
{
    auto shape = GetShapeImpl<Is...>(layout);
    auto stride = GetStrideImpl<Is...>(layout);
    auto coCoord = CoshapeCompute{}(shape, stride);
    return coCoord + Std::Int<1>{};
}

template <size_t... Is, typename ShapeType, typename StrideType>
__aicore__ inline constexpr auto CosizeImpl(const Layout<ShapeType, StrideType>& layout)
{
    return TupleSize(CoshapeImpl<Is...>(layout));
}

// NZ
template <typename T, size_t row, size_t column>
using NZShapeFormat = Shape<Shape<Std::Int<FRACTAL_FIXED>, Std::Int<row / FRACTAL_FIXED>>,
    Shape<Std::Int<C0_SIZE / sizeof(T)>, Std::Int<column / (C0_SIZE / sizeof(T))>>>;

template <typename T, size_t row, size_t column>
using NZStrideFormat = Stride<Stride<Std::Int<C0_SIZE / sizeof(T)>, Std::Int<C0_SIZE / sizeof(T) * FRACTAL_FIXED>>,
    Stride<Std::Int<1>, Std::Int<C0_SIZE / sizeof(T) * row>>>;

// ND
template <typename T, size_t row, size_t column>
using NDShapeFormat = Shape<Shape<Std::Int<1>, Std::Int<row>>, Shape<Std::Int<1>, Std::Int<column>>>;

template <typename T, size_t row, size_t column>
using NDStrideFormat = Stride<Stride<Std::Int<0>, Std::Int<column>>, Stride<Std::Int<0>, Std::Int<1>>>;

// DN
template <typename T, size_t row, size_t column>
using DNShapeFormat = Shape<Shape<Std::Int<1>, Std::Int<row>>, Shape<Std::Int<1>, Std::Int<column>>>;

template <typename T, size_t row, size_t column>
using DNStrideFormat = Stride<Stride<Std::Int<0>, Std::Int<1>>, Stride<Std::Int<0>, Std::Int<row>>>;

// ZN
template <typename T, size_t  row, size_t  column>
using ZNShapeFormat = Shape<Shape<Std::Int<C0_SIZE / sizeof(T)>, Std::Int<row / (C0_SIZE / sizeof(T))>>,
    Shape<Std::Int<FRACTAL_FIXED>, Std::Int<column / FRACTAL_FIXED>>>;
template <typename T, size_t  row, size_t  column>
using ZNStrideFormat = Stride<Stride<Std::Int<1>, Std::Int<C0_SIZE / sizeof(T) * column>>,
    Stride<Std::Int<C0_SIZE / sizeof(T)>, Std::Int<C0_SIZE / sizeof(T) * FRACTAL_FIXED>>>;

// ZZ
template <typename T, size_t row, size_t column>
using ZZShapeFormat = Shape<Shape<Std::Int<FRACTAL_FIXED>, Std::Int<row / FRACTAL_FIXED>>,
    Shape<Std::Int<C0_SIZE / sizeof(T)>, Std::Int<column / (C0_SIZE / sizeof(T))>>>;
template <typename T, size_t row, size_t column>
using ZZStrideFormat = Stride<Stride<Std::Int<C0_SIZE / sizeof(T)>, Std::Int<FRACTAL_FIXED * column>>,
    Stride<Std::Int<1>, Std::Int<C0_SIZE / sizeof(T) * FRACTAL_FIXED>>>;

// layout_construct.h
struct MakeTupleCons {
    template <typename... Ts>
    __aicore__ inline decltype(auto) operator()(Ts&&... ts) {
        return Std::make_tuple(Std::forward<Ts>(ts)...);
    }
};

template <typename F, typename T>
__aicore__ inline decltype(auto) Make2Params2Tuple(F&& f, T&& t);

template <typename F, typename T0, typename T1>
__aicore__ inline decltype(auto) Make2Params2Tuple(F&& f, T0&& t0, T1&& t1);

template <typename F, typename T0, typename T1, typename... Ts>
__aicore__ inline decltype(auto) Make2Params2Tuple(F&& f, T0&& t0, T1&& t1, Ts&&... ts);

template <typename T0, typename T1, typename T2, typename T3, typename... Ts>
__aicore__ inline decltype(auto) LayoutConstructor(T0&& t0, T1&& t1, T2&& t2, T3&& t3, Ts&&... ts) {
    auto shape = Make2Params2Tuple(MakeTupleCons{}, t0, t1, t2, t3);
    auto stride = Make2Params2Tuple(MakeTupleCons{}, ts...);
    return Layout(shape, stride);
}

template <typename F, typename T>
__aicore__ inline decltype(auto) Make2Params2Tuple(F&& f, T&& t) {
    return t;
}

template <typename F, typename T0, typename T1>
__aicore__ inline decltype(auto) Make2Params2Tuple(F&& f, T0&& t0, T1&& t1) {
    return f(t0, t1);
}

template <typename F, typename T0, typename T1, typename... Ts>
__aicore__ inline decltype(auto) Make2Params2Tuple(F&& f, T0&& t0, T1&& t1, Ts&&... ts) {
    auto tuple1 = Make2Params2Tuple(f, t0, t1);
    auto tuple2 = Make2Params2Tuple(f, ts...);
    return Make2Params2Tuple(f, tuple1, tuple2);
}

// layout_disptach.h
template <LayoutFormat format, typename T>
struct LayoutDispatcher;

template <typename T>
struct LayoutDispatcher<LayoutFormat::NZ, T> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return LayoutConstructor(Std::Int<FRACTAL_FIXED>{},  CeilDivision(row, FRACTAL_FIXED), 
                                Std::Int<C0_SIZE / sizeof(T)>{},  CeilDivision(column, (C0_SIZE / sizeof(T))), 
                                Std::Int<C0_SIZE / sizeof(T)>{},  Std::Int<C0_SIZE / sizeof(T) * FRACTAL_FIXED>{},
                                Std::Int<1>{},  C0_SIZE / sizeof(T) * row); 
    }
};

template <typename T>
struct LayoutDispatcher<LayoutFormat::ZN, T> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return LayoutConstructor(Std::Int<C0_SIZE / sizeof(T)>{},  CeilDivision(row, (C0_SIZE / sizeof(T))),
                                Std::Int<FRACTAL_FIXED>{},  CeilDivision(column, FRACTAL_FIXED),
                                Std::Int<1>{},  C0_SIZE / sizeof(T) * column,
                                Std::Int<C0_SIZE / sizeof(T)>{},  Std::Int<C0_SIZE / sizeof(T) * FRACTAL_FIXED>{});
    }
};

template <typename T>
struct LayoutDispatcher<LayoutFormat::DN, T> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return LayoutConstructor(Std::Int<1>{}, row, Std::Int<1>{}, column,
                                    Std::Int<0>{}, Std::Int<1>{}, Std::Int<0>{}, row);
    }
};

template <>
struct LayoutDispatcher<LayoutFormat::DN, fp8_e8m0_t> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return LayoutConstructor(row, Std::Int<1>{}, Std::Int<2>{}, CeilDivision(column, MX_SCALE_K0),
                                    Std::Int<MX_SCALE_K0>{}, row * column, Std::Int<1>{}, MX_SCALE_K0 * row);
    }
};

template <typename T>
struct LayoutDispatcher<LayoutFormat::ND, T> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return LayoutConstructor(Std::Int<1>{}, row, Std::Int<1>{}, column,
                                    Std::Int<0>{}, column, Std::Int<0>{}, Std::Int<1>{});
    }
};

template <typename T>
struct LayoutDispatcher<LayoutFormat::ZZ, T> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return LayoutConstructor(Std::Int<FRACTAL_FIXED>{}, CeilDivision(row, FRACTAL_FIXED),
                                    Std::Int<C0_SIZE / sizeof(T)>{}, CeilDivision(column, (C0_SIZE / sizeof(T))),
                                    Std::Int<C0_SIZE / sizeof(T)>{}, FRACTAL_FIXED * column,
                                    Std::Int<1>{}, Std::Int<C0_SIZE / sizeof(T) * FRACTAL_FIXED>{});
    }
};

template <>
struct LayoutDispatcher<LayoutFormat::ZZ, fp8_e8m0_t> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return LayoutConstructor(Std::Int<FRACTAL_FIXED>{}, CeilDivision(row, FRACTAL_FIXED),
                                    Std::Int<MX_SCALE_K0>{}, CeilDivision(column, MX_SCALE_K0),
                                    Std::Int<MX_SCALE_K0>{}, column * FRACTAL_FIXED,
                                    Std::Int<1>{}, Std::Int<C0_SIZE>{});
    }
};

template <>
struct LayoutDispatcher<LayoutFormat::NN, fp8_e8m0_t> {
    __aicore__ inline static decltype(auto) apply(size_t row, size_t column) {
        return LayoutConstructor(Std::Int<MX_SCALE_K0>{}, CeilDivision(row, MX_SCALE_K0),
                                    Std::Int<FRACTAL_FIXED>{}, column * MX_SCALE_K0,
                                    Std::Int<FRACTAL_FIXED>{}, FRACTAL_FIXED * MX_SCALE_K0,
                                    Std::Int<1>{}, CeilDivision(row, MX_SCALE_K0));
    }
};

// make_coord_impl.h
template <typename T, typename U, typename S>
__aicore__ inline constexpr auto Crd2IdxImpl(const T& coord, const U& shape, const S& stride);

template <typename T, typename U, typename S, size_t... Is>
__aicore__ inline constexpr auto Crd2IdxTTT(const T& coord, const U& shape, const S& stride,
    Std::index_sequence<Is...>)
{
    return (... + Crd2IdxImpl(Std::get<Is>(coord), Std::get<Is>(shape), Std::get<Is>(stride)));
}

template <typename T, typename U, typename S, size_t I0, size_t... Is>
__aicore__ inline constexpr auto Crd2IdxITT(const T& coord, const U& shape, const S& stride,
    Std::index_sequence<I0,Is...>)
{
    if constexpr (sizeof...(Is) == 0) {  // Avoid recursion and mod on single/last iter
        return Crd2IdxImpl(coord, Std::get<I0>(shape), Std::get<I0>(stride));
    } else if constexpr (Std::is_constant<0, T>::value) {
        return Crd2IdxImpl(Std::Int<0>{}, Std::get<I0>(shape), Std::get<I0>(stride)) +
            (Std::Int<0>{} + ... + Crd2IdxImpl(Std::Int<0>{}, Std::get<Is>(shape), Std::get<Is>(stride)));
    } else { // General case
        auto prod = Product{}(Std::get<I0>(shape));
        auto div = coord / prod;
        auto mod = coord % prod;
        return Crd2IdxImpl(mod, Std::get<I0>(shape), Std::get<I0>(stride)) +
            Crd2IdxITT(div, shape, stride, Std::index_sequence<Is...>{});
    }
}

template <typename T, typename U, typename S>
__aicore__ inline constexpr auto Crd2IdxImpl(const T& coord, const U& shape, const S& stride)
{
    if constexpr (Std::is_tuple_v<T>) {
        if constexpr (Std::is_tuple_v<U>) { // tuple tuple tuple
            static_assert(Std::tuple_size_v<T> == Std::tuple_size_v<U>, "Shape and Coord Mismatched Ranks");
            static_assert(Std::tuple_size_v<T> == Std::tuple_size_v<S>, "Stride and Coord Mismatched Ranks");
            return Crd2IdxTTT(coord, shape, stride, tuple_sequence<T>{});
        } else { // tuple "int" "int"
            static_assert(sizeof(T) == 0, "Invalid parameters, U is not tuple!");
        }
    } else {
        if constexpr (Std::is_tuple_v<U>) { // "int" tuple tuple
            static_assert(Std::tuple_size_v<U> == Std::tuple_size_v<S>, "Shape and Stride Mismatched Ranks");
            return Crd2IdxITT(coord, shape, stride, tuple_sequence<U>{});
        } else { // "int" "int" "int"
            return coord * stride;
        }
    }
}

template <typename T, typename U, typename S>
__aicore__ inline constexpr auto Crd2IdxImpl(const T& coord, const Layout<U, S>& layout)
{
    return Crd2IdxImpl(coord, layout.Shape(), layout.Stride());
}

// make_layout.h
template <typename... Ts>
__aicore__ inline constexpr Shape<Ts...> MakeShape(const Ts&... t)
{
    return MakeShapeImpl<Ts...>(t...);
}

template <typename... Ts>
__aicore__ inline constexpr Stride<Ts...> MakeStride(const Ts&... t)
{
    return MakeStrideImpl<Ts...>(t...);
}

template <typename... Ts>
__aicore__ inline constexpr Tile<Ts...>  MakeTile(const Ts&... t)
{
    return MakeTileImpl<Ts...>(t...);
}

template <typename... Ts>
__aicore__ inline constexpr Coord<Ts...> MakeCoord(const Ts&... t)
{
    return MakeCoordImpl<Ts...>(t...);
}

template <typename T, typename U>
__aicore__ inline constexpr auto MakeLayout(const T& shape, const U& stride)
{
    return MakeLayoutImpl(shape, stride);
}

template <typename T>
__aicore__ inline constexpr auto MakeLayout(const T& shape)
{
    return MakeLayoutImpl(shape);
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Rank(const Layout<Shape, Stride>& layout)
{
    return RankImpl<Is...>(layout);
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto GetShape(const Layout<Shape, Stride>& layout)
{
    return GetShapeImpl<Is...>(layout);
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto GetShape(Layout<Shape, Stride>& layout)
{
    return GetShapeImpl<Is...>(layout);
}

template <typename Tuple>
__aicore__ inline constexpr auto GetShape(const Tuple& shape)
{
    return GetShapeImpl(shape);
}

template <size_t I, size_t... Is, typename Tuple>
__aicore__ inline constexpr auto GetShape(const Tuple& shape)
{
    return GetShapeImpl<I, Is...>(shape);
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto GetStride(const Layout<Shape, Stride>& layout)
{
    return GetStrideImpl<Is...>(layout);
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto GetStride(Layout<Shape, Stride>& layout)
{
    return GetStrideImpl<Is...>(layout);
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Select(const Layout<Shape, Stride>& layout)
{
    return SelectImpl<Is...>(layout);
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Get(const Layout<Shape, Stride>& layout)
{
    return GetImpl<Is...>(layout);
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Size(const Layout<Shape, Stride>& layout)
{
    return ShapeSizeImpl<Is...>(layout);
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Capacity(const Layout<Shape, Stride>& layout)
{
    return CapacityImpl<Is...>(layout);
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Coshape(const Layout<Shape, Stride>& layout)
{
    return CoshapeImpl<Is...>(layout);
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Cosize(const Layout<Shape, Stride>& layout)
{
    return CosizeImpl<Is...>(layout);
}

template <typename T, typename U, typename S>
__aicore__ inline constexpr auto Crd2Idx(const T& coord, const Layout<U, S>& layout)
{
    return Crd2IdxImpl(coord, layout);
}

template <typename T, typename Shape, typename Stride>
__aicore__ inline constexpr auto Crd2Idx(const T& coord, const Shape& shape, const Stride& stride)
{
    return Crd2IdxImpl(coord, shape, stride);
}

// make_fractal.h
template <typename T>
__aicore__ inline decltype(auto) MakeNzLayout(size_t row, size_t column) {
    return LayoutDispatcher<LayoutFormat::NZ, T>::apply(row, column);
}

template <>
__aicore__ inline decltype(auto) MakeNzLayout<Std::ignore_t>(size_t row, size_t column) {
    return MakeNzLayout<uint16_t>(row, column);
}

__aicore__ inline decltype(auto) MakeL0CLayout(size_t row, size_t column) {
    return MakeNzLayout<uint16_t>(row, column);
}

template <typename T>
__aicore__ inline decltype(auto) MakeNDLayout(size_t row, size_t column) {
    return LayoutDispatcher<LayoutFormat::ND, T>::apply(row, column);
}

template <typename T>
__aicore__ inline decltype(auto) MakeDNLayout(size_t row, size_t column) {
    return LayoutDispatcher<LayoutFormat::DN, T>::apply(row, column);
}

template <typename T>
__aicore__ inline decltype(auto) MakeZnLayout(size_t row, size_t column) {
    return LayoutDispatcher<LayoutFormat::ZN, T>::apply(row, column);
}

template <typename T>
__aicore__ inline decltype(auto) MakeZzLayout(size_t row, size_t column) {
    return LayoutDispatcher<LayoutFormat::ZZ, T>::apply(row, column);
}

template <typename T>
__aicore__ inline decltype(auto) MakeNnLayout(size_t row, size_t column) {
    return LayoutDispatcher<LayoutFormat::NN, T>::apply(row, column);
}

template <typename T>
__aicore__ inline decltype(auto) MakeScaleANDLayout(size_t row, size_t column) {
    return LayoutDispatcher<LayoutFormat::ND, T>::apply(row, column);
}

template <typename T>
__aicore__ inline decltype(auto) MakeScaleADNLayout(size_t row, size_t column) {
    return LayoutDispatcher<LayoutFormat::DN, T>::apply(row, column);
}

template <typename T>
__aicore__ inline decltype(auto) MakeScaleBNDLayout(size_t row, size_t column) {
    return LayoutDispatcher<LayoutFormat::ND, T>::apply(column, row);
}

template <typename T>
__aicore__ inline decltype(auto) MakeScaleBDNLayout(size_t row, size_t column) {
    return LayoutDispatcher<LayoutFormat::DN, T>::apply(column, row);
}

class DimConversion {
public:
    template<typename T>
    __aicore__ inline auto Run(const T& tensor) {
        return ConvertTwoDim2FourDim(tensor);
    }
    
private:
    template<typename T>
    __aicore__ inline auto ConvertTwoDim2FourDim(const T& tensor) {
        auto layout = tensor.Layout();
        auto row = Std::get<0>(layout.Shape());
        auto column = Std::get<1>(layout.Shape());
        auto fourDimLayout = MakeNDLayout<Std::ignore_t>(row, column);
        return MakeTensorImpl(tensor.Engine().Begin(), fourDimLayout);
    }
};

template <typename T>
__aicore__ inline auto PreProcess(const T& tensor) {
    using  tensorShape = Std::remove_cvref_t<decltype(tensor.Shape())>;
    if constexpr (nesting_depth_v<tensorShape> == TWO_DIM_DATA) {
        return DimConversion{}.Run(tensor);
    } else {
        static_assert(nesting_depth_v<tensorShape> == FOUR_DIM_DATA, "Only support two or four dim LayoutType");
        return tensor;
    }
}

template <typename Layout, typename TileShape>
__aicore__ inline decltype(auto) MakeTileLayout(const Layout& layout, const TileShape& tileShape) {
    static_assert(Std::is_tuple_v<TileShape>);

    using OriginShape = Std::remove_cvref_t<decltype(layout.Shape())>;
    if constexpr (nesting_depth_v<TileShape> == nesting_depth_v<OriginShape>
                  && Std::tuple_size_v<TileShape> == Std::tuple_size_v<OriginShape>) {
        return MakeLayout(tileShape, layout.Stride());
    } else {
        static_assert(Std::tuple_size_v<TileShape> == TWO_DIM_DATA);

        const uint32_t rows = Std::get<0>(tileShape);
        const uint32_t cols = Std::get<1>(tileShape);

        const auto& innerRow = Std::get<0>(Std::get<0>(layout.Shape()));
        const auto& innerCol = Std::get<0>(Std::get<1>(layout.Shape()));

        using InnerRowType = Std::remove_cvref_t<decltype(innerRow)>;
        using InnerColType = Std::remove_cvref_t<decltype(innerCol)>;

        if constexpr (IsIntegralConstantV<InnerRowType> && IsIntegralConstantV<InnerColType>) {
            return MakeLayout(
                MakeShape(MakeShape(Std::Int<InnerRowType::value>{}, CeilDivision(rows, InnerRowType::value)),
                          MakeShape(Std::Int<InnerColType::value>{}, CeilDivision(cols, InnerColType::value))),
                layout.Stride());
        } else {
            return MakeLayout(
                MakeShape(MakeShape(innerRow, CeilDivision(rows, innerRow)),
                                        MakeShape(innerCol, CeilDivision(cols, innerCol))),
                layout.Stride());
        }
    }
}

} // namespace Te
} // namespace AscendC

#endif // IMPL_TENSOR_API_TENSOR_MAKE_LAYOUT_IMPL_H
