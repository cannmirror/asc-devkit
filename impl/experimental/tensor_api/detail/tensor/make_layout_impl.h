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
#include "include/experimental/tensor_api/tensor/local_tensor.h"

namespace AscendC {
namespace TensorInternal {

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

template <typename Shape>
__aicore__ inline constexpr auto MakeLayoutImpl(const Shape& shape)
{
    static_assert(Std::is_tuple_v<Shape>, "Shape  is not tuple!");
    return MakeLayoutImpl(shape, CompactMajor<LayoutLeft>(shape));
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto RankImpl(const Layout<Shape, Stride>& layout)
{
    return layout.template Rank<Is...>();
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto GetShapeImpl(Layout<Shape, Stride>& layout)
{
    static_assert(
        Std::is_same_v<Layout<Shape, Stride>, Std::remove_cvref_t<decltype(layout)>>,
        "GetShape() called with a  layout of wrong type!"
    );
    return layout.template GetShape<Is...>();
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto GetShapeImpl(const Layout<Shape, Stride>& layout)
{
    static_assert(
        Std::is_same_v<Layout<Shape, Stride>, Std::remove_cvref_t<decltype(layout)>>,
        "GetShape() called with a  layout of wrong type!"
    );
    return layout.template GetShape<Is...>();
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto GetStrideImpl(Layout<Shape, Stride>& layout)
{
    static_assert(
        Std::is_same_v<Layout<Shape, Stride>, Std::remove_cvref_t<decltype(layout)>>,
        "GetStride() called with a  layout of wrong type!"
    );
    return layout.template GetStride<Is...>();
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto GetStrideImpl(const Layout<Shape, Stride>& layout)
{
    static_assert(
        Std::is_same_v<Layout<Shape, Stride>, Std::remove_cvref_t<decltype(layout)>>,
        "GetStride() called with a  layout of wrong type!"
    );
    return layout.template GetStride<Is...>();
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto SelectImpl(const Layout<Shape, Stride>& layout)
{
    static_assert(
        Std::is_same_v<Layout<Shape, Stride>, Std::remove_cvref_t<decltype(layout)>>,
        "Select() called with a  layout of wrong type!"
    );
    return MakeLayoutImpl(SelectTuple<Is...>(layout.GetShape()),
                            SelectTuple<Is...>(layout.GetStride()));
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto ShapeSizeImpl(const Layout<Shape, Stride>& layout)
{
    static_assert(
        Std::is_same_v<Layout<Shape, Stride>, Std::remove_cvref_t<decltype(layout)>>,
        "Size() called with a  layout of wrong type!"
    );
    return layout.template ShapeSize<Is...>();
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto CapicityImpl(const Layout<Shape, Stride>& layout)
{
    static_assert(
        Std::is_same_v<Layout<Shape, Stride>, Std::remove_cvref_t<decltype(layout)>>,
        "Capicity() called with a  layout of wrong type!"
    );
    return layout.template GetSize<Is...>();
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto CoshapeImpl(const Layout<Shape, Stride>& layout)
{
    auto m1Shapes = TransformLeaf(GetShapeImpl<Is...>(layout), [](auto s) { return s - Std::Int<1>{};});
    auto absStrides = TransformLeaf(GetStrideImpl<Is...>(layout), [](auto s) { return s < 0 ? -s : s;});
    auto coCoord = InnerProduct(m1Shapes, absStrides);
    return TransformLeaf(coCoord, [](auto c) { return c + Std::Int<1>{}; });
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto CosizeImpl(const Layout<Shape, Stride>& layout)
{
    return TupleSize(CoshapeImpl<Is...>(layout));
}

// NZ
template <typename T>
__aicore__ inline decltype(auto) MakeLayoutForNZ(size_t row, size_t column)
{
    auto shape = MakeShapeImpl(MakeShapeImpl(Std::Int<FRACTAL_FIXED>{}, row / FRACTAL_FIXED),
        MakeShapeImpl(Std::Int<C0_SIZE / sizeof(T)>{}, column / (C0_SIZE / sizeof(T))));
    auto stride = MakeStrideImpl(MakeStrideImpl(Std::Int<C0_SIZE / sizeof(T)>{}, Std::Int<C0_SIZE / sizeof(T) * FRACTAL_FIXED>{}),
        MakeStrideImpl(Std::Int<1>{}, C0_SIZE / sizeof(T) * row));
    return MakeLayoutImpl(shape, stride);
}

template <typename T, size_t row, size_t column>
using NZShapeFormat = Shape<Shape<Std::Int<FRACTAL_FIXED>, Std::Int<row / FRACTAL_FIXED>>,
    Shape<Std::Int<C0_SIZE / sizeof(T)>, Std::Int<column / (C0_SIZE / sizeof(T))>>>;

template <typename T, size_t row, size_t column>
using NZStrideFormat = Stride<Stride<Std::Int<C0_SIZE / sizeof(T)>, Std::Int<C0_SIZE / sizeof(T) * FRACTAL_FIXED>>,
    Stride<Std::Int<1>, Std::Int<C0_SIZE / sizeof(T) * row>>>;

// ND
template <typename T>
__aicore__ inline decltype(auto) MakeLayoutForND(size_t row, size_t column)
{
    auto shape = MakeShapeImpl(MakeShapeImpl(Std::Int<1>{}, row), MakeShapeImpl(Std::Int<1>{}, column));
    auto stride = MakeStrideImpl(MakeStrideImpl(Std::Int<0>{}, column), MakeStrideImpl(Std::Int<0>{},  Std::Int<1>{}));
    return MakeLayoutImpl(shape, stride);
}

template <typename T, size_t row, size_t column>
using NDShapeFormat = Shape<Shape<Std::Int<1>, Std::Int<row>>, Shape<Std::Int<1>, Std::Int<column>>>;

template <typename T, size_t row, size_t column>
using NDStrideFormat = Stride<Stride<Std::Int<0>, Std::Int<column>>, Stride<Std::Int<0>, Std::Int<1>>>;

// DN
template <typename T>
__aicore__ inline decltype(auto) MakeLayoutForDN(size_t row, size_t column)
{
    auto shape = MakeShapeImpl(MakeShapeImpl(Std::Int<1>{}, row), MakeShapeImpl(Std::Int<1>{}, column));
    auto stride = MakeStrideImpl(MakeStrideImpl(Std::Int<0>{}, Std::Int<1>{}), MakeStrideImpl(Std::Int<0>{}, row));
    return MakeLayoutImpl(shape, stride);
}

template <typename T, size_t row, size_t column>
using DNShapeFormat = Shape<Shape<Std::Int<1>, Std::Int<row>>, Shape<Std::Int<1>, Std::Int<column>>>;

template <typename T, size_t row, size_t column>
using DNStrideFormat = Stride<Stride<Std::Int<0>, Std::Int<1>>, Stride<Std::Int<0>, Std::Int<row>>>;

// ZN
template <typename T>
__aicore__ inline decltype(auto) MakeLayoutForZN(size_t row, size_t  column)
{
    auto shape = MakeShapeImpl(
        MakeShapeImpl(Std::Int<C0_SIZE / sizeof(T)>{}, row / (C0_SIZE / sizeof(T))),
        MakeShapeImpl(Std::Int<FRACTAL_FIXED>{}, column / FRACTAL_FIXED)
    );
    auto stride = MakeStrideImpl(
        MakeStrideImpl(Std::Int<1>{}, C0_SIZE / sizeof(T) * column),
        MakeStrideImpl(Std::Int<C0_SIZE / sizeof(T)>{}, Std::Int<C0_SIZE / sizeof(T) * FRACTAL_FIXED>{})
    );
    return MakeLayoutImpl(shape, stride);
}

template <typename T, size_t  row, size_t  column>
using ZNShapeFormat = Shape<Shape<Std::Int<C0_SIZE / sizeof(T)>, Std::Int<row / (C0_SIZE / sizeof(T))>>,
    Shape<Std::Int<FRACTAL_FIXED>, Std::Int<column / FRACTAL_FIXED>>>;
template <typename T, size_t  row, size_t  column>
using ZNStrideFormat = Stride<Stride<Std::Int<1>, Std::Int<C0_SIZE / sizeof(T) * column>>,
    Stride<Std::Int<C0_SIZE / sizeof(T)>, Std::Int<C0_SIZE / sizeof(T) * FRACTAL_FIXED>>>;

template <typename T>
__aicore__ inline decltype(auto) MakeLayoutForZZ(size_t row, size_t column) {
    auto shape = MakeShapeImpl(MakeShapeImpl(Std::Int<FRACTAL_FIXED>{}, row / FRACTAL_FIXED),
        MakeShapeImpl(Std::Int<C0_SIZE / sizeof(T)>{}, column / (C0_SIZE / sizeof(T))));
    auto stride = MakeStrideImpl(MakeStrideImpl(Std::Int<C0_SIZE / sizeof(T)>{}, FRACTAL_FIXED * column),
        MakeStrideImpl(Std::Int<1>{}, Std::Int<C0_SIZE / sizeof(T) * FRACTAL_FIXED>{}));
    return MakeLayoutImpl(shape, stride);
}

template <typename T, size_t row, size_t column>
using ZZShapeFormat = Shape<Shape<Std::Int<FRACTAL_FIXED>, Std::Int<row / FRACTAL_FIXED>>,
    Shape<Std::Int<C0_SIZE / sizeof(T)>, Std::Int<column / (C0_SIZE / sizeof(T))>>>;
template <typename T, size_t row, size_t column>
using ZZStrideFormat = Stride<Stride<Std::Int<C0_SIZE / sizeof(T)>, Std::Int<FRACTAL_FIXED * column>>,
    Stride<Std::Int<1>, Std::Int<C0_SIZE / sizeof(T) * FRACTAL_FIXED>>>;

}
} // namespace AscendC

// make_coord_impl.h
namespace AscendC {
namespace TensorInternal
{
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
    return Crd2IdxImpl(coord, layout.GetShape(), layout.GetStride());
}
}
} // namespace AscendC

// make_layout.h
namespace AscendC {
template <typename... Ts>
__aicore__ inline constexpr Shape<Ts...> MakeShape(const Ts&... t)
{
    return TensorInternal::MakeShapeImpl<Ts...>(t...);
}

template <typename... Ts>
__aicore__ inline constexpr Stride<Ts...> MakeStride(const Ts&... t)
{
    return TensorInternal::MakeStrideImpl<Ts...>(t...);
}

template <typename... Ts>
__aicore__ inline constexpr Tile<Ts...>  MakeTile(const Ts&... t)
{
    return TensorInternal::MakeTileImpl<Ts...>(t...);
}

template <typename... Ts>
__aicore__ inline constexpr Coord<Ts...> MakeCoord(const Ts&... t)
{
    return TensorInternal::MakeCoordImpl<Ts...>(t...);
}

template <typename T, typename U>
__aicore__ inline constexpr auto MakeLayout(const T& shape, const U& stride)
{
    return TensorInternal::MakeLayoutImpl(shape, stride);
}

template <typename T>
__aicore__ inline constexpr auto MakeLayout(const T& shape)
{
    return TensorInternal::MakeLayoutImpl(shape);
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Rank(const Layout<Shape, Stride>& layout)
{
    return TensorInternal::RankImpl<Is...>(layout);
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto GetShape(const Layout<Shape, Stride>& layout)
{
    return TensorInternal::GetShapeImpl<Is...>(layout);
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto GetShape(Layout<Shape, Stride>& layout)
{
    return TensorInternal::GetShapeImpl<Is...>(layout);
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto GetStride(const Layout<Shape, Stride>& layout)
{
    return TensorInternal::GetStrideImpl<Is...>(layout);
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto GetStride(Layout<Shape, Stride>& layout)
{
    return TensorInternal::GetStrideImpl<Is...>(layout);
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Select(const Layout<Shape, Stride>& layout)
{
    return TensorInternal::SelectImpl<Is...>(layout);
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Size(const Layout<Shape, Stride>& layout)
{
    return TensorInternal::ShapeSizeImpl<Is...>(layout);
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Capicity(const Layout<Shape, Stride>& layout)
{
    return TensorInternal::CapicityImpl<Is...>(layout);
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Coshape(const Layout<Shape, Stride>& layout)
{
    return TensorInternal::CoshapeImpl<Is...>(layout);
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Cosize(const Layout<Shape, Stride>& layout)
{
    return TensorInternal::CosizeImpl<Is...>(layout);
}

template <typename T, typename U, typename S>
__aicore__ inline constexpr auto Crd2Idx(const T& coord, const Layout<U, S>& layout)
{
    return TensorInternal::Crd2IdxImpl(coord, layout);
}

template <typename T, typename Shape, typename Stride>
__aicore__ inline constexpr auto Crd2Idx(const T& coord, const Shape& shape, const Stride& stride)
{
    return TensorInternal::Crd2IdxImpl(coord, shape, stride);
}
} // namespace AscendC

// make_fractal.h
namespace AscendC {
template <typename T>
__aicore__ inline decltype(auto) MakeNZLayout(size_t row, size_t column) {
    return TensorInternal::MakeLayoutForNZ<T>(row, column);
}

template <>
__aicore__ inline decltype(auto) MakeNZLayout<Std::ignore_t>(size_t row, size_t column) {
    return TensorInternal::MakeLayoutForNZ<uint16_t>(row, column);
}

template <typename T>
__aicore__ inline decltype(auto) MakeRowMajorLayout(size_t row, size_t column) {
    return TensorInternal::MakeLayoutForND<T>(row, column);
}

template <typename T>
__aicore__ inline decltype(auto) MakeColumnMajorLayout(size_t row, size_t column) {
    return TensorInternal::MakeLayoutForDN<T>(row, column);
}

template <typename T>
__aicore__ inline decltype(auto) MakeZNLayout(size_t row, size_t column) {
    return TensorInternal::MakeLayoutForZN<T>(row, column);
}

template <typename T>
__aicore__ inline decltype(auto) MakeZZLayout(size_t row, size_t column) {
    return TensorInternal::MakeLayoutForZZ<T>(row, column);
}
} // namespace AscendC

#endif // IMPL_TENSOR_API_TENSOR_MAKE_LAYOUT_IMPL_H
