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
 * \file tensor_tile_make_layout.h
 * \brief
 */
#ifndef IMPL_EXPERIMENTAL_TENSOR_API_STRUCT_CONSTRUCTOR_TENSOR_TILE_MAKE_LAYOUT_H
#define IMPL_EXPERIMENTAL_TENSOR_API_STRUCT_CONSTRUCTOR_TENSOR_TILE_MAKE_LAYOUT_H

#include "impl/experimental/tensor_api/struct/definition/tensor_tile_tensor.h"

namespace AscendC {
namespace TileInternal
{

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
#endif
