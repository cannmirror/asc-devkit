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
 * \file layout.h
 * \brief
 */
#ifndef INCLUDE_TENSOR_API_TENSOR_LAYOUT_H
#define INCLUDE_TENSOR_API_TENSOR_LAYOUT_H

#include "impl/experimental/tensor_api/tensor/layout_impl.h"

namespace AscendC {
namespace Te {

// make_layout.h
template <typename... Ts>
__aicore__ inline constexpr Shape<Ts...> MakeShape(const Ts&... t);

template <typename... Ts>
__aicore__ inline constexpr Stride<Ts...> MakeStride(const Ts&... t);

template <typename... Ts>
__aicore__ inline constexpr Tile<Ts...>  MakeTile(const Ts&... t);

template <typename... Ts>
__aicore__ inline constexpr Coord<Ts...> MakeCoord(const Ts&... t);

template <typename T, typename U>
__aicore__ inline constexpr auto MakeLayout(const T& shape, const U& stride);

template <typename T>
__aicore__ inline constexpr auto MakeLayout(const T& shape);

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Rank(const Layout<Shape, Stride>& layout);

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto GetShape(const Layout<Shape, Stride>& layout);

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto GetShape(Layout<Shape, Stride>& layout);

template <typename Tuple>
__aicore__ inline constexpr auto GetShape(const Tuple& shape);

template <size_t I, size_t... Is, typename Tuple>
__aicore__ inline constexpr auto GetShape(const Tuple& shape);

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto GetStride(const Layout<Shape, Stride>& layout);

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto GetStride(Layout<Shape, Stride>& layout);

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Select(const Layout<Shape, Stride>& layout);

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Get(const Layout<Shape, Stride>& layout);

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Size(const Layout<Shape, Stride>& layout);

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Capacity(const Layout<Shape, Stride>& layout);

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Coshape(const Layout<Shape, Stride>& layout);

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Cosize(const Layout<Shape, Stride>& layout);

template <typename T, typename U, typename S>
__aicore__ inline constexpr auto Crd2Idx(const T& coord, const Layout<U, S>& layout);

template <typename T, typename Shape, typename Stride>
__aicore__ inline constexpr auto Crd2Idx(const T& coord, const Shape& shape, const Stride& stride);

// make_fractal.h
template <typename T>
__aicore__ inline decltype(auto) MakeNZLayout(size_t row, size_t column);

__aicore__ inline decltype(auto) MakeL0CLayout(size_t row, size_t column);

template <typename T>
__aicore__ inline decltype(auto) MakeRowMajorLayout(size_t row, size_t column);

template <typename T>
__aicore__ inline decltype(auto) MakeColumnMajorLayout(size_t row, size_t column);

template <typename T>
__aicore__ inline decltype(auto) MakeZNLayout(size_t row, size_t column);

template <typename T>
__aicore__ inline decltype(auto) MakeZZLayout(size_t row, size_t column);

template <typename T, size_t row, size_t column, typename Enable = void>
struct NZLayoutFormat;

template <typename T, size_t row, size_t column>
struct NZLayoutFormat<T, row, column, typename Std::enable_if<!Std::is_same_v<T, Std::ignore_t>>::type> {
    using type = Layout<NZShapeFormat<T, row, column>, NZStrideFormat<T, row, column>>;
};

template <typename T, size_t row, size_t column>
struct NZLayoutFormat<T, row, column, typename Std::enable_if<Std::is_same_v<T, Std::ignore_t>>::type> {
    using type = Layout<NZShapeFormat<uint16_t, row, column>, NZStrideFormat<uint16_t, row, column>>;
};

template <typename T, size_t row, size_t column>
using NDLayoutFormat = Layout<NDShapeFormat<T, row, column>, NDStrideFormat<T, row, column>>;

template <typename T, size_t row, size_t column>
using DNLayoutFormat = Layout<DNShapeFormat<T, row, column>, DNStrideFormat<T, row, column>>;

template <typename T, size_t row, size_t column>
using ZNLayoutFormat = Layout<ZNShapeFormat<T, row, column>, ZNStrideFormat<T, row, column>>;

template <typename T, size_t row, size_t column>
using ZZLayoutFormat = Layout<ZZShapeFormat<T, row, column>, ZZStrideFormat<T, row, column>>;

template <size_t row, size_t column>
using L0CLayoutFormat = NZLayoutFormat<Std::ignore_t, row, column>;

 template <typename Layout, typename TileShape>
__aicore__ inline decltype(auto) MakeTileLayout(const Layout& layout, const TileShape& tileShape);

} // namespace Te
} // namespace AscendC

#endif // INCLUDE_TENSOR_API_TENSOR_LAYOUT_H
