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
 * \file make_layout.h
 * \brief
 */
#ifndef EXPERIMENTAL_TENSOR_API_TENSOR_MAKE_LAYOUT_H
#define EXPERIMENTAL_TENSOR_API_TENSOR_MAKE_LAYOUT_H

#include "impl/experimental/tensor_api/detail/tensor/make_layout_impl.h"

namespace AscendC {
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

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto GetStride(const Layout<Shape, Stride>& layout);

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto GetStride(Layout<Shape, Stride>& layout);

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Select(const Layout<Shape, Stride>& layout);

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Size(const Layout<Shape, Stride>& layout);

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Capicity(const Layout<Shape, Stride>& layout);

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Coshape(const Layout<Shape, Stride>& layout);

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Cosize(const Layout<Shape, Stride>& layout);

template <typename T, typename U, typename S>
__aicore__ inline constexpr auto Crd2Idx(const T& coord, const Layout<U, S>& layout);

template <typename T, typename Shape, typename Stride>
__aicore__ inline constexpr auto Crd2Idx(const T& coord, const Shape& shape, const Stride& stride);
} // namespace AscendC

// make_fractal.h
namespace AscendC {
template <typename T>
__aicore__ inline decltype(auto) MakeNZLayout(size_t row, size_t column);

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
    using type = Layout<TensorInternal::NZShapeFormat<T, row, column>, TensorInternal::NZStrideFormat<T, row, column>>;
};

template <typename T, size_t row, size_t column>
struct NZLayoutFormat<T, row, column, typename Std::enable_if<Std::is_same_v<T, Std::ignore_t>>::type> {
    using type = Layout<TensorInternal::NZShapeFormat<uint16_t, row, column>, TensorInternal::NZStrideFormat<uint16_t, row, column>>;
};

template <typename T, size_t row, size_t column>
using NDLayoutFormat = Layout<TensorInternal::NDShapeFormat<T, row, column>, TensorInternal::NDStrideFormat<T, row, column>>;

template <typename T, size_t row, size_t column>
using DNLayoutFormat = Layout<TensorInternal::DNShapeFormat<T, row, column>, TensorInternal::DNStrideFormat<T, row, column>>;

template <typename T, size_t row, size_t column>
using ZNLayoutFormat = Layout<TensorInternal::ZNShapeFormat<T, row, column>, TensorInternal::ZNStrideFormat<T, row, column>>;

template <typename T, size_t row, size_t column>
using ZZLayoutFormat = Layout<TensorInternal::ZZShapeFormat<T, row, column>, TensorInternal::ZZStrideFormat<T, row, column>>;
} // namespace AscendC

#endif // EXPERIMENTAL_TENSOR_API_TENSOR_MAKE_LAYOUT_H
