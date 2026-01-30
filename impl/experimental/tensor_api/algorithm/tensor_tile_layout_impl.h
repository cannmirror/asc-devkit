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
 * \file tensor_tile_layout_impl.h
 * \brief
 */
#ifndef IMPL_EXPERIMENTAL_TENSOR_API_ALGORITHM_TENSOR_TILE_LAYOUT_IMPL_H
#define IMPL_EXPERIMENTAL_TENSOR_API_ALGORITHM_TENSOR_TILE_LAYOUT_IMPL_H

#include "impl/experimental/tensor_api/utils/tensor_tile_utils.h"
#include "impl/experimental/tensor_api/struct/tensor_tile_struct.h"

namespace AscendC {

template <typename... Ts>
__aicore__ inline constexpr Shape<Ts...> MakeShape(const Ts&... t)
{
    return TileInternal::MakeShapeImpl<Ts...>(t...);
}

template <typename... Ts>
__aicore__ inline constexpr Stride<Ts...> MakeStride(const Ts&... t)
{
    return TileInternal::MakeStrideImpl<Ts...>(t...);
}

template <typename... Ts>
__aicore__ inline constexpr Tile<Ts...>  MakeTile(const Ts&... t)
{
    return TileInternal::MakeTileImpl<Ts...>(t...);
}

template <typename... Ts>
__aicore__ inline constexpr Coord<Ts...> MakeCoord(const Ts&... t)
{
    return TileInternal::MakeCoordImpl<Ts...>(t...);
}

template <typename T, typename U>
__aicore__ inline constexpr auto MakeLayout(const T& shape, const U& stride)
{
    return TileInternal::MakeLayoutImpl(shape, stride);
}

template <typename T>
__aicore__ inline constexpr auto MakeLayout(const T& shape)
{
    return TileInternal::MakeLayoutImpl(shape);
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Rank(const Layout<Shape, Stride>& layout)
{
    return TileInternal::RankImpl<Is...>(layout);
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto GetShape(const Layout<Shape, Stride>& layout)
{
    return TileInternal::GetShapeImpl<Is...>(layout);
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto GetShape(Layout<Shape, Stride>& layout)
{
    return TileInternal::GetShapeImpl<Is...>(layout);
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto GetStride(const Layout<Shape, Stride>& layout)
{
    return TileInternal::GetStrideImpl<Is...>(layout);
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto GetStride(Layout<Shape, Stride>& layout)
{
    return TileInternal::GetStrideImpl<Is...>(layout);
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Select(const Layout<Shape, Stride>& layout)
{
    return TileInternal::SelectImpl<Is...>(layout);
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Size(const Layout<Shape, Stride>& layout)
{
    return TileInternal::ShapeSizeImpl<Is...>(layout);
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Capicity(const Layout<Shape, Stride>& layout)
{
    return TileInternal::CapicityImpl<Is...>(layout);
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Coshape(const Layout<Shape, Stride>& layout)
{
    return TileInternal::CoshapeImpl<Is...>(layout);
}

template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Cosize(const Layout<Shape, Stride>& layout)
{
    return TileInternal::CosizeImpl<Is...>(layout);
}

template <typename T, typename U, typename S>
__aicore__ inline constexpr auto Crd2Idx(const T& coord, const Layout<U, S>& layout)
{
    return TileInternal::Crd2IdxImpl(coord, layout);
}

template <typename T, typename Shape, typename Stride>
__aicore__ inline constexpr auto Crd2Idx(const T& coord, const Shape& shape, const Stride& stride)
{
    return TileInternal::Crd2IdxImpl(coord, shape, stride);
}
} // namespace AscendC

#endif // IMPL_EXPERIMENTAL_TENSOR_API_ALGORITHM_TENSOR_TILE_LAYOUT_IMPL_H
