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
 * \file tensor_tile_layout.h
 * \brief
 */
#ifndef IMPL_EXPERIMENTAL_TENSOR_API_STRUCT_TENSOR_TILE_LAYOUT_H
#define IMPL_EXPERIMENTAL_TENSOR_API_STRUCT_TENSOR_TILE_LAYOUT_H

#include "tensor_tile_StaticLayoutSize.h"
#include "impl/experimental/tensor_api/struct/constructor/tensor_tile_make_stride.h"

namespace AscendC {

template <typename... Shapes>
using Shape = Std::tuple<Shapes...>;

template <typename... Strides>
using Stride = Std::tuple<Strides...>;

template <typename... Layouts>
using Tile = Std::tuple<Layouts...>;

template <typename... Coords>
using Coord = Std::tuple<Coords...>;

template <typename T, typename U = TileInternal::LayoutLeft::Apply<T>>
struct Layout : private Std::tuple<T, U>
{
    static constexpr auto size = TileInternal::StaticLayoutSize<T, U>::size;

    __aicore__ inline constexpr Layout(const T& shape  = {}, const U& stride = {})
        : Std::tuple<T, U>(shape, stride)
    {
        static_assert(Std::is_tuple_v<T> && Std::is_tuple_v<U>, "Shape or Stride is not tuple!");
    }

    __aicore__ inline constexpr decltype(auto) GetSize() const
    {
        return TileInternal::GetCapicitySize(GetShape(), GetStride());
    }

    __aicore__ inline constexpr decltype(auto) layout()
    {
        return *this;
    }

    __aicore__ inline constexpr decltype(auto) layout() const
    {
        return *this;
    }

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) GetShape()
    {
        return TileInternal::GetValue<0, I...>(static_cast<Std::tuple<T, U>&>(*this));
    }

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) GetShape() const
    {
        return TileInternal::GetValue<0, I...>(static_cast<const Std::tuple<T, U>&>(*this));
    }

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) GetStride()
    {
        return TileInternal::GetValue<1, I...>(static_cast<Std::tuple<T, U>&>(*this));
    }

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) GetStride() const
    {
        return TileInternal::GetValue<1, I...>(static_cast<const Std::tuple<T, U>&>(*this));
    }

    template <typename S>
    __aicore__ inline constexpr auto operator()(const S& coord) const
    {
        return Crd2Idx(coord, *this);
    }

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) Rank() const
    {
        static_assert(Std::tuple_size_v<T> == Std::tuple_size_v<U>, "The dimensions of the Shape and Stride are not the same.");
        return TileInternal::GetRank<I...>(GetShape());
    }

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) ShapeSize() const
    {
        return TileInternal::TupleSize<I...>(GetShape());
    }

    static constexpr auto rank = Std::tuple_size_v<T>;
};

template <typename T>
struct is_layout : Std::false_type {};

template <typename T, typename U>
struct is_layout<Layout<T, U>> : Std::true_type {};

template <typename T>
constexpr bool is_layout_v = is_layout<T>::value;

} // namespace AscendC
#endif
