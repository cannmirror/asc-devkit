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
 * \file tensor_tile_make_stride.h
 * \brief
 */
#ifndef IMPL_EXPERIMENTAL_TENSOR_API_STRUCT_CONSTRUCTOR_TENSOR_TILE_MAKE_STRIDE_H
#define IMPL_EXPERIMENTAL_TENSOR_API_STRUCT_CONSTRUCTOR_TENSOR_TILE_MAKE_STRIDE_H

#include "tensor_tile_make_tuple.h"

namespace AscendC {
namespace TileInternal
{
template <typename TupleType>
using tuple_sequence = Std::make_index_sequence<Std::tuple_size_v<Std::remove_cvref_t<TupleType>>>;

template<typename Major>
struct CompactLambda;

template <typename Major, typename Shape, typename Current>
__aicore__ inline constexpr auto Compact(const Shape& shape, const Current& current)
{
    if constexpr (Std::is_tuple_v<Shape>) {
        using Lambda =CompactLambda<Major>;
        using Seq = typename Lambda::template seq<Shape>;
        return TileInternal::Fold(shape, Std::make_tuple(Std::make_tuple(), current), Lambda{}, Seq{});
    } else {
        if constexpr (Std::is_constant<1, Shape>::value) {
            return Std::make_tuple(Std::Int<0>{}, current);
        } else {
            return Std::make_tuple(current, current * shape);
        }
    }
}

template <typename Major, typename Shape, typename Current = Std::Int<1>>
__aicore__ inline constexpr auto CompactMajor(const Shape& shape, const Current& current = {})
{
    if constexpr (Std::is_tuple_v<Current>) {
        static_assert(Std::is_tuple_v<Shape>, "Invalid parameters");
        static_assert(Std::tuple_size_v<Shape> == Std::tuple_size_v<Current>, "Mismatched Ranks");
        return TileInternal::Transform(shape, current, [&](auto const& s, auto const& c){ return CompactMajor<Major>(s, c);});
    }else {
        static_assert(Std::is_tuple_v<Shape> || Std::is_integral_v<Shape>, "Shape is not tuple or integer");
        return Std::get<0>(Compact<Major>(shape, current));
    }
}

struct LayoutLeft {
    template <typename Shape>
    using Apply = decltype(CompactMajor<LayoutLeft>(Std::declval<Shape>()));
};

template<>
struct CompactLambda<LayoutLeft>
{
    template <typename Init, typename Shape>
    __aicore__ inline constexpr auto operator()(const Init& init, const Shape& shape) {
        auto result = Compact<LayoutLeft>(shape, Std::get<1>(init));
        return Std::make_tuple(TileInternal::Append(Std::get<0>(init), Std::get<0>(result)), Std::get<1>(result));
    }
    template <typename Shape>
    using seq = tuple_sequence<Shape>;
};
}
} 
#endif
