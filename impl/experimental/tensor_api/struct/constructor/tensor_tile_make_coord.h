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
 * \file tensor_tile_make_coord.h
 * \brief
 */
#ifndef IMPL_EXPERIMENTAL_TENSOR_API_STRUCT_CONSTRUCTOR_TENSOR_TILE_MAKE_COORD_H
#define IMPL_EXPERIMENTAL_TENSOR_API_STRUCT_CONSTRUCTOR_TENSOR_TILE_MAKE_COORD_H

#include "impl/experimental/tensor_api/struct/definition/tensor_tile_tensor.h"

namespace AscendC {
namespace TileInternal
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
#endif