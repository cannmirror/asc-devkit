/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file kernel_operator_layout.h
 * \brief
 */
#ifndef TIKCFW_IMPL_KERNEL_OPERATOR_LAYOUT_H
#define TIKCFW_IMPL_KERNEL_OPERATOR_LAYOUT_H

#include "include/utils/std/tuple.h"
#include "include/utils/std/type_traits.h"
#include "include/utils/std/utility.h"

namespace AscendC {

template <typename T, typename U, typename S>
__aicore__ inline constexpr auto Crd2Idx(const T& coord, const U& shape, const S& stride);

template <typename... Shapes>
using Shape = Std::tuple<Shapes...>;

template <typename... Strides>
using Stride = Std::tuple<Strides...>;

template <typename... Ts>
__aicore__ inline constexpr Shape<Ts...> MakeShape(const Ts&... t)
{
    return {t...};
}

template <typename... Ts>
__aicore__ inline constexpr Stride<Ts...> MakeStride(const Ts&... t)
{
    return {t...};
}

template <typename T, typename U>
struct Layout : private Std::tuple<T, U>
{
    __aicore__ inline constexpr Layout(const T& shape  = {}, const U& stride = {})
        : Std::tuple<T, U>(shape, stride)
    {
            static_assert(Std::is_tuple_v<T> && Std::is_tuple_v<U>, "Shape or Stride is not tuple!");
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
        return GetValue<0, I...>(static_cast<Std::tuple<T, U>&>(*this));
    }

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) GetShape() const
    {
        return GetValue<0, I...>(static_cast<const Std::tuple<T, U>&>(*this));
    }

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) GetStride()
    {
        return GetValue<1, I...>(static_cast<Std::tuple<T, U>&>(*this));
    }

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) GetStride() const
    {
        return GetValue<1, I...>(static_cast<const Std::tuple<T, U>&>(*this));
    }

    template <typename S>
    __aicore__ inline constexpr auto operator()(const S& coord) const
    {
        return Crd2Idx(coord, GetShape(), GetStride());
    }

private:
    template<size_t index, size_t I, size_t... Is, typename Tuple>
    __aicore__ inline constexpr decltype(auto) GetValue(const Tuple& t)
    {
        auto tupleEle = Std::get<index>(t);
        return Std::make_tuple(Std::get<I>(tupleEle), Std::get<Is>(tupleEle)...);
    }

    template<size_t index, size_t I, size_t... Is, typename Tuple>
    __aicore__ inline constexpr decltype(auto) GetValue(const Tuple& t) const
    {
        auto tupleEle = Std::get<index>(t);
        return Std::make_tuple(Std::get<I>(tupleEle), Std::get<Is>(tupleEle)...);
    }

    template<size_t index, typename Tuple>
    __aicore__ inline constexpr decltype(auto) GetValue(const Tuple& t)
    {
        return Std::get<index>(t);
    }

    template<size_t index, typename Tuple>
    __aicore__ inline constexpr decltype(auto) GetValue(const Tuple& t) const
    {
        return Std::get<index>(t);
    }
};

template <typename T, typename U>
__aicore__ inline constexpr auto MakeLayout(const T& shape, const U& stride)
{
    static_assert(Std::is_tuple_v<T> && Std::is_tuple_v<U>, "Shape or Stride is not tuple!");
    return Layout<T, U>(shape, stride);
}

template <typename T>
struct is_layout : Std::false_type {};

template <typename T, typename U>
struct is_layout<Layout<T, U>> : Std::true_type {};

template <typename T>
constexpr bool is_layout_v = is_layout<T>::value;

} // namespace AscendC
#endif
