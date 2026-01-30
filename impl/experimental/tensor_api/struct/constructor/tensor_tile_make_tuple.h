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
 * \file tensor_tile_make_tuple.h
 * \brief
 */
#ifndef IMPL_EXPERIMENTAL_TENSOR_API_STRUCT_CONSTRUCTOR_TENSOR_TILE_MAKE_TUPLE_H
#define IMPL_EXPERIMENTAL_TENSOR_API_STRUCT_CONSTRUCTOR_TENSOR_TILE_MAKE_TUPLE_H

#include "tensor_tile_algorithm.h"

namespace AscendC {
namespace TileInternal
{
struct MultipliesUnaryLeftFold {
    template <typename... T>
    __aicore__ inline constexpr auto operator()(T&&... t) const {
        return (... * t);
    }
};

struct Product {
    template <typename T>
    __aicore__ inline constexpr auto operator()(const T& intT) const
    {
        if constexpr (Std::is_tuple_v<T>) {
            if constexpr (Std::tuple_size_v<T> == 0) {
                return Std::Int<1>{};
            } else {
                return TileInternal::TransformApply(intT, Product{}, MultipliesUnaryLeftFold{});
            }
        } else if constexpr (Std::is_integral<T>::value) {
            return intT;
        } else {
            static_assert(sizeof(T) == 0, "Invalid Product parameters");
        }
    }
};

static constexpr Product product;

template <size_t I, typename Tuple>
__aicore__ inline constexpr auto GetTuple(Tuple&& t) 
{
    static_assert(Std::is_tuple_v<Std::remove_cvref_t<Tuple>>, "Shape or Stride is not Tuple");
    auto&& tt = Std::get<I>(static_cast<Tuple&&>(t));
    if constexpr (Std::is_tuple_v<Std::remove_cvref_t<decltype(tt)>>) {
        return tt;
    } else {
        return Std::make_tuple(tt);
    }
}

template <size_t I0, size_t I1, size_t... Is, typename Tuple>
__aicore__ inline constexpr auto GetTuple(Tuple&& t) 
{
    static_assert(Std::is_tuple_v<Std::remove_cvref_t<Tuple>>, "Shape or Stride is not Tuple!");
    return GetTuple<I1, Is...>(GetTuple<I0>(static_cast<Tuple&&>(t)));
}

template <typename Tuple>
__aicore__ inline constexpr auto GetTuple(Tuple&& t) 
{
    static_assert(Std::is_tuple_v<Std::remove_cvref_t<Tuple>>, "Shape or Stride is not Tuple!");
    return static_cast<Tuple&&>(t);
}

template <size_t... Is, typename Tuple>
__aicore__ inline constexpr auto GetRank(const Tuple& t)
{
    static_assert(Std::is_tuple_v<Tuple>, "Shape or Stride is not Tuple!");
    if constexpr (sizeof...(Is) == 0) {
        return Std::Int<Std::tuple_size_v<Tuple>>{};
    } else {
        return GetRank(GetTuple<Is...>(t));
    }
}

template <size_t... Is, typename Tuple>
__aicore__ inline constexpr auto TupleSize(const Tuple& t)
{
    if constexpr (sizeof...(Is) == 0) {
        return Product{}(t);
    } else {
        return TupleSize(GetTuple<Is...>(t));
    }
}

template <size_t I, typename Tuple>
__aicore__ inline constexpr auto SelectTuple(Tuple&& t)
{
    static_assert(Std::is_tuple_v<Std::remove_cvref_t<Tuple>>, "Shape or Stride is not tuple");
    auto&& tt = Std::get<I>(static_cast<Tuple&&>(t));
    if constexpr (Std::is_tuple_v<Std::remove_cvref_t<decltype(tt)>>) {
        return tt;
    }else {
        return Std::make_tuple(tt);
    }
}

template <size_t I0, size_t I1, size_t... Is, typename Tuple>
__aicore__ inline constexpr auto SelectTuple(Tuple&& t)
{
     static_assert(Std::is_tuple_v<Std::remove_cvref_t<Tuple>>, "Shape or Stride is not tuple");
     return Std::make_tuple(Std::get<I0>(static_cast<Tuple&&>(t)), Std::get<I1>(static_cast<Tuple&&>(t)), Std::get<Is>(static_cast<Tuple&&>(t))...);
}

template <typename Tuple>
__aicore__ inline constexpr auto SelectTuple(Tuple&& t)
{
    static_assert(Std::is_tuple_v<Std::remove_cvref_t<Tuple>>, "Shape or Stride is not tuple");
    return static_cast<Tuple&&>(t);
}

template<size_t index, size_t I, size_t... Is, typename Tuple>
__aicore__ inline constexpr decltype(auto) GetValue(const Tuple& t)
{
    auto tupleEle = Std::get<index>(t);
    return Std::make_tuple(Std::get<I>(tupleEle), Std::get<Is>(tupleEle)...);
}

template<size_t index, typename Tuple>
__aicore__ inline constexpr decltype(auto) GetValue(const Tuple& t)
{
    return Std::get<index>(t);
}

template<typename Tuple>
__aicore__ inline constexpr auto GetShape(const Tuple& s)
{
    if constexpr (Std::is_tuple_v<Tuple>) {
        return TileInternal::Transform(s, [](const auto& a) { return GetShape(a);});
    } else {
        return s;
    }
}

template<size_t I, size_t... Is, typename Tuple>
__aicore__ inline constexpr auto GetShape(const Tuple& shape)
{
    if constexpr (Std::is_tuple_v<Tuple>) {
        return GetShape<Is...>(Std::get<I>(shape));
    } else {
        return GetTuple<I,Is...>(shape);
    }
}

template<typename T0, typename... Ts>
__aicore__ inline constexpr auto GetMax(const T0& t0, const Ts&... ts)
{
    if constexpr (Std::is_tuple_v<T0>) {
        return GetMax(Apply(t0, [](auto const&... a){ return GetMax(a...);}), ts...);
    } else if constexpr (sizeof...(Ts) == 0) {
        return t0;
    } else {
        return Std::max(t0, GetMax(ts...));
    }
}

template<typename TupleA, typename TupleB>
__aicore__ inline constexpr auto GetCapicitySize(const TupleA& a, const TupleB& b)
{
    if constexpr (Std::is_tuple_v<TupleA> && Std::is_tuple_v<TupleB>) {
        static_assert(Std::tuple_size_v<TupleA> == Std::tuple_size_v<TupleB>, "Mismatched ranks");
        return TransformApply(a, b, [](const auto& x, const auto& y) { return GetCapicitySize(x,y);},
                                    [](const auto&... v) { return GetMax(v...);});
    } else {
        return a * b;
    }
}

template<typename TupleA, typename TupleB>
__aicore__ inline constexpr auto InnerProduct(const TupleA& a, const TupleB& b)
{
    if constexpr (Std::is_tuple_v<TupleA> && Std::is_tuple_v<TupleB>) {
        static_assert(Std::tuple_size_v<TupleA> == Std::tuple_size_v<TupleB>, "Mismatched ranks");
        return TileInternal::TransformApply(a, b, [](const auto& x, const auto& y) { return InnerProduct(x,y);},
                                    [](const auto&... v) { return (Std::Int<0>{} + ... + v);});
    } else {
        return a * b;
    }
}
}
} 
#endif
