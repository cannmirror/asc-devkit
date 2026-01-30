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
 * \file tensor_tile_algorithm.h
 * \brief
 */
#ifndef IMPL_EXPERIMENTAL_TENSOR_API_STRUCT_CONSTRUCTOR_TENSOR_TILE_ALGORITHM_H
#define IMPL_EXPERIMENTAL_TENSOR_API_STRUCT_CONSTRUCTOR_TENSOR_TILE_ALGORITHM_H

namespace AscendC {
namespace TileInternal
{
template <typename Tuple>
using tuple_sequence = Std::make_index_sequence<Std::tuple_size_v<Std::remove_cvref_t<Tuple>>>;

template <typename T, typename F, size_t... I>
__aicore__ inline constexpr auto Apply(T&& t, F&& f, Std::index_sequence<I...>)
{
    return f(Std::get<I>(static_cast<T&&>(t))...);
}

template <typename T, typename F>
__aicore__ inline constexpr auto Apply(T&& t, F&& f)
{
    return Apply(static_cast<T&&>(t), f, tuple_sequence<T>{});
}

template <typename T, typename F, typename G, size_t... I>
__aicore__ inline constexpr auto TupleApply(T&& t, F&& f, G&& g, Std::index_sequence<I...>)
{
    return g(f(Std::get<I>(static_cast<T&&>(t)))...);
}

template <typename T0, typename T1, typename F, typename G, size_t... I>
__aicore__ inline constexpr auto TupleApply(T0&& t0, T1&& t1, F&& f, G&& g, Std::index_sequence<I...>)
{
    return g(f(Std::get<I>(static_cast<T0&&>(t0)),
                Std::get<I>(static_cast<T1&&>(t1)))...);
}

template <typename T, typename F>
__aicore__ inline constexpr auto Transform(const T& t, F&& f)
{
    if constexpr(Std::is_tuple_v<T>) {
        return TupleApply(t, f, [](auto const&... a){ return Std::make_tuple(a...);}, tuple_sequence<T>{});
    } else {
        return f(t);
    }
}

template <typename T, typename F, typename G>
__aicore__ inline constexpr auto TransformApply(T&& t, F&& f, G&& g)
{
    if constexpr (Std::is_tuple_v<Std::remove_cvref_t<T>>) {
        return TupleApply(static_cast<T&&>(t), f, g, tuple_sequence<T>{});
    } else {
        return g(f(static_cast<T&&>(t)));
    }
}

template <typename T0, typename T1, typename F, typename G>
__aicore__ inline constexpr auto TransformApply(T0&& t0, T1&& t1, F&& f, G&& g)
{
    if constexpr (Std::is_tuple_v<Std::remove_cvref_t<T0>>) {
        return TupleApply(static_cast<T0&&>(t0), static_cast<T1&&>(t1), f, g, tuple_sequence<T0>{});
    } else {
        return g(f(static_cast<T0&&>(t0), static_cast<T1&&>(t1)));
    }
}

template <typename Fn, typename Val>
struct FoldAdaptor {
    template <typename X>
    __aicore__ inline constexpr auto operator|(X&& x) {
        auto r =fn_(val_, static_cast<X&&>(x));
        return FoldAdaptor<Fn, decltype(r)>{fn_, r};
    }
    Fn fn_;
    Val val_;
};

template <typename T, typename V, typename F, size_t... Is>
__aicore__ inline constexpr auto Fold(T&& t, const V& v, F&& f, Std::index_sequence<Is...>)
{
    return (FoldAdaptor<F, V>{f,v}| ... | Std::get<Is>(static_cast<T&&>(t))).val_;
}

template <typename T, typename X, size_t... I, size_t... J, size_t... K>
__aicore__ inline constexpr auto Construct(const T& t, const X& x, Std::index_sequence<I...>, Std::index_sequence<J...>, Std::index_sequence<K...>)
{
    return Std::make_tuple(Std::get<I>(t)..., (void(J),x)..., Std::get<K>(t)...);
}

template<typename T, typename X>
__aicore__ inline constexpr auto Append(const T& a, const X& x)
{
    if constexpr (Std::is_tuple_v<T>) {
        return Construct(a, x, Std::make_index_sequence<Std::tuple_size_v<T>>{}, Std::index_sequence<0>{}, Std::index_sequence<>{});
    } else {
        return Std::make_tuple(a, x);
    }
}

template <typename T, typename F>
__aicore__ inline constexpr auto TransformLeaf(const T& t, F&& f)
{
    if constexpr (Std::is_tuple_v<T>) {
        return Transform(t, [&](const auto& a) { return TransformLeaf(a,f);});
    } else {
        return f(t);
    }
}
}
} 
#endif
