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
 * \file basic_algorithm.h
 * \brief
 */
#ifndef EXPERIMENTAL_TENSOR_API_DETAIL_TENSOR_BASIC_ALGORITHM_H
#define EXPERIMENTAL_TENSOR_API_DETAIL_TENSOR_BASIC_ALGORITHM_H

#include "include/experimental/tensor_api/utils/utils.h"

namespace AscendC {
namespace TensorInternal
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

// make_tuple.h
namespace AscendC {
namespace TensorInternal {
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
                return TensorInternal::TransformApply(intT, Product{}, MultipliesUnaryLeftFold{});
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
        return TensorInternal::Transform(s, [](const auto& a) { return GetShape(a);});
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
        return TensorInternal::TransformApply(a, b, [](const auto& x, const auto& y) { return InnerProduct(x,y);},
                                    [](const auto&... v) { return (Std::Int<0>{} + ... + v);});
    } else {
        return a * b;
    }
}
}
} 

// make_stride.h
namespace AscendC {
namespace TensorInternal {

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
        return TensorInternal::Fold(shape, Std::make_tuple(Std::make_tuple(), current), Lambda{}, Seq{});
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
        return TensorInternal::Transform(shape, current, [&](auto const& s, auto const& c){ return CompactMajor<Major>(s, c);});
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
        return Std::make_tuple(TensorInternal::Append(Std::get<0>(init), Std::get<0>(result)), Std::get<1>(result));
    }
    template <typename Shape>
    using seq = tuple_sequence<Shape>;
};
}
}

// static_layout_size.h
namespace AscendC {
namespace TensorInternal {

template <typename T>
struct nesting_depth {
    static constexpr size_t value = 1;
};

template <>
struct nesting_depth<Std::tuple<>> {
    static constexpr size_t value = 0;
};

template <typename... Args>
struct nesting_depth<Std::tuple<Args...>> {
    static constexpr size_t value = (nesting_depth<Args>::value + ...);
};

template <typename T>
constexpr size_t nesting_depth_v = nesting_depth<T>::value;

template <size_t Dim, typename T, typename U>
struct IsStaticLayout {
private:
    template<typename T1>
    struct include_dynamic_type : Std::true_type {};

    template<size_t v>
    struct include_dynamic_type<Std::Int<v>> : Std::false_type {};

    template <typename... Args>
    struct include_dynamic_type<Std::tuple<Args...>> : Std::bool_constant<(include_dynamic_type<Args>::value || ...)> {};

    __aicore__ inline static constexpr auto TestStaticLayout()
    {
        if constexpr (nesting_depth_v<T> == Dim && 
            !(include_dynamic_type<T>::value || include_dynamic_type<U>::value)) {
            return true;
        }
        return false;
    }
public:
    static constexpr bool value = TestStaticLayout();
};

template<typename T, typename U>
struct StaticLayoutSize {
private:
    __aicore__ inline static constexpr auto GetFourDimStaticLayoutSize() 
    {
        using rowShapeType = typename Std::tuple_element<0, T>::type;
        using colShapeType = typename Std::tuple_element<1, T>::type;
        using rowStrideType = typename Std::tuple_element<0, U>::type;
        using colStrideType = typename Std::tuple_element<1, U>::type;

        using outterRowNumType = typename Std::tuple_element<1, rowShapeType>::type;
        using outterRowStrideType = typename Std::tuple_element<1, rowStrideType>::type;
        using outterColNumType = typename Std::tuple_element<1, colShapeType>::type;
        using outterColStrideType = typename Std::tuple_element<1, colStrideType>::type;

        return (outterRowNumType {} * outterRowStrideType {}) > (outterColNumType {} * outterColStrideType {}) ? 
            (outterRowNumType {} * outterRowStrideType {}) : (outterColNumType {} * outterColStrideType {});
    }

    __aicore__ inline static constexpr auto GetTwoDimStaticLayoutSize() 
    {
        using rowNumType = typename Std::tuple_element<0, T>::type;
        using colNumType = typename Std::tuple_element<1, T>::type;
        using rowStrideType = typename Std::tuple_element<0, U>::type;
        using colStrideType = typename Std::tuple_element<1, U>::type;

        return (rowNumType {} * rowStrideType {}) > (colNumType {} * colStrideType {}) ? 
            (rowNumType {} * rowStrideType {}) : (colNumType {} * colStrideType {});
    }

    __aicore__ inline static constexpr auto GetStaticLayoutSize() {
        if constexpr (IsStaticLayout<TensorInternal::FOUR_DIM_DATA, T, U>::value) {
            return GetFourDimStaticLayoutSize();
        } else if constexpr (IsStaticLayout<TensorInternal::TWO_DIM_DATA, T, U>::value) {
            return GetTwoDimStaticLayoutSize();
        } else {
            return Std::Int<0>{};
        }
    }
public:
    static constexpr size_t size = GetStaticLayoutSize();
};
}
} // namespace AscendC
#endif
