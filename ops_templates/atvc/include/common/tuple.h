/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file tuple.h
 * \brief
 */
#ifndef ATVC_TUPLE_H
#define ATVC_TUPLE_H

#include "type_list.h"

namespace ATVC {

/////////////////////////////////////////////////////////////////////////////////////
template <typename... Ts>
struct Tuple {};

template <typename T, typename... Ts>
struct Tuple<T, Ts...> {
    T head;
    Tuple<Ts...> tail;
    
    __aicore__ inline Tuple() : head(), tail() {}
};

template <>
struct Tuple<> {};

/////////////////////////////////////////////////////////////////////////////////////
template <typename List>
struct TupleFromTypeList {};

template <typename... Ts>
struct TupleFromTypeList<TypeList<Ts...>> {
    using Type = Tuple<Ts...>;
};

/////////////////////////////////////////////////////////////////////////////////////
template <typename TupleType>
struct TupleSize {};

template <typename... Ts>
struct TupleSize<Tuple<Ts...>> {
    static constexpr size_t value = sizeof...(Ts);
};

/////////////////////////////////////////////////////////////////////////////////////
// 获取 Tuple 中第 N 个元素的类型
template <std::size_t N, typename TupleType>
struct TupleElemType {};

template <typename T, typename... Ts>
struct TupleElemType<0, Tuple<T, Ts...>> {
    using Type = T;
};

template <std::size_t N, typename T, typename... Ts>
struct TupleElemType<N, Tuple<T, Ts...>> {
    using Type = typename TupleElemType<N - 1, Tuple<Ts...>>::Type;
};

/////////////////////////////////////////////////////////////////////////////////////
// 获取特定索引的 Tuple 元素的引用
template <std::size_t N, typename TupleType>
__aicore__ inline typename TupleElemType<N, TupleType>::Type& TupleElemGet(TupleType& tuple) {
    if constexpr (N == 0) {
        return tuple.head;
    } else {
        return TupleElemGet<N - 1>(tuple.tail);
    }
}

}
#endif
