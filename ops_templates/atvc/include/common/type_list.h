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

#ifndef ATVC_TYPE_LIST_H
#define ATVC_TYPE_LIST_H

#include <cstddef>

namespace ATVC {
template <size_t N>
struct SizeValue {
    static const size_t value = N;
};

template <typename Acc, typename T>
struct SumSizes {
    using Type = SizeValue<Acc::value + sizeof(T)>;
};
/////////////////////////////////////////////////////////////////////////////////////
// TypeList 定义
template <typename... Ts>
struct TypeList {};

/////////////////////////////////////////////////////////////////////////////////////
// Size 元结构
template <typename List>
struct TypeListSize {};

template <typename... Ts>
struct TypeListSize<TypeList<Ts...>> {
    static constexpr size_t value = sizeof...(Ts);
};

/////////////////////////////////////////////////////////////////////////////////////
// Get 元结构
template <typename TypeList, std::size_t N>
struct TypeListGet {};

template <typename T, typename... Ts, std::size_t N>
struct TypeListGet<TypeList<T, Ts...>, N> {
    using Type = typename TypeListGet<TypeList<Ts...>, N - 1>::Type;
};

template <typename T, typename... Ts>
struct TypeListGet<TypeList<T, Ts...>, 0> {
    using Type = T;
};

template <std::size_t N>
struct TypeListGet<TypeList<>, N> {
    static_assert(N < 0, "Index out of bounds in TypeListGet");
};

/////////////////////////////////////////////////////////////////////////////////////
// ByteOffset 元结构
template <typename TypeList, std::size_t N>
struct TypeListByteOffset {};

template <std::size_t N>
struct TypeListByteOffset<TypeList<>, N> {
    static_assert(N < 0, "Index out of range for empty TypeList.");
    static constexpr std::size_t value = 0;
};

template <typename Head, typename... Tail>
struct TypeListByteOffset<TypeList<Head, Tail...>, 0> {
    static constexpr std::size_t value = 0;
};

template <typename Head, typename... Tail, std::size_t N>
struct TypeListByteOffset<TypeList<Head, Tail...>, N> {
    static_assert(N < sizeof...(Tail) + 1, "Index out of range for TypeList.");
    static constexpr std::size_t value = sizeof(Head) + TypeListByteOffset<TypeList<Tail...>, N - 1>::value;
};

/////////////////////////////////////////////////////////////////////////////////////
// Prepend 元结构
template <typename T, typename List>
struct TypeListPrepend {};

template <typename T, typename... Ts>
struct TypeListPrepend<T, TypeList<Ts...>> {
    using Type = TypeList<T, Ts...>;
};

/////////////////////////////////////////////////////////////////////////////////////
// Map 元结构
template <typename List, template <typename> class Mapper>
struct TypeListMap {};

template <template <typename> class Mapper>
struct TypeListMap<TypeList<>, Mapper> {
    using Type = TypeList<>;
};

template <typename Head, typename... Tail, template <typename> class Mapper>
struct TypeListMap<TypeList<Head, Tail...>, Mapper> {
private:
    using MappedHead = typename Mapper<Head>::Type;
    using MappedTail = typename TypeListMap<TypeList<Tail...>, Mapper>::Type;

public:
    using Type = typename TypeListPrepend<MappedHead, MappedTail>::Type;
};

/////////////////////////////////////////////////////////////////////////////////////
// Reduce 元结构
template <typename List, typename Init, template <typename, typename> class Reducer>
struct TypeListReduce {};

template <typename Init, template <typename, typename> class Reducer>
struct TypeListReduce<TypeList<>, Init, Reducer> {
    using Type = Init;
};

template <typename Head, typename... Tail, typename Init, template <typename, typename> class Reducer>
struct TypeListReduce<TypeList<Head, Tail...>, Init, Reducer> {
private:
    using NewInit = typename Reducer<Init, Head>::Type;
    using ReducedTail = typename TypeListReduce<TypeList<Tail...>, NewInit, Reducer>::Type;

public:
    using Type = ReducedTail;
};
}
#endif
