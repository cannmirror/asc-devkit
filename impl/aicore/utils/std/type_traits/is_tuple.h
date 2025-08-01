//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/*!
 * \file is_tuple.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_TYPE_TRAITS_IS_TUPLE_H
#define AICORE_UTILS_STD_TYPE_TRAITS_IS_TUPLE_H

#include <cstdint>
#include "integral_constant.h"

namespace AscendC {
namespace Std {

template <typename T>
struct IsTupleImpl {
private:
    template <typename Ts>
    ASCENDC_HOST_AICORE inline static auto HasTupleSize(int32_t) -> bool_constant<(tuple_size<Ts>::value >= 0)>;

    template <typename Ts>
    ASCENDC_HOST_AICORE inline static auto HasTupleSize(uint32_t) -> false_type;

public:
    static constexpr bool value = decltype(HasTupleSize<T>(static_cast<int32_t>(0)))::value;
};

template <typename T>
struct is_tuple : bool_constant<IsTupleImpl<T>::value> {};

template <typename T>
constexpr bool is_tuple_v = is_tuple<T>::value;

} // namespace Std
} // namespace AscendC

#endif // AICORE_UTILS_STD_TYPE_TRAITS_IS_TUPLE_H
