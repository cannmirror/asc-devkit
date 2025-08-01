//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/* !
 * \file is_base_of.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_TYPE_TRAITS_IS_BASE_OF_H
#define AICORE_UTILS_STD_TYPE_TRAITS_IS_BASE_OF_H

#include <cstdint>
#include "integral_constant.h"
#include "is_class.h"

namespace AscendC {
namespace Std {

template <typename Base, typename Derived>
struct IsBaseOfImpl {
private:
    template <typename B>
    ASCENDC_HOST_AICORE inline static true_type TestPtrConv(const volatile B*);

    template <typename B>
    ASCENDC_HOST_AICORE inline static false_type TestPtrConv(const volatile void*);

    template <typename B, typename D>
    ASCENDC_HOST_AICORE inline static auto IsBaseOf(int32_t) -> decltype(TestPtrConv<B>(static_cast<D*>(nullptr)));

    template <typename B, typename D>
    ASCENDC_HOST_AICORE inline static auto IsBaseOf(uint32_t) -> true_type; // private or ambiguous base

public:
    static constexpr bool value = decltype(IsBaseOf<Base, Derived>(0))::value;
};

template <typename Base, typename Derived>
struct is_base_of : bool_constant<is_class_v<Base> && is_class_v<Derived> && IsBaseOfImpl<Base, Derived>::value> {};

template <typename Base, typename Derived>
constexpr bool is_base_of_v = is_base_of<Base, Derived>::value;

} // namespace Std
} // namespace AscendC

#endif // AICORE_UTILS_STD_TYPE_TRAITS_IS_BASE_OF_H
