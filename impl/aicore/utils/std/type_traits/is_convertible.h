//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/* !
 * \file is_convertible.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_TYPE_TRAITS_IS_CONVERTIBLE_H
#define AICORE_UTILS_STD_TYPE_TRAITS_IS_CONVERTIBLE_H

#include <cstdint>
#include "integral_constant.h"
#include "is_void.h"
#include "add_rvalue_reference.h"
#include "../utility/declval.h"

namespace AscendC {
namespace Std {

template <typename From, typename To>
struct IsConvertibleImpl {
private:
    template <typename T>
    ASCENDC_HOST_AICORE inline static auto TestReturnable(int32_t)
        -> decltype(void(static_cast<T (*)()>(nullptr)), true_type{});

    template <typename T>
    ASCENDC_HOST_AICORE inline static auto TestReturnable(uint32_t) -> false_type;

    template <typename F, typename T>
    ASCENDC_HOST_AICORE inline static auto TestImplicitlyConvertible(int32_t)
        -> decltype(void(declval<void (&)(T)>()(declval<F>())), true_type{});

    template <typename F, typename T>
    ASCENDC_HOST_AICORE inline static auto TestImplicitlyConvertible(uint32_t) -> false_type;

public:
    static constexpr bool value =
        decltype(TestReturnable<To>(0))::value && decltype(TestImplicitlyConvertible<From, To>(0))::value;
};

template <typename From, typename To>
struct is_convertible : bool_constant<(is_void_v<From> && is_void_v<To>) || IsConvertibleImpl<From, To>::value> {};

template <typename From, typename To>
constexpr bool is_convertible_v = is_convertible<From, To>::value;

template <typename Ty>
struct is_convertible<Ty&, volatile Ty&> : true_type {};

template <typename Ty>
struct is_convertible<volatile Ty&, volatile Ty&> : true_type {};

template <typename Ty>
struct is_convertible<Ty&, const volatile Ty&> : true_type {};

template <typename Ty>
struct is_convertible<volatile Ty&, const volatile Ty&> : true_type {};

template <typename Ty>
constexpr bool is_convertible_v<Ty&, volatile Ty&> = true;

template <typename Ty>
constexpr bool is_convertible_v<volatile Ty&, volatile Ty&> = true;

template <typename Ty>
constexpr bool is_convertible_v<Ty&, const volatile Ty&> = true;

template <typename Ty>
constexpr bool is_convertible_v<volatile Ty&, const volatile Ty&> = true;

} // namespace Std
} // namespace AscendC

#endif // AICORE_UTILS_STD_TYPE_TRAITS_IS_CONVERTIBLE_H
