//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/* !
 * \file integral_constant.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_TYPE_TRAITS_INTEGRAL_CONSTANT_H
#define AICORE_UTILS_STD_TYPE_TRAITS_INTEGRAL_CONSTANT_H

namespace AscendC {
namespace Std {

template <typename Tp, Tp v>
struct integral_constant {
    static constexpr const Tp value = v;

    using value_type = Tp;
    using type = integral_constant;

    ASCENDC_HOST_AICORE inline constexpr operator value_type() const noexcept
    {
        return value;
    }

    ASCENDC_HOST_AICORE inline constexpr value_type operator()() const noexcept
    {
        return value;
    }
};

template <typename Tp, Tp v>
constexpr const Tp integral_constant<Tp, v>::value;

using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;

template <bool b>
using bool_constant = integral_constant<bool, b>;

template <size_t v>
using Int = integral_constant<size_t, v>;

} // namespace Std
} // namespace AscendC

#endif // AICORE_UTILS_STD_TYPE_TRAITS_INTEGRAL_CONSTANT_H
