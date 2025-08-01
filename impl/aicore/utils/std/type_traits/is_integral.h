//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/* !
 * \file is_integral.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_TYPE_TRAITS_IS_INTEGRAL_H
#define AICORE_UTILS_STD_TYPE_TRAITS_IS_INTEGRAL_H

#include "is_same.h"
#include "remove_cv.h"
#include "integral_constant.h"

namespace AscendC {
namespace Std {

template <typename T>
struct is_integral {
private:
    template <typename Tp, typename... Tps>
    ASCENDC_HOST_AICORE inline static constexpr bool IsUnqualifiedAnyOf()
    {
        return (... || is_same_v<remove_cv_t<Tp>, Tps>);
    }

public:
    static constexpr bool value = IsUnqualifiedAnyOf<T, bool, unsigned long long, long long, unsigned long, long,
        unsigned int, int, unsigned short, short, unsigned char, signed char, char>();
};

template <typename T, T v>
struct is_integral<integral_constant<T, v>> : true_type {};

template <typename T>
constexpr bool is_integral_v = is_integral<T>::value;

} // namespace Std
} // namespace AscendC

#endif // AICORE_UTILS_STD_TYPE_TRAITS_IS_INTEGRAL_H
