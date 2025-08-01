//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/* !
 * \file is_floating_point.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_TYPE_TRAITS_IS_FLOATING_POINT_H
#define AICORE_UTILS_STD_TYPE_TRAITS_IS_FLOATING_POINT_H

#include "is_same.h"
#include "remove_cv.h"

namespace AscendC {
namespace Std {

template <typename T>
struct is_floating_point {
private:
    template <typename Head, typename... Args>
    ASCENDC_HOST_AICORE inline static constexpr bool IsUnqualifiedAnyOf()
    {
        return (... || is_same_v<remove_cv_t<Head>, Args>);
    }

public:
    static constexpr bool value = IsUnqualifiedAnyOf<T, float, double, long double, half>();
};

template <typename Tp>
constexpr bool is_floating_point_v = is_floating_point<Tp>::value;

} // namespace Std
} // namespace AscendC

#endif // AICORE_UTILS_STD_TYPE_TRAITS_IS_FLOATING_POINT_H
