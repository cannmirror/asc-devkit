//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/* !
 * \file forward.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_UTILITY_FORWARD_H
#define AICORE_UTILS_STD_UTILITY_FORWARD_H

#include "../type_traits/remove_reference.h"
#include "../type_traits/is_reference.h"

namespace AscendC {
namespace Std {

template <typename Tp>
ASCENDC_HOST_AICORE inline constexpr Tp&& forward(remove_reference_t<Tp>& t) noexcept
{
    return static_cast<Tp&&>(t);
}

template <typename Tp>
ASCENDC_HOST_AICORE inline constexpr Tp&& forward(remove_reference_t<Tp>&& t) noexcept
{
    static_assert(!is_lvalue_reference<Tp>::value, "cannot forward an rvalue as an lvalue");
    return static_cast<Tp&&>(t);
}

} // namespace Std
} // namespace AscendC

#endif // AICORE_UTILS_STD_UTILITY_FORWARD_H
