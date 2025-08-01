//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/* !
 * \file move.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_UTILITY_MOVE_H
#define AICORE_UTILS_STD_UTILITY_MOVE_H

#include "../type_traits/remove_reference.h"

namespace AscendC {
namespace Std {

template <typename Tp>
ASCENDC_HOST_AICORE inline constexpr remove_reference_t<Tp>&& move(Tp&& t) noexcept
{
    using Up = remove_reference_t<Tp>;
    return static_cast<Up&&>(t);
}

} // namespace Std
} // namespace AscendC

#endif // AICORE_UTILS_STD_UTILITY_MOVE_H
