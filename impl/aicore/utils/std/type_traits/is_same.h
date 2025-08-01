//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/* !
 * \file is_same.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_TYPE_TRAITS_IS_SAME_H
#define AICORE_UTILS_STD_TYPE_TRAITS_IS_SAME_H

#include "integral_constant.h"

namespace AscendC {
namespace Std {

template <typename TP, typename Up>
struct is_same : public false_type {};

template <typename TP>
struct is_same<TP, TP> : public true_type {};

template <typename TP, typename Up>
constexpr bool is_same_v = false;

template <typename TP>
constexpr bool is_same_v<TP, TP> = true;

template <typename Tp, typename Up>
using IsSame = bool_constant<is_same_v<Tp, Up>>;

template <typename Tp, typename Up>
using IsNotSame = bool_constant<!is_same_v<Tp, Up>>;

} // namespace Std
} // namespace AscendC

#endif // AICORE_UTILS_STD_TYPE_TRAITS_IS_SAME_H
