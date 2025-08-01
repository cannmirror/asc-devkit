//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/* !
 * \file is_const.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_TYPE_TRAITS_IS_CONST_H
#define AICORE_UTILS_STD_TYPE_TRAITS_IS_CONST_H

#include "integral_constant.h"

namespace AscendC {
namespace Std {

template <typename Tp>
struct is_const : public false_type {};

template <typename Tp>
struct is_const<Tp const> : public true_type {};

template <typename Tp>
constexpr bool is_const_v = is_const<Tp>::value;

} // namespace Std
} // namespace AscendC

#endif // AICORE_UTILS_STD_TYPE_TRAITS_IS_CONST_H
