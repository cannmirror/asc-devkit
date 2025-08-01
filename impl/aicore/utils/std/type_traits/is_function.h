//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/* !
 * \file is_function.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_TYPE_TRAITS_IS_FUNCTION_H
#define AICORE_UTILS_STD_TYPE_TRAITS_IS_FUNCTION_H

#include "integral_constant.h"
#include "is_const.h"
#include "is_reference.h"

namespace AscendC {
namespace Std {

template <typename T>
struct is_function : public bool_constant<!(is_reference_v<T> || is_const_v<const T>)> {};

template <typename Tp>
constexpr bool is_function_v = is_function<Tp>::value;

} // namespace Std
} // namespace AscendC

#endif // AICORE_UTILS_STD_TYPE_TRAITS_IS_FUNCTION_H
