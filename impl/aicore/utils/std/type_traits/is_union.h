//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/* !
 * \file is_union.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_TYPE_TRAITS_IS_UNION_H
#define AICORE_UTILS_STD_TYPE_TRAITS_IS_UNION_H

#include <type_traits>
#include "integral_constant.h"
#include "remove_cv.h"

namespace AscendC {
namespace Std {

template <typename Tp>
struct is_union : bool_constant<std::is_union<Tp>::value> {};

template <typename Tp>
constexpr bool is_union_v = is_union<Tp>::value;

} // namespace Std
} // namespace AscendC

#endif // AICORE_UTILS_STD_TYPE_TRAITS_IS_UNION_H
