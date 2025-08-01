//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/* !
 * \file is_reference.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_TYPE_TRAITS_IS_REFERENCE_H
#define AICORE_UTILS_STD_TYPE_TRAITS_IS_REFERENCE_H

#include "integral_constant.h"

namespace AscendC {
namespace Std {

template <typename Tp>
struct is_lvalue_reference : public false_type {};

template <typename Tp>
struct is_lvalue_reference<Tp&> : public true_type {};

template <typename Tp>
struct is_rvalue_reference : public false_type {};

template <typename Tp>
struct is_rvalue_reference<Tp&&> : public true_type {};

template <typename Tp>
struct is_reference : public false_type {};

template <typename Tp>
struct is_reference<Tp&> : public true_type {};

template <typename Tp>
struct is_reference<Tp&&> : public true_type {};

template <typename Tp>
constexpr bool is_lvalue_reference_v = is_lvalue_reference<Tp>::value;

template <typename Tp>
constexpr bool is_rvalue_reference_v = is_rvalue_reference<Tp>::value;

template <typename Tp>
constexpr bool is_reference_v = is_reference<Tp>::value;

} // namespace Std
} // namespace AscendC

#endif // AICORE_UTILS_STD_TYPE_TRAITS_IS_REFERENCE_H
