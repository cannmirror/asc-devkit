//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/*!
 * \file is_constant.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_TYPE_TRAITS_IS_CONSTANT_H
#define AICORE_UTILS_STD_TYPE_TRAITS_IS_CONSTANT_H

#include "integral_constant.h"

namespace AscendC {
namespace Std {

template <auto n, typename T>
struct is_constant : false_type {};

template <auto n, typename T>
struct is_constant<n, T const> : is_constant<n, T> {};

template <auto n, typename T>
struct is_constant<n, T const&> : is_constant<n, T> {};

template <auto n, typename T>
struct is_constant<n, T&> : is_constant<n, T> {};

template <auto n, typename T>
struct is_constant<n, T&&> : is_constant<n, T> {};

template <auto n, typename T, T v>
struct is_constant<n, integral_constant<T, v>> : bool_constant<v == n> {};

} // namespace Std
} // namespace AscendC

#endif // AICORE_UTILS_STD_TYPE_TRAITS_IS_CONSTANT_H
