//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/* !
 * \file is_pointer.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_TYPE_TRAITS_IS_POINTER_H
#define AICORE_UTILS_STD_TYPE_TRAITS_IS_POINTER_H

#include "integral_constant.h"
#include "remove_cv.h"

namespace AscendC {
namespace Std {

template <typename Tp>
struct IsPointerImpl : public false_type {};

template <typename Tp>
struct IsPointerImpl<Tp*> : public true_type {};

template <typename Tp>
struct is_pointer : public IsPointerImpl<remove_cv_t<Tp>> {};

template <typename Tp>
constexpr bool is_pointer_v = is_pointer<Tp>::value;

} // namespace Std
} // namespace AscendC

#endif // AICORE_UTILS_STD_TYPE_TRAITS_IS_POINTER_H
