//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/* !
 * \file is_void.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_TYPE_TRAITS_IS_VOID_H
#define AICORE_UTILS_STD_TYPE_TRAITS_IS_VOID_H

#include "is_same.h"
#include "remove_cv.h"

namespace AscendC {
namespace Std {

template <typename Tp>
struct is_void : public is_same<remove_cv_t<Tp>, void> {};

template <typename Tp>
constexpr bool is_void_v = is_void<Tp>::value;

} // namespace Std
} // namespace AscendC

#endif // AICORE_UTILS_STD_TYPE_TRAITS_IS_VOID_H
