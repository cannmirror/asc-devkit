//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/* !
 * \file is_class.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_TYPE_TRAITS_IS_CLASS_H
#define AICORE_UTILS_STD_TYPE_TRAITS_IS_CLASS_H

#include "integral_constant.h"
#include "is_union.h"

namespace AscendC {
namespace Std {

namespace IsClassImpl {

template <typename Tp>
ASCENDC_HOST_AICORE inline bool_constant<!is_union_v<Tp>> Test(int32_t Tp::*);

template <typename Tp>
ASCENDC_HOST_AICORE inline false_type Test(uint32_t);

} // namespace IsClassImpl

template <typename Tp>
struct is_class : decltype(IsClassImpl::Test<Tp>(nullptr)) {};

template <typename Tp>
constexpr bool is_class_v = is_class<Tp>::value;

} // namespace Std
} // namespace AscendC

#endif // AICORE_UTILS_STD_TYPE_TRAITS_IS_CLASS_H
