//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/* !
 * \file is_referenceable.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_TYPE_TRAITS_IS_REFERENCEABLE_H
#define AICORE_UTILS_STD_TYPE_TRAITS_IS_REFERENCEABLE_H

#include "integral_constant.h"
#include "is_same.h"

namespace AscendC {
namespace Std {

struct IsReferenceableImpl {
    template <typename Tp>
    ASCENDC_HOST_AICORE inline static Tp& Test(int32_t);

    template <typename Tp>
    ASCENDC_HOST_AICORE inline static false_type Test(uint32_t);
};

template <typename Tp>
struct is_referenceable
    : integral_constant<bool, IsNotSame<decltype(IsReferenceableImpl::Test<Tp>(0)), false_type>::value> {};

} // namespace Std
} // namespace AscendC

#endif // AICORE_UTILS_STD_TYPE_TRAITS_IS_REFERENCEABLE_H
