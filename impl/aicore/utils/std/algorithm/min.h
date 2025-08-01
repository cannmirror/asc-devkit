//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/*!
 * \file min.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_ALGORITHM_MIN_H
#define AICORE_UTILS_STD_ALGORITHM_MIN_H
#include "../type_traits/is_same.h"

namespace AscendC {
namespace Std {
template <typename T, typename U>
ASCENDC_HOST_AICORE inline T min(const T src0, const U src1)
{
    static_assert(Std::is_same<T, U>::value, "Only support compare with same type!");
    return (src0 < src1) ? src0 : src1;
}
} // namespace Std
} // namespace AscendC
#endif
