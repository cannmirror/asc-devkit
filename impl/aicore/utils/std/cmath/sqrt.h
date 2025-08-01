//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/*!
 * \file sqrt.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_CMATH_SQRT_H
#define AICORE_UTILS_STD_CMATH_SQRT_H

namespace AscendC {
namespace Std {
template <typename T>
__aicore__ inline T sqrt(const T src)
{
    static_assert(SupportType<T, float, int64_t>(), "current data type is not supported on current device!");
    return ::sqrt(src);
}
} // namespace Std
} // namespace AscendC
#endif
