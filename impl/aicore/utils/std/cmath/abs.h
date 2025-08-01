//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/*!
 * \file abs.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_CMATH_ABS_H
#define AICORE_UTILS_STD_CMATH_ABS_H

namespace AscendC {
namespace Std {
template <typename T>
__aicore__ inline T abs(const T src)
{
    static_assert(SupportType<T, int8_t, int16_t, int32_t, float, int64_t>(),
        "current data type is not supported on current device!");
    return ::abs(src);
}
} // namespace Std
} // namespace AscendC
#endif
