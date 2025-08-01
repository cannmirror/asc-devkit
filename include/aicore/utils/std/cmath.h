//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/*!
 * \file cmath.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_CMATH_H
#define AICORE_UTILS_STD_CMATH_H
#include "utils/std/cmath/sqrt.h"
#include "utils/std/cmath/abs.h"

namespace AscendC {
namespace Std {
template <typename T>
__aicore__ inline T sqrt(const T src);
template <typename T>
__aicore__ inline T abs(const T src);
} // namespace Std
} // namespace AscendC
#endif // AICORE_UTILS_STD_CMATH_H
