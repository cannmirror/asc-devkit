//===----------------------------------------------------------------------===//
//
// Part of LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Huawei Technologies Co., Ltd.
//
//===----------------------------------------------------------------------===//

/*!
 * \file algorithm.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_ALGORITHM_H
#define AICORE_UTILS_STD_ALGORITHM_H
#include "utils/std/algorithm/max.h"
#include "utils/std/algorithm/min.h"

namespace AscendC {
namespace Std {
template <typename T, typename U>
ASCENDC_HOST_AICORE inline T min(const T src0, const U src1);
template <typename T, typename U>
ASCENDC_HOST_AICORE inline T max(const T src0, const U src1);
} // namespace Std
} // namespace AscendC
#endif // AICORE_UTILS_STD_ALGORITHM_H
