/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cmath.h
 * \brief
 */
#ifndef AICORE_UTILS_STD_CMATH_H
#define AICORE_UTILS_STD_CMATH_H
#include "impl/utils/std/cmath/sqrt.h"
#include "impl/utils/std/cmath/abs.h"

namespace AscendC {
namespace Std {
template <typename T> __aicore__ inline T sqrt(const T src);
template <typename T> __aicore__ inline T abs(const T src);
}
}
#endif  // AICORE_UTILS_STD_CMATH_H
