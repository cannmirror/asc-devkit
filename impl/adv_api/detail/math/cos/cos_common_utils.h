/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file cos_common_utils.h
 * \brief
 */

#ifndef IMPL_MATH_COS_COS_COMMON_UTILS_H
#define IMPL_MATH_COS_COS_COMMON_UTILS_H

namespace AscendC {
enum class CosAlgo {
    POLYNOMIAL_APPROXIMATION = 0,
    RADIAN_REDUCTION,
};

struct CosConfig {
    CosAlgo algo = CosAlgo::POLYNOMIAL_APPROXIMATION;
};

constexpr CosConfig defaultCosConfig = { CosAlgo::POLYNOMIAL_APPROXIMATION };
}

#endif // IMPL_MATH_COS_COS_COMMON_UTILS_H