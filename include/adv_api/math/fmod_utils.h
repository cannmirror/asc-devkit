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
 * \file fmod_utils.h
 * \brief
 */
#ifndef LIB_MATH_FMOD_UTILS_H
#define LIB_MATH_FMOD_UTILS_H

namespace AscendC {
constexpr uint32_t FMOD_ITERATION_NUM_MAX = 11;
enum class FmodAlgo {
    NORMAL = 0,
    ITERATION_COMPENSATION = 1,
};

struct FmodConfig {
    FmodAlgo algo = FmodAlgo::NORMAL;
    uint32_t iterationNum = FMOD_ITERATION_NUM_MAX;
};

constexpr FmodConfig DEFAULT_FMOD_CONFIG = { FmodAlgo::NORMAL, FMOD_ITERATION_NUM_MAX };
}; // namespace AscendC
#endif // LIB_MATH_FMOD_UTILS_H