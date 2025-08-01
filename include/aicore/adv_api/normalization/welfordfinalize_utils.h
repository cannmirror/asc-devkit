/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file welfordfinalize_utils.h
 * \brief
 */
#ifndef AICORE_ADV_API_NORMALIZATION_WELFORDFINALIZE_UTILS_H
#define AICORE_ADV_API_NORMALIZATION_WELFORDFINALIZE_UTILS_H

namespace AscendC {
struct WelfordFinalizeConfig {
    __aicore__ constexpr WelfordFinalizeConfig(const bool isCorrectionIn)
    {
        isCorrection = isCorrectionIn;
    }
    bool isCorrection = false;
};

constexpr WelfordFinalizeConfig WFFINALIZE_DEFAULT_CFG = {false};

struct WelfordFinalizePara {
    uint32_t rnLength;
    uint32_t abLength;
    uint32_t headCount;
    uint32_t headCountLength;
    uint32_t tailCount;
    uint32_t tailCountLength;
    float abRec;
    float rRec;
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
    float rRecWithCorrection;
#endif
};

};     // namespace AscendC
#endif // AICORE_ADV_API_NORMALIZATION_WELFORDFINALIZE_UTILS_H