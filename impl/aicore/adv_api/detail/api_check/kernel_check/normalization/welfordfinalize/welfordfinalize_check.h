/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file welfordfinalize_check.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_NORMALIZATION_WELFORDFINALIZE_WELFORDFINALIZE_CHECK_H
#define AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_NORMALIZATION_WELFORDFINALIZE_WELFORDFINALIZE_CHECK_H

#include "normalization/welfordfinalize_utils.h"
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220)
#include "welfordfinalize_check_common.h"
#elif defined(__DAV_C310__) || defined(__DAV_310R6__)
#include "welfordfinalize_check_310.h"
#endif

namespace AscendC {
namespace HighLevelApiCheck {

template <bool isReuseSource = false>
__aicore__ inline void CheckFuncWelfordFinalize(__gm__ const char* apiName, const LocalTensor<float>& outputMean,
    const LocalTensor<float>& outputVariance, const LocalTensor<float>& inputMean,
    const LocalTensor<float>& inputVariance, const LocalTensor<int32_t>& counts,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const WelfordFinalizePara& para)
{
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220)
    CheckFuncClassWelfordFinalizeCounts<isReuseSource> checkFun(apiName);
    checkFun.VerifyingParameters(outputMean, outputVariance, inputMean, inputVariance, counts, sharedTmpBuffer, para);
#endif
}

template <bool isReuseSource = false>
__aicore__ inline void CheckFuncWelfordFinalize(__gm__ const char* apiName, const LocalTensor<float>& outputMean,
    const LocalTensor<float>& outputVariance, const LocalTensor<float>& inputMean,
    const LocalTensor<float>& inputVariance, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const WelfordFinalizePara& para)
{
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220)
    CheckFuncClassWelfordFinalize<isReuseSource> checkFun(apiName);
    checkFun.VerifyingParameters(outputMean, outputVariance, inputMean, inputVariance, sharedTmpBuffer, para);
#endif
}

} // namespace HighLevelApiCheck
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_NORMALIZATION_WELFORDFINALIZE_WELFORDFINALIZE_CHECK_H
