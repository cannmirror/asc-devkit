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
 * \file rmsnorm_check.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_NORMALIZATION_RMSNORM_RMSNORM_CHECK_H
#define AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_NORMALIZATION_RMSNORM_RMSNORM_CHECK_H

#include "kernel_tiling/kernel_tiling.h"
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220)
#include "rmsnorm_check_common.h"
#elif defined(__DAV_C310__) || defined(__DAV_310R6__)
#include "rmsnorm_check_310.h"
#endif

namespace AscendC {
namespace HighLevelApiCheck {

template <typename T, bool isBasicBlock = false>
__aicore__ inline void CheckFuncRmsNorm(__gm__ const char* apiName, const LocalTensor<T>& dstLocal,
    const LocalTensor<T>& srcLocal, const LocalTensor<T>& gammaLocal, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const T epsilon, const RmsNormTiling& tiling)
{
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220)
    CheckFuncClassRmsNorm<T, isBasicBlock> checkFun(apiName);
    checkFun.VerifyingParameters(dstLocal, srcLocal, gammaLocal, sharedTmpBuffer, epsilon, tiling);
#endif
}

} // namespace HighLevelApiCheck
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_NORMALIZATION_RMSNORM_RMSNORM_CHECK_H
