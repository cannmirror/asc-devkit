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
 * \file softmax_grad_check.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_SOFTMAX_GRAD_SOFTMAX_GRAD_CHECK_H
#define AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_SOFTMAX_GRAD_SOFTMAX_GRAD_CHECK_H

#include "kernel_tiling/kernel_tiling.h"
#include "activation/softmax_utils.h"
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220 || __CCE_AICORE__ == 300)
#include "softmax_grad_check_common.h"
#elif defined(__DAV_C310__) || defined(__DAV_310R6__)
#include "softmax_grad_check_310.h"
#endif

namespace AscendC {
namespace HighLevelApiCheck {

template <typename T, bool isReuseSource, bool isDataFormatNZ = false>
__aicore__ inline void CheckFuncSoftmaxGrad(__gm__ const char* apiName, const LocalTensor<T>& dstTensor,
    const LocalTensor<T>& gradTensor, const LocalTensor<T>& srcTensor, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling, bool isFront, const SoftMaxShapeInfo& softmaxShapeInfo)
{
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220 || __CCE_AICORE__ == 300)
    CheckFuncClassSoftmaxGrad<T, isReuseSource, isDataFormatNZ> checkFun(apiName);
    checkFun.VerifyingParameters(dstTensor, gradTensor, srcTensor, workLocal, tiling, isFront, softmaxShapeInfo);
#endif
}

} // namespace HighLevelApiCheck
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_SOFTMAX_GRAD_SOFTMAX_GRAD_CHECK_H
