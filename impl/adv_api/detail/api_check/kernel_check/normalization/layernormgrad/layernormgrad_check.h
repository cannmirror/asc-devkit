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
 * \file layernormgrad_check.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_NORMALIZATION_LAYERNORMGRAD_LAYERNORMGRAD_CHECK_H_
#define IMPL_API_CHECK_KERNEL_CHECK_NORMALIZATION_LAYERNORMGRAD_LAYERNORMGRAD_CHECK_H_

#include "kernel_tiling/kernel_tiling.h"
#include "include/adv_api/normalization/layernormgrad_utils.h"
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002 || __NPU_ARCH__ == 2201)
#include "layernormgrad_check_common.h"
#else
#include "layernormgrad_check_aicore.h"
#endif

namespace AscendC {
namespace HighLevelApiCheck {

template <typename T, bool isReuseSource = false>
__aicore__ inline void CheckFuncLayerNormGrad(__gm__ const char *apiName, const LocalTensor<T> &outputPdX, const LocalTensor<T> &resForGamma,
    const LocalTensor<T> &inputDy, const LocalTensor<T> &inputX, const LocalTensor<T> &inputVariance,
    const LocalTensor<T> &inputMean, const LocalTensor<T> &inputGamma, LocalTensor<uint8_t> &sharedTmpBuffer, T epsilon,
    LayerNormGradTiling &tiling, const LayerNormGradShapeInfo &shapeInfo = {})
{
    CheckFuncClassLayerNormGrad<T, isReuseSource> checkFun(apiName);
    checkFun.VerifyingParameters(outputPdX, resForGamma, inputDy, inputX, inputVariance, inputMean, inputGamma,
        sharedTmpBuffer, epsilon, tiling, shapeInfo);
}

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_NORMALIZATION_LAYERNORMGRAD_LAYERNORMGRAD_CHECK_H_
