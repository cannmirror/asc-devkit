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
 * \file normalize_check.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_NORMALIZATION_NORMALIZE_NORMALIZE_CHECK_H_
#define IMPL_API_CHECK_KERNEL_CHECK_NORMALIZATION_NORMALIZE_NORMALIZE_CHECK_H_

#include "include/adv_api/normalization/normalize_utils.h"
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002 || __NPU_ARCH__ == 2201)
#include "normalize_check_common.h"
#else
#include "normalize_check_aicore.h"
#endif

namespace AscendC {
namespace HighLevelApiCheck {

template <typename U, typename T, bool isReuseSource, const NormalizeConfig& config>
__aicore__ inline void CheckFuncNormalize(__gm__ const char *apiName, const LocalTensor<T>& output, const LocalTensor<float>& outputRstd,
    const LocalTensor<float>& inputMean, const LocalTensor<float>& inputVariance, const LocalTensor<T>& inputX,
    const LocalTensor<U>& gamma, const LocalTensor<U>& beta, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const float epsilon, const NormalizePara& para)
{
    CheckFuncClassNormalize<U, T, isReuseSource, config> checkFun(apiName);
    checkFun.VerifyingParameters(output, outputRstd, inputMean, inputVariance, inputX, gamma, beta,
        sharedTmpBuffer, epsilon, para);
}

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_NORMALIZATION_NORMALIZE_NORMALIZE_CHECK_H_
