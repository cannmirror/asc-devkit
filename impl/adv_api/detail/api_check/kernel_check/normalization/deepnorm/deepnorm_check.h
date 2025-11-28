/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file deepnorm_check.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_NORMALIZATION_DEEPNORM_DEEPNORM_CHECK_H_
#define IMPL_API_CHECK_KERNEL_CHECK_NORMALIZATION_DEEPNORM_DEEPNORM_CHECK_H_

#include "kernel_tiling/kernel_tiling.h"
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002 || __NPU_ARCH__ == 2201)
#include "deepnorm_check_common.h"
#else
#include "deepnorm_check_aicore.h"
#endif

namespace AscendC {
namespace HighLevelApiCheck {

template <typename T, bool isReuseSrc, bool isBasicBlock>
__aicore__ inline void CheckFuncDeepNorm(__gm__ const char *apiName, const LocalTensor<T>& dstLocal, const LocalTensor<T>& meanLocal,
    const LocalTensor<T>& rstdLocal, const LocalTensor<T>& srcLocal, const LocalTensor<T>& gxLocal,
    const LocalTensor<T>& betaLocal, const LocalTensor<T>& gammaLocal, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const T alpha, const T epsilon, DeepNormTiling& tiling)
{
    CheckFuncClassDeepNorm<T, isReuseSrc, isBasicBlock> checkFun(apiName);
    checkFun.VerifyingParameters(dstLocal, meanLocal, rstdLocal, srcLocal, gxLocal, betaLocal,
        gammaLocal, sharedTmpBuffer, alpha, epsilon, tiling);
}

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_NORMALIZATION_DEEPNORM_DEEPNORM_CHECK_H_
