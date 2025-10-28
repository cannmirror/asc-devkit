/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file layernormgradbeta_check.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_NORMALIZATION_LAYERNORMGRADBETA_LAYERNORMGRADBETA_CHECK_H_
#define IMPL_API_CHECK_KERNEL_CHECK_NORMALIZATION_LAYERNORMGRADBETA_LAYERNORMGRADBETA_CHECK_H_

#include "kernel_tiling/kernel_tiling.h"
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002 || __NPU_ARCH__ == 2201)
#include "layernormgradbeta_check_common.h"
#else
#include "layernormgradbeta_check_aicore.h"
#endif

namespace AscendC {
namespace HighLevelApiCheck {

template <typename T, bool isReuseSource = false>
__aicore__ inline void CheckFuncLayerNormGradBeta(__gm__ const char *apiName, const LocalTensor<T> &outputPdGamma, const LocalTensor<T> &outputPdBeta,
    const LocalTensor<T> &resForGamma, const LocalTensor<T> &inputDy, const LocalTensor<uint8_t> &sharedTmpBuffer,
    const LayerNormGradBetaTiling &tiling)
{
    CheckFuncClassLayerNormGradBeta<T, isReuseSource> checkFun(apiName);
    checkFun.VerifyingParameters(outputPdGamma, outputPdBeta, resForGamma, inputDy, sharedTmpBuffer, tiling);
}

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_NORMALIZATION_LAYERNORMGRADBETA_LAYERNORMGRADBETA_CHECK_H_
