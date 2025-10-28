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
 * \file softmax_check.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_SOFTMAX_SOFTMAX_CHECK_H_
#define IMPL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_SOFTMAX_SOFTMAX_CHECK_H_

#include "kernel_tiling/kernel_tiling.h"
#include "include/adv_api/activation/softmax_utils.h"
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002 || __NPU_ARCH__ == 2201 || __NPU_ARCH__ == 3002)
#include "softmax_check_common.h"
#else
#include "softmax_check_aicore.h"
#endif

namespace AscendC {
namespace HighLevelApiCheck {

template <typename T1, typename T2, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void CheckFuncSoftMax(__gm__ const char *apiName, const LocalTensor<T1>& dst, const LocalTensor<T2>& sumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& src, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo)
{
    CheckFuncClassSoftMax<T1, T2, isReuseSource, isBasicBlock, isDataFormatNZ, config> checkFun(apiName);
    checkFun.VerifyingParameters(dst, sumTensor, maxTensor, src, workLocal, tiling, softmaxShapeInfo);
}

template <typename T, bool isReuseSource = false, bool isBasicBlock = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void CheckFuncSoftMax(__gm__ const char *apiName, const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<float>& workLocal, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo)
{
    CheckFuncClassSoftMaxSameT<T, isReuseSource, isBasicBlock, config> checkFun(apiName);
    checkFun.VerifyingParameters(dst, src, workLocal, tiling, softmaxShapeInfo);
}

}
}
#endif // IMPL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_SOFTMAX_SOFTMAX_CHECK_H_
