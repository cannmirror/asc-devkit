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
 * \file softmax_flash_check.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_SOFTMAX_FLASH_SOFTMAX_FLASH_CHECK_H
#define AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_SOFTMAX_FLASH_SOFTMAX_FLASH_CHECK_H

#include "kernel_tiling/kernel_tiling.h"
#include "activation/softmax_utils.h"
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220)
#include "softmax_flash_check_common.h"
#elif defined(__DAV_C310__) || defined(__DAV_310R6__)
#include "softmax_flash_check_310.h"
#endif

namespace AscendC {
namespace HighLevelApiCheck {

template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void CheckFuncSoftmaxFlash(__gm__ const char* apiName, const LocalTensor<T>& dstTensor,
    const LocalTensor<T>& sumTensor, const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inSumTensor, const LocalTensor<T>& inMaxTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling, bool isUpdate,
    const SoftMaxShapeInfo& softmaxShapeInfo)
{
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220)
    CheckFuncClassSoftmaxFlashSameT<T, isReuseSource, isBasicBlock> checkFun(apiName);
    checkFun.VerifyingParameters(dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor, inSumTensor, inMaxTensor,
        sharedTmpBuffer, tiling, isUpdate, softmaxShapeInfo);
#endif
}

template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void CheckFuncSoftmaxFlash(__gm__ const char* apiName, const LocalTensor<half>& dstTensor,
    const LocalTensor<float>& sumTensor, const LocalTensor<float>& maxTensor, const LocalTensor<half>& srcTensor,
    const LocalTensor<half>& expMaxTensor, const LocalTensor<float>& inSumTensor, const LocalTensor<float>& inMaxTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling, bool isUpdate,
    const SoftMaxShapeInfo& softmaxShapeInfo)
{
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220)
    CheckFuncClassSoftmaxFlash<T, isReuseSource, isBasicBlock> checkFun(apiName);
    checkFun.VerifyingParameters(dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor, inSumTensor, inMaxTensor,
        sharedTmpBuffer, tiling, isUpdate, softmaxShapeInfo);
#endif
}

template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void CheckFuncSoftmaxFlash(__gm__ const char* apiName, const LocalTensor<T>& dstTensor,
    const LocalTensor<T>& sumTensor, const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<T>& expMaxTensor, const LocalTensor<T>& inSumTensor, const LocalTensor<T>& inMaxTensor,
    const SoftMaxTiling& tiling, bool isUpdate, const SoftMaxShapeInfo& softmaxShapeInfo)
{
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220)
    CheckFuncClassSoftmaxFlashSameT<T, isReuseSource, isBasicBlock> checkFun(apiName);
    checkFun.VerifyingParameters(dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor, inSumTensor, inMaxTensor,
        tiling, isUpdate, softmaxShapeInfo);
#endif
}

template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void CheckFuncSoftmaxFlash(__gm__ const char* apiName, const LocalTensor<half>& dstTensor,
    const LocalTensor<float>& sumTensor, const LocalTensor<float>& maxTensor, const LocalTensor<half>& srcTensor,
    const LocalTensor<half>& expMaxTensor, const LocalTensor<float>& inSumTensor, const LocalTensor<float>& inMaxTensor,
    const SoftMaxTiling& tiling, bool isUpdate, const SoftMaxShapeInfo& softmaxShapeInfo)
{
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 200 || __CCE_AICORE__ == 220)
    CheckFuncClassSoftmaxFlash<T, isReuseSource, isBasicBlock> checkFun(apiName);
    checkFun.VerifyingParameters(dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor, inSumTensor, inMaxTensor,
        tiling, isUpdate, softmaxShapeInfo);
#endif
}

} // namespace HighLevelApiCheck
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_SOFTMAX_FLASH_SOFTMAX_FLASH_CHECK_H
