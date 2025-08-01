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
 * \file softmax_flash_check_310.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_SOFTMAX_FLASH_SOFTMAX_FLASH_CHECK_310_H
#define AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_SOFTMAX_FLASH_SOFTMAX_FLASH_CHECK_310_H

#include "kernel_tiling/kernel_tiling.h"
#include "activation/softmax_utils.h"

namespace AscendC {
namespace HighLevelApiCheck {

template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
class CheckFuncClassSoftmaxFlashSameT {
public:
    __aicore__ inline CheckFuncClassSoftmaxFlashSameT(){};
    __aicore__ inline CheckFuncClassSoftmaxFlashSameT(__gm__ const char* apiName){};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<T>& dstTensor, const LocalTensor<T>& sumTensor,
        const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor,
        const LocalTensor<T>& inSumTensor, const LocalTensor<T>& inMaxTensor,
        const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling, bool isUpdate,
        const SoftMaxShapeInfo& softmaxShapeInfo){};
};

template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
class CheckFuncClassSoftmaxFlash {
public:
    __aicore__ inline CheckFuncClassSoftmaxFlash(){};
    __aicore__ inline CheckFuncClassSoftmaxFlash(__gm__ const char* apiName){};

public:
    __aicore__ inline void VerifyingParameters(const LocalTensor<half>& dstTensor, const LocalTensor<float>& sumTensor,
        const LocalTensor<float>& maxTensor, const LocalTensor<half>& srcTensor, const LocalTensor<half>& expMaxTensor,
        const LocalTensor<float>& inSumTensor, const LocalTensor<float>& inMaxTensor,
        const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling, bool isUpdate,
        const SoftMaxShapeInfo& softmaxShapeInfo){};
};

} // namespace HighLevelApiCheck
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_API_CHECK_KERNEL_CHECK_ACTIVATION_SOFTMAX_SOFTMAX_FLASH_SOFTMAX_FLASH_CHECK_310_H
