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

/* !
 * \file softmax_flashv3_base_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_SOFTMAX_FLASHV3_BASE_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_SOFTMAX_FLASHV3_BASE_IMPL_H

#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
#include "regbase/c310/softmax_flashv3_impl.h"
#elif __CCE_AICORE__ == 300
#include "regbase/v300/softmax_flashv3_impl.h"
#elif __CCE_AICORE__ == 220
#include "membase/v220/softmax_flashv3_impl.h"
#elif __CCE_AICORE__ == 200
#include "membase/v200/softmax_flashv3_impl.h"
#endif
#include "../../api_check/kernel_api_check.h"

namespace AscendC {
template <typename T, typename U, bool isUpdate, bool isReuseSource, bool isBasicBlock, bool isDataFormatNZ, const SoftmaxConfig& config>
__aicore__ inline void SoftmaxFlashV3Impl(const LocalTensor<T>& dstTensor, const LocalTensor<U>& meanTensor,
    const LocalTensor<U>& expSumTensor, const LocalTensor<U>& maxTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<T>& expMaxTensor, const LocalTensor<U>& inMeanTensor, const LocalTensor<U>& inexpSumTensor,
    const LocalTensor<U>& inMaxTensor, const LocalTensor<float>& workLocal, const SoftMaxTiling& tiling,
    const SoftMaxParams& params)
{
    CHECK_FUNC_HIGHLEVEL_API(SoftmaxFlashV3, (T, U, isUpdate, isReuseSource, isBasicBlock, isDataFormatNZ, config), (
        dstTensor, meanTensor, expSumTensor, maxTensor, srcTensor, expMaxTensor,
        inMeanTensor, inexpSumTensor, inMaxTensor, workLocal, tiling, params));
    static_assert((SupportType<Tuple<T, U>, Tuple<half, float>>()), "Failed to check dtype in SoftmaxFlashV3, "
        "Current api support dtype combination is T : half, U : float");

    LastAxisShapeND originalSrcShape = { params.oriSrcM, params.oriSrcK };
    if (params.srcM == 0 || params.srcK == 0) {
        ShapeInfo srcShape = srcTensor.GetShapeInfo();
        originalSrcShape = GetLastAxisOriginShapeND(srcShape);
    }
    SoftmaxFlashV3Process<T, U, isUpdate, isBasicBlock, config>(dstTensor, meanTensor, expSumTensor, maxTensor, srcTensor,
        expMaxTensor, inMeanTensor, inexpSumTensor, inMaxTensor, workLocal, originalSrcShape, tiling, params);
}

template <typename T, typename U, bool isUpdate, bool isReuseSource, bool isBasicBlock, bool isDataFormatNZ, const SoftmaxConfig& config>
__aicore__ inline void SoftmaxFlashV3Impl(const LocalTensor<T>& dstTensor, const LocalTensor<U>& meanTensor,
    const LocalTensor<U>& expSumTensor, const LocalTensor<U>& maxTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<T>& expMaxTensor, const LocalTensor<U>& inMeanTensor, const LocalTensor<U>& inexpSumTensor,
    const LocalTensor<U>& inMaxTensor, const SoftMaxTiling& tiling, const SoftMaxParams& params)
{
    LocalTensor<float> workLocal;
    bool ans = PopStackBuffer<float, TPosition::LCM>(workLocal);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "SoftmaxFlashv3 PopStackBuffer Error!"); });
    SoftmaxFlashV3Impl<T, U, isUpdate, isReuseSource, isBasicBlock, isDataFormatNZ, config>(dstTensor, meanTensor, expSumTensor, maxTensor,
        srcTensor, expMaxTensor, inMeanTensor, inexpSumTensor, inMaxTensor, workLocal, tiling, params);
}

template <typename T, typename U, bool isUpdate, bool isReuseSource, bool isBasicBlock, bool isDataFormatNZ, const SoftmaxConfig& config>
__aicore__ inline void SoftmaxFlashV3Impl(const LocalTensor<T>& dstTensor, const LocalTensor<U>& meanTensor,
    const LocalTensor<U>& expSumTensor, const LocalTensor<U>& maxTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<T>& expMaxTensor, const LocalTensor<U>& inMeanTensor, const LocalTensor<U>& inexpSumTensor,
    const LocalTensor<U>& inMaxTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling,
    const SoftMaxParams& params)
{
    auto workLocal = sharedTmpBuffer.ReinterpretCast<float>();
    SoftmaxFlashV3Impl<T, U, isUpdate, isReuseSource, isBasicBlock, isDataFormatNZ, config>(dstTensor, meanTensor, expSumTensor, maxTensor,
        srcTensor, expMaxTensor, inMeanTensor, inexpSumTensor, inMaxTensor, workLocal, tiling, params);
}
}
#endif // IMPL_ACTIVATION_SOFTMAX_SOFTMAX_FLASHV3_BASE_IMPL_H