/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file logsoftmax_base_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_LOGSOFTMAX_BASE_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_LOGSOFTMAX_BASE_IMPL_H

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002 || __NPU_ARCH__ == 2201)
#include "membase/common/logsoftmax_common_impl.h"
#endif
#include "../../api_check/kernel_api_check.h"

namespace AscendC {
template <typename T, bool isReuseSource = false, bool isDataFormatNZ = false>
__aicore__ inline void LogSoftMaxImpl(const LocalTensor<T>& dst, const LocalTensor<T>& sumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& src, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const LogSoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo = {})
{
    CHECK_FUNC_HIGHLEVEL_API(LogSoftMax, (T, isReuseSource, isDataFormatNZ), (dst, sumTensor, maxTensor, src,
                            sharedTmpBuffer, tiling, softmaxShapeInfo));

    LocalTensor<float> tempBuffer = sharedTmpBuffer.ReinterpretCast<float>();
    tempBuffer.SetSize(sharedTmpBuffer.GetSize() / B32_BYTE_SIZE);
    ShapeInfo srcShape = src.GetShapeInfo();
    uint32_t elementNumPerBlk = ONE_BLK_SIZE / sizeof(T);
    // check the src shape info, last axis must be 32B aligned
    ASCENDC_ASSERT((srcShape.shape[srcShape.shapeDim - 1] * sizeof(T) % ONE_BLK_SIZE == 0), {
        KERNEL_LOG(KERNEL_ERROR, "src local's last axis is %d, which must be 32B aligned.",
            srcShape.shape[srcShape.shapeDim - 1]);
    });
    LastAxisShapeND srcNDinfo;
    LastAxisShapeND originalSrcShape;
    if (softmaxShapeInfo.srcM == 0 || softmaxShapeInfo.srcK == 0) {
        srcNDinfo = GetLastAxisShapeND(srcShape);
        originalSrcShape = GetLastAxisOriginShapeND(srcShape);
    } else {
        srcNDinfo = { softmaxShapeInfo.srcM, softmaxShapeInfo.srcK };
        originalSrcShape = { softmaxShapeInfo.oriSrcM, softmaxShapeInfo.oriSrcK };
    }

    SoftMaxTiling newTiling;
    SoftMaxTilingFunc(tempBuffer.GetSize(), { srcNDinfo.m, srcNDinfo.k, originalSrcShape.m, srcNDinfo.k },
        newTiling, sizeof(T), sizeof(T), isDataFormatNZ);
    if constexpr (isDataFormatNZ) {
        LogSoftMaxNZImpl(dst, sumTensor, maxTensor, src, tempBuffer, originalSrcShape, newTiling);
    } else {
        LogSoftMaxNDImpl(dst, sumTensor, maxTensor, src, tempBuffer, originalSrcShape, newTiling);
    }
}
}
#endif // IMPL_ACTIVATION_SOFTMAX_LOGSOFTMAX_BASE_IMPL_H