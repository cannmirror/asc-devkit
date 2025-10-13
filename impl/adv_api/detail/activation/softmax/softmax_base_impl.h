/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_base_impl.h
 * \brief
 */
#ifndef IMPL_ACTIVATION_SOFTMAX_SOFTMAX_BASE_IMPL_H
#define IMPL_ACTIVATION_SOFTMAX_SOFTMAX_BASE_IMPL_H

#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
#include "regbase/c310/softmax_impl.h"
#elif __CCE_AICORE__ == 300
#include "regbase/v300/softmax_impl.h"
#elif __CCE_AICORE__ == 220
#include "membase/v220/softmax_impl.h"
#elif __CCE_AICORE__ == 200
#include "membase/v200/softmax_impl.h"
#endif
#include "../../api_check/kernel_api_check.h"

namespace AscendC {
template <typename T, bool isReuseSource = false, bool isBasicBlock = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftMaxImpl(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<float>& workLocal, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo)
{
    CHECK_FUNC_HIGHLEVEL_API(SoftMax, (T, isReuseSource, isBasicBlock, config), (dst, src, workLocal, tiling, softmaxShapeInfo));
    SetMaskNorm();
    ResetMask();
    ShapeInfo srcShape = src.GetShapeInfo();
    LastAxisShapeND srcNDinfo;
    LastAxisShapeND originalSrcShape;
    if (softmaxShapeInfo.srcM == 0 || softmaxShapeInfo.srcK == 0) {
        srcNDinfo = GetLastAxisShapeND(srcShape);
        originalSrcShape = GetLastAxisOriginShapeND(srcShape);
    } else {
        srcNDinfo = { softmaxShapeInfo.srcM, softmaxShapeInfo.srcK };
        originalSrcShape = { softmaxShapeInfo.oriSrcM, softmaxShapeInfo.oriSrcK };
    }
    // when the shape is changed, need recalculate the softmax's tiling
    if (unlikely(srcNDinfo.k != tiling.srcK || srcNDinfo.m != tiling.srcM)) {
        SoftMaxTiling newTiling = tiling;
        SoftMaxTilingFunc(workLocal.GetSize(), { srcNDinfo.m, srcNDinfo.k, originalSrcShape.m, srcNDinfo.k },
            newTiling, sizeof(T), sizeof(float), isBasicBlock);
        SoftMaxNDImpl<T, isReuseSource, isBasicBlock, config>(dst, src, workLocal, originalSrcShape, newTiling);
    } else {
        SoftMaxNDImpl<T, isReuseSource, isBasicBlock, config>(dst, src, workLocal, originalSrcShape, tiling);
    }
}
template <typename T, bool isReuseSource = false, bool isBasicBlock = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftMaxImpl(const LocalTensor<T>& dst, const LocalTensor<T>& src, const SoftMaxTiling& tiling,
    const SoftMaxShapeInfo& softmaxShapeInfo)
{
    LocalTensor<float> workLocal;
    PopStackBuffer<float, TPosition::LCM>(workLocal);
    SoftMaxImpl<T, isReuseSource, isBasicBlock, config>(dst, src, workLocal, tiling, softmaxShapeInfo);
}
template <typename T, bool isReuseSource = false, bool isBasicBlock = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftMaxImpl(const LocalTensor<T>& dst, const LocalTensor<T>& src,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo)
{
    auto workLocal = sharedTmpBuffer.ReinterpretCast<float>();
    SoftMaxImpl<T, isReuseSource, isBasicBlock, config>(dst, src, workLocal, tiling, softmaxShapeInfo);
}

template <typename T1, typename T2, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftMaxImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& sumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& src, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo)
{
    CHECK_FUNC_HIGHLEVEL_API(SoftMax, (T1, T2, isReuseSource, isBasicBlock, isDataFormatNZ, config),
        (dst, sumTensor, maxTensor, src, workLocal, tiling, softmaxShapeInfo));
    SetMaskNorm();
    ResetMask();
    ShapeInfo srcShape = src.GetShapeInfo();
    LastAxisShapeND srcNDinfo;
    LastAxisShapeND originalSrcShape;
    if (softmaxShapeInfo.srcM == 0 || softmaxShapeInfo.srcK == 0) {
        srcNDinfo = GetLastAxisShapeND(srcShape);
        originalSrcShape = GetLastAxisOriginShapeND(srcShape);
    } else {
        srcNDinfo = { softmaxShapeInfo.srcM, softmaxShapeInfo.srcK };
        originalSrcShape = { softmaxShapeInfo.oriSrcM, softmaxShapeInfo.oriSrcK };
    }
    if constexpr (isDataFormatNZ) {
        // when the shape is changed, need recalculate the softmax's tiling
        if (unlikely(srcNDinfo.k != tiling.srcK || originalSrcShape.m != tiling.srcM)) {
            SoftMaxTiling newTiling = tiling;
            SoftMaxTilingFunc(workLocal.GetSize(), { srcNDinfo.m, srcNDinfo.k, originalSrcShape.m, srcNDinfo.k },
                newTiling, sizeof(T1), sizeof(T2), false, isDataFormatNZ);
            SoftMaxNZImpl<T1, T2, isBasicBlock>(dst, sumTensor, maxTensor, src, workLocal, originalSrcShape, newTiling);
        } else {
            SoftMaxNZImpl<T1, T2, isBasicBlock>(dst, sumTensor, maxTensor, src, workLocal, originalSrcShape, tiling);
        }
    } else {
        // when the shape is changed, need recalculate the softmax's tiling
        if (unlikely(srcNDinfo.k != tiling.srcK || srcNDinfo.m != tiling.srcM)) {
            SoftMaxTiling newTiling = tiling;
            SoftMaxTilingFunc(workLocal.GetSize(), { srcNDinfo.m, srcNDinfo.k, originalSrcShape.m, srcNDinfo.k },
                newTiling, sizeof(T1), sizeof(T2), isBasicBlock);
            SoftMaxNDImpl<T1, T2, isBasicBlock, config>(dst, sumTensor, maxTensor, src, workLocal, originalSrcShape,
                newTiling);
        } else {
            SoftMaxNDImpl<T1, T2, isBasicBlock, config>(dst, sumTensor, maxTensor, src, workLocal, originalSrcShape,
                tiling);
        }
    }
}

template <typename T1, typename T2, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftMaxImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& sumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& src, const SoftMaxTiling& tiling,
    const SoftMaxShapeInfo& softmaxShapeInfo)
{
    LocalTensor<float> workLocal;
    PopStackBuffer<float, TPosition::LCM>(workLocal);
    SoftMaxImpl<T1, T2, isReuseSource, isBasicBlock, isDataFormatNZ, config>(dst, sumTensor, maxTensor, src, workLocal, tiling,
        softmaxShapeInfo);
}

template <typename T1, typename T2, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftMaxImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& sumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& src, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo)
{
    auto workLocal = sharedTmpBuffer.ReinterpretCast<float>();
    SoftMaxImpl<T1, T2, isReuseSource, isBasicBlock, isDataFormatNZ, config>(dst, sumTensor, maxTensor, src, workLocal, tiling,
        softmaxShapeInfo);
}

template <typename T1, typename T2, bool isDataFormatNZ = false, uint8_t stepSizeMode = 0>
__aicore__ inline bool AdjustSoftMaxResImpl(const LocalTensor<T1>& softMaxRes, const LocalTensor<T2>& maxTensor,
    const uint32_t from, const T1 to, const SoftMaxShapeInfo& softmaxShapeInfo)
{
    CHECK_FUNC_HIGHLEVEL_API(AdjustSoftMaxRes, (T1, T2, isDataFormatNZ, stepSizeMode), (softMaxRes, maxTensor, from, to, softmaxShapeInfo));
    return AdjustSoftMaxResBaseImpl<T1, T2, isDataFormatNZ, stepSizeMode>(softMaxRes, maxTensor, from, to,
        softmaxShapeInfo);
}
}

#endif // IMPL_ACTIVATION_SOFTMAX_SOFTMAX_BASE_IMPL_H