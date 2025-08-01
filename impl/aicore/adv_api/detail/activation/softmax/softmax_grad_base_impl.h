/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file softmax_grad_base_impl.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_ACTIVATION_SOFTMAX_SOFTMAX_GRAD_BASE_IMPL_H
#define AICORE_ADV_API_DETAIL_ACTIVATION_SOFTMAX_SOFTMAX_GRAD_BASE_IMPL_H

#if defined(__DAV_C310__) || defined(__DAV_310R6__)
#include "regbase/c310/softmax_grad_impl.h"
#elif __CCE_AICORE__ == 300
#include "regbase/v300/softmax_grad_impl.h"
#elif __CCE_AICORE__ == 220
#include "membase/v220/softmax_grad_impl.h"
#elif __CCE_AICORE__ == 200
#include "membase/v200/softmax_grad_impl.h"
#endif
#include "../../api_check/kernel_api_check.h"

namespace AscendC {
template <typename T, bool isBasicBlock = false, bool isDataFormatNZ = false>
__aicore__ inline void SoftmaxGradFrontImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<float>& workLocal, const SoftMaxTiling& tiling,
    const SoftMaxShapeInfo& softmaxShapeInfo)
{
    CHECK_FUNC_HIGHLEVEL_API(SoftmaxGradFront, (T, isBasicBlock, isDataFormatNZ),
        (dstTensor, gradTensor, srcTensor, workLocal, tiling, softmaxShapeInfo));
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
    CheckTensorPos<T>(dstTensor, Hardware::UB, "dstTensor", "VECIN / VECCALC / VECOUT", "SoftmaxGradFront");
    CheckTensorPos<T>(srcTensor, Hardware::UB, "srcTensor", "VECIN / VECCALC / VECOUT", "SoftmaxGradFront");
    CheckTensorPos<T>(gradTensor, Hardware::UB, "gradTensor", "VECIN / VECCALC / VECOUT", "SoftmaxGradFront");
    ASCENDC_ASSERT((softmaxShapeInfo.srcK * sizeof(T) % ONE_BLK_SIZE == 0),
        { KERNEL_LOG(KERNEL_ERROR, "srcK should be 32B aligned, current srcK is %u", softmaxShapeInfo.srcK); });

    SoftmaxApiSupportedTypeCheck<T>();
#endif
    ShapeInfo srcShape = srcTensor.GetShapeInfo();
    uint32_t elementNumPerBlk = ONE_BLK_SIZE / sizeof(T);
    LastAxisShapeND srcNDinfo;
    LastAxisShapeND originalSrcShape;
    if (softmaxShapeInfo.srcM == 0 || softmaxShapeInfo.srcK == 0) {
        srcNDinfo = GetLastAxisShapeND(srcShape);
        originalSrcShape = GetLastAxisOriginShapeND(srcShape);
    } else {
        srcNDinfo = {softmaxShapeInfo.srcM, softmaxShapeInfo.srcK};
        originalSrcShape = {softmaxShapeInfo.oriSrcM, softmaxShapeInfo.oriSrcK};
    }
    if constexpr (isDataFormatNZ) {
        if (unlikely(srcNDinfo.k != tiling.srcK || srcNDinfo.m != tiling.srcM)) {
            SoftMaxTiling newTiling = tiling;
            SoftMaxGradTilingFunc(workLocal.GetSize(), srcNDinfo, newTiling, elementNumPerBlk, true, false, true);
            SoftmaxGradFrontNZImpl(dstTensor, gradTensor, srcTensor, workLocal, originalSrcShape, newTiling);
        } else {
            SoftmaxGradFrontNZImpl(dstTensor, gradTensor, srcTensor, workLocal, originalSrcShape, tiling);
        }
    } else {
        if (unlikely(srcNDinfo.k != tiling.srcK || srcNDinfo.m != tiling.srcM)) {
            SoftMaxTiling newTiling = tiling;
            SoftMaxGradTilingFunc(workLocal.GetSize(), srcNDinfo, newTiling, elementNumPerBlk, true, isBasicBlock);
            SoftmaxGradFrontNDImpl<T, isBasicBlock>(
                dstTensor, gradTensor, srcTensor, workLocal, newTiling, originalSrcShape);
        } else {
            SoftmaxGradFrontNDImpl<T, isBasicBlock>(
                dstTensor, gradTensor, srcTensor, workLocal, tiling, originalSrcShape);
        }
    }
}

template <typename T, bool isBasicBlock = false, bool isDataFormatNZ = false>
__aicore__ inline void SoftmaxGradFrontImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor,
    const LocalTensor<T>& srcTensor, const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo)
{
    LocalTensor<float> workLocal;
    PopStackBuffer<float, TPosition::LCM>(workLocal);
    SoftmaxGradFrontImpl<T, isBasicBlock, isDataFormatNZ>(
        dstTensor, gradTensor, srcTensor, workLocal, tiling, softmaxShapeInfo);
}

template <typename T, bool isBasicBlock = false, bool isDataFormatNZ = false>
__aicore__ inline void SoftmaxGradFrontImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling,
    const SoftMaxShapeInfo& softmaxShapeInfo)
{
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
    CheckTensorPos<uint8_t>(
        sharedTmpBuffer, Hardware::UB, "sharedTmpBuffer", "VECIN / VECCALC / VECOUT", "SoftmaxGradFront");
#endif
    auto workLocal = sharedTmpBuffer.ReinterpretCast<float>();
    SoftmaxGradFrontImpl<T, isBasicBlock, isDataFormatNZ>(
        dstTensor, gradTensor, srcTensor, workLocal, tiling, softmaxShapeInfo);
}

template <typename T, bool isReuseSource = false, bool isDataFormatNZ = false>
__aicore__ inline void SoftmaxGradImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<float>& workLocal, const SoftMaxTiling& tiling, bool isFront,
    const SoftMaxShapeInfo& softmaxShapeInfo)
{
    CHECK_FUNC_HIGHLEVEL_API(SoftmaxGrad, (T, isReuseSource, isDataFormatNZ),
        (dstTensor, gradTensor, srcTensor, workLocal, tiling, isFront, softmaxShapeInfo));
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
    CheckTensorPos<T>(dstTensor, Hardware::UB, "dstTensor", "VECIN / VECCALC / VECOUT", "SoftmaxGrad");
    CheckTensorPos<T>(gradTensor, Hardware::UB, "gradTensor", "VECIN / VECCALC / VECOUT", "SoftmaxGrad");
    CheckTensorPos<T>(srcTensor, Hardware::UB, "srcTensor", "VECIN / VECCALC / VECOUT", "SoftmaxGrad");
    ASCENDC_ASSERT((softmaxShapeInfo.srcK * sizeof(T) % ONE_BLK_SIZE == 0),
        { KERNEL_LOG(KERNEL_ERROR, "srcK should be 32B aligned, current srcK is %u", softmaxShapeInfo.srcK); });

    SoftmaxApiSupportedTypeCheck<T>();
#endif
    ShapeInfo srcShape = srcTensor.GetShapeInfo();
    uint32_t elementNumPerBlk = ONE_BLK_SIZE / sizeof(T);

    LastAxisShapeND srcNDinfo;
    LastAxisShapeND originalSrcShape;
    if (softmaxShapeInfo.srcM == 0 || softmaxShapeInfo.srcK == 0) {
        srcNDinfo = GetLastAxisShapeND(srcShape);
        originalSrcShape = GetLastAxisOriginShapeND(srcShape);
    } else {
        srcNDinfo = {softmaxShapeInfo.srcM, softmaxShapeInfo.srcK};
        originalSrcShape = {softmaxShapeInfo.oriSrcM, softmaxShapeInfo.oriSrcK};
    }
    if constexpr (isDataFormatNZ) {
        if (unlikely(srcNDinfo.k != tiling.srcK || srcNDinfo.m != tiling.srcM)) {
            SoftMaxTiling newTiling = tiling;
            SoftMaxGradTilingFunc(workLocal.GetSize(), srcNDinfo, newTiling, elementNumPerBlk, isFront, false, true);
            SoftmaxGradNZImpl(dstTensor, gradTensor, srcTensor, workLocal, originalSrcShape, newTiling, isFront);
        } else {
            SoftmaxGradNZImpl(dstTensor, gradTensor, srcTensor, workLocal, originalSrcShape, tiling, isFront);
        }
    } else {
        if (unlikely(srcNDinfo.k != tiling.srcK || srcNDinfo.m != tiling.srcM)) {
            SoftMaxTiling newTiling = tiling;
            SoftMaxGradTilingFunc(workLocal.GetSize(), srcNDinfo, newTiling, elementNumPerBlk, isFront, false);
            SoftmaxGradPostProcess<T>(
                dstTensor, gradTensor, srcTensor, workLocal, newTiling, originalSrcShape, isFront);
        } else {
            SoftmaxGradPostProcess<T>(dstTensor, gradTensor, srcTensor, workLocal, tiling, originalSrcShape, isFront);
        }
    }
}

template <typename T, bool isReuseSource = false, bool isDataFormatNZ = false>
__aicore__ inline void SoftmaxGradImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor,
    const LocalTensor<T>& srcTensor, const SoftMaxTiling& tiling, bool isFront,
    const SoftMaxShapeInfo& softmaxShapeInfo)
{
    LocalTensor<float> workLocal;
    PopStackBuffer<float, TPosition::LCM>(workLocal);
    SoftmaxGradImpl<T, isReuseSource, isDataFormatNZ>(
        dstTensor, gradTensor, srcTensor, workLocal, tiling, isFront, softmaxShapeInfo);
}
template <typename T, bool isReuseSource = false, bool isDataFormatNZ = false>
__aicore__ inline void SoftmaxGradImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor,
    const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling,
    bool isFront, const SoftMaxShapeInfo& softmaxShapeInfo)
{
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
    CheckTensorPos<uint8_t>(
        sharedTmpBuffer, Hardware::UB, "sharedTmpBuffer", "VECIN / VECCALC / VECOUT", "SoftmaxGrad");
#endif
    auto workLocal = sharedTmpBuffer.ReinterpretCast<float>();
    SoftmaxGradImpl<T, isReuseSource, isDataFormatNZ>(
        dstTensor, gradTensor, srcTensor, workLocal, tiling, isFront, softmaxShapeInfo);
}
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_ACTIVATION_SOFTMAX_SOFTMAX_GRAD_BASE_IMPL_H