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
 * \file simple_softmax_base_impl.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_ACTIVATION_SOFTMAX_SIMPLE_SOFTMAX_BASE_IMPL_H
#define AICORE_ADV_API_DETAIL_ACTIVATION_SOFTMAX_SIMPLE_SOFTMAX_BASE_IMPL_H

#if defined(__DAV_C310__) || defined(__DAV_310R6__)
#include "regbase/c310/simple_softmax_impl.h"
#elif __CCE_AICORE__ == 300
#include "regbase/v300/simple_softmax_impl.h"
#elif __CCE_AICORE__ == 220
#include "membase/v220/simple_softmax_impl.h"
#elif __CCE_AICORE__ == 200
#include "membase/v200/simple_softmax_impl.h"
#endif
#include "../../api_check/kernel_api_check.h"

namespace AscendC {
template <typename T1, typename T2, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SimpleSoftMaxImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& inSumTensor,
    const LocalTensor<T2>& inMaxTensor, const LocalTensor<T1>& src, const LocalTensor<float>& workLocal,
    const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo)
{
    CHECK_FUNC_HIGHLEVEL_API(SimpleSoftMax, (T1, T2, isReuseSource, isBasicBlock, isDataFormatNZ, config),
        (dst, inSumTensor, inMaxTensor, src, workLocal, tiling, softmaxShapeInfo));
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
    CheckTensorPos<T1>(dst, Hardware::UB, "dstTensor", "VECIN / VECCALC / VECOUT", "SimpleSoftMax");
    CheckTensorPos<T2>(inSumTensor, Hardware::UB, "inSumTensor", "VECIN / VECCALC / VECOUT", "SimpleSoftMax");
    CheckTensorPos<T2>(inMaxTensor, Hardware::UB, "inMaxTensor", "VECIN / VECCALC / VECOUT", "SimpleSoftMax");
    CheckTensorPos<T1>(src, Hardware::UB, "srcTensor", "VECIN / VECCALC / VECOUT", "SimpleSoftMax");
    ASCENDC_ASSERT((softmaxShapeInfo.srcK * sizeof(T1) % ONE_BLK_SIZE == 0),
        { KERNEL_LOG(KERNEL_ERROR, "srcK should be 32B aligned, current srcK is %u", softmaxShapeInfo.srcK); });
    SoftmaxApiSupportedTypeCheck<T1>();
    SoftmaxApiSupportedTypeCheck<T2>();
#endif
    SetMaskNorm();
    ResetMask();
    ShapeInfo srcShape = src.GetShapeInfo();
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
            SoftMaxTilingFunc(workLocal.GetSize(), {srcNDinfo.m, srcNDinfo.k, originalSrcShape.m, srcNDinfo.k},
                newTiling, sizeof(T1), sizeof(T2), false, isDataFormatNZ);
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
            SimpleSoftMaxNZImpl<T1, T2>(dst, inSumTensor, inMaxTensor, src, workLocal, newTiling, originalSrcShape);
#else
            SimpleSoftMaxNZImpl<T1, T2>(dst, inSumTensor, inMaxTensor, src, workLocal, newTiling);
#endif
        } else {
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
            SimpleSoftMaxNZImpl<T1, T2>(dst, inSumTensor, inMaxTensor, src, workLocal, tiling, originalSrcShape);
#else
            SimpleSoftMaxNZImpl<T1, T2>(dst, inSumTensor, inMaxTensor, src, workLocal, tiling);
#endif
        }
    } else {
        if (unlikely(srcNDinfo.k != tiling.srcK || srcNDinfo.m != tiling.srcM)) {
            SoftMaxTiling newTiling = tiling;
            SoftMaxTilingFunc(workLocal.GetSize(), {srcNDinfo.m, srcNDinfo.k, originalSrcShape.m, srcNDinfo.k},
                newTiling, sizeof(T1), sizeof(T2), isBasicBlock);
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
            SimpleSoftMaxNDImpl<T1, T2, isBasicBlock, config>(dst, inSumTensor, inMaxTensor, src, workLocal, newTiling);
#else
            SimpleSoftMaxNDImpl<T1, isBasicBlock, config>(dst, inSumTensor, inMaxTensor, src, workLocal, newTiling);
#endif
        } else {
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
            SimpleSoftMaxNDImpl<T1, T2, isBasicBlock, config>(dst, inSumTensor, inMaxTensor, src, workLocal, tiling);
#else
            SimpleSoftMaxNDImpl<T1, isBasicBlock, config>(dst, inSumTensor, inMaxTensor, src, workLocal, tiling);
#endif
        }
    }
}

template <typename T1, typename T2, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SimpleSoftMaxImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& inSumTensor,
    const LocalTensor<T2>& inMaxTensor, const LocalTensor<T1>& src, const SoftMaxTiling& tiling,
    const SoftMaxShapeInfo& softmaxShapeInfo)
{
    LocalTensor<float> workLocal;
    PopStackBuffer<float, TPosition::LCM>(workLocal);
    SimpleSoftMaxImpl<T1, T2, isReuseSource, isBasicBlock, isDataFormatNZ, config>(
        dst, inSumTensor, inMaxTensor, src, workLocal, tiling, softmaxShapeInfo);
}

template <typename T1, typename T2, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SimpleSoftMaxImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& inSumTensor,
    const LocalTensor<T2>& inMaxTensor, const LocalTensor<T1>& src, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo)
{
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
    CheckTensorPos<uint8_t>(
        sharedTmpBuffer, Hardware::UB, "sharedTmpBuffer", "VECIN / VECCALC / VECOUT", "SimpleSoftMax");
#endif
    auto workLocal = sharedTmpBuffer.ReinterpretCast<float>();
    SimpleSoftMaxImpl<T1, T2, isReuseSource, isBasicBlock, isDataFormatNZ, config>(
        dst, inSumTensor, inMaxTensor, src, workLocal, tiling, softmaxShapeInfo);
}
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_ACTIVATION_SOFTMAX_SIMPLE_SOFTMAX_BASE_IMPL_H