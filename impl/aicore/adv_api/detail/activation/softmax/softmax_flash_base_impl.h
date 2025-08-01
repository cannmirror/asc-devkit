/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file softmax_flash_base_impl.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_ACTIVATION_SOFTMAX_SOFTMAX_FLASH_BASE_IMPL_H
#define AICORE_ADV_API_DETAIL_ACTIVATION_SOFTMAX_SOFTMAX_FLASH_BASE_IMPL_H
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
#include "regbase/c310/softmax_flash_impl.h"
#else
#include "softmax_flash_base_impl/softmax_flash_nd_process_impl.h"
#endif
#include "../../api_check/kernel_api_check.h"

namespace AscendC {
template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void SoftmaxFlashImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& sumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor,
    const LocalTensor<T>& inSumTensor, const LocalTensor<T>& inMaxTensor, const SoftMaxTiling& tiling, bool isUpdate,
    const SoftMaxShapeInfo& softmaxShapeInfo)
{
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
    CheckTensorPos<T>(dstTensor, Hardware::UB, "dstTensor", "VECIN / VECCALC / VECOUT", "SoftmaxFlash");
    CheckTensorPos<T>(sumTensor, Hardware::UB, "sumTensor", "VECIN / VECCALC / VECOUT", "SoftmaxFlash");
    CheckTensorPos<T>(maxTensor, Hardware::UB, "maxTensor", "VECIN / VECCALC / VECOUT", "SoftmaxFlash");
    CheckTensorPos<T>(srcTensor, Hardware::UB, "srcTensor", "VECIN / VECCALC / VECOUT", "SoftmaxFlash");
    CheckTensorPos<T>(expMaxTensor, Hardware::UB, "expMaxTensor", "VECIN / VECCALC / VECOUT", "SoftmaxFlash");
    CheckTensorPos<T>(inSumTensor, Hardware::UB, "inSumTensor", "VECIN / VECCALC / VECOUT", "SoftmaxFlash");
    CheckTensorPos<T>(inMaxTensor, Hardware::UB, "inMaxTensor", "VECIN / VECCALC / VECOUT", "SoftmaxFlash");
    ASCENDC_ASSERT((softmaxShapeInfo.srcK * sizeof(T) % ONE_BLK_SIZE == 0),
        { KERNEL_LOG(KERNEL_ERROR, "srcK should be 32B aligned, current srcK is %u", softmaxShapeInfo.srcK); });

    SoftmaxApiSupportedTypeCheck<T>();
#endif
    CHECK_FUNC_HIGHLEVEL_API(SoftmaxFlash, (T, isReuseSource, isBasicBlock),
        (dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor, inSumTensor, inMaxTensor, tiling, isUpdate,
            softmaxShapeInfo));
    const uint32_t elementNumPerBlk = ONE_BLK_SIZE / sizeof(T);
    const uint32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(T);
    LocalTensor<float> workLocal;
    PopStackBuffer<float, TPosition::LCM>(workLocal);
    uint32_t workLocalSize = workLocal.GetSize();
    LastAxisShapeND srcNDinfo;
    LastAxisShapeND originalSrcShape;
    if (softmaxShapeInfo.srcM == 0 || softmaxShapeInfo.srcK == 0) {
        ShapeInfo srcShape = srcTensor.GetShapeInfo();
        srcNDinfo = GetLastAxisShapeND(srcShape);
        originalSrcShape = GetLastAxisOriginShapeND(srcShape);
    } else {
        srcNDinfo = {softmaxShapeInfo.srcM, softmaxShapeInfo.srcK};
        originalSrcShape = {softmaxShapeInfo.oriSrcM, softmaxShapeInfo.oriSrcK};
    }
    if (unlikely(srcNDinfo.k != tiling.srcK || srcNDinfo.m != tiling.srcM)) {
        SoftMaxTiling new_tiling = tiling;
        SoftMaxFlashTilingFunc(workLocalSize, srcNDinfo, new_tiling, elementNumPerBlk, isUpdate, isBasicBlock);
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
        SoftmaxFlashPostProcess<T, T, isBasicBlock>(dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor,
            inSumTensor, inMaxTensor, workLocal, originalSrcShape, new_tiling, isUpdate);
#else
        SoftmaxFlashPostProcess<T, isBasicBlock>(dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor, inSumTensor,
            inMaxTensor, workLocal, originalSrcShape, new_tiling, isUpdate);
#endif
    } else {
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
        SoftmaxFlashPostProcess<T, T, isBasicBlock>(dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor,
            inSumTensor, inMaxTensor, workLocal, originalSrcShape, tiling, isUpdate);
#else
        SoftmaxFlashPostProcess<T, isBasicBlock>(dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor, inSumTensor,
            inMaxTensor, workLocal, originalSrcShape, tiling, isUpdate);
#endif
    }
}
template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void SoftmaxFlashImpl(const LocalTensor<half>& dstTensor, const LocalTensor<float>& sumTensor,
    const LocalTensor<float>& maxTensor, const LocalTensor<half>& srcTensor, const LocalTensor<half>& expMaxTensor,
    const LocalTensor<float>& inSumTensor, const LocalTensor<float>& inMaxTensor, const SoftMaxTiling& tiling,
    bool isUpdate, const SoftMaxShapeInfo& softmaxShapeInfo)
{
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
    CheckTensorPos<half>(dstTensor, Hardware::UB, "dstTensor", "VECIN / VECCALC / VECOUT", "SoftmaxFlash");
    CheckTensorPos<float>(sumTensor, Hardware::UB, "sumTensor", "VECIN / VECCALC / VECOUT", "SoftmaxFlash");
    CheckTensorPos<float>(maxTensor, Hardware::UB, "maxTensor", "VECIN / VECCALC / VECOUT", "SoftmaxFlash");
    CheckTensorPos<half>(srcTensor, Hardware::UB, "srcTensor", "VECIN / VECCALC / VECOUT", "SoftmaxFlash");
    CheckTensorPos<half>(expMaxTensor, Hardware::UB, "expMaxTensor", "VECIN / VECCALC / VECOUT", "SoftmaxFlash");
    CheckTensorPos<float>(inSumTensor, Hardware::UB, "inSumTensor", "VECIN / VECCALC / VECOUT", "SoftmaxFlash");
    CheckTensorPos<float>(inMaxTensor, Hardware::UB, "inMaxTensor", "VECIN / VECCALC / VECOUT", "SoftmaxFlash");
    ASCENDC_ASSERT((softmaxShapeInfo.srcK * sizeof(half) % ONE_BLK_SIZE == 0),
        { KERNEL_LOG(KERNEL_ERROR, "srcK should be 32B aligned, current srcK is %u", softmaxShapeInfo.srcK); });
#endif
    CHECK_FUNC_HIGHLEVEL_API(SoftmaxFlash, (T, isReuseSource, isBasicBlock),
        (dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor, inSumTensor, inMaxTensor, tiling, isUpdate,
            softmaxShapeInfo));
    LocalTensor<float> workLocal;
    PopStackBuffer<float, TPosition::LCM>(workLocal);
    uint32_t workLocalSize = workLocal.GetSize();

    LastAxisShapeND srcNDinfo;
    LastAxisShapeND originalSrcShape;
    if (softmaxShapeInfo.srcM == 0 || softmaxShapeInfo.srcK == 0) {
        ShapeInfo srcShape = srcTensor.GetShapeInfo();
        srcNDinfo = GetLastAxisShapeND(srcShape);
        originalSrcShape = GetLastAxisOriginShapeND(srcShape);
    } else {
        srcNDinfo = {softmaxShapeInfo.srcM, softmaxShapeInfo.srcK};
        originalSrcShape = {softmaxShapeInfo.oriSrcM, softmaxShapeInfo.oriSrcK};
    }
    if (srcNDinfo.k != tiling.srcK || srcNDinfo.m != tiling.srcM) {
        SoftMaxTiling newTiling = tiling;
        SoftMaxFlashTilingFunc(workLocalSize, srcNDinfo, newTiling, FLOAT_NUM_PER_BLK, isUpdate, isBasicBlock);
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
        SoftmaxFlashPostProcess<half, float, isBasicBlock>(dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor,
            inSumTensor, inMaxTensor, workLocal, originalSrcShape, newTiling, isUpdate);
#else
        if (!isUpdate) {
            SoftMaxNDImpl<half, float>(
                dstTensor, sumTensor, maxTensor, srcTensor, workLocal, originalSrcShape, newTiling);
        } else {
            SoftmaxFlashNDImpl<isBasicBlock>(dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor, inSumTensor,
                inMaxTensor, workLocal, originalSrcShape, newTiling);
        }
#endif
    } else {
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
        SoftmaxFlashPostProcess<half, float, isBasicBlock>(dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor,
            inSumTensor, inMaxTensor, workLocal, originalSrcShape, tiling, isUpdate);
#else
        if (!isUpdate) {
            SoftMaxNDImpl<half, float>(dstTensor, sumTensor, maxTensor, srcTensor, workLocal, originalSrcShape, tiling);
        } else {
            SoftmaxFlashNDImpl<isBasicBlock>(dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor, inSumTensor,
                inMaxTensor, workLocal, originalSrcShape, tiling);
        }
#endif
    }
}

template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void SoftmaxFlashImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& sumTensor,
    const LocalTensor<T>& maxTensor, const LocalTensor<T>& srcTensor, const LocalTensor<T>& expMaxTensor,
    const LocalTensor<T>& inSumTensor, const LocalTensor<T>& inMaxTensor, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const SoftMaxTiling& tiling, bool isUpdate, const SoftMaxShapeInfo& softmaxShapeInfo)
{
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
    CheckTensorPos<T>(dstTensor, Hardware::UB, "dstTensor", "VECIN / VECCALC / VECOUT", "SoftmaxFlash");
    CheckTensorPos<T>(sumTensor, Hardware::UB, "sumTensor", "VECIN / VECCALC / VECOUT", "SoftmaxFlash");
    CheckTensorPos<T>(maxTensor, Hardware::UB, "maxTensor", "VECIN / VECCALC / VECOUT", "SoftmaxFlash");
    CheckTensorPos<T>(srcTensor, Hardware::UB, "srcTensor", "VECIN / VECCALC / VECOUT", "SoftmaxFlash");
    CheckTensorPos<T>(expMaxTensor, Hardware::UB, "expMaxTensor", "VECIN / VECCALC / VECOUT", "SoftmaxFlash");
    CheckTensorPos<T>(inSumTensor, Hardware::UB, "inSumTensor", "VECIN / VECCALC / VECOUT", "SoftmaxFlash");
    CheckTensorPos<T>(inMaxTensor, Hardware::UB, "inMaxTensor", "VECIN / VECCALC / VECOUT", "SoftmaxFlash");
    CheckTensorPos<uint8_t>(
        sharedTmpBuffer, Hardware::UB, "sharedTmpBuffer", "VECIN / VECCALC / VECOUT", "SoftmaxFlash");
    ASCENDC_ASSERT((softmaxShapeInfo.srcK * sizeof(T) % ONE_BLK_SIZE == 0),
        { KERNEL_LOG(KERNEL_ERROR, "srcK should be 32B aligned, current srcK is %u", softmaxShapeInfo.srcK); });

    SoftmaxApiSupportedTypeCheck<T>();
#endif
    CHECK_FUNC_HIGHLEVEL_API(SoftmaxFlash, (T, isReuseSource, isBasicBlock),
        (dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor, inSumTensor, inMaxTensor, sharedTmpBuffer, tiling,
            isUpdate, softmaxShapeInfo));
    auto tempBuffer = sharedTmpBuffer.ReinterpretCast<float>();
    const uint32_t elementNumPerBlk = ONE_BLK_SIZE / sizeof(T);
    const uint32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(T);
    uint32_t workLocalSize = tempBuffer.GetSize();
    LastAxisShapeND srcNDinfo;
    LastAxisShapeND originalSrcShape;
    if (softmaxShapeInfo.srcM == 0 || softmaxShapeInfo.srcK == 0) {
        ShapeInfo srcShape = srcTensor.GetShapeInfo();
        srcNDinfo = GetLastAxisShapeND(srcShape);
        originalSrcShape = GetLastAxisOriginShapeND(srcShape);
    } else {
        srcNDinfo = {softmaxShapeInfo.srcM, softmaxShapeInfo.srcK};
        originalSrcShape = {softmaxShapeInfo.oriSrcM, softmaxShapeInfo.oriSrcK};
    }
    if (unlikely(srcNDinfo.k != tiling.srcK || srcNDinfo.m != tiling.srcM)) {
        SoftMaxTiling newTiling = tiling;
        SoftMaxFlashTilingFunc(workLocalSize, srcNDinfo, newTiling, elementNumPerBlk, isUpdate, isBasicBlock);
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
        SoftmaxFlashPostProcess<T, T, isBasicBlock>(dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor,
            inSumTensor, inMaxTensor, tempBuffer, originalSrcShape, newTiling, isUpdate);
#else
        SoftmaxFlashPostProcess<T, isBasicBlock>(dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor, inSumTensor,
            inMaxTensor, tempBuffer, originalSrcShape, newTiling, isUpdate);
#endif
    } else {
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
        SoftmaxFlashPostProcess<T, T, isBasicBlock>(dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor,
            inSumTensor, inMaxTensor, tempBuffer, originalSrcShape, tiling, isUpdate);
#else
        SoftmaxFlashPostProcess<T, isBasicBlock>(dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor, inSumTensor,
            inMaxTensor, tempBuffer, originalSrcShape, tiling, isUpdate);
#endif
    }
}

template <typename T, bool isReuseSource = false, bool isBasicBlock = false>
__aicore__ inline void SoftmaxFlashImpl(const LocalTensor<half>& dstTensor, const LocalTensor<float>& sumTensor,
    const LocalTensor<float>& maxTensor, const LocalTensor<half>& srcTensor, const LocalTensor<half>& expMaxTensor,
    const LocalTensor<float>& inSumTensor, const LocalTensor<float>& inMaxTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const SoftMaxTiling& tiling, bool isUpdate,
    const SoftMaxShapeInfo& softmaxShapeInfo)
{
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
    CheckTensorPos<half>(dstTensor, Hardware::UB, "dstTensor", "VECIN / VECCALC / VECOUT", "SoftmaxFlash");
    CheckTensorPos<float>(sumTensor, Hardware::UB, "sumTensor", "VECIN / VECCALC / VECOUT", "SoftmaxFlash");
    CheckTensorPos<float>(maxTensor, Hardware::UB, "maxTensor", "VECIN / VECCALC / VECOUT", "SoftmaxFlash");
    CheckTensorPos<half>(srcTensor, Hardware::UB, "srcTensor", "VECIN / VECCALC / VECOUT", "SoftmaxFlash");
    CheckTensorPos<half>(expMaxTensor, Hardware::UB, "expMaxTensor", "VECIN / VECCALC / VECOUT", "SoftmaxFlash");
    CheckTensorPos<float>(inSumTensor, Hardware::UB, "inSumTensor", "VECIN / VECCALC / VECOUT", "SoftmaxFlash");
    CheckTensorPos<float>(inMaxTensor, Hardware::UB, "inMaxTensor", "VECIN / VECCALC / VECOUT", "SoftmaxFlash");
    CheckTensorPos<uint8_t>(
        sharedTmpBuffer, Hardware::UB, "sharedTmpBuffer", "VECIN / VECCALC / VECOUT", "SoftmaxFlash");
    ASCENDC_ASSERT((softmaxShapeInfo.srcK * sizeof(half) % ONE_BLK_SIZE == 0),
        { KERNEL_LOG(KERNEL_ERROR, "srcK should be 32B aligned, current srcK is %u", softmaxShapeInfo.srcK); });
#endif
    CHECK_FUNC_HIGHLEVEL_API(SoftmaxFlash, (T, isReuseSource, isBasicBlock),
        (dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor, inSumTensor, inMaxTensor, sharedTmpBuffer, tiling,
            isUpdate, softmaxShapeInfo));
    auto tempBuffer = sharedTmpBuffer.ReinterpretCast<float>();
    LastAxisShapeND srcNDinfo;
    LastAxisShapeND originalSrcShape;
    if (softmaxShapeInfo.srcM == 0 || softmaxShapeInfo.srcK == 0) {
        ShapeInfo srcShape = srcTensor.GetShapeInfo();
        srcNDinfo = GetLastAxisShapeND(srcShape);
        originalSrcShape = GetLastAxisOriginShapeND(srcShape);
    } else {
        srcNDinfo = {softmaxShapeInfo.srcM, softmaxShapeInfo.srcK};
        originalSrcShape = {softmaxShapeInfo.oriSrcM, softmaxShapeInfo.oriSrcK};
    }
    uint32_t workLocalSize = tempBuffer.GetSize();
    if (srcNDinfo.k != tiling.srcK || srcNDinfo.m != tiling.srcM) {
        SoftMaxTiling newTiling = tiling;
        SoftMaxFlashTilingFunc(workLocalSize, srcNDinfo, newTiling, FLOAT_NUM_PER_BLK, isUpdate, isBasicBlock);
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
        SoftmaxFlashPostProcess<half, float, isBasicBlock>(dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor,
            inSumTensor, inMaxTensor, tempBuffer, originalSrcShape, newTiling, isUpdate);
#else
        if (!isUpdate) {
            SoftMaxNDImpl<half, float>(
                dstTensor, sumTensor, maxTensor, srcTensor, tempBuffer, originalSrcShape, newTiling);
        } else {
            SoftmaxFlashNDImpl<isBasicBlock>(dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor, inSumTensor,
                inMaxTensor, tempBuffer, originalSrcShape, newTiling);
        }
#endif
    } else {
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
        SoftmaxFlashPostProcess<half, float, isBasicBlock>(dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor,
            inSumTensor, inMaxTensor, tempBuffer, originalSrcShape, tiling, isUpdate);
#else
        if (!isUpdate) {
            SoftMaxNDImpl<half, float>(
                dstTensor, sumTensor, maxTensor, srcTensor, tempBuffer, originalSrcShape, tiling);
        } else {
            SoftmaxFlashNDImpl<isBasicBlock>(dstTensor, sumTensor, maxTensor, srcTensor, expMaxTensor, inSumTensor,
                inMaxTensor, tempBuffer, originalSrcShape, tiling);
        }
#endif
    }
}
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_ACTIVATION_SOFTMAX_SOFTMAX_FLASH_BASE_IMPL_H