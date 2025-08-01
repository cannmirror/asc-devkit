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
 * \file softmax_base_impl.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_ACTIVATION_SOFTMAX_SOFTMAX_BASE_IMPL_H
#define AICORE_ADV_API_DETAIL_ACTIVATION_SOFTMAX_SOFTMAX_BASE_IMPL_H

#if defined(__DAV_C310__) || defined(__DAV_310R6__)
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
    CHECK_FUNC_HIGHLEVEL_API(
        SoftMax, (T, isReuseSource, isBasicBlock, config), (dst, src, workLocal, tiling, softmaxShapeInfo));
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
    CheckTensorPos<T>(dst, Hardware::UB, "dstTensor", "VECIN / VECCALC / VECOUT", "SoftMax");
    CheckTensorPos<T>(src, Hardware::UB, "srcTensor", "VECIN / VECCALC / VECOUT", "SoftMax");
    ASCENDC_ASSERT((softmaxShapeInfo.srcK * sizeof(T) % ONE_BLK_SIZE == 0),
        { KERNEL_LOG(KERNEL_ERROR, "srcK should be 32B aligned, current srcK is %u", softmaxShapeInfo.srcK); });
    SoftmaxApiSupportedTypeCheck<T>();
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
    // when the shape is changed, need recalculate the softmax's tiling
    if (unlikely(srcNDinfo.k != tiling.srcK || srcNDinfo.m != tiling.srcM)) {
        SoftMaxTiling newTiling = tiling;
        SoftMaxTilingFunc(workLocal.GetSize(), {srcNDinfo.m, srcNDinfo.k, originalSrcShape.m, srcNDinfo.k}, newTiling,
            sizeof(T), sizeof(float), isBasicBlock);
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
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
    CheckTensorPos<uint8_t>(sharedTmpBuffer, Hardware::UB, "sharedTmpBuffer", "VECIN / VECCALC / VECOUT", "SoftMax");
#endif
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
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
    CheckTensorPos<T1>(dst, Hardware::UB, "dstTensor", "VECIN / VECCALC / VECOUT", "SoftMax");
    CheckTensorPos<T2>(sumTensor, Hardware::UB, "sumTensor", "VECIN / VECCALC / VECOUT", "SoftMax");
    CheckTensorPos<T2>(maxTensor, Hardware::UB, "maxTensor", "VECIN / VECCALC / VECOUT", "SoftMax");
    CheckTensorPos<T1>(src, Hardware::UB, "srcTensor", "VECIN / VECCALC / VECOUT", "SoftMax");
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
        // when the shape is changed, need recalculate the softmax's tiling
        if (unlikely(srcNDinfo.k != tiling.srcK || originalSrcShape.m != tiling.srcM)) {
            SoftMaxTiling newTiling = tiling;
            SoftMaxTilingFunc(workLocal.GetSize(), {srcNDinfo.m, srcNDinfo.k, originalSrcShape.m, srcNDinfo.k},
                newTiling, sizeof(T1), sizeof(T2), false, isDataFormatNZ);
            SoftMaxNZImpl<T1, T2, isBasicBlock>(dst, sumTensor, maxTensor, src, workLocal, originalSrcShape, newTiling);
        } else {
            SoftMaxNZImpl<T1, T2, isBasicBlock>(dst, sumTensor, maxTensor, src, workLocal, originalSrcShape, tiling);
        }
    } else {
        // when the shape is changed, need recalculate the softmax's tiling
        if (unlikely(srcNDinfo.k != tiling.srcK || srcNDinfo.m != tiling.srcM)) {
            SoftMaxTiling newTiling = tiling;
            SoftMaxTilingFunc(workLocal.GetSize(), {srcNDinfo.m, srcNDinfo.k, originalSrcShape.m, srcNDinfo.k},
                newTiling, sizeof(T1), sizeof(T2), isBasicBlock);
            SoftMaxNDImpl<T1, T2, isBasicBlock, config>(
                dst, sumTensor, maxTensor, src, workLocal, originalSrcShape, newTiling);
        } else {
            SoftMaxNDImpl<T1, T2, isBasicBlock, config>(
                dst, sumTensor, maxTensor, src, workLocal, originalSrcShape, tiling);
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
    SoftMaxImpl<T1, T2, isReuseSource, isBasicBlock, isDataFormatNZ, config>(
        dst, sumTensor, maxTensor, src, workLocal, tiling, softmaxShapeInfo);
}

template <typename T1, typename T2, bool isReuseSource = false, bool isBasicBlock = false, bool isDataFormatNZ = false,
    const SoftmaxConfig& config = SOFTMAX_DEFAULT_CFG>
__aicore__ inline void SoftMaxImpl(const LocalTensor<T1>& dst, const LocalTensor<T2>& sumTensor,
    const LocalTensor<T2>& maxTensor, const LocalTensor<T1>& src, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const SoftMaxTiling& tiling, const SoftMaxShapeInfo& softmaxShapeInfo)
{
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
    CheckTensorPos<uint8_t>(sharedTmpBuffer, Hardware::UB, "sharedTmpBuffer", "VECIN / VECCALC / VECOUT", "SoftMax");
#endif
    auto workLocal = sharedTmpBuffer.ReinterpretCast<float>();
    SoftMaxImpl<T1, T2, isReuseSource, isBasicBlock, isDataFormatNZ, config>(
        dst, sumTensor, maxTensor, src, workLocal, tiling, softmaxShapeInfo);
}

#if defined(__DAV_C310__) || defined(__DAV_310R6__)
template <typename T1, typename T2, uint32_t stepSize, uint32_t stride>
__aicore__ inline void AdjustSoftMaxResNZImpl(__local_mem__ T1* resUb, __local_mem__ T2* maxUb,
    __local_mem__ uint64_t* maskUb, const uint32_t from, const T1 to, const uint32_t dataBlock,
    const uint16_t mRepeatTimes, const uint16_t kRepeatTimes)
{
    MicroAPI::RegTensor<T1> srcVreg;
    MicroAPI::RegTensor<T1> tmpVreg;
    MicroAPI::RegTensor<T1> dstVreg;
    MicroAPI::RegTensor<T2> maxVreg;
    MicroAPI::MaskReg cmpMaskReg;
    MicroAPI::MaskReg cmpMaskReg0;
    MicroAPI::MaskReg cmpMaskReg1;
    MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<T1, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg dstMask = MicroAPI::CreateMask<T1, MicroAPI::MaskPattern::ALLF>();

    bool isUpdateNeedCheck = false;
    for (uint16_t i = 0; i < mRepeatTimes; ++i) {
        if constexpr (sizeof(T2) == sizeof(float)) {
            MicroAPI::DataCopy<T2, MicroAPI::LoadDist::DIST_BRC_B32>(maxVreg, maxUb + i * stepSize);
            // either full mask or zero mask
            MicroAPI::CompareScalar(cmpMaskReg, (MicroAPI::RegTensor<uint32_t>&)maxVreg, from, maskFull);
        } else if constexpr (sizeof(T2) == sizeof(half)) {
            MicroAPI::DataCopy<T2, MicroAPI::LoadDist::DIST_BRC_B16>(maxVreg, maxUb + i * stepSize);
            // either full mask or zero mask
            MicroAPI::CompareScalar(cmpMaskReg, (MicroAPI::RegTensor<uint16_t>&)maxVreg, (uint16_t)from, maskFull);
        }
        if constexpr (sizeof(T1) != sizeof(T2)) {
            MicroAPI::MaskPack(cmpMaskReg0, cmpMaskReg);
            MicroAPI::MaskPack<MicroAPI::HighLowPart::HIGHEST>(cmpMaskReg1, cmpMaskReg);
            MicroAPI::MaskOr(cmpMaskReg, cmpMaskReg0, cmpMaskReg1, maskFull);
        }
        MicroAPI::MaskOr(dstMask, dstMask, cmpMaskReg, maskFull);
        MicroAPI::Duplicate(tmpVreg, to, maskFull);
        for (uint16_t j = 0; j < kRepeatTimes; ++j) {
            MicroAPI::DataCopy(srcVreg, resUb + i * stride + j * dataBlock);
            MicroAPI::Select(dstVreg, tmpVreg, srcVreg, cmpMaskReg);
            MicroAPI::DataCopy(resUb + i * stride + j * dataBlock, dstVreg, maskFull);
        }
    }
    MicroAPI::DataCopy((__local_mem__ uint8_t*)maskUb, dstMask);
}

template <typename T1, typename T2, uint32_t stepSize, uint32_t stride>
__aicore__ inline void AdjustSoftMaxResNDImpl(__local_mem__ T1* resUb, __local_mem__ T2* maxUb,
    __local_mem__ uint64_t* maskUb, const uint32_t from, const T1 to, const uint32_t srcK, const uint16_t srcM,
    const uint16_t repeatTimes)
{
    MicroAPI::RegTensor<T1> srcVreg;
    MicroAPI::RegTensor<T1> tmpVreg;
    MicroAPI::RegTensor<T1> dstVreg;
    MicroAPI::RegTensor<T2> maxVreg;
    MicroAPI::MaskReg maskReg;
    MicroAPI::MaskReg cmpMaskReg;
    MicroAPI::MaskReg cmpMaskReg0;
    MicroAPI::MaskReg cmpMaskReg1;
    MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<T1, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg dstMask = MicroAPI::CreateMask<T1, MicroAPI::MaskPattern::ALLF>();

    for (uint16_t i = 0; i < srcM; i++) {
        if constexpr (sizeof(T2) == sizeof(float)) {
            MicroAPI::DataCopy<T2, MicroAPI::LoadDist::DIST_BRC_B32>(maxVreg, maxUb + i * stepSize);
            // either full mask or zero mask
            MicroAPI::CompareScalar(cmpMaskReg, (MicroAPI::RegTensor<uint32_t>&)maxVreg, from, maskFull);
        } else if constexpr (sizeof(T2) == sizeof(half)) {
            MicroAPI::DataCopy<T2, MicroAPI::LoadDist::DIST_BRC_B16>(maxVreg, maxUb + i * stepSize);
            // either full mask or zero mask
            MicroAPI::CompareScalar(cmpMaskReg, (MicroAPI::RegTensor<uint16_t>&)maxVreg, (uint16_t)from, maskFull);
        }
        if constexpr (sizeof(T1) == sizeof(half) && sizeof(T2) == sizeof(float)) {
            MicroAPI::MaskPack(cmpMaskReg0, cmpMaskReg);
            MicroAPI::MaskPack<MicroAPI::HighLowPart::HIGHEST>(cmpMaskReg1, cmpMaskReg);
            MicroAPI::MaskOr(cmpMaskReg, cmpMaskReg0, cmpMaskReg1, maskFull);
        }
        MicroAPI::MaskOr(dstMask, dstMask, cmpMaskReg, maskFull);
        MicroAPI::Duplicate(tmpVreg, to, maskFull);
        uint32_t sreg = srcK;
        for (uint16_t j = 0; j < repeatTimes; j++) {
            maskReg = MicroAPI::UpdateMask<T1>(sreg);
            MicroAPI::DataCopy(srcVreg, resUb + i * srcK + j * stride);
            MicroAPI::Select(dstVreg, tmpVreg, srcVreg, cmpMaskReg);
            MicroAPI::DataCopy(resUb + i * srcK + j * stride, dstVreg, maskReg);
        }
    }
    MicroAPI::DataCopy((__local_mem__ uint8_t*)maskUb, dstMask);
}
#else
template <typename T1, typename T2, uint8_t stepSizeMode = 0>
__aicore__ inline bool AdjustSoftMaxResNZImpl(const LocalTensor<T1>& softMaxRes, const LocalTensor<T2>& maxTensor,
    const uint32_t from, const T1 to, const SoftMaxShapeInfo& softmaxShapeInfo)
{
    uint32_t floatStepSize = ONE_BLK_FLOAT_NUM;
    uint32_t halfStepSize = ONE_BLK_HALF_NUM;

    bool isUpdateNeedCheck = false;
    const uint32_t splitNZBlockCount = softmaxShapeInfo.srcK / SOFTMAX_SHAPE_NZ_BASIC_COUNT;
    SetVectorMask<float>(SOFTMAX_SHAPE_NZ_BASIC_COUNT);
    for (uint32_t j = 0; j < softmaxShapeInfo.srcM; j++) {
        uint32_t offset = j * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
        uint32_t splitCount = softmaxShapeInfo.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
        if constexpr (sizeof(T2) == sizeof(float)) {
            T2 maxValue = maxTensor.GetValue(j * floatStepSize);
            uint32_t checkValue = *reinterpret_cast<uint32_t*>(&maxValue);
            if (checkValue == from) {
                for (uint32_t k = 0; k < splitNZBlockCount; k++) {
                    Duplicate<T1, false>(
                        softMaxRes[offset + splitCount * k], to, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
                }
                isUpdateNeedCheck = true;
            }
        } else {
            T2 maxValue = maxTensor.GetValue(j * halfStepSize);
            uint16_t checkValue = *reinterpret_cast<uint16_t*>(&maxValue);
            if (checkValue == (uint16_t)from) {
                for (uint32_t k = 0; k < splitNZBlockCount; k++) {
                    Duplicate<T1, false>(
                        softMaxRes[offset + splitCount * k], to, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
                }
                isUpdateNeedCheck = true;
            }
        }
    }
    ResetMask();
    return isUpdateNeedCheck;
}

template <typename T1, typename T2, uint8_t stepSizeMode = 0>
__aicore__ inline bool AdjustSoftMaxResNDImpl(const LocalTensor<T1>& softMaxRes, const LocalTensor<T2>& maxTensor,
    const uint32_t from, const T1 to, const SoftMaxShapeInfo& softmaxShapeInfo)
{
    uint32_t floatStepSize = ONE_BLK_FLOAT_NUM;
    uint32_t halfStepSize = ONE_BLK_HALF_NUM;
    if constexpr (stepSizeMode) {
        floatStepSize = 1;
        halfStepSize = 1;
    }

    bool isUpdateNeedCheck = false;
    SetMaskCount();
    SetVectorMask<float, MaskMode::COUNTER>(0, softmaxShapeInfo.srcK);
    for (uint32_t j = 0; j < softmaxShapeInfo.srcM; j++) {
        if constexpr (sizeof(T2) == sizeof(float)) {
            T2 maxValue = maxTensor.GetValue(j * floatStepSize);
            uint32_t checkValue = *reinterpret_cast<uint32_t*>(&maxValue);
            if (checkValue == from) {
                Duplicate<T1, false>(
                    softMaxRes[j * softmaxShapeInfo.srcK], to, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
                isUpdateNeedCheck = true;
            }
        } else {
            T2 maxValue = maxTensor.GetValue(j * halfStepSize);
            uint16_t checkValue = *reinterpret_cast<uint16_t*>(&maxValue);
            if (checkValue == (uint16_t)from) {
                Duplicate<T1, false>(
                    softMaxRes[j * softmaxShapeInfo.srcK], to, MASK_PLACEHOLDER, 1, 1, DEFAULT_REPEAT_STRIDE);
                isUpdateNeedCheck = true;
            }
        }
    }
    SetMaskNorm();
    ResetMask();
    return isUpdateNeedCheck;
}
#endif

template <typename T1, typename T2, bool isDataFormatNZ = false, uint8_t stepSizeMode = 0>
__aicore__ inline bool AdjustSoftMaxResImpl(const LocalTensor<T1>& softMaxRes, const LocalTensor<T2>& maxTensor,
    const uint32_t from, const T1 to, const SoftMaxShapeInfo& softmaxShapeInfo)
{
    CHECK_FUNC_HIGHLEVEL_API(
        AdjustSoftMaxRes, (T1, T2, isDataFormatNZ, stepSizeMode), (softMaxRes, maxTensor, from, to, softmaxShapeInfo));
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
    SoftmaxApiSupportedTypeCheck<T1>();
    SoftmaxApiSupportedTypeCheck<T2>();
    static_assert((SupportType<Tuple<T1, T2>, Tuple<half, float>, Tuple<half, half>, Tuple<float, float>>()),
        "Failed to check dtype in SimpleSoftMax, current api "
        "support dtype combination is T1 : half, T2 : float; T1 : half, T2 : half; "
        "T1 : float, T2 : float");
    constexpr uint32_t stride = GetVecLen() / sizeof(T1);
    __local_mem__ T1* resUb = (__local_mem__ T1*)softMaxRes.GetPhyAddr();
    __local_mem__ T2* maxUb = (__local_mem__ T2*)maxTensor.GetPhyAddr();
    __local_mem__ uint64_t* maskBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(TMP_UB_OFFSET, 4);

    if constexpr (isDataFormatNZ) {
        constexpr uint32_t stepSize = GetDataBlockSizeInBytes() / sizeof(T2);
        uint32_t dataBlock = softmaxShapeInfo.srcM * SOFTMAX_SHAPE_NZ_BASIC_COUNT;
        uint16_t mRepeatTimes = static_cast<uint16_t>(CeilDivision(dataBlock, stride));
        uint16_t kRepeatTimes = static_cast<uint16_t>(softmaxShapeInfo.srcK / SOFTMAX_SHAPE_NZ_BASIC_COUNT);
        VF_CALL<AdjustSoftMaxResNZImpl<T1, T2, stepSize, stride>>(
            resUb, maxUb, maskBuf, from, to, dataBlock, mRepeatTimes, kRepeatTimes);
    } else {
        uint32_t srcK = softmaxShapeInfo.srcK;
        uint16_t srcM = softmaxShapeInfo.srcM;
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(softmaxShapeInfo.srcK, stride));
        if constexpr (stepSizeMode != 0) {
            constexpr uint32_t stepSize = 1;
            VF_CALL<AdjustSoftMaxResNDImpl<T1, T2, stepSize, stride>>(
                resUb, maxUb, maskBuf, from, to, srcK, srcM, repeatTimes);
        } else {
            constexpr uint32_t stepSize = GetDataBlockSizeInBytes() / sizeof(T2);
            VF_CALL<AdjustSoftMaxResNDImpl<T1, T2, stepSize, stride>>(
                resUb, maxUb, maskBuf, from, to, srcK, srcM, repeatTimes);
        }
    }
    auto eventID = GetTPipePtr()->FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(eventID);
    WaitFlag<HardEvent::V_S>(eventID);
    bool isUpdateNeedCheck = *((__local_mem__ uint8_t*)maskBuf);
    AscendCUtils::FreeTemporaryBuffer<uint64_t>(maskBuf);
    return isUpdateNeedCheck;
#else
    SetMaskNorm();
    ResetMask();
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
    bool isUpdateNeedCheck = false;
    if constexpr (isDataFormatNZ) {
        isUpdateNeedCheck =
            AdjustSoftMaxResNZImpl<T1, T2, stepSizeMode>(softMaxRes, maxTensor, from, to, softmaxShapeInfo);
    } else {
        isUpdateNeedCheck =
            AdjustSoftMaxResNDImpl<T1, T2, stepSizeMode>(softMaxRes, maxTensor, from, to, softmaxShapeInfo);
    }
    return isUpdateNeedCheck;
#endif
}
} // namespace AscendC

#endif // AICORE_ADV_API_DETAIL_ACTIVATION_SOFTMAX_SOFTMAX_BASE_IMPL_H