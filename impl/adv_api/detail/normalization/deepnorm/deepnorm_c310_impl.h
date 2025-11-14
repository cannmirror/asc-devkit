/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file deepnorm_common_impl.h
 * \brief
 */
#ifndef IMPL_NORMALIZATION_DEEPNORM_DEEPNORM_C310_IMPL_H
#define IMPL_NORMALIZATION_DEEPNORM_DEEPNORM_C310_IMPL_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "kernel_pop_stack_buffer.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../../api_check/kernel_api_check.h"

namespace AscendC {
namespace DeepNormAPI {

constexpr int32_t oneRepSize = GetVecLen() / sizeof(float);

namespace Internal {
struct DeepnormPara {
    uint32_t hRepeatTimes;
    uint32_t hTailSize;
    uint32_t hRepeatCtrl;
    uint32_t hTailCtrl;
    uint32_t hTailOffset;
    uint32_t hDim;
};

__aicore__ inline void GetDeepnormPara(DeepnormPara& para, DeepNormTiling& tiling)
{
    para.hRepeatTimes = tiling.hLength / static_cast<uint32_t>(oneRepSize);
    para.hTailSize = tiling.hLength % oneRepSize;
    para.hDim = tiling.hLength;
    para.hRepeatCtrl = 1;
    para.hTailCtrl = 1;
    para.hTailOffset = para.hRepeatTimes * oneRepSize;
    if (para.hRepeatTimes == 0) {
        para.hRepeatCtrl = 0;
    }
    if (para.hTailSize == 0) {
        para.hTailCtrl = 0;
    }
}
}

template <typename T>
__simd_callee__ inline void CopyInFloat(MicroAPI::RegTensor<float>& reg, __local_mem__ T* ub,
    MicroAPI::MaskReg& hFloatAllMask)
{
    MicroAPI::DataCopy(reg, ub);
}

template <typename T>
__simd_callee__ inline void LoadDataWithT(
    __local_mem__ T* src, MicroAPI::RegTensor<float>& dstReg, MicroAPI::MaskReg& mask, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<T> srcOrigin;
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcOrigin, src + offset);
        Cast<float, T, LayoutZMrgZRndRSatNS>(dstReg, srcOrigin, mask);
    } else {
        DataCopy(dstReg, src + offset);
    }
}

template <typename T>
__simd_callee__ inline void SaveDataWithT(
    __local_mem__ T* dst, MicroAPI::RegTensor<float>& srcReg, MicroAPI::MaskReg& mask, uint32_t offset)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<T> regT;
        MicroAPI::Cast<T, float, LayoutZMrgZRndRSatNS>(regT, srcReg, mask);
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(dst + offset, regT, mask);
    } else {
        MicroAPI::DataCopy(dst + offset, srcReg, mask);
    }
}

//tmpLocal = alpha * srcLocal + gxLocal
template <typename T>
__simd_vf__ inline void ComputeSum(__local_mem__ T* srcLocal, __local_mem__ T* gxLocal, __local_mem__ float* tmpLocal,
    const float alpha, uint32_t bLength, uint32_t sLength, uint32_t hLength, uint32_t oriHLength, uint32_t tailCount)
{
    uint16_t mainRepeatTime = static_cast<uint16_t>(oriHLength / oneRepSize);
    uint16_t tailRepeatTime = static_cast<uint16_t>(CeilDivision(tailCount, oneRepSize));
    MicroAPI::RegTensor<float> srcReg;
    MicroAPI::RegTensor<float> tmpReg;
    MicroAPI::RegTensor<float> gxReg;
    MicroAPI::RegTensor<float> dstReg;
    MicroAPI::RegTensor<float> dstTailReg;
    MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::MaskReg maskOne = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
    MicroAPI::MaskReg maskReg = MicroAPI::UpdateMask<float>(tailCount);
    MicroAPI::Duplicate(tmpReg, alpha, maskFull);
    for (uint16_t bIdx = 0; bIdx < bLength; bIdx++) {
        for (uint16_t sIdx = 0; sIdx < sLength; sIdx++) {
            for (uint16_t i = 0; i < mainRepeatTime; i++) {
                LoadDataWithT(srcLocal, srcReg, maskFull, (bIdx * sLength + sIdx) * hLength + i * oneRepSize);
                LoadDataWithT(gxLocal, gxReg, maskFull, (bIdx * sLength + sIdx) * hLength + i * oneRepSize);
                // step 1: alpha * x
                MicroAPI::Mul(srcReg, srcReg, tmpReg, maskFull);
                // step 2: x + gx
                MicroAPI::Add(dstReg, gxReg, srcReg, maskFull);
                SaveDataWithT(tmpLocal, dstReg, maskFull, (bIdx * sLength + sIdx) * hLength + i * oneRepSize);
            }
            for (uint16_t i = 0; i < tailRepeatTime; i++) {
                LoadDataWithT(srcLocal, srcReg, maskReg, (bIdx * sLength + sIdx) * hLength + mainRepeatTime * oneRepSize);
                LoadDataWithT(gxLocal, gxReg, maskReg, (bIdx * sLength + sIdx) * hLength + mainRepeatTime * oneRepSize);
                // step 1: alpha * x
                MicroAPI::Mul(srcReg, srcReg, tmpReg, maskReg);
                // step 2: x + gx
                MicroAPI::Add(dstTailReg, gxReg, srcReg, maskReg);
                MicroAPI::Select(dstReg, dstTailReg, dstReg, maskReg);
                SaveDataWithT(tmpLocal, dstReg, maskReg, (bIdx * sLength + sIdx) * hLength + mainRepeatTime * oneRepSize);
            }
        }
    }
}

__simd_callee__ inline void CalcHMean(MicroAPI::RegTensor<float>& outputMean, __local_mem__ float* inputX,
    Internal::DeepnormPara para)
{
    MicroAPI::RegTensor<float> hDim;
    MicroAPI::Duplicate(hDim, para.hDim);
    MicroAPI::RegTensor<float> sumResultH;
    MicroAPI::Duplicate(sumResultH, 0);
    MicroAPI::MaskReg hFloatAllMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    uint32_t hTailSizeForMask = static_cast<uint32_t>(para.hTailSize);
    MicroAPI::MaskReg hTailFloatMask = MicroAPI::UpdateMask<float>(hTailSizeForMask);
    for (uint32_t repeat = 0; repeat < para.hRepeatCtrl; ++repeat) {
        // Copy first block to sumResultH.
        CopyInFloat(sumResultH, inputX, hFloatAllMask);
        // Calc x/H in first block
        MicroAPI::Div(sumResultH, sumResultH, hDim, hFloatAllMask);
        for (uint32_t i = 1; i < para.hRepeatTimes; ++i) {
            MicroAPI::RegTensor<float> inputMeanTempReg;
            // Copy new block to inputMeanTempReg.
            CopyInFloat(inputMeanTempReg, inputX + i * oneRepSize, hFloatAllMask);
            // Calc x/H in new block
            MicroAPI::Div(inputMeanTempReg, inputMeanTempReg, hDim, hFloatAllMask);
            // Accumulate new data onto sumResultH
            MicroAPI::Add(sumResultH, sumResultH, inputMeanTempReg, hFloatAllMask);
        }
    }
    for (uint32_t tail = 0; tail < para.hTailCtrl; ++tail) {
        MicroAPI::RegTensor<float> inputMeanTempReg;
        // Copy tail block to inputMeanTempReg.
        CopyInFloat(inputMeanTempReg, inputX + para.hTailOffset, hTailFloatMask);
        // Calc x/H in tail block
        MicroAPI::Div(inputMeanTempReg, inputMeanTempReg, hDim, hTailFloatMask);
        // Accumulate tail data onto sumResultH
        MicroAPI::Add(sumResultH, sumResultH, inputMeanTempReg, hFloatAllMask);
    }
    MicroAPI::ReduceSum(outputMean, sumResultH, hFloatAllMask);
}

__simd_callee__ inline void CalcHVariance(MicroAPI::RegTensor<float>& outputVariance, MicroAPI::RegTensor<float>& meanReg,
    __local_mem__ float* inputX, Internal::DeepnormPara para)
{
    MicroAPI::RegTensor<float> sumVarianceResultH;
    MicroAPI::Duplicate(sumVarianceResultH, 0);
    MicroAPI::RegTensor<float> hDim;
    MicroAPI::Duplicate(hDim, para.hDim);
    uint32_t hTailSizeForMask = static_cast<uint32_t>(para.hTailSize);
    MicroAPI::MaskReg hTailFloatMask = MicroAPI::UpdateMask<float>(hTailSizeForMask);
    MicroAPI::MaskReg hFloatAllMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    for (uint32_t repeat = 0; repeat < para.hRepeatCtrl; ++repeat) {
        // Copy first block to sumVarianceResultH.
        CopyInFloat(sumVarianceResultH, inputX, hFloatAllMask);
        // Calc x - mean in first block
        MicroAPI::Sub(sumVarianceResultH, sumVarianceResultH, meanReg, hFloatAllMask);
        // Calc (x - mean)^2 in first block
        MicroAPI::Mul(sumVarianceResultH, sumVarianceResultH, sumVarianceResultH, hFloatAllMask);
        // Calc (x - mean)^2 / H in first block
        MicroAPI::Div(sumVarianceResultH, sumVarianceResultH, hDim, hFloatAllMask);
        for (uint32_t i = 1; i < para.hRepeatTimes; ++i) {
            MicroAPI::RegTensor<float> inputVarianceReg;
            // Copy new block to inputVarianceReg.
            CopyInFloat(inputVarianceReg, inputX + i * oneRepSize, hFloatAllMask);
            // Calc x - mean in new block
            MicroAPI::Sub(inputVarianceReg, inputVarianceReg, meanReg, hFloatAllMask);
            // Calc (x - mean)^2 in new block
            MicroAPI::Mul(inputVarianceReg, inputVarianceReg, inputVarianceReg, hFloatAllMask);
            // Calc (x - mean)^2 / H in new block
            MicroAPI::Div(inputVarianceReg, inputVarianceReg, hDim, hFloatAllMask);
            // Accumulate new data onto sumVarianceResultH
            MicroAPI::Add(sumVarianceResultH, sumVarianceResultH, inputVarianceReg, hFloatAllMask);
        }
    }
    for (uint32_t tail = 0; tail < para.hTailCtrl; ++tail) {
        MicroAPI::RegTensor<float> inputVarianceReg;
        // Copy tail block to inputVarianceReg.
        CopyInFloat(inputVarianceReg, inputX + para.hTailOffset, hTailFloatMask);
        // Calc x - mean in tail block
        MicroAPI::Sub(inputVarianceReg, inputVarianceReg, meanReg, hTailFloatMask);
        // Calc (x - mean)^2 in tail block
        MicroAPI::Mul(inputVarianceReg, inputVarianceReg, inputVarianceReg, hTailFloatMask);
        // Calc (x - mean)^2 / H in tail block
        MicroAPI::Div(inputVarianceReg, inputVarianceReg, hDim, hTailFloatMask);
        // Accumulate new data onto sumVarianceResultH
        MicroAPI::Add(sumVarianceResultH, sumVarianceResultH, inputVarianceReg, hFloatAllMask);
    }
    MicroAPI::ReduceSum(outputVariance, sumVarianceResultH, hFloatAllMask);
}

template <typename T>
__simd_callee__ inline void CalcHSingleBlockOutPut(__local_mem__ T* output, MicroAPI::RegTensor<float>& meanReg,
    MicroAPI::RegTensor<float>& varianceReg, __local_mem__ float* inputX, __local_mem__ T* gamma, __local_mem__ T* beta,
    MicroAPI::RegTensor<float>& sdReg, MicroAPI::MaskReg& hFloatMask)
{
    MicroAPI::RegTensor<float> resultH;
    if constexpr (SupportType<T, half>()) {
        MicroAPI::DataCopy(resultH, inputX);
        // Calc x - mean in first block.
        MicroAPI::Sub(resultH, resultH, meanReg, hFloatMask);
        // Calc (x - mean) / sdReg in first block.
        MicroAPI::Div(resultH, resultH, sdReg, hFloatMask);
        MicroAPI::RegTensor<T> oriGamaH;
        MicroAPI::RegTensor<float> gammaReg;
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_US_B16>(oriGamaH, gamma);
        MicroAPI::Cast<float, T, layoutZMrgZ>(gammaReg, oriGamaH, hFloatMask);
        // Calc (x - mean) / sdReg * gamma in first block.
        MicroAPI::Mul(resultH, resultH, gammaReg, hFloatMask);
        MicroAPI::RegTensor<T> oriBataH;
        MicroAPI::RegTensor<float> betaReg;
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_US_B16>(oriBataH, beta);
        MicroAPI::Cast<float, T, layoutZMrgZ>(betaReg, oriBataH, hFloatMask);
        // Calc (x - mean) * sdReg * gamma + in first block.
        MicroAPI::Add(resultH, resultH, betaReg, hFloatMask);
        MicroAPI::RegTensor<T> oriOutputH;
        MicroAPI::Cast<T, float, LayoutZMrgZRndRSatNS>(oriOutputH, resultH, hFloatMask);
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(output, oriOutputH, hFloatMask);
    } else {
        MicroAPI::DataCopy(resultH, inputX);
        // Calc x - mean in first block.
        MicroAPI::Sub(resultH, resultH, meanReg, hFloatMask);
        // Calc (x - mean) / sdReg in first block.
        MicroAPI::Div(resultH, resultH, sdReg, hFloatMask);
        MicroAPI::RegTensor<float> gammaReg;
        MicroAPI::DataCopy(gammaReg, gamma);
        // Calc (x - mean) * sdReg * gamma in first block.
        MicroAPI::Mul(resultH, resultH, gammaReg, hFloatMask);
        MicroAPI::RegTensor<float> betaReg;
        MicroAPI::DataCopy(betaReg, beta);
        // Calc (x - mean) * sdReg * gamma + in first block.
        MicroAPI::Add(resultH, resultH, betaReg, hFloatMask);
        MicroAPI::DataCopy(output, resultH, hFloatMask);
    }
}

template <typename T>
__simd_callee__ inline void CalcHOutPut(__local_mem__ T* output, MicroAPI::RegTensor<float>& meanReg,
    MicroAPI::RegTensor<float>& varianceReg, __local_mem__ float* inputX, __local_mem__ T* gamma,
    __local_mem__ T* beta, const T epsilon, Internal::DeepnormPara para)
{
    MicroAPI::MaskReg hFloatAllMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<float> sdReg;    // The standard deviation.
    // Calc variance + epsilon.
    MicroAPI::Adds(sdReg, varianceReg, epsilon, hFloatAllMask);
    // Calc (variance + epsilon)^(1/2).
    MicroAPI::Sqrt(sdReg, sdReg, hFloatAllMask);
    for (uint32_t i = 0; i < para.hRepeatTimes; ++i) {
        CalcHSingleBlockOutPut(output + i * oneRepSize, meanReg, varianceReg,
            inputX + i * oneRepSize, gamma + i * oneRepSize, beta + i * oneRepSize,
            sdReg, hFloatAllMask);
    }
    for (uint32_t tail = 0; tail < para.hTailCtrl; ++tail) {
        uint32_t hTailSizeForMask = static_cast<uint32_t>(para.hTailSize);
        MicroAPI::MaskReg hTailFloatMask = MicroAPI::UpdateMask<float>(hTailSizeForMask);
        CalcHSingleBlockOutPut(output + para.hTailOffset, meanReg, varianceReg, inputX + para.hTailOffset,
            gamma + para.hTailOffset, beta + para.hTailOffset, sdReg, hTailFloatMask);
    }
}

template <typename T>
__simd_vf__ inline void DeepNormImplVfHalf(__local_mem__ T* output, __local_mem__ T* outputMean,
    __local_mem__ T* outputVariance, __local_mem__ float* inputX, __local_mem__ T* gamma,
    __local_mem__ T* beta, const T epsilon, Internal::DeepnormPara para, DeepNormTiling tiling)
{
    MicroAPI::MaskReg floatLowestMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
    MicroAPI::MaskReg srcLowestMask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL1>();
    MicroAPI::MaskReg hFloatAllMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    uint32_t bLength = tiling.bLength;
    uint32_t sLength = tiling.sLength;
    uint32_t hLength = tiling.hLength;
    uint32_t bTotalLength = sLength * hLength;
    for (uint32_t bIdx = 0; bIdx < bLength; ++bIdx) {
        for (uint32_t sIdx = 0; sIdx < sLength; ++sIdx) {
            MicroAPI::RegTensor<float> outputMeanReg;
            CalcHMean(outputMeanReg, inputX + bIdx * bTotalLength + sIdx * hLength, para);
            MicroAPI::RegTensor<float> meanRegForNextCalc;
            MicroAPI::Duplicate(meanRegForNextCalc, outputMeanReg, hFloatAllMask);
            if constexpr (SupportType<T, half>()) {
                MicroAPI::RegTensor<T> oriTypeOutputMean;
                MicroAPI::Cast<T, float, LayoutZMrgZRndRSatNS>(oriTypeOutputMean, outputMeanReg, floatLowestMask);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B16>(
                    outputMean + bIdx * sLength + sIdx, oriTypeOutputMean, srcLowestMask);
            } else {
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    outputMean + bIdx * sLength + sIdx, outputMeanReg, floatLowestMask);
            }
            MicroAPI::RegTensor<float> outputVarianceReg;
            CalcHVariance(outputVarianceReg, meanRegForNextCalc, inputX + bIdx * bTotalLength + sIdx * hLength, para);
            MicroAPI::RegTensor<float> varianceRegNextCalc;
            MicroAPI::Duplicate(varianceRegNextCalc, outputVarianceReg, hFloatAllMask);
            if constexpr (SupportType<T, half>()) {
                MicroAPI::RegTensor<T> oriTypeOutputVariance;
                MicroAPI::Cast<T, float, LayoutZMrgZRndRSatNS>(oriTypeOutputVariance, outputVarianceReg, floatLowestMask);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B16>(
                    outputVariance + bIdx * sLength + sIdx, oriTypeOutputVariance, srcLowestMask);
            } else {
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    outputVariance + bIdx * sLength + sIdx, outputVarianceReg, floatLowestMask);
            }
            CalcHOutPut(output + bIdx * bTotalLength + sIdx * hLength, meanRegForNextCalc, varianceRegNextCalc,
                inputX + bIdx * bTotalLength + sIdx * hLength, gamma, beta, epsilon, para);
        }
    }
}

template <typename T>
__aicore__ inline void DeepNormImplVf(__local_mem__ T* srcLocal, __local_mem__ T* gxLocal,
    __local_mem__ float* tmpLocal, const T alpha, const DeepNormTiling& tiling)
{
    uint32_t bLength = tiling.bLength;
    uint32_t sLength = tiling.sLength;
    uint32_t hLength = tiling.hLength;
    uint32_t oriHLength = tiling.originalHLength;
    uint32_t tailCount = oriHLength % oneRepSize;
    if constexpr (IsSameType<T, half>::value) {
        float alp = static_cast<float>(alpha);
        ComputeSum<T>(srcLocal, gxLocal, tmpLocal, alp, bLength, sLength, hLength, oriHLength, tailCount);
    } else {
        ComputeSum<T>(srcLocal, gxLocal, tmpLocal, alpha, bLength, sLength, hLength, oriHLength, tailCount);
    }
}

template <typename T, bool isBasicBlock = false>
__aicore__ inline bool IsDeepNormParamValid(DeepNormTiling& tiling)
{
    ASCENDC_ASSERT((IsSameType<T, half>::value || IsSameType<T, float>::value),
        {KERNEL_LOG(KERNEL_ERROR, "DeepNorm only support data type: float/half");});
    ASCENDC_ASSERT(tiling.oneTmpSize > 0,
        {KERNEL_LOG(KERNEL_ERROR, "In DeepNorm, Reduce axis is too long to put it in Pop Stack Buffer!");});
    const bool hDivBy64 = (tiling.hLength % 64 == 0) &&
        (tiling.originalHLength % 64 == 0);
    const bool bsDivBy8 = ((tiling.bLength * tiling.sLength) % 8 == 0);
    if constexpr (isBasicBlock) {
        ASCENDC_ASSERT(hDivBy64 && bsDivBy8,
            {KERNEL_LOG(KERNEL_ERROR, "In DeepNorm, when isBasicBlock is true, input must have hLength %% 64 = 0, " \
                "originalHLength %% 64 = 0 and (bLength * sLength) %% 8 = 0 !");});
    }
    return true;
}

template <typename T>
__aicore__ inline void DeepNormImplHalf(__local_mem__ T* output, __local_mem__ T* outputMean,
    __local_mem__ T* outputVariance, __local_mem__ float* inputX, __local_mem__ T* gamma, 
    __local_mem__ T* beta, const T epsilon, Internal::DeepnormPara& para, DeepNormTiling& tiling)
{
    DeepNormImplVfHalf<T>(output, outputMean, outputVariance, inputX, gamma,
        beta, epsilon, para, tiling);
}

template <typename T, bool isReuseSrc, bool isBasicBlock>
__aicore__ inline void DeepNormImpl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& meanLocal,
    const LocalTensor<T>& rstdLocal, const LocalTensor<T>& srcLocal, const LocalTensor<T>& gxLocal,
    const LocalTensor<T>& betaLocal, const LocalTensor<T>& gammaLocal, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const T alpha, const T epsilon, DeepNormTiling& tiling)
{
    static_assert(SupportType<T, half, float>(), "template parameter (T) is not half or float");
    if constexpr (isReuseSrc) {
        static_assert(SupportType<T, float>(), "isReuseSrc is only supported for float on current device!");
    }
    CHECK_FUNC_HIGHLEVEL_API(DeepNorm, (T, isReuseSrc, isBasicBlock), (dstLocal, meanLocal, rstdLocal, srcLocal, gxLocal, betaLocal, gammaLocal,
        sharedTmpBuffer, alpha, epsilon, tiling));
    if (!DeepNormAPI::IsDeepNormParamValid<T, isBasicBlock>(tiling)) {
        return;
    }
    ASCENDC_ASSERT((sharedTmpBuffer.GetSize() > 0), { KERNEL_LOG(KERNEL_ERROR, "sharedTmpBuffer size must > 0!"); });
    if(IsSameType<T, half>::value) {
        Internal::DeepnormPara para;
        Internal::GetDeepnormPara(para, tiling);
        LocalTensor<float> tmpLocal = sharedTmpBuffer.ReinterpretCast<float>();
        LocalTensor<float> ssdLocal = tmpLocal[tiling.firstTmpStartPos];
        DeepNormImplVf<T>((__local_mem__ T*)srcLocal.GetPhyAddr(), (__local_mem__ T*)gxLocal.GetPhyAddr(),
            (__local_mem__ float*)ssdLocal.GetPhyAddr(), alpha, tiling);
        DeepNormImplHalf<T>((__local_mem__ T*)dstLocal.GetPhyAddr(), (__local_mem__ T*)meanLocal.GetPhyAddr(), (__local_mem__ T*)rstdLocal.GetPhyAddr(), 
            (__local_mem__ float*)ssdLocal.GetPhyAddr(), (__local_mem__ T*)gammaLocal.GetPhyAddr(), (__local_mem__ T*)betaLocal.GetPhyAddr(), epsilon, para, tiling);
    } else {
        DeepNormImplVf<T>((__local_mem__ T*)srcLocal.GetPhyAddr(), (__local_mem__ T*)gxLocal.GetPhyAddr(),
            (__local_mem__ float*)dstLocal.GetPhyAddr(), alpha, tiling);
        Internal::DeepnormPara para;
        Internal::GetDeepnormPara(para, tiling);
        DeepNormImplHalf<T>((__local_mem__ T*)dstLocal.GetPhyAddr(), (__local_mem__ T*)meanLocal.GetPhyAddr(), (__local_mem__ T*)rstdLocal.GetPhyAddr(), 
            (__local_mem__ float*)dstLocal.GetPhyAddr(), (__local_mem__ T*)gammaLocal.GetPhyAddr(), (__local_mem__ T*)betaLocal.GetPhyAddr(), epsilon, para, tiling);   
    }
}

template <typename T, bool isReuseSrc, bool isBasicBlock>
__aicore__ inline void DeepNormImpl(const LocalTensor<T>& dstLocal, const LocalTensor<T>& meanLocal,
    const LocalTensor<T>& rstdLocal, const LocalTensor<T>& srcLocal, const LocalTensor<T>& gxLocal,
    const LocalTensor<T>& betaLocal, const LocalTensor<T>& gammaLocal, const T alpha, const T epsilon,
    DeepNormTiling& tiling)
{
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    DeepNormImpl<T, isReuseSrc, isBasicBlock>(dstLocal, meanLocal, rstdLocal, srcLocal, gxLocal, betaLocal,
        gammaLocal, sharedTmpBuffer, alpha, epsilon, tiling);
}

} // namespace DeepNormAPI
} // namespace AscendC
#endif // IMPL_NORMALIZATION_DEEPNORM_DEEPNORM_COMMON_IMPL_H
