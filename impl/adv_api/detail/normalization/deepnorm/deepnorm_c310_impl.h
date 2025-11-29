/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
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
    float hDim;
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
__simd_callee__ inline void CopyInFloat(MicroAPI::RegTensor<float>& reg, __ubuf__ T* ub,
    MicroAPI::MaskReg& hFloatAllMask)
{
    if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<T> oriInputH;
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_US_B16>(oriInputH, ub);
        MicroAPI::Cast<float, T, layoutZMrgZ>(reg, oriInputH, hFloatAllMask);
    } else {
        MicroAPI::DataCopy(reg, ub);
    }
}

template <typename T>
__simd_callee__ inline void CalcHMean(MicroAPI::RegTensor<float>& outputMean, __ubuf__ T* inputX, __ubuf__ T* gxLocal,
    const float alpha, Internal::DeepnormPara para)
{
    MicroAPI::RegTensor<float> hDim, dupAlpha, gxReg;
    MicroAPI::Duplicate(hDim, para.hDim);
    MicroAPI::Duplicate(dupAlpha, alpha);
    MicroAPI::RegTensor<float> sumResultH;
    MicroAPI::Duplicate(sumResultH, 0);
    MicroAPI::MaskReg hFloatAllMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    uint32_t hTailSizeForMask = static_cast<uint32_t>(para.hTailSize);
    MicroAPI::MaskReg hTailFloatMask = MicroAPI::UpdateMask<float>(hTailSizeForMask);
    for (uint32_t repeat = 0; repeat < para.hRepeatCtrl; ++repeat) {
        // Copy first block to sumResultH.
        CopyInFloat(sumResultH, inputX, hFloatAllMask);
        CopyInFloat(gxReg, gxLocal, hFloatAllMask);
        MicroAPI::Mul(sumResultH, sumResultH, dupAlpha, hFloatAllMask);
        MicroAPI::Add(sumResultH, sumResultH, gxReg, hFloatAllMask);
        // Calc x/H in first block
        MicroAPI::Div(sumResultH, sumResultH, hDim, hFloatAllMask);
        for (uint32_t i = 1; i < para.hRepeatTimes; ++i) {
            MicroAPI::RegTensor<float> inputMeanTempReg;
            // Copy new block to inputMeanTempReg.
            CopyInFloat(inputMeanTempReg, inputX + i * oneRepSize, hFloatAllMask);
            CopyInFloat(gxReg, gxLocal + i * oneRepSize, hFloatAllMask);
            MicroAPI::Mul(inputMeanTempReg, inputMeanTempReg, dupAlpha, hFloatAllMask);
            MicroAPI::Add(inputMeanTempReg, inputMeanTempReg, gxReg, hFloatAllMask);
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
        CopyInFloat(gxReg, gxLocal + para.hTailOffset, hTailFloatMask);
        MicroAPI::Mul(inputMeanTempReg, inputMeanTempReg, dupAlpha, hTailFloatMask);
        MicroAPI::Add(inputMeanTempReg, inputMeanTempReg, gxReg, hTailFloatMask);
        // Calc x/H in tail block
        MicroAPI::Div(inputMeanTempReg, inputMeanTempReg, hDim, hTailFloatMask);
        // Accumulate tail data onto sumResultH
        MicroAPI::Add(sumResultH, sumResultH, inputMeanTempReg, hFloatAllMask);
    }
    MicroAPI::ReduceSum(outputMean, sumResultH, hFloatAllMask);
}

template <typename T>
__simd_callee__ inline void CalcHVariance(MicroAPI::RegTensor<float>& outputVariance, MicroAPI::RegTensor<float>& meanReg,
    __ubuf__ T* inputX, __ubuf__ T* gxLocal, const float alpha, Internal::DeepnormPara para)
{
    MicroAPI::RegTensor<float> sumVarianceResultH, dupAlpha, gxReg;
    MicroAPI::Duplicate(sumVarianceResultH, 0);
    MicroAPI::Duplicate(dupAlpha, alpha);
    MicroAPI::RegTensor<float> hDim;
    MicroAPI::Duplicate(hDim, para.hDim);
    uint32_t hTailSizeForMask = static_cast<uint32_t>(para.hTailSize);
    MicroAPI::MaskReg hTailFloatMask = MicroAPI::UpdateMask<float>(hTailSizeForMask);
    MicroAPI::MaskReg hFloatAllMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    for (uint32_t repeat = 0; repeat < para.hRepeatCtrl; ++repeat) {
        // Copy first block to sumVarianceResultH.
        CopyInFloat(sumVarianceResultH, inputX, hFloatAllMask);
        CopyInFloat(gxReg, gxLocal, hFloatAllMask);
        MicroAPI::Mul(sumVarianceResultH, sumVarianceResultH, dupAlpha, hFloatAllMask);
        MicroAPI::Add(sumVarianceResultH, sumVarianceResultH, gxReg, hFloatAllMask);
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
            CopyInFloat(gxReg, gxLocal + i * oneRepSize, hFloatAllMask);
            MicroAPI::Mul(inputVarianceReg, inputVarianceReg, dupAlpha, hFloatAllMask);
            MicroAPI::Add(inputVarianceReg, inputVarianceReg, gxReg, hFloatAllMask);
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
        CopyInFloat(gxReg, gxLocal + para.hTailOffset, hTailFloatMask);
        MicroAPI::Mul(inputVarianceReg, inputVarianceReg, dupAlpha, hTailFloatMask);
        MicroAPI::Add(inputVarianceReg, inputVarianceReg, gxReg, hTailFloatMask);
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
__simd_callee__ inline void CalcHSingleBlockOutPut(__ubuf__ T* output, MicroAPI::RegTensor<float>& meanReg,
    MicroAPI::RegTensor<float>& varianceReg, __ubuf__ T* inputX, __ubuf__ T* gxLocal, const float alpha, __ubuf__ T* gamma, __ubuf__ T* beta,
    MicroAPI::RegTensor<float>& sdReg, MicroAPI::MaskReg& hFloatMask)
{
    MicroAPI::RegTensor<float> resultH, dupAlpha, gxReg;
    MicroAPI::Duplicate(dupAlpha, alpha);
    if constexpr (SupportType<T, half>()) {
        MicroAPI::RegTensor<T> oriInputH;
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_US_B16>(oriInputH, inputX);
        MicroAPI::Cast<float, T, layoutZMrgZ>(resultH, oriInputH, hFloatMask);
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_US_B16>(oriInputH, gxLocal);
        MicroAPI::Cast<float, T, layoutZMrgZ>(gxReg, oriInputH, hFloatMask);
        MicroAPI::Mul(resultH, resultH, dupAlpha, hFloatMask);
        MicroAPI::Add(resultH, resultH, gxReg, hFloatMask);
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
        MicroAPI::DataCopy(gxReg, gxLocal);
        MicroAPI::Mul(resultH, resultH, dupAlpha, hFloatMask);
        MicroAPI::Add(resultH, resultH, gxReg, hFloatMask);
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
__simd_callee__ inline void CalcHOutPut(__ubuf__ T* output, MicroAPI::RegTensor<float>& meanReg,
    MicroAPI::RegTensor<float>& varianceReg, __ubuf__ T* inputX, __ubuf__ T* gxLocal, const float alpha, __ubuf__ T* gamma,
    __ubuf__ T* beta, const T epsilon, Internal::DeepnormPara para)
{
    MicroAPI::MaskReg hFloatAllMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<float> sdReg;    // The standard deviation.
    // Calc variance + epsilon.
    MicroAPI::Adds(sdReg, varianceReg, epsilon, hFloatAllMask);
    // Calc (variance + epsilon)^(1/2).
    MicroAPI::Sqrt(sdReg, sdReg, hFloatAllMask);
    for (uint32_t i = 0; i < para.hRepeatTimes; ++i) {
        CalcHSingleBlockOutPut(output + i * oneRepSize, meanReg, varianceReg,
            inputX + i * oneRepSize, gxLocal + i * oneRepSize, alpha, gamma + i * oneRepSize, beta + i * oneRepSize,
            sdReg, hFloatAllMask);
    }
    for (uint32_t tail = 0; tail < para.hTailCtrl; ++tail) {
        uint32_t hTailSizeForMask = static_cast<uint32_t>(para.hTailSize);
        MicroAPI::MaskReg hTailFloatMask = MicroAPI::UpdateMask<float>(hTailSizeForMask);
        CalcHSingleBlockOutPut(output + para.hTailOffset, meanReg, varianceReg, inputX + para.hTailOffset, gxLocal + para.hTailOffset, alpha,
            gamma + para.hTailOffset, beta + para.hTailOffset, sdReg, hTailFloatMask);
    }
}

template <typename T>
__simd_vf__ inline void DeepNormImplVfHalf(__ubuf__ T* output, __ubuf__ T* outputMean,
    __ubuf__ T* outputVariance, __ubuf__ T* inputX, __ubuf__ T* gxLocal, __ubuf__ T* gamma,
    __ubuf__ T* beta, const T epsilon, Internal::DeepnormPara para, DeepNormTiling tiling, const float alpha)
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
            CalcHMean(outputMeanReg, inputX + bIdx * bTotalLength + sIdx * hLength, gxLocal + bIdx * bTotalLength + sIdx * hLength, alpha, para);
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
            CalcHVariance(outputVarianceReg, meanRegForNextCalc, inputX + bIdx * bTotalLength + sIdx * hLength, gxLocal + bIdx * bTotalLength + sIdx * hLength, alpha, para);
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
                inputX + bIdx * bTotalLength + sIdx * hLength, gxLocal + bIdx * bTotalLength + sIdx * hLength, alpha, gamma, beta, epsilon, para);
        }
    }
}

template <typename T, bool isBasicBlock = false>
__aicore__ inline bool IsDeepNormParamValid(DeepNormTiling& tiling)
{
    ASCENDC_ASSERT((IsSameType<T, half>::value || IsSameType<T, float>::value),
        {KERNEL_LOG(KERNEL_ERROR, "DeepNorm only support data type: float/half");});
    ASCENDC_ASSERT(tiling.oneTmpSize > 0,
        {KERNEL_LOG(KERNEL_ERROR, "In DeepNorm, each tmpsize in sharedTmpBuffer must > 0!");});
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
__aicore__ inline void DeepNormImplHalf(__ubuf__ T* output, __ubuf__ T* outputMean,
    __ubuf__ T* outputVariance, __ubuf__ T* inputX, __ubuf__ T* gxLocal, __ubuf__ T* gamma, 
    __ubuf__ T* beta, const T epsilon, Internal::DeepnormPara& para, DeepNormTiling& tiling, const T alpha)
{
    if(IsSameType<T, float>::value) {
        DeepNormImplVfHalf<T>(output, outputMean, outputVariance, inputX, gxLocal, gamma,
            beta, epsilon, para, tiling, alpha);
    } else {
        float alp = alpha;
        DeepNormImplVfHalf<T>(output, outputMean, outputVariance, inputX, gxLocal, gamma,
            beta, epsilon, para, tiling, alp);
    }
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
    Internal::DeepnormPara para;
    Internal::GetDeepnormPara(para, tiling);
    DeepNormImplHalf<T>((__ubuf__ T*)dstLocal.GetPhyAddr(), (__ubuf__ T*)meanLocal.GetPhyAddr(), (__ubuf__ T*)rstdLocal.GetPhyAddr(), 
        (__ubuf__ T*)srcLocal.GetPhyAddr(), (__ubuf__ T*)gxLocal.GetPhyAddr(), (__ubuf__ T*)gammaLocal.GetPhyAddr(), (__ubuf__ T*)betaLocal.GetPhyAddr(), epsilon, para, tiling, alpha);   
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
