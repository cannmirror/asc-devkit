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
 * \file layernorm_variance_impl.h
 * \brief
 */
#ifndef IMPL_NORMALIZATION_LAYERNORM_LAYERNORM_VARIANCE_IMPL_H
#define IMPL_NORMALIZATION_LAYERNORM_LAYERNORM_VARIANCE_IMPL_H

#include "kernel_tensor.h"
#include "kernel_tiling/kernel_tiling.h"

namespace AscendC {
namespace Internal {
constexpr int32_t oneRegSize = GetVecLen() / sizeof(float);
constexpr MicroAPI::CastTrait float2HalfCastTrait = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

struct LayerNormInternalPara {
    uint32_t hRepeatTimes;
    uint32_t hTailSize;
    uint32_t hRepeatCtrl;
    uint32_t hTailCtrl;
    uint32_t hTailOffset;
    uint32_t hDim;
};

template <typename T>
__aicore__ inline void GetLayerNormInternalPara(LayerNormInternalPara& para, const LayerNormTiling& tiling)
{
    para.hRepeatTimes = tiling.hLength / static_cast<uint32_t>(oneRegSize);
    para.hTailSize = tiling.hLength % oneRegSize;
    para.hDim = tiling.hLength;
    para.hRepeatCtrl = 1;
    para.hTailCtrl = 1;
    para.hTailOffset = para.hRepeatTimes * oneRegSize;
    if (para.hRepeatTimes == 0) {
        para.hRepeatCtrl = 0;
    }
    if (para.hTailSize == 0) {
        para.hTailCtrl = 0;
    }
}

template <typename T>
__simd_callee__ inline void CopyInFloatData(MicroAPI::RegTensor<float>& reg, __local_mem__ T* ub,
    MicroAPI::MaskReg& hFloatAllMask)
{
    if constexpr (SupportType<T, half>()) {
        MicroAPI::RegTensor<T> oriInputH;
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_US_B16>(oriInputH, ub);
        MicroAPI::Cast<float, T, layoutZMrgZ>(reg, oriInputH, hFloatAllMask);
    } else {
        MicroAPI::DataCopy(reg, ub);
    }
}
}

template <typename T>
__simd_callee__ inline void CalcHMean(MicroAPI::RegTensor<float>& outputMean, __local_mem__ T* inputX,
    Internal::LayerNormInternalPara& para)
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
        Internal::CopyInFloatData(sumResultH, inputX, hFloatAllMask);
        // Calc x/H in first block
        MicroAPI::Div(sumResultH, sumResultH, hDim, hFloatAllMask);

        for (uint32_t i = 1; i < para.hRepeatTimes; ++i) {
            MicroAPI::RegTensor<float> inputMeanTempReg;
            // Copy new block to inputMeanTempReg.
            Internal::CopyInFloatData(inputMeanTempReg, inputX + i * Internal::oneRegSize, hFloatAllMask);
            // Calc x/H in new block
            MicroAPI::Div(inputMeanTempReg, inputMeanTempReg, hDim, hFloatAllMask);
            // Accumulate new data onto sumResultH
            MicroAPI::Add(sumResultH, sumResultH, inputMeanTempReg, hFloatAllMask);
        }
    }

    for (uint32_t tail = 0; tail < para.hTailCtrl; ++tail) {
        MicroAPI::RegTensor<float> inputMeanTempReg;
        // Copy tail block to inputMeanTempReg.
        Internal::CopyInFloatData(inputMeanTempReg, inputX + para.hTailOffset, hTailFloatMask);

        // Calc x/H in tail block
        MicroAPI::Div(inputMeanTempReg, inputMeanTempReg, hDim, hTailFloatMask);
        // Accumulate tail data onto sumResultH
        MicroAPI::Add(sumResultH, sumResultH, inputMeanTempReg, hFloatAllMask);
    }
    MicroAPI::ReduceSum(outputMean, sumResultH, hFloatAllMask);
}

template <typename T>
__simd_callee__ inline void CalcHVariance(MicroAPI::RegTensor<float>& outputVariance, MicroAPI::RegTensor<float>& meanReg,
    __local_mem__ T* inputX, Internal::LayerNormInternalPara& para)
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
        Internal::CopyInFloatData(sumVarianceResultH, inputX, hFloatAllMask);

        // Calc x - mean in first block
        MicroAPI::Sub(sumVarianceResultH, sumVarianceResultH, meanReg, hFloatAllMask);
        // Calc (x - mean)^2 in first block
        MicroAPI::Mul(sumVarianceResultH, sumVarianceResultH, sumVarianceResultH, hFloatAllMask);
        // Calc (x - mean)^2 / H in first block
        MicroAPI::Div(sumVarianceResultH, sumVarianceResultH, hDim, hFloatAllMask);

        for (uint32_t i = 1; i < para.hRepeatTimes; ++i) {
            MicroAPI::RegTensor<float> inputVarianceReg;
            // Copy new block to inputVarianceReg.
            Internal::CopyInFloatData(inputVarianceReg, inputX + i * Internal::oneRegSize, hFloatAllMask);

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
        Internal::CopyInFloatData(inputVarianceReg, inputX + para.hTailOffset, hTailFloatMask);

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
    MicroAPI::RegTensor<float>& varianceReg, __local_mem__ T* inputX, __local_mem__ T* gamma, __local_mem__ T* beta,
    MicroAPI::RegTensor<float>& sdReg, MicroAPI::MaskReg& hFloatMask)
{
    MicroAPI::RegTensor<float> resultH;

    if constexpr (SupportType<T, half>()) {
        MicroAPI::RegTensor<T> oriInputH;
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_US_B16>(oriInputH, inputX);
        MicroAPI::Cast<float, T, layoutZMrgZ>(resultH, oriInputH, hFloatMask);

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
        MicroAPI::Cast<T, float, Internal::float2HalfCastTrait>(oriOutputH, resultH, hFloatMask);
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
    MicroAPI::RegTensor<float>& varianceReg, __local_mem__ T* inputX, __local_mem__ T* gamma,
    __local_mem__ T* beta, const T epsilon, Internal::LayerNormInternalPara& para)
{
    MicroAPI::MaskReg hFloatAllMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::RegTensor<float> sdReg;    // The standard deviation.

    // Calc variance + epsilon.
    MicroAPI::Adds(sdReg, varianceReg, epsilon, hFloatAllMask);
    // Calc (variance + epsilon)^(1/2).
    MicroAPI::Sqrt(sdReg, sdReg, hFloatAllMask);

    for (uint32_t i = 0; i < para.hRepeatTimes; ++i) {
        CalcHSingleBlockOutPut(output + i * Internal::oneRegSize, meanReg, varianceReg,
            inputX + i * Internal::oneRegSize, gamma + i * Internal::oneRegSize, beta + i * Internal::oneRegSize,
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
__simd_vf__ inline void LayerNormImplVf(__local_mem__ T* output, __local_mem__ T* outputMean,
    __local_mem__ T* outputVariance, __local_mem__ T* inputX, __local_mem__ T* gamma,
    __local_mem__ T* beta, const T epsilon, Internal::LayerNormInternalPara para, LayerNormTiling tiling)
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
                MicroAPI::Cast<T, float, Internal::float2HalfCastTrait>(oriTypeOutputMean, outputMeanReg, floatLowestMask);
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
                MicroAPI::Cast<T, float, Internal::float2HalfCastTrait>(oriTypeOutputVariance, outputVarianceReg, floatLowestMask);
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

template <typename T, bool isReuseSource = false>
__aicore__ inline void LayerNormImpl(const LocalTensor<T>& output, const LocalTensor<T>& outputMean,
    const LocalTensor<T>& outputVariance, const LocalTensor<T>& inputX, const LocalTensor<T>& gamma,
    const LocalTensor<T>& beta, const LocalTensor<uint8_t>& sharedTmpBuffer, const T epsilon, LayerNormTiling& tiling)
{
    static_assert(SupportType<T, half, float>(), "current data type is not supported on current device!");
    CHECK_FUNC_HIGHLEVEL_API(LayerNorm, (T), (output, outputMean, outputVariance, inputX, gamma, beta, sharedTmpBuffer, epsilon, tiling));
    Internal::LayerNormInternalPara para{};
    Internal::GetLayerNormInternalPara<T>(para, tiling);

    LayerNormImplVf<T>((__local_mem__ T*)output.GetPhyAddr(), (__local_mem__ T*)outputMean.GetPhyAddr(),
        (__local_mem__ T*)outputVariance.GetPhyAddr(), (__local_mem__ T *)inputX.GetPhyAddr(),
        (__local_mem__ T*)gamma.GetPhyAddr(), (__local_mem__ T*)beta.GetPhyAddr(), epsilon, para, tiling);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void LayerNormImpl(const LocalTensor<T>& output, const LocalTensor<T>& outputMean,
    const LocalTensor<T>& outputVariance, const LocalTensor<T>& inputX, const LocalTensor<T>& gamma,
    const LocalTensor<T>& beta, const T epsilon, LayerNormTiling& tiling)
{
    static_assert(SupportType<T, half, float>(), "current data type is not supported on current device!");
    const LocalTensor<uint8_t> sharedTmpBuffer; // Not used, no need to alloc memory.
    CHECK_FUNC_HIGHLEVEL_API(LayerNorm, (T), (output, outputMean, outputVariance, inputX, gamma, beta, sharedTmpBuffer, epsilon, tiling));
    Internal::LayerNormInternalPara para{};
    Internal::GetLayerNormInternalPara<T>(para, tiling);

    LayerNormImplVf<T>((__local_mem__ T*)output.GetPhyAddr(), (__local_mem__ T*)outputMean.GetPhyAddr(),
        (__local_mem__ T*)outputVariance.GetPhyAddr(), (__local_mem__ T *)inputX.GetPhyAddr(),
        (__local_mem__ T*)gamma.GetPhyAddr(), (__local_mem__ T*)beta.GetPhyAddr(), epsilon, para, tiling);
}

} // namespace AscendC
#endif // IMPL_NORMALIZATION_LAYERNORM_LAYERNORM_VARIANCE_IMPL_H