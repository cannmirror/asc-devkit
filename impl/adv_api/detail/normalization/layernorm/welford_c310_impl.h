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
 * \file welford_c310_impl.h
 * \brief
 */
#ifndef IMPL_NORMALIZATION_WELFORD_C310_IMPL_H
#define IMPL_NORMALIZATION_WELFORD_C310_IMPL_H

#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"
#include "kernel_tiling/kernel_tiling.h"
#include "include/adv_api/normalization/normalize.h"
#include "layernorm_c310_utils.h"
#include "../../api_check/kernel_api_check.h"

namespace AscendC {

// Unified helper function for repeated WelfordUpdateImplForB16VF/B32VF core logic
template <typename T, bool IsB16>
__aicore__ inline void WelfordUpdateImplForVFCommon(MicroAPI::MaskReg& preg, MicroAPI::RegTensor<float>& meanVreg,
    MicroAPI::RegTensor<float>& varVreg, MicroAPI::RegTensor<float>& tmpVreg,
    typename std::conditional<IsB16, MicroAPI::RegTensor<float>&, MicroAPI::RegTensor<T>&>::type srcVreg,
    MicroAPI::RegTensor<float>& outMeanreg, MicroAPI::RegTensor<float>& outVarreg, MicroAPI::RegTensor<float>& f32vreg,
    __local_mem__ float* const outMean, __local_mem__ float* const outVar, __local_mem__ float* const inMean,
    __local_mem__ float* const inVar, uint32_t offset, float nRec, uint32_t sreg)
{
    MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(meanVreg, inMean + offset);
    MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(varVreg, inVar + offset);
    MicroAPI::Sub(tmpVreg, srcVreg, meanVreg, preg);
    MicroAPI::Muls(outMeanreg, tmpVreg, nRec, preg);
    MicroAPI::Add(outMeanreg, outMeanreg, meanVreg, preg);
    MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(outMean + offset, outMeanreg, preg);

    MicroAPI::Sub(f32vreg, srcVreg, outMeanreg, preg);
    MicroAPI::Mul(f32vreg, tmpVreg, f32vreg, preg);
    MicroAPI::Add(outVarreg, f32vreg, varVreg, preg);
    MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(outVar + offset, outVarreg, preg);
}

// Helper for in-place copy logic in B16/B32
__aicore__ inline void WelfordUpdateImplInplaceCopy(MicroAPI::MaskReg& preg, MicroAPI::RegTensor<float>& meanVreg,
    MicroAPI::RegTensor<float>& varVreg, __local_mem__ float* const outMean, __local_mem__ float* const outVar,
    __local_mem__ float* const inMean, __local_mem__ float* const inVar, uint32_t abLength, uint32_t inPlaceLength,
    uint16_t repeatInplace, uint32_t sregLower, uint32_t dstOffset)
{
    for (uint16_t i = 0; i < 1; ++i) {
        uint32_t sreg = inPlaceLength;
        uint32_t rowOffset = i * abLength;
        for (uint16_t j = 0; j < repeatInplace; ++j) {
            preg = MicroAPI::UpdateMask<uint32_t>(sreg);
            uint32_t srcOffset = rowOffset + j * sregLower;
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(meanVreg, inMean + dstOffset + srcOffset);
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(varVreg, inVar + dstOffset + srcOffset);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(
                outMean + dstOffset + srcOffset, meanVreg, preg);
            MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(
                outVar + dstOffset + srcOffset, varVreg, preg);
        }
    }
}

// VF helper extracted from WelfordUpdateImplForB16
template <typename T, const WelfordUpdateConfig& config = WFUPDATE_DEFAULT_CFG>
__aicore__ inline void WelfordUpdateImplForB16VF(__local_mem__ float* const outMean, __local_mem__ float* const outVar,
    __local_mem__ T* const src, __local_mem__ float* const inMean, __local_mem__ float* const inVar,
    const WelfordUpdateParam& para, const uint16_t sregLowerB32, const uint32_t sregLower, const uint32_t K)
{
    MicroAPI::MaskReg preg;

    MicroAPI::RegTensor<T> b16vreg, vreg1, vreg2;
    MicroAPI::RegTensor<float> f32vreg, tmpVreg, srcVreg, meanVreg, varVreg, outMeanreg, outVarreg;
    MicroAPI::RegTensor<uint16_t> zeroReg;

    if constexpr (config.isInplace) {
        uint32_t inPlaceLength = AlignUp(para.abLength - para.abComputeLength, 8);
        uint16_t repeatInplace = static_cast<uint16_t>(CeilDivision(inPlaceLength, Internal::LAYERNORM_B32_VF_LEN));
        uint32_t dstOffset = para.abLength - inPlaceLength;
        WelfordUpdateImplInplaceCopy(preg, meanVreg, varVreg, outMean, outVar, inMean, inVar, para.abLength,
            inPlaceLength, repeatInplace, sregLower, dstOffset);
    }

    MicroAPI::Duplicate(zeroReg, (uint16_t)0x0000);
    uint16_t repeat = static_cast<uint16_t>(CeilDivision(K, sregLower));
    for (uint16_t i = 0; i < 1; ++i) {
        uint32_t rowOffset = i * para.abLength;
        uint32_t sreg = static_cast<uint32_t>(K);
        for (uint16_t j = 0; j < static_cast<uint16_t>(repeat); ++j) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(b16vreg, src + rowOffset + j * sregLower);
            MicroAPI::Interleave<uint16_t>((MicroAPI::RegTensor<uint16_t>&)vreg1, (MicroAPI::RegTensor<uint16_t>&)vreg2,
                (MicroAPI::RegTensor<uint16_t>&)b16vreg, (MicroAPI::RegTensor<uint16_t>&)zeroReg);

            // First half (64 F32 elements)
            preg = MicroAPI::UpdateMask<uint32_t>(sreg);
            MicroAPI::Cast<float, T, layoutZMrgZ>(srcVreg, vreg1, preg);
            WelfordUpdateImplForVFCommon<T, true>(preg, meanVreg, varVreg, tmpVreg, srcVreg, outMeanreg, outVarreg,
                f32vreg, outMean, outVar, inMean, inVar, rowOffset + (2 * j) * sregLowerB32,
                static_cast<float>(para.nRec), sreg);

            // Second half
            preg = MicroAPI::UpdateMask<uint32_t>(sreg);
            MicroAPI::Cast<float, T, layoutZMrgZ>(srcVreg, vreg2, preg);
            WelfordUpdateImplForVFCommon<T, true>(preg, meanVreg, varVreg, tmpVreg, srcVreg, outMeanreg, outVarreg,
                f32vreg, outMean, outVar, inMean, inVar, rowOffset + (2 * j + 1) * sregLowerB32,
                static_cast<float>(para.nRec), sreg);
        }
    }
}

// VF helper extracted from WelfordUpdateImplForB32
template <typename T, const WelfordUpdateConfig& config = WFUPDATE_DEFAULT_CFG>
__aicore__ inline void WelfordUpdateImplForB32VF(__local_mem__ float* const outMean, __local_mem__ float* const outVar,
    __local_mem__ T* const src, __local_mem__ float* const inMean, __local_mem__ float* const inVar,
    const WelfordUpdateParam& para, const uint32_t sregLower, const uint32_t K)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<T> srcVreg;
    MicroAPI::RegTensor<float> f32vreg;
    MicroAPI::RegTensor<float> tmpVreg;

    MicroAPI::RegTensor<float> meanVreg;
    MicroAPI::RegTensor<float> varVreg;
    MicroAPI::RegTensor<float> outMeanreg;
    MicroAPI::RegTensor<float> outVarreg;

    if constexpr (config.isInplace) {
        uint32_t inPlaceLength = AlignUp(para.abLength - para.abComputeLength, 8);
        uint16_t repeatInplace = static_cast<uint16_t>(CeilDivision(inPlaceLength, Internal::LAYERNORM_B32_VF_LEN));
        uint32_t dstOffset = para.abLength - inPlaceLength;
        WelfordUpdateImplInplaceCopy(preg, meanVreg, varVreg, outMean, outVar, inMean, inVar, para.abLength,
            inPlaceLength, repeatInplace, sregLower, dstOffset);
    }

    uint16_t repeat = static_cast<uint16_t>(CeilDivision(K, sregLower));
    for (uint16_t i = 0; i < 1; ++i) {
        uint32_t rowOffset = i * para.abLength;
        uint32_t sreg = static_cast<uint32_t>(K);
        for (uint16_t j = 0; j < static_cast<uint16_t>(repeat); ++j) {
            preg = MicroAPI::UpdateMask<uint32_t>(sreg);
            uint32_t offset = rowOffset + j * sregLower;
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(srcVreg, src + offset);
            WelfordUpdateImplForVFCommon<T, false>(preg, meanVreg, varVreg, tmpVreg, srcVreg, outMeanreg, outVarreg,
                f32vreg, outMean, outVar, inMean, inVar, offset, static_cast<float>(para.nRec), sreg);
        }
    }
}

template <typename T, const WelfordUpdateConfig& config = WFUPDATE_DEFAULT_CFG>
__aicore__ inline void WelfordUpdateImplForB16(__local_mem__ float* const outMean, __local_mem__ float* const outVar,
    __local_mem__ T* const src, __local_mem__ float* const inMean, __local_mem__ float* const inVar,
    const WelfordUpdateParam& para)
{
    const uint16_t sregLowerB32 = static_cast<uint16_t>(GetVecLen() / sizeof(float)); // 64
    const uint32_t sregLower = static_cast<uint32_t>(Internal::LAYERNORM_B16_VF_LEN);
    const uint32_t K = para.abComputeLength;

    VF_CALL<WelfordUpdateImplForB16VF<T, config>>(
        outMean, outVar, src, inMean, inVar, para, sregLowerB32, sregLower, K);
}

template <typename T, const WelfordUpdateConfig& config = WFUPDATE_DEFAULT_CFG>
__aicore__ inline void WelfordUpdateImplForB32(__local_mem__ float* const outMean, __local_mem__ float* const outVar,
    __local_mem__ T* const src, __local_mem__ float* const inMean, __local_mem__ float* const inVar,
    const WelfordUpdateParam& para)
{
    const uint32_t sregLower = static_cast<uint32_t>(Internal::LAYERNORM_B32_VF_LEN);
    const uint32_t K = para.abComputeLength;

    VF_CALL<WelfordUpdateImplForB32VF<T, config>>(outMean, outVar, src, inMean, inVar, para, sregLower, K);
}

template <typename T, typename U = float, bool isReuseSource = false,
    const WelfordUpdateConfig& config = WFUPDATE_DEFAULT_CFG>
__aicore__ inline void WelfordUpdateImpl(const LocalTensor<U>& outputMean, const LocalTensor<U>& outputVariance,
    const LocalTensor<U>& inputMean, const LocalTensor<U>& inputVariance, const LocalTensor<T>& inputX,
    const WelfordUpdateParam& para)
{
    CHECK_FUNC_HIGHLEVEL_API(WelfordUpdate, (T, U, isReuseSource, config), (outputMean, outputVariance, inputMean, inputVariance, inputX, para));

    static_assert(SupportType<U, float>(), "current data type is not supported on current device!");
    __local_mem__ T* srcUb = (__local_mem__ T*)inputX.GetPhyAddr();
    __local_mem__ float* inMean = (__local_mem__ float*)inputMean.GetPhyAddr();
    __local_mem__ float* inVar = (__local_mem__ float*)inputVariance.GetPhyAddr();
    __local_mem__ float* outMean = (__local_mem__ float*)outputMean.GetPhyAddr();
    __local_mem__ float* outVar = (__local_mem__ float*)outputVariance.GetPhyAddr();

    if constexpr (SupportType<T, half, bfloat16_t>()) {
        WelfordUpdateImplForB16<T, config>(outMean, outVar, srcUb, inMean, inVar, para);
    } else { // fp32
        WelfordUpdateImplForB32<T, config>(outMean, outVar, srcUb, inMean, inVar, para);
    }
}
template <typename T, typename U = float, bool isReuseSource = false,
    const WelfordUpdateConfig& config = WFUPDATE_DEFAULT_CFG>
__aicore__ inline void WelfordUpdateImpl(const LocalTensor<U>& outputMean, const LocalTensor<U>& outputVariance,
    const LocalTensor<U>& inputMean, const LocalTensor<U>& inputVariance, const LocalTensor<T>& inputX,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const WelfordUpdateParam& para)
{
    static_assert(SupportType<T, half, bfloat16_t, float>(), "current data type is not supported on current device!");
    WelfordUpdateImpl<T, float, isReuseSource, config>(
        outputMean, outputVariance, inputMean, inputVariance, inputX, para);
}

} // namespace AscendC
#endif // IMPL_NORMALIZATION_WELFORD_C310_IMPL_H
