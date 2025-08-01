/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file welfordfinalize_c310_impl.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_NORMALIZATION_WELFORDFINALIZE_WELFORDFINALIZE_C310_IMPL_H
#define AICORE_ADV_API_DETAIL_NORMALIZATION_WELFORDFINALIZE_WELFORDFINALIZE_C310_IMPL_H

#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"
#include "kernel_tiling/kernel_tiling.h"
#include "normalization/welfordfinalize_utils.h"

namespace AscendC {
namespace Internal {
const uint16_t kWelfordFinalizeFoldNum = 2;
constexpr uint32_t WELFORD_B32_VF_LEN = GetVecLen() / sizeof(uint32_t);
} // namespace Internal

template <uint32_t HalfAddTimes>
__aicore__ constexpr inline uint16_t CalculatePower()
{
    constexpr uint16_t kMaxOffset = 16;
    uint16_t fold = 0;
    for (uint16_t i = 1; i < kMaxOffset; i++) {
        if ((HalfAddTimes >> i) == 0) {
            break;
        }
        fold++;
    }
    return fold;
}

// Calculate the sum of two points based on count. The main block size is ex.2000->1024 900->512.
__aicore__ inline uint32_t CalculateMainBlock(uint32_t count)
{
    count |= count >> 1;
    count |= count >> 2;
    count |= count >> 4;
    count |= count >> 8;
    count |= count >> 16;
    return (count + 1) >> 1;
}

// only support rLength <= 64
template <typename T, bool isCorrection = false>
__aicore__ inline void ComputeMean64(__local_mem__ float* meanUb, __local_mem__ float* varianceUb,
    __local_mem__ T* srcUb, const uint32_t aLength, const uint32_t rLength, const uint32_t rLengthWithPadding,
    const float k2Rec, const float k2RRec, const float rRecWithCorrection)
{
    constexpr uint16_t sregLower = static_cast<uint16_t>(VECTOR_REG_WIDTH / sizeof(float)); // 64
    uint32_t count = rLength;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> src0Reg;
        MicroAPI::RegTensor<float> src1Reg;
        MicroAPI::RegTensor<float> dstReg;
        MicroAPI::RegTensor<float> meanReg;
        MicroAPI::RegTensor<float> varianceReg;

        MicroAPI::MaskReg preg = MicroAPI::UpdateMask<float>(count);
        MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregOne = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
        for (uint16_t i = 0; i < static_cast<uint16_t>(aLength); i++) {
            LoadDataWithT<T>(srcUb, src0Reg, preg, i * rLengthWithPadding);
            Muls(src1Reg, src0Reg, k2Rec, preg);
            ReduceSum(dstReg, src1Reg, preg);
            if constexpr (isCorrection) {
                Muls(meanReg, dstReg, rRecWithCorrection, pregOne);
            } else {
                Muls(meanReg, dstReg, k2RRec, pregOne);
            }
            DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((meanUb + i), meanReg, pregOne);
        }
    }
}

template <typename T, bool isCorrection = false>
__aicore__ inline void ComputeMean(__local_mem__ float* meanUb, __local_mem__ float* varianceUb, __local_mem__ T* srcUb,
    __local_mem__ float* workUbOrigin, const uint32_t k, const uint32_t aLength, const uint32_t rLength,
    const uint32_t rLengthWithPadding, const uint32_t rHeadLength, const float k2Rec, const float k2RRec,
    float rRecWithCorrection)
{
    constexpr uint16_t sregLower = static_cast<uint16_t>(VECTOR_REG_WIDTH / sizeof(float)); // 64
    const uint32_t m = rLength - rHeadLength;
    uint32_t count;
    const uint32_t mVL = static_cast<uint32_t>(CeilDivision(m, sregLower) * sregLower);
    const uint32_t mainTailCount = rHeadLength - mVL;
    uint32_t workCount = static_cast<uint32_t>(
        CeilDivision(rHeadLength / sregLower * sizeof(float), sregLower * Internal::kWelfordFinalizeFoldNum)
        * (sregLower * Internal::kWelfordFinalizeFoldNum)); // 256 * 4 = 1024; 1024/128 * 128
    int16_t halfAddRepeatTimes = static_cast<int16_t>(k - CalculatePower<sregLower * sregLower>());
    if (halfAddRepeatTimes < 0) {
        halfAddRepeatTimes = 0;
    }
    const uint16_t halfAddCount = static_cast<uint16_t>(CeilDivision(rHeadLength, sregLower)); // total count
    const uint16_t halfAddTimes = static_cast<uint16_t>(CeilDivision(halfAddCount, sregLower));
    int16_t lastCount = static_cast<int16_t>(halfAddCount); // last vcadd times less than 64
    if (lastCount > sregLower) {
        lastCount = sregLower;
    }
    uint16_t repeatTimes1 = static_cast<uint16_t>(CeilDivision(m, sregLower));
    uint16_t repeatTimes2 = static_cast<uint16_t>(CeilDivision(mainTailCount, sregLower));

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> src0Reg;
        MicroAPI::RegTensor<float> src1Reg;
        MicroAPI::RegTensor<float> dstReg;
        MicroAPI::RegTensor<float> meanReg;

        MicroAPI::MaskReg preg;
        MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregOne = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();

        for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
            // tail block add to main block
            count = m;
            for (uint16_t i = 0; i < repeatTimes1; i++) {
                preg = MicroAPI::UpdateMask<float>(count);
                LoadDataWithT<T>(srcUb, srcUb, src0Reg, src1Reg, pregFull, preg, j * rLengthWithPadding + i * sregLower,
                    j * rLengthWithPadding + rHeadLength + i * sregLower);
                Muls(src0Reg, src0Reg, k2Rec, pregFull);
                Muls(src1Reg, src1Reg, k2Rec, preg);
                Add(dstReg, src0Reg, src1Reg, pregFull);
                ReduceSum(dstReg, dstReg, pregFull);
                DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((workUbOrigin + i), dstReg, pregOne);
            }

            // Processes the remaining data of the entire block.
            count = mainTailCount;
            for (uint16_t i = 0; i < repeatTimes2; i++) {
                preg = MicroAPI::UpdateMask<float>(count);
                LoadDataWithT<T>(srcUb, src0Reg, pregFull, j * rLengthWithPadding + mVL + i * sregLower);
                Muls(dstReg, src0Reg, k2Rec, pregFull);
                ReduceSum(dstReg, dstReg, pregFull);
                DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    (workUbOrigin + repeatTimes1 + i), dstReg, pregOne);
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            // Fold the tmpbuffer in half.
            uint16_t currentHalfAddTimes = halfAddTimes;
            for (uint16_t k = 0; k < static_cast<uint16_t>(halfAddRepeatTimes); k++) {
                currentHalfAddTimes = currentHalfAddTimes / Internal::kWelfordFinalizeFoldNum; // Fold
                for (uint16_t i = 0; i < currentHalfAddTimes; i++) {
                    DataCopy(src0Reg, workUbOrigin + i * sregLower);
                    DataCopy(src1Reg, workUbOrigin + (currentHalfAddTimes + i) * sregLower);
                    Add(dstReg, src0Reg, src1Reg, pregFull);
                    DataCopy(workUbOrigin + i * sregLower, dstReg, pregFull);
                }
                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            }
            // the last reducesum
            count = lastCount;
            preg = MicroAPI::UpdateMask<float>(count);
            DataCopy(src0Reg, workUbOrigin);
            ReduceSum(dstReg, src0Reg, preg);
            // save mean
            if constexpr (isCorrection) {
                Muls(meanReg, dstReg, rRecWithCorrection, preg);
            } else {
                Muls(meanReg, dstReg, k2RRec, preg);
            }
            DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((meanUb + j), meanReg, pregOne);
        }
    }
}

template <bool isCorrection = false>
__aicore__ inline void BinaryReduceSum(__ubuf__ float* dstUb, __ubuf__ float* srcUb, __ubuf__ float* workUbOrigin,
    uint32_t rLength, uint32_t rHeadLength, float k2Rec, float k2RRec, float rRecWithCorrection)
{
    constexpr uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(float));
    if (rLength < sregLower) {
        ComputeMean64<float, isCorrection>(
            dstUb, nullptr, srcUb, 1, rLength, rLength, k2Rec, k2RRec, rRecWithCorrection);
    } else {
        uint32_t rLengthWithPadding = static_cast<uint32_t>(CeilDivision(rLength, sregLower)) * sregLower;
        uint32_t rHeadLengthTmp = sregLower;
        uint32_t k = CalculatePower<sregLower>();
        for (uint32_t i = 0; i < rLengthWithPadding; i++) {
            if (rHeadLengthTmp * Internal::kWelfordFinalizeFoldNum > rLengthWithPadding) {
                k += i;
                break;
            }
            rHeadLengthTmp *= Internal::kWelfordFinalizeFoldNum;
        }
        ComputeMean<float, isCorrection>(dstUb, nullptr, srcUb, workUbOrigin, k, 1, rLength, rLengthWithPadding,
            rHeadLength, k2Rec, k2RRec, rRecWithCorrection);
    }
}

template <bool isReuseSource = false, const WelfordFinalizeConfig& config = WFFINALIZE_DEFAULT_CFG>
__aicore__ inline void WelfordFinalizeWithCounts(__local_mem__ float* outMean, __local_mem__ float* outVar,
    __local_mem__ int32_t* counts, __local_mem__ float* inMean, __local_mem__ float* inVar,
    __local_mem__ float* tmpbuffer, __local_mem__ float* sumTmpbuffer, const WelfordFinalizePara& para)
{
    // K is actually abLength, which needs to be aligned with 32 bytes.
    uint32_t K = para.headCountLength + para.tailCountLength;

    uint32_t sregLower = (uint32_t)Internal::WELFORD_B32_VF_LEN;
    uint16_t repeat = static_cast<uint16_t>(CeilDivision(K, sregLower));

    for (uint16_t m = 0; m < 1; m++) {
        __VEC_SCOPE__
        {
            MicroAPI::MaskReg preg;
            MicroAPI::RegTensor<int32_t> srcVreg;
            MicroAPI::RegTensor<float> f32vreg;
            MicroAPI::RegTensor<float> tmpVreg;

            MicroAPI::RegTensor<float> meanVreg;
            MicroAPI::RegTensor<float> varVreg;
            MicroAPI::RegTensor<float> outMeanreg;
            MicroAPI::RegTensor<float> outVarreg;

            uint32_t sreg = static_cast<uint32_t>(K);

            for (uint16_t i = 0; i < repeat; ++i) {
                preg = MicroAPI::UpdateMask<uint32_t>(sreg);
                MicroAPI::DataCopy<int32_t, MicroAPI::LoadDist::DIST_NORM>(srcVreg, counts + i * sregLower);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(meanVreg, inMean + i * sregLower + m * K);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(varVreg, inVar + i * sregLower + m * K);

                MicroAPI::Cast<float, int32_t, MrgZRndA>(f32vreg, srcVreg, preg);

                MicroAPI::Mul(outMeanreg, f32vreg, meanVreg, preg);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(
                    tmpbuffer + i * sregLower, outMeanreg, preg);
            }
        }
        uint32_t k = CalculateMainBlock(K);
        uint32_t kOverflow = k + 1;
        BinaryReduceSum(outMean + m, tmpbuffer, sumTmpbuffer, K, k, 1 / (float)kOverflow, para.rRec * kOverflow,
            para.rRecWithCorrection * kOverflow);

        __VEC_SCOPE__
        {
            MicroAPI::MaskReg preg;
            MicroAPI::RegTensor<int32_t> srcVreg;
            MicroAPI::RegTensor<float> f32vreg;
            MicroAPI::RegTensor<float> meanVreg;
            MicroAPI::RegTensor<float> varVreg;
            MicroAPI::RegTensor<float> outMeanreg;
            MicroAPI::RegTensor<float> outVarreg;
            MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(outMeanreg, outMean + m);
            uint32_t sreg = (uint32_t)K;
            for (uint16_t i = 0; i < repeat; ++i) {
                preg = MicroAPI::UpdateMask<uint32_t>(sreg);
                MicroAPI::DataCopy<int32_t, MicroAPI::LoadDist::DIST_NORM>(srcVreg, counts + i * sregLower);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(varVreg, inVar + i * sregLower + m * K);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(meanVreg, inMean + i * sregLower + m * K);

                MicroAPI::Cast<float, int32_t, MrgZRndA>(f32vreg, srcVreg, preg);
                MicroAPI::Sub(meanVreg, meanVreg, outMeanreg, preg);
                MicroAPI::Mul(meanVreg, meanVreg, meanVreg, preg);

                MicroAPI::Mul(meanVreg, meanVreg, f32vreg, preg);
                MicroAPI::Add(outVarreg, meanVreg, varVreg, preg);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(
                    tmpbuffer + i * sregLower, outVarreg, preg);
            }
        }
        BinaryReduceSum<config.isCorrection>(outVar + m, tmpbuffer, sumTmpbuffer, K, k, 1 / (float)kOverflow,
            para.rRec * kOverflow, para.rRecWithCorrection * kOverflow);
    }
}

template <bool isReuseSource = false, const WelfordFinalizeConfig& config = WFFINALIZE_DEFAULT_CFG>
__aicore__ inline void WelfordFinalizeForB32(__local_mem__ float* outMean, __local_mem__ float* outVar,
    __local_mem__ float* inMean, __local_mem__ float* inVar, __local_mem__ float* tmpbuffer,
    __local_mem__ float* sumTmpbuffer, const WelfordFinalizePara& para)
{
    if (para.tailCount == 0 || para.tailCountLength == 0) {
        uint32_t k = CalculateMainBlock(para.headCountLength);
        uint32_t kOverflow = k + 1;
        for (uint16_t m = 0; m < 1; m++) {
            BinaryReduceSum(outMean + m, inMean + m * para.headCountLength, sumTmpbuffer, para.headCountLength, k,
                1 / (float)kOverflow, para.abRec * kOverflow, para.rRecWithCorrection * kOverflow);

            __VEC_SCOPE__
            {
                MicroAPI::RegTensor<float> outmeanReg;
                MicroAPI::RegTensor<float> inMeanReg;
                MicroAPI::RegTensor<float> invarReg;
                MicroAPI::RegTensor<float> outVarreg;
                MicroAPI::RegTensor<float> tmpVreg;
                uint32_t K = para.headCountLength;
                uint32_t sregLower = static_cast<uint32_t>(Internal::WELFORD_B32_VF_LEN);
                uint16_t repeat = static_cast<uint16_t>(CeilDivision(K, sregLower));

                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(outmeanReg, outMean + m);
                uint32_t sreg = static_cast<uint32_t>(K);
                float rn = para.rnLength;
                for (uint16_t i = 0; i < repeat; ++i) {
                    MicroAPI::MaskReg preg = MicroAPI::UpdateMask<uint32_t>(sreg);
                    MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(
                        inMeanReg, inMean + i * sregLower + m * para.headCountLength);
                    MicroAPI::Sub(inMeanReg, inMeanReg, outmeanReg, preg);
                    MicroAPI::Mul(inMeanReg, inMeanReg, inMeanReg, preg);
                    MicroAPI::Muls(inMeanReg, inMeanReg, rn, preg);
                    MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(
                        invarReg, inVar + i * sregLower + m * para.headCountLength);
                    MicroAPI::Add(outVarreg, invarReg, inMeanReg, preg);
                    MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(
                        tmpbuffer + i * sregLower, outVarreg, preg);
                }
            }
            BinaryReduceSum<config.isCorrection>(outVar + m, tmpbuffer, sumTmpbuffer, para.headCountLength, k,
                1 / (float)kOverflow, para.rRec * kOverflow, para.rRecWithCorrection * kOverflow);
        }
    } else {
        uint32_t K = para.abLength;
        uint32_t sregLower = (uint32_t)Internal::WELFORD_B32_VF_LEN;
        uint16_t abRepeat = static_cast<uint16_t>(CeilDivision(K, sregLower));
        uint16_t hRepeat = static_cast<uint16_t>(CeilDivision(para.headCountLength, sregLower));
        uint32_t k = CalculateMainBlock(K);
        uint32_t kOverflow = k + 1;
        for (uint16_t m = 0; m < 1; m++) {
            __VEC_SCOPE__
            {
                MicroAPI::RegTensor<float> inMeanReg;
                MicroAPI::RegTensor<float> headReg;
                MicroAPI::RegTensor<float> tailReg;
                MicroAPI::MaskReg preg;
                Duplicate(headReg, (float)para.headCount / (float)para.tailCount);
                Duplicate(tailReg, para.tailCount);
                uint32_t sreg = K;
                for (uint16_t i = 0; i < abRepeat; ++i) {
                    preg = MicroAPI::UpdateMask<uint32_t>(sreg);
                    MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(inMeanReg, inMean + i * sregLower + m * K);
                    MicroAPI::Mul(inMeanReg, inMeanReg, tailReg, preg);
                    MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(
                        tmpbuffer + i * sregLower, inMeanReg, preg);
                }
                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
                sreg = (uint32_t)para.headCountLength;
                for (uint16_t i = 0; i < hRepeat; ++i) {
                    preg = MicroAPI::UpdateMask<uint32_t>(sreg);
                    MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(inMeanReg, tmpbuffer + i * sregLower);
                    MicroAPI::Mul(inMeanReg, inMeanReg, headReg, preg);
                    MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(
                        tmpbuffer + i * sregLower, inMeanReg, preg);
                }
            }

            BinaryReduceSum(outMean + m, tmpbuffer, sumTmpbuffer, K, k, 1 / (float)kOverflow, para.rRec * kOverflow,
                para.rRecWithCorrection * kOverflow);

            __VEC_SCOPE__
            {
                MicroAPI::RegTensor<float> outmeanReg;
                MicroAPI::RegTensor<float> inMeanReg;
                MicroAPI::RegTensor<float> invarReg;
                MicroAPI::RegTensor<float> outVarreg;
                MicroAPI::RegTensor<float> tmpVreg;
                MicroAPI::MaskReg preg;
                MicroAPI::RegTensor<float> headReg;
                MicroAPI::RegTensor<float> tailReg;
                Duplicate(headReg, (float)para.headCount / (float)para.tailCount);
                Duplicate(tailReg, para.tailCount);

                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(outmeanReg, outMean + m);
                uint32_t sreg = static_cast<uint32_t>(K);
                for (uint16_t i = 0; i < abRepeat; ++i) {
                    preg = MicroAPI::UpdateMask<uint32_t>(sreg);
                    MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(inMeanReg, inMean + i * sregLower + m * K);
                    MicroAPI::Sub(inMeanReg, inMeanReg, outmeanReg, preg);
                    MicroAPI::Mul(inMeanReg, inMeanReg, inMeanReg, preg);
                    MicroAPI::Mul(inMeanReg, inMeanReg, tailReg, preg);
                    MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(
                        tmpbuffer + i * sregLower, inMeanReg, preg);
                }
                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
                sreg = static_cast<uint32_t>(para.headCountLength);
                for (uint16_t i = 0; i < hRepeat; ++i) {
                    preg = MicroAPI::UpdateMask<uint32_t>(sreg);
                    MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(inMeanReg, tmpbuffer + i * sregLower);
                    MicroAPI::Mul(inMeanReg, inMeanReg, headReg, preg);
                    MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(
                        tmpbuffer + i * sregLower, inMeanReg, preg);
                }
                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
                sreg = (uint32_t)K;
                for (uint16_t i = 0; i < abRepeat; ++i) {
                    preg = MicroAPI::UpdateMask<uint32_t>(sreg);
                    MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(tmpVreg, tmpbuffer + i * sregLower);
                    MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(outVarreg, inVar + i * sregLower + m * K);
                    MicroAPI::Add(outVarreg, outVarreg, tmpVreg, preg);
                    MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(
                        tmpbuffer + i * sregLower, outVarreg, preg);
                }
            }
            BinaryReduceSum<config.isCorrection>(outVar + m, tmpbuffer, sumTmpbuffer, K, k, 1 / (float)kOverflow,
                para.rRec * kOverflow, para.rRecWithCorrection * kOverflow);
        }
    }
}

__aicore__ inline void CheckWelfordFinalizePara(const WelfordFinalizePara& para)
{
    bool ans = para.abLength > 0 && (para.abLength * sizeof(float) % ONE_BLK_SIZE == 0);
    ASCENDC_ASSERT(ans, { KERNEL_LOG(KERNEL_ERROR, "abLength is %u, is not 32B aligned.", para.abLength); });
    ans = para.abLength == para.headCountLength + para.tailCountLength;
    ASCENDC_ASSERT(ans, {
        KERNEL_LOG(KERNEL_ERROR, "abLength is %u, not equal to the sum of headCountLength %u and tailCountLength %u.",
            para.abLength, para.headCountLength, para.tailCountLength);
    });
    if (para.tailCount == 0) {
        ans = para.tailCountLength != 0;
        ASCENDC_ASSERT(ans, { KERNEL_LOG(KERNEL_ERROR, "tailCountLength cannot be zero when tailCount is zero."); });
    }
}

template <bool isReuseSource = false, const WelfordFinalizeConfig& config = WFFINALIZE_DEFAULT_CFG>
__aicore__ inline void WelfordFinalizeImpl(const LocalTensor<float>& outputMean,
    const LocalTensor<float>& outputVariance, const LocalTensor<float>& inputMean,
    const LocalTensor<float>& inputVariance, const LocalTensor<uint8_t>& sharedTmpBuffer, WelfordFinalizePara& para)
{
#if ASCENDC_CPU_DEBUG
    CheckWelfordFinalizePara(para);
#endif
    __local_mem__ float* inMean = (__local_mem__ float*)inputMean.GetPhyAddr();
    __local_mem__ float* inVar = (__local_mem__ float*)inputVariance.GetPhyAddr();
    __local_mem__ float* outMean = (__local_mem__ float*)outputMean.GetPhyAddr();
    __local_mem__ float* outVar = (__local_mem__ float*)outputVariance.GetPhyAddr();
    LocalTensor<float> stackBuffer = sharedTmpBuffer.ReinterpretCast<float>();
    __local_mem__ float* tmpbuffer1 = (__local_mem__ float*)stackBuffer.GetPhyAddr();
    __local_mem__ float* tmpbuffer2 = (__local_mem__ float*)stackBuffer[para.abLength].GetPhyAddr();

    WelfordFinalizeForB32<isReuseSource, config>(outMean, outVar, inMean, inVar, tmpbuffer1, tmpbuffer2, para);
}

template <bool isReuseSource = false, const WelfordFinalizeConfig& config = WFFINALIZE_DEFAULT_CFG>
__aicore__ inline void WelfordFinalizeImpl(const LocalTensor<float>& outputMean,
    const LocalTensor<float>& outputVariance, const LocalTensor<float>& inputMean,
    const LocalTensor<float>& inputVariance, WelfordFinalizePara& para)
{
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    WelfordFinalizeImpl<isReuseSource, config>(
        outputMean, outputVariance, inputMean, inputVariance, sharedTmpBuffer, para);
}

template <bool isReuseSource = false, const WelfordFinalizeConfig& config = WFFINALIZE_DEFAULT_CFG>
__aicore__ inline void WelfordFinalizeImpl(const LocalTensor<float>& outputMean,
    const LocalTensor<float>& outputVariance, const LocalTensor<float>& inputMean,
    const LocalTensor<float>& inputVariance, const LocalTensor<int32_t>& counts,
    const LocalTensor<uint8_t>& sharedTmpBuffer, WelfordFinalizePara& para)
{
#if ASCENDC_CPU_DEBUG
    CheckWelfordFinalizePara(para);
#endif
    LocalTensor<float> stackBuffer = sharedTmpBuffer.ReinterpretCast<float>();
    __local_mem__ int32_t* countsUb = (__local_mem__ int32_t*)counts.GetPhyAddr();
    __local_mem__ float* inMean = (__local_mem__ float*)inputMean.GetPhyAddr();
    __local_mem__ float* inVar = (__local_mem__ float*)inputVariance.GetPhyAddr();
    __local_mem__ float* outMean = (__local_mem__ float*)outputMean.GetPhyAddr();
    __local_mem__ float* outVar = (__local_mem__ float*)outputVariance.GetPhyAddr();
    __local_mem__ float* tmpbuffer1 = (__local_mem__ float*)stackBuffer.GetPhyAddr();
    __local_mem__ float* tmpbuffer2 = (__local_mem__ float*)stackBuffer[para.abLength].GetPhyAddr();

    WelfordFinalizeWithCounts<isReuseSource, config>(
        outMean, outVar, countsUb, inMean, inVar, tmpbuffer1, tmpbuffer2, para);
}

template <bool isReuseSource = false, const WelfordFinalizeConfig& config = WFFINALIZE_DEFAULT_CFG>
__aicore__ inline void WelfordFinalizeImpl(const LocalTensor<float>& outputMean,
    const LocalTensor<float>& outputVariance, const LocalTensor<float>& inputMean,
    const LocalTensor<float>& inputVariance, const LocalTensor<int32_t>& counts, WelfordFinalizePara& para)
{
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    WelfordFinalizeImpl<isReuseSource, config>(
        outputMean, outputVariance, inputMean, inputVariance, counts, sharedTmpBuffer, para);
}
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_NORMALIZATION_WELFORDFINALIZE_WELFORDFINALIZE_C310_IMPL_H
