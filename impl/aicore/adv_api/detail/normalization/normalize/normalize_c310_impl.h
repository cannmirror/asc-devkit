/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file normalize_c310_impl.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_NORMALIZATION_NORMALIZE_NORMALIZE_C310_IMPL_H
#define AICORE_ADV_API_DETAIL_NORMALIZATION_NORMALIZE_NORMALIZE_C310_IMPL_H

#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"
#include "normalization/normalize_utils.h"

namespace AscendC {

template <typename T = MicroAPI::DefaultType, MicroAPI::MaskMergeMode mode = MicroAPI::MaskMergeMode::ZEROING,
    typename RegT>
__aicore__ inline void RsqrtUtil(RegT& dstReg, RegT& srcReg, MicroAPI::MaskReg& mask)
{
    constexpr static float POS_INF = 3.40282366920938E+38;
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, MicroAPI::DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<T, float>(), "RsqrtUtil for high precision mode only supports float.");
    MicroAPI::RegTensor<float> regOne;
    MicroAPI::RegTensor<float> regZero;
    MicroAPI::RegTensor<float> regInf;
    MicroAPI::RegTensor<float> r;
    MicroAPI::RegTensor<float> y;
    MicroAPI::RegTensor<float> s;
    MicroAPI::RegTensor<float> t;
    MicroAPI::RegTensor<float> n15;
    MicroAPI::RegTensor<float> n1;
    MicroAPI::RegTensor<float> n05;

    MicroAPI::RegTensor<float> calReg;
    MicroAPI::RegTensor<float> cal1Reg;
    MicroAPI::MaskReg cmpRegZero;
    MicroAPI::MaskReg cmpRegInf;

    MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

    Duplicate(regOne, float(1), pregFull);
    Duplicate(regInf, POS_INF, pregFull);
    Duplicate(regZero, float(0), pregFull);
    Duplicate(n15, float(1.5), mask);
    Duplicate(n05, float(0.5), mask);
    Duplicate(s, float(1), mask);

    Div(r, regOne, srcReg, mask);
    Sqrt(y, r, mask);
    // y = y * (1.5 - 0.5*x*y*y)
    Muls(t, srcReg, float(-0.5), mask);   // -0.5*x
    Mul(t, t, y, mask);                   // -0.5*x*y
    MicroAPI::MulAddDst(n15, t, y, mask); // 1.5 + (-0.5*x*y) * y
    Mul(y, y, n15, mask);                 // y = y * (1.5 + (-0.5*x*y) * y)
    // s = 1 - x*r
    Muls(n1, srcReg, float(-1.0), mask); // -x
    MulAddDst(s, n1, r, mask);           // s = 1 + (-x) * r, (ps: s = 1)
    // t = r - y*y => r = r + (-y) * y
    Muls(n1, y, float(-1.0), mask); // -y
    MulAddDst(r, n1, y, mask);      // t = r + (-y) * y
    // e = s + x * t => s = s + x * t
    MulAddDst(s, srcReg, r, mask);
    // y = y + y*e*0.5
    Mul(s, s, y, mask);         // y*e => y*s(e)
    MulAddDst(y, s, n05, mask); // y = y + s*0.5

    // move to the last
    // if x == float(inf): return 0.0f // if mask is 0, then default select srcReg Value
    CompareScalar(cmpRegZero, srcReg, POS_INF, mask);
    Select(dstReg, regZero, y, cmpRegZero);
    // if x == 0.0f: return float(inf)
    CompareScalar(cmpRegInf, srcReg, (float)0, mask);
    Select(dstReg, regInf, dstReg, cmpRegInf);
}

template <typename U, typename T, bool isReuseSource = false, const NormalizeConfig& config = NLCFG_NORM>
__aicore__ inline void NormalizeImpl(const LocalTensor<T>& output, const LocalTensor<float>& outputRstd,
    const LocalTensor<float>& inputMean, const LocalTensor<float>& inputVariance, const LocalTensor<T>& inputX,
    const LocalTensor<U>& gamma, const LocalTensor<U>& beta, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const float epsilon, const NormalizePara& para)
{
    static_assert(SupportType<T, half, float, bfloat16_t>(), "current data type is not supported on current device!");
    if constexpr (IsSameType<T, half>::value) {
        static_assert(SupportType<U, half, float>(), "current data type is not supported on current device!");
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        static_assert(SupportType<U, bfloat16_t, float>(), "current data type is not supported on current device!");
    } else if constexpr (IsSameType<T, float>::value) {
        static_assert(SupportType<U, float>(), "current data type is not supported on current device!");
    }
    static_assert(config.isOnlyOutput == false, "current value is not supported on current device!");

    static_assert(SupportEnum<config.reducePattern, ReducePattern::AR>(),
        "current api only supported pattern AR on current device!");
    if constexpr (config.aLength != -1) {
        ASCENDC_ASSERT((config.aLength == para.aLength), { KERNEL_LOG(KERNEL_ERROR, "current aLength not match!"); });
    }
    constexpr uint16_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(float));

    uint16_t aLength = static_cast<uint16_t>(para.aLength);
    uint32_t rLength = para.rLength;
    uint32_t rLengthWithPadding = para.rLengthWithPadding;
    uint16_t repeatTimes = CeilDivision(rLength, sregLower);
    uint16_t tailARepeatTimes = aLength % 2;
    uint32_t count;

    uint32_t halfA = aLength / 2;

    LocalTensor<float> workLocal = sharedTmpBuffer.ReinterpretCast<float>();

    __local_mem__ float* rstdUb = (__local_mem__ float*)outputRstd.GetPhyAddr();
    __local_mem__ float* meanUb = (__local_mem__ float*)inputMean.GetPhyAddr();
    __local_mem__ float* varianceUb = (__local_mem__ float*)inputVariance.GetPhyAddr();

    __local_mem__ float* rstdUb2 = rstdUb + halfA;
    __local_mem__ float* rstdUbTail = rstdUb + aLength - 1;
    __local_mem__ float* meanUb2 = meanUb + halfA;
    __local_mem__ float* meanUbTail = meanUb + aLength - 1;

    __local_mem__ float* varianceUb2 = varianceUb + halfA;
    __local_mem__ float* varianceUbTail = varianceUb + aLength - 1;

    __local_mem__ T* inputXUb = (__local_mem__ T*)inputX.GetPhyAddr();
    __local_mem__ T* outputUb = (__local_mem__ T*)output.GetPhyAddr();
    __local_mem__ U* gammaUb = (__local_mem__ U*)gamma.GetPhyAddr();
    __local_mem__ U* betaUb = (__local_mem__ U*)beta.GetPhyAddr();
    __local_mem__ float* workUb = (__local_mem__ float*)workLocal.GetPhyAddr();

    __local_mem__ T* inputXUb2 = inputXUb + halfA * rLengthWithPadding;
    __local_mem__ T* inputXUbTail = inputXUb + (aLength - 1) * rLengthWithPadding;
    __local_mem__ T* outputUb2 = outputUb + halfA * rLengthWithPadding;
    __local_mem__ T* outputUbTail = outputUb + (aLength - 1) * rLengthWithPadding;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> inputReg1;
        MicroAPI::RegTensor<float> inputReg2;
        MicroAPI::RegTensor<float> inputReg;
        MicroAPI::RegTensor<float> gammaReg;
        MicroAPI::RegTensor<float> betaReg;
        MicroAPI::RegTensor<float> dstReg;
        MicroAPI::RegTensor<float> dstReg1;
        MicroAPI::RegTensor<float> dstReg2;

        MicroAPI::RegTensor<float> meanReg;
        MicroAPI::RegTensor<float> varianceReg;
        MicroAPI::RegTensor<float> rstdReg;

        MicroAPI::RegTensor<float> meanReg1;
        MicroAPI::RegTensor<float> varianceReg1;
        MicroAPI::RegTensor<float> rstdReg1;
        MicroAPI::RegTensor<float> meanReg2;
        MicroAPI::RegTensor<float> varianceReg2;
        MicroAPI::RegTensor<float> rstdReg2;

        MicroAPI::MaskReg preg;
        MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregOne = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();

        for (uint16_t j = 0; j < static_cast<uint16_t>(halfA); j++) {
            DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(varianceReg1, varianceUb + j);
            DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(varianceReg2, varianceUb2 + j);
            Adds(varianceReg1, varianceReg1, epsilon, pregFull);
            Adds(varianceReg2, varianceReg2, epsilon, pregFull);
            RsqrtUtil<float>(rstdReg1, varianceReg1, pregFull);
            RsqrtUtil<float>(rstdReg2, varianceReg2, pregFull);
            DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdUb + j, rstdReg1, pregOne);
            DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdUb2 + j, rstdReg2, pregOne);
        }
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t j = 0; j < static_cast<uint16_t>(halfA); j++) {
            count = rLength;
            DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(rstdReg1, rstdUb + j);
            DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(rstdReg2, rstdUb2 + j);
            DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanReg1, meanUb + j);
            DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanReg2, meanUb2 + j);
            for (uint16_t i = 0; i < repeatTimes; i++) {
                preg = MicroAPI::UpdateMask<float>(count);
                Internal::LoadDataWithT<T>(inputXUb, inputReg1, preg, j * rLengthWithPadding + i * sregLower);
                Internal::LoadDataWithT<T>(inputXUb2, inputReg2, preg, j * rLengthWithPadding + i * sregLower);

                if constexpr (IsSameType<U, half>::value || IsSameType<U, bfloat16_t>::value) {
                    if constexpr (!config.isNoGamma) {
                        MicroAPI::RegTensor<U> gammaRegOrigin;
                        DataCopy<U, MicroAPI::LoadDist::DIST_UNPACK_B16>(gammaRegOrigin, gammaUb + i * sregLower);
                        Cast<float, U, layoutZMrgZ>(gammaReg, gammaRegOrigin, preg);
                    }
                    if constexpr (!config.isNoBeta) {
                        MicroAPI::RegTensor<U> betaRegOrigin;
                        DataCopy<U, MicroAPI::LoadDist::DIST_UNPACK_B16>(betaRegOrigin, betaUb + i * sregLower);
                        Cast<float, U, layoutZMrgZ>(betaReg, betaRegOrigin, preg);
                    }
                } else {
                    if constexpr (!config.isNoGamma) {
                        DataCopy(gammaReg, gammaUb + i * sregLower);
                    }
                    if constexpr (!config.isNoBeta) {
                        DataCopy(betaReg, betaUb + i * sregLower);
                    }
                }
                Sub<float, MicroAPI::MaskMergeMode::ZEROING>(dstReg1, inputReg1, meanReg1, preg);
                Mul<float, MicroAPI::MaskMergeMode::ZEROING>(dstReg1, dstReg1, rstdReg1, preg);
                Sub<float, MicroAPI::MaskMergeMode::ZEROING>(dstReg2, inputReg2, meanReg2, preg);
                Mul<float, MicroAPI::MaskMergeMode::ZEROING>(dstReg2, dstReg2, rstdReg2, preg);
                // FusedMulAdd: Vd = Vn * Vd + Vm, dst = gamma * dst + beta
                if constexpr (!config.isNoGamma && !config.isNoBeta) {
                    FusedMulDstAdd(dstReg1, gammaReg, betaReg, pregFull);
                    FusedMulDstAdd(dstReg2, gammaReg, betaReg, pregFull);
                } else {
                    if constexpr (!config.isNoGamma) {
                        Mul<float, MicroAPI::MaskMergeMode::ZEROING>(dstReg1, dstReg1, gammaReg, preg);
                        Mul<float, MicroAPI::MaskMergeMode::ZEROING>(dstReg2, dstReg2, gammaReg, preg);
                    }
                    if constexpr (!config.isNoBeta) {
                        Add<float, MicroAPI::MaskMergeMode::ZEROING>(dstReg1, dstReg1, betaReg, preg);
                        Add<float, MicroAPI::MaskMergeMode::ZEROING>(dstReg2, dstReg2, betaReg, preg);
                    }
                }
                if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
                    MicroAPI::RegTensor<T> yRegOrigin;
                    Cast<T, float, LayoutZMrgZRndRSatNS>(yRegOrigin, dstReg1, preg);
                    DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(
                        outputUb + j * rLengthWithPadding + i * sregLower, yRegOrigin, preg);
                } else {
                    DataCopy(outputUb + j * rLengthWithPadding + i * sregLower, dstReg1, preg);
                }
                if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
                    MicroAPI::RegTensor<T> yRegOrigin;
                    Cast<T, float, LayoutZMrgZRndRSatNS>(yRegOrigin, dstReg2, preg);
                    DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(
                        outputUb2 + j * rLengthWithPadding + i * sregLower, yRegOrigin, preg);
                } else {
                    DataCopy(outputUb2 + j * rLengthWithPadding + i * sregLower, dstReg2, preg);
                }
            }
        }
        for (uint16_t j = 0; j < tailARepeatTimes; j++) {
            DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(varianceReg, varianceUbTail);
            Adds(varianceReg, varianceReg, epsilon, pregFull);
            RsqrtUtil<float>(rstdReg, varianceReg, pregFull);
            DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(rstdUbTail, rstdReg, pregOne);
        }
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        for (uint16_t j = 0; j < tailARepeatTimes; j++) {
            DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanReg, meanUbTail);
            DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(rstdReg, rstdUbTail);
            count = rLength;
            for (uint16_t i = 0; i < repeatTimes; i++) {
                MicroAPI::MaskReg preg = MicroAPI::UpdateMask<float>(count);
                Internal::LoadDataWithT<T>(inputXUbTail, inputReg, preg, i * sregLower);

                if constexpr (IsSameType<U, half>::value || IsSameType<U, bfloat16_t>::value) {
                    if constexpr (!config.isNoGamma) {
                        MicroAPI::RegTensor<U> gammaRegOrigin;
                        DataCopy<U, MicroAPI::LoadDist::DIST_UNPACK_B16>(gammaRegOrigin, gammaUb + i * sregLower);
                        Cast<float, U, layoutZMrgZ>(gammaReg, gammaRegOrigin, preg);
                    }
                    if constexpr (!config.isNoBeta) {
                        MicroAPI::RegTensor<U> betaRegOrigin;
                        DataCopy<U, MicroAPI::LoadDist::DIST_UNPACK_B16>(betaRegOrigin, betaUb + i * sregLower);
                        Cast<float, U, layoutZMrgZ>(betaReg, betaRegOrigin, preg);
                    }
                } else {
                    if constexpr (!config.isNoGamma) {
                        DataCopy(gammaReg, gammaUb + i * sregLower);
                    }
                    if constexpr (!config.isNoBeta) {
                        DataCopy(betaReg, betaUb + i * sregLower);
                    }
                }
                Sub<float, MicroAPI::MaskMergeMode::ZEROING>(dstReg, inputReg, meanReg, preg);
                Mul<float, MicroAPI::MaskMergeMode::ZEROING>(dstReg, dstReg, rstdReg, preg);
                // FusedMulAdd: Vd = Vn * Vd + Vm, dst = gamma * dst + beta
                if constexpr (!config.isNoGamma && !config.isNoBeta) {
                    FusedMulDstAdd(dstReg, gammaReg, betaReg, pregFull);
                } else {
                    if constexpr (!config.isNoGamma) {
                        Mul<float, MicroAPI::MaskMergeMode::ZEROING>(dstReg, dstReg, gammaReg, preg);
                    }
                    if constexpr (!config.isNoBeta) {
                        Add<float, MicroAPI::MaskMergeMode::ZEROING>(dstReg, dstReg, betaReg, preg);
                    }
                }
                if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
                    MicroAPI::RegTensor<T> yRegOrigin;
                    Cast<T, float, LayoutZMrgZRndRSatNS>(yRegOrigin, dstReg, preg);
                    DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(outputUbTail + i * sregLower, yRegOrigin, preg);
                } else {
                    DataCopy(outputUbTail + i * sregLower, dstReg, preg);
                }
            }
        }
    }
}

template <typename U, typename T, bool isReuseSource = false, const NormalizeConfig& config = NLCFG_NORM>
__aicore__ inline void NormalizeImpl(const LocalTensor<T>& output, const LocalTensor<float>& outputRstd,
    const LocalTensor<float>& inputMean, const LocalTensor<float>& inputVariance, const LocalTensor<T>& inputX,
    const LocalTensor<U>& gamma, const LocalTensor<U>& beta, const float epsilon, const NormalizePara& para)
{
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    NormalizeImpl<U, T, isReuseSource, config>(
        output, outputRstd, inputMean, inputVariance, inputX, gamma, beta, sharedTmpBuffer, epsilon, para);
}
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_NORMALIZATION_NORMALIZE_NORMALIZE_C310_IMPL_H
