/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file layernorm_c310_impl.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_NORMALIZATION_LAYERNORM_LAYERNORM_C310_IMPL_H
#define AICORE_ADV_API_DETAIL_NORMALIZATION_LAYERNORM_LAYERNORM_C310_IMPL_H

#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"
#include "kernel_tiling/kernel_tiling.h"
#include "normalization/normalize.h"

namespace AscendC {
namespace Internal {
const uint16_t kLayernormFoldNum = 2;
constexpr uint32_t LAYERNORM_B16_VF_LEN = GetVecLen() / sizeof(uint16_t);
constexpr uint32_t LAYERNORM_B32_VF_LEN = GetVecLen() / sizeof(uint32_t);
} // namespace Internal

template <typename T>
__aicore__ inline void LoadDataWithT(__local_mem__ T* src0, __local_mem__ T* src1, MicroAPI::RegTensor<float>& dstReg0,
    MicroAPI::RegTensor<float>& dstReg1, MicroAPI::MaskReg& dst0Preg, MicroAPI::MaskReg& dst1Preg, uint32_t src0Offset,
    uint32_t src1Offset)
{
    if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<T> src0Origin;
        MicroAPI::RegTensor<T> src1Origin;
        DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(src0Origin, src0 + src0Offset);
        DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(src1Origin, src1 + src1Offset);
        Cast<float, T, layoutZMrgZ>(dstReg0, src0Origin, dst0Preg);
        Cast<float, T, layoutZMrgZ>(dstReg1, src1Origin, dst1Preg);
    } else { // this branch: only support float
        DataCopy(dstReg0, src0 + src0Offset);
        DataCopy(dstReg1, src1 + src1Offset);
    }
}

template <typename T>
__aicore__ inline void LoadDataWithT(
    __local_mem__ T* src, MicroAPI::RegTensor<float>& dstReg, MicroAPI::MaskReg& dstPreg, uint32_t srcOffset)
{
    if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<T> srcOrigin;
        DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcOrigin, src + srcOffset);
        Cast<float, T, layoutZMrgZ>(dstReg, srcOrigin, dstPreg);
    } else { // this branch: only support float
        DataCopy(dstReg, src + srcOffset);
    }
}

__aicore__ inline uint16_t CalculateHalfAddRepeatTimes(uint32_t halfAddTimes)
{
    constexpr uint16_t kMaxOffset = 16;
    uint16_t fold = 0;
    for (uint16_t i = 1; i < kMaxOffset; i++) {
        if ((halfAddTimes >> i) == 0) {
            break;
        }
        fold++;
    }
    return fold;
}

// only support rLength <= 64
template <typename T, bool isOutputVariance = true, bool isCorrection = false>
__aicore__ inline void ComputeMeanVariance64(__local_mem__ float* meanUb, __local_mem__ float* varianceUb,
    __local_mem__ T* srcUb, const uint32_t aLength, const uint32_t rLength, const uint32_t rLengthWithPadding,
    const float k2Rec, const float k2RRec, const float rRecWithCorrection)
{
    constexpr uint16_t sregLower = static_cast<uint16_t>(GetVecLen() / sizeof(float)); // 64
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
            Muls(meanReg, dstReg, k2RRec, pregOne);
            DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((meanUb + i), meanReg, pregOne);
            if constexpr (isOutputVariance) {
                Duplicate(meanReg, meanReg, pregFull);
                Sub(src0Reg, src0Reg, meanReg, pregFull);
                Mul(src0Reg, src0Reg, src0Reg, pregFull);
                Muls(src0Reg, src0Reg, k2Rec, pregFull);
                ReduceSum(dstReg, src0Reg, preg);
                if constexpr (isCorrection) {
                    Muls(varianceReg, dstReg, rRecWithCorrection, pregOne);
                } else {
                    Muls(varianceReg, dstReg, k2RRec, pregOne);
                }
                DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((varianceUb + i), varianceReg, pregOne);
            }
        }
    }
}
// only support rLength <= 128
template <typename T, bool isOutputVariance = true>
__aicore__ inline void ComputeMeanVariance128(__local_mem__ float* meanUb, __local_mem__ float* varianceUb,
    __local_mem__ T* srcUb, const uint32_t aLength, const uint32_t rLength, const uint32_t rLengthWithPadding,
    const float k2Rec, const float k2RRec)
{
    constexpr uint16_t sregLower = (uint32_t)(GetVecLen() / sizeof(float)); // 64
    uint32_t count = rLength - sregLower;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> src0Reg;
        MicroAPI::RegTensor<float> src1Reg;
        MicroAPI::RegTensor<float> src0CalReg;
        MicroAPI::RegTensor<float> src1CalReg;
        MicroAPI::RegTensor<float> dstReg;
        MicroAPI::RegTensor<float> meanReg;
        MicroAPI::RegTensor<float> varianceReg;

        MicroAPI::MaskReg preg = MicroAPI::UpdateMask<float>(count);
        MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregOne = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
        for (uint16_t i = 0; i < static_cast<uint16_t>(aLength); i++) {
            LoadDataWithT<T>(srcUb, srcUb, src0Reg, src1Reg, pregFull, preg, i * rLengthWithPadding,
                i * rLengthWithPadding + sregLower);
            Muls(src0CalReg, src0Reg, k2Rec, pregFull);
            Muls(src1CalReg, src1Reg, k2Rec, preg);
            Add(dstReg, src0CalReg, src1CalReg, pregFull);
            ReduceSum(dstReg, dstReg, pregFull);
            Muls(meanReg, dstReg, k2RRec, pregOne);
            DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((meanUb + i), meanReg, pregOne);
            if constexpr (isOutputVariance) {
                Duplicate(meanReg, meanReg, pregFull);
                Sub(src0CalReg, src0Reg, meanReg, pregFull);
                Mul(src0CalReg, src0CalReg, src0CalReg, pregFull);
                Muls(src0CalReg, src0CalReg, k2Rec, pregFull);

                Sub(src1CalReg, src1Reg, meanReg, preg);
                Mul(src1CalReg, src1CalReg, src1CalReg, preg);
                Muls(src1CalReg, src1CalReg, k2Rec, preg);

                Add(dstReg, src0CalReg, src1CalReg, pregFull);

                ReduceSum(dstReg, dstReg, pregFull);
                Muls(varianceReg, dstReg, k2RRec, pregOne);
                DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((varianceUb + i), varianceReg, pregOne);
            }
        }
    }
}

template <typename T, bool isOutputVariance = true>
__aicore__ inline void ComputeMeanVarianceUseY(__local_mem__ float* meanUb, __local_mem__ float* varianceUb,
    __local_mem__ T* srcUb, __local_mem__ T* workUbYOrigin, const uint32_t k, const uint32_t aLength,
    const uint32_t rLength, const uint32_t rLengthWithPadding, const uint32_t rHeadLength, const float k2Rec,
    const float k2RRec)
{
    constexpr uint16_t sregLower = static_cast<uint16_t>(GetVecLen() / sizeof(float)); // 64
    const uint32_t m = rLength - rHeadLength;
    uint32_t count;
    const uint16_t halfAddCount = CeilDivision(rHeadLength / 2, sregLower); // total count
    const uint16_t halfAddTimes = CeilDivision(halfAddCount, sregLower);

    const uint16_t varianceOffset = CeilDivision(aLength, sregLower);
    const uint16_t halfAddRepeatTimes = CalculateHalfAddRepeatTimes(halfAddTimes);
    int16_t lastCount = halfAddCount; // last vcadd times less than 64
    uint16_t repeatTimes1 = CeilDivision(m, sregLower) / 2;
    uint16_t count2 = m % (sregLower * 2);
    if (repeatTimes1 * sregLower * 2 > m) {
        count2 = 0;
    }
    uint16_t repeatTimes2 = CeilDivision(count2, sregLower);
    const uint32_t mainTailCount = rHeadLength - repeatTimes1 * 2 * sregLower - repeatTimes2 * sregLower * 2;

    const uint32_t mVL = (repeatTimes1 + repeatTimes2) * sregLower * 2;

    uint16_t repeatTimes3 = mainTailCount / sregLower / 2;
    uint32_t lastCountTmp = static_cast<uint32_t>(lastCount);

    if (halfAddTimes == 1) {
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> v0, v1, v2, v3, v4, v5, v6, v7;
            MicroAPI::RegTensor<float> src0Reg0;
            MicroAPI::RegTensor<float> src1Reg0;
            MicroAPI::RegTensor<float> dstReg0;
            MicroAPI::RegTensor<float> src0Reg1;
            MicroAPI::RegTensor<float> src1Reg1;
            MicroAPI::RegTensor<float> dstReg1;
            MicroAPI::RegTensor<float> dstReg;
            MicroAPI::RegTensor<float> meanReg;

            MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg pregOne = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            MicroAPI::MaskReg pregLastCount = MicroAPI::UpdateMask<float>(lastCountTmp);
            count = count2;
            MicroAPI::MaskReg preg2 = MicroAPI::UpdateMask<float>(count);
            for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                uint32_t mTmp = m;
                __local_mem__ float* workUbOrigin = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);
                // tail block add to main block
                for (uint16_t i = 0; i < repeatTimes1; i++) {
                    MicroAPI::MaskReg preg = MicroAPI::UpdateMask<float>(mTmp);
                    LoadDataWithT<T>(srcUb, srcUb, src0Reg0, src1Reg0, pregFull, preg,
                        j * rLengthWithPadding + (2 * i) * sregLower,
                        j * rLengthWithPadding + rHeadLength + (2 * i) * sregLower);
                    Muls(dstReg0, src1Reg0, k2Rec, preg);
                    Axpy(dstReg0, src0Reg0, k2Rec, pregFull);
                    preg = MicroAPI::UpdateMask<float>(mTmp);
                    LoadDataWithT<T>(srcUb, srcUb, src0Reg1, src1Reg1, pregFull, preg,
                        j * rLengthWithPadding + (2 * i + 1) * sregLower,
                        j * rLengthWithPadding + rHeadLength + (2 * i + 1) * sregLower);
                    Muls(dstReg1, src1Reg1, k2Rec, preg);
                    Axpy(dstReg1, src0Reg1, k2Rec, pregFull);
                    Add(dstReg, dstReg0, dstReg1, pregFull);
                    ReduceSum(dstReg, dstReg, pregFull);
                    DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((workUbOrigin + i), dstReg, pregOne);
                }
            }
            for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                __local_mem__ float* workUbOrigin = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);
                for (uint16_t i = 0; i < repeatTimes2; i++) {
                    LoadDataWithT<T>(srcUb, srcUb, src0Reg0, src1Reg0, pregFull, preg2,
                        j * rLengthWithPadding + repeatTimes1 * 2 * sregLower,
                        j * rLengthWithPadding + rHeadLength + repeatTimes1 * 2 * sregLower);
                    Muls(dstReg0, src1Reg0, k2Rec, preg2);
                    Axpy(dstReg0, src0Reg0, k2Rec, pregFull);

                    LoadDataWithT<T>(
                        srcUb, src0Reg1, pregFull, j * rLengthWithPadding + repeatTimes1 * 2 * sregLower + sregLower);
                    Muls(dstReg1, src0Reg1, k2Rec, pregFull);

                    Add(dstReg, dstReg0, dstReg1, pregFull);
                    ReduceSum(dstReg, dstReg, pregFull);
                    DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                        (workUbOrigin + repeatTimes1 + i), dstReg, pregOne);
                }
            }
            for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                __local_mem__ float* workUbOrigin = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);
                // Processes the remaining data of the entire block.
                for (uint16_t i = 0; i < repeatTimes3; i++) {
                    LoadDataWithT<T>(srcUb, src0Reg0, pregFull, j * rLengthWithPadding + mVL + 2 * i * sregLower);
                    LoadDataWithT<T>(srcUb, src0Reg1, pregFull, j * rLengthWithPadding + mVL + (2 * i + 1) * sregLower);
                    Muls(dstReg0, src0Reg0, k2Rec, pregFull);
                    Muls(dstReg1, src0Reg1, k2Rec, pregFull);
                    Add(dstReg, dstReg0, dstReg1, pregFull);
                    ReduceSum(dstReg, dstReg, pregFull);
                    DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                        (workUbOrigin + repeatTimes1 + repeatTimes2 + i), dstReg, pregOne);
                }
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                __local_mem__ float* workUbOrigin = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);
                DataCopy(v0, workUbOrigin);
                // reduce
                ReduceSum(dstReg, v0, pregLastCount);
                Muls(meanReg, dstReg, k2RRec, pregOne);
                // save mean
                DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((meanUb + j), meanReg, pregOne);
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

            if constexpr (isOutputVariance) {
                for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                    uint32_t mTmp = m;
                    __local_mem__ float* workUbOrigin = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);
                    DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanReg, meanUb + j);
                    // tail block add to main block
                    for (uint16_t i = 0; i < repeatTimes1; i++) {
                        MicroAPI::MaskReg preg = MicroAPI::UpdateMask<float>(mTmp);
                        LoadDataWithT<T>(srcUb, srcUb, src0Reg0, src1Reg0, pregFull, preg,
                            j * rLengthWithPadding + (2 * i) * sregLower,
                            j * rLengthWithPadding + rHeadLength + (2 * i) * sregLower);

                        Sub(src0Reg0, src0Reg0, meanReg, pregFull);
                        Sub(src1Reg0, src1Reg0, meanReg, preg);

                        Mul(src0Reg0, src0Reg0, src0Reg0, pregFull);
                        Mul(src1Reg0, src1Reg0, src1Reg0, pregFull);
                        Muls(dstReg0, src1Reg0, k2Rec, pregFull);
                        Axpy(dstReg0, src0Reg0, k2Rec, pregFull);

                        preg = MicroAPI::UpdateMask<float>(mTmp);
                        LoadDataWithT<T>(srcUb, srcUb, src0Reg1, src1Reg1, pregFull, preg,
                            j * rLengthWithPadding + (2 * i + 1) * sregLower,
                            j * rLengthWithPadding + rHeadLength + (2 * i + 1) * sregLower);

                        Sub(src0Reg1, src0Reg1, meanReg, pregFull);
                        Sub(src1Reg1, src1Reg1, meanReg, preg);

                        Mul(src0Reg1, src0Reg1, src0Reg1, pregFull);
                        Mul(src1Reg1, src1Reg1, src1Reg1, pregFull);
                        Muls(dstReg1, src1Reg1, k2Rec, pregFull);
                        Axpy(dstReg1, src0Reg1, k2Rec, pregFull);

                        Add(dstReg, dstReg0, dstReg1, pregFull);
                        ReduceSum(dstReg, dstReg, pregFull);
                        DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                            (workUbOrigin + i), dstReg, pregOne);
                    }
                }
                for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                    __local_mem__ float* workUbOrigin = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);
                    DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanReg, meanUb + j);
                    for (uint16_t i = 0; i < repeatTimes2; i++) {
                        LoadDataWithT<T>(srcUb, srcUb, src0Reg0, src1Reg0, pregFull, preg2,
                            j * rLengthWithPadding + repeatTimes1 * 2 * sregLower,
                            j * rLengthWithPadding + rHeadLength + repeatTimes1 * 2 * sregLower);

                        Sub(src0Reg0, src0Reg0, meanReg, pregFull);
                        Sub(src1Reg0, src1Reg0, meanReg, pregFull);

                        Mul(src0Reg0, src0Reg0, src0Reg0, pregFull);
                        Mul(src1Reg0, src1Reg0, src1Reg0, pregFull);
                        Muls(dstReg0, src1Reg0, k2Rec, preg2);
                        Axpy(dstReg0, src0Reg0, k2Rec, pregFull);

                        LoadDataWithT<T>(srcUb, src0Reg1, pregFull,
                            j * rLengthWithPadding + repeatTimes1 * 2 * sregLower + sregLower);

                        Sub(src0Reg1, src0Reg1, meanReg, pregFull);

                        Mul(src0Reg1, src0Reg1, src0Reg1, pregFull);

                        Muls(dstReg1, src0Reg1, k2Rec, pregFull);

                        Add(dstReg, dstReg0, dstReg1, pregFull);
                        ReduceSum(dstReg, dstReg, pregFull);
                        DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                            (workUbOrigin + repeatTimes1 + i), dstReg, pregOne);
                    }
                }
                for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                    __local_mem__ float* workUbOrigin = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);
                    DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanReg, meanUb + j);
                    // Processes the remaining data of the entire block.
                    for (uint16_t i = 0; i < repeatTimes3; i++) {
                        LoadDataWithT<T>(srcUb, src0Reg0, pregFull, j * rLengthWithPadding + mVL + (2 * i) * sregLower);
                        LoadDataWithT<T>(
                            srcUb, src0Reg1, pregFull, j * rLengthWithPadding + mVL + (2 * i + 1) * sregLower);

                        Sub(src0Reg0, src0Reg0, meanReg, pregFull);
                        Mul(src0Reg0, src0Reg0, src0Reg0, pregFull);
                        Muls(dstReg0, src0Reg0, k2Rec, pregFull);
                        Sub(src0Reg1, src0Reg1, meanReg, pregFull);
                        Mul(src0Reg1, src0Reg1, src0Reg1, pregFull);
                        Muls(dstReg1, src0Reg1, k2Rec, pregFull);

                        Add(dstReg, dstReg0, dstReg1, pregFull);

                        ReduceSum(dstReg, dstReg, pregFull);
                        DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                            (workUbOrigin + repeatTimes1 + repeatTimes2 + i), dstReg, pregOne);
                    }
                }
                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
                for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                    __local_mem__ float* workUbOrigin = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);
                    DataCopy(v0, workUbOrigin);
                    // reduce
                    ReduceSum(dstReg, v0, pregLastCount);
                    Muls(dstReg, dstReg, k2RRec, pregOne);
                    // save variance
                    DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((varianceUb + j), dstReg, pregOne);
                }
            }
        }
    } else if (halfAddTimes == 2) {
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> v0, v1, v2, v3, v4, v5, v6, v7;
            MicroAPI::RegTensor<float> src0Reg0;
            MicroAPI::RegTensor<float> src1Reg0;
            MicroAPI::RegTensor<float> dstReg0;
            MicroAPI::RegTensor<float> src0Reg1;
            MicroAPI::RegTensor<float> src1Reg1;
            MicroAPI::RegTensor<float> dstReg1;
            MicroAPI::RegTensor<float> dstReg;
            MicroAPI::RegTensor<float> meanReg;

            MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg pregOne = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            MicroAPI::MaskReg pregLastCount = MicroAPI::UpdateMask<float>(lastCountTmp);
            count = count2;
            MicroAPI::MaskReg preg2 = MicroAPI::UpdateMask<float>(count);

            for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                uint32_t mTmp = m;
                __local_mem__ float* workUbOrigin = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);
                // tail block add to main block
                for (uint16_t i = 0; i < repeatTimes1; i++) {
                    MicroAPI::MaskReg preg = MicroAPI::UpdateMask<float>(mTmp);
                    LoadDataWithT<T>(srcUb, srcUb, src0Reg0, src1Reg0, pregFull, preg,
                        j * rLengthWithPadding + (2 * i) * sregLower,
                        j * rLengthWithPadding + rHeadLength + (2 * i) * sregLower);
                    Muls(dstReg0, src1Reg0, k2Rec, preg);
                    Axpy(dstReg0, src0Reg0, k2Rec, pregFull);

                    preg = MicroAPI::UpdateMask<float>(mTmp);
                    LoadDataWithT<T>(srcUb, srcUb, src0Reg1, src1Reg1, pregFull, preg,
                        j * rLengthWithPadding + (2 * i + 1) * sregLower,
                        j * rLengthWithPadding + rHeadLength + (2 * i + 1) * sregLower);
                    Muls(dstReg1, src1Reg1, k2Rec, preg);
                    Axpy(dstReg1, src0Reg1, k2Rec, pregFull);
                    Add(dstReg, dstReg0, dstReg1, pregFull);
                    ReduceSum(dstReg, dstReg, pregFull);
                    DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((workUbOrigin + i), dstReg, pregOne);
                }
            }
            for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                __local_mem__ float* workUbOrigin = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);
                for (uint16_t i = 0; i < repeatTimes2; i++) {
                    LoadDataWithT<T>(srcUb, srcUb, src0Reg0, src1Reg0, pregFull, preg2,
                        j * rLengthWithPadding + repeatTimes1 * 2 * sregLower,
                        j * rLengthWithPadding + rHeadLength + repeatTimes1 * 2 * sregLower);
                    Muls(dstReg0, src1Reg0, k2Rec, preg2);
                    Axpy(dstReg0, src0Reg0, k2Rec, pregFull);

                    LoadDataWithT<T>(
                        srcUb, src0Reg1, pregFull, j * rLengthWithPadding + repeatTimes1 * 2 * sregLower + sregLower);
                    Muls(dstReg1, src0Reg1, k2Rec, pregFull);

                    Add(dstReg, dstReg0, dstReg1, pregFull);
                    ReduceSum(dstReg, dstReg, pregFull);
                    DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                        (workUbOrigin + repeatTimes1 + i), dstReg, pregOne);
                }
            }
            for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                __local_mem__ float* workUbOrigin = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);
                // Processes the remaining data of the entire block.
                for (uint16_t i = 0; i < repeatTimes3; i++) {
                    LoadDataWithT<T>(srcUb, src0Reg0, pregFull, j * rLengthWithPadding + mVL + 2 * i * sregLower);
                    LoadDataWithT<T>(srcUb, src0Reg1, pregFull, j * rLengthWithPadding + mVL + (2 * i + 1) * sregLower);
                    Muls(dstReg0, src0Reg0, k2Rec, pregFull);
                    Muls(dstReg1, src0Reg1, k2Rec, pregFull);
                    Add(dstReg, dstReg0, dstReg1, pregFull);
                    ReduceSum(dstReg, dstReg, pregFull);

                    DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                        (workUbOrigin + repeatTimes1 + repeatTimes2 + i), dstReg, pregOne);
                }
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                __local_mem__ float* workUbOrigin = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);
                DataCopy(v0, workUbOrigin);
                DataCopy(v1, workUbOrigin + sregLower);
                // 0~1=>0
                Add(v0, v0, v1, pregFull);
                // reduce
                ReduceSum(dstReg, v0, pregLastCount);
                Muls(meanReg, dstReg, k2RRec, pregOne);
                // save mean
                DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((meanUb + j), meanReg, pregOne);
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            if constexpr (isOutputVariance) {
                for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                    uint32_t mTmp = m;
                    __local_mem__ float* workUbOrigin = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);
                    DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanReg, meanUb + j);
                    // tail block add to main block
                    for (uint16_t i = 0; i < repeatTimes1; i++) {
                        MicroAPI::MaskReg preg = MicroAPI::UpdateMask<float>(mTmp);
                        LoadDataWithT<T>(srcUb, srcUb, src0Reg0, src1Reg0, pregFull, preg,
                            j * rLengthWithPadding + (2 * i) * sregLower,
                            j * rLengthWithPadding + rHeadLength + (2 * i) * sregLower);

                        Sub(src0Reg0, src0Reg0, meanReg, pregFull);
                        Sub(src1Reg0, src1Reg0, meanReg, preg);

                        Mul(src0Reg0, src0Reg0, src0Reg0, pregFull);
                        Mul(src1Reg0, src1Reg0, src1Reg0, pregFull);
                        Muls(dstReg0, src1Reg0, k2Rec, pregFull);
                        Axpy(dstReg0, src0Reg0, k2Rec, pregFull);

                        preg = MicroAPI::UpdateMask<float>(mTmp);
                        LoadDataWithT<T>(srcUb, srcUb, src0Reg1, src1Reg1, pregFull, preg,
                            j * rLengthWithPadding + (2 * i + 1) * sregLower,
                            j * rLengthWithPadding + rHeadLength + (2 * i + 1) * sregLower);

                        Sub(src0Reg1, src0Reg1, meanReg, pregFull);
                        Sub(src1Reg1, src1Reg1, meanReg, preg);

                        Mul(src0Reg1, src0Reg1, src0Reg1, pregFull);
                        Mul(src1Reg1, src1Reg1, src1Reg1, pregFull);
                        Muls(dstReg1, src1Reg1, k2Rec, pregFull);
                        Axpy(dstReg1, src0Reg1, k2Rec, pregFull);

                        Add(dstReg, dstReg0, dstReg1, pregFull);
                        ReduceSum(dstReg, dstReg, pregFull);
                        DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                            (workUbOrigin + i), dstReg, pregOne);
                    }
                }
                for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                    __local_mem__ float* workUbOrigin = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);
                    DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanReg, meanUb + j);
                    for (uint16_t i = 0; i < repeatTimes2; i++) {
                        LoadDataWithT<T>(srcUb, srcUb, src0Reg0, src1Reg0, pregFull, preg2,
                            j * rLengthWithPadding + repeatTimes1 * 2 * sregLower,
                            j * rLengthWithPadding + rHeadLength + repeatTimes1 * 2 * sregLower);

                        Sub(src0Reg0, src0Reg0, meanReg, pregFull);
                        Sub(src1Reg0, src1Reg0, meanReg, pregFull);

                        Mul(src0Reg0, src0Reg0, src0Reg0, pregFull);
                        Mul(src1Reg0, src1Reg0, src1Reg0, pregFull);
                        Muls(dstReg0, src1Reg0, k2Rec, preg2);
                        Axpy(dstReg0, src0Reg0, k2Rec, pregFull);

                        LoadDataWithT<T>(srcUb, src0Reg1, pregFull,
                            j * rLengthWithPadding + repeatTimes1 * 2 * sregLower + sregLower);

                        Sub(src0Reg1, src0Reg1, meanReg, pregFull);

                        Mul(src0Reg1, src0Reg1, src0Reg1, pregFull);

                        Muls(dstReg1, src0Reg1, k2Rec, pregFull);

                        Add(dstReg, dstReg0, dstReg1, pregFull);
                        ReduceSum(dstReg, dstReg, pregFull);
                        DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                            (workUbOrigin + repeatTimes1 + i), dstReg, pregOne);
                    }
                }
                for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                    __local_mem__ float* workUbOrigin = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);
                    DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanReg, meanUb + j);
                    // Processes the remaining data of the entire block.
                    for (uint16_t i = 0; i < repeatTimes3; i++) {
                        LoadDataWithT<T>(srcUb, src0Reg0, pregFull, j * rLengthWithPadding + mVL + (2 * i) * sregLower);
                        LoadDataWithT<T>(
                            srcUb, src0Reg1, pregFull, j * rLengthWithPadding + mVL + (2 * i + 1) * sregLower);

                        Sub(src0Reg0, src0Reg0, meanReg, pregFull);
                        Mul(src0Reg0, src0Reg0, src0Reg0, pregFull);
                        Muls(dstReg0, src0Reg0, k2Rec, pregFull);
                        Sub(src0Reg1, src0Reg1, meanReg, pregFull);
                        Mul(src0Reg1, src0Reg1, src0Reg1, pregFull);
                        Muls(dstReg1, src0Reg1, k2Rec, pregFull);

                        Add(dstReg, dstReg0, dstReg1, pregFull);

                        ReduceSum(dstReg, dstReg, pregFull);
                        DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                            (workUbOrigin + repeatTimes1 + repeatTimes2 + i), dstReg, pregOne);
                    }
                }
                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
                for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                    __local_mem__ float* workUbOrigin = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);
                    DataCopy(v0, workUbOrigin);
                    DataCopy(v1, workUbOrigin + sregLower);
                    // 0~1=>0
                    Add(v0, v0, v1, pregFull);
                    // reduce
                    ReduceSum(dstReg, v0, pregLastCount);
                    Muls(dstReg, dstReg, k2RRec, pregOne);
                    // save variance
                    DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((varianceUb + j), dstReg, pregOne);
                }
            }
        }
    } else if (halfAddTimes == 4) {
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> v0, v1, v2, v3, v4, v5, v6, v7;
            MicroAPI::RegTensor<float> src0Reg0;
            MicroAPI::RegTensor<float> src1Reg0;
            MicroAPI::RegTensor<float> dstReg0;
            MicroAPI::RegTensor<float> src0Reg1;
            MicroAPI::RegTensor<float> src1Reg1;
            MicroAPI::RegTensor<float> dstReg1;
            MicroAPI::RegTensor<float> dstReg;
            MicroAPI::RegTensor<float> meanReg;

            MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg pregOne = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            MicroAPI::MaskReg pregLastCount = MicroAPI::UpdateMask<float>(lastCountTmp);
            count = count2;
            MicroAPI::MaskReg preg2 = MicroAPI::UpdateMask<float>(count);

            for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                uint32_t mTmp = m;
                __local_mem__ float* workUbOrigin = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);
                // tail block add to main block
                for (uint16_t i = 0; i < repeatTimes1; i++) {
                    MicroAPI::MaskReg preg = MicroAPI::UpdateMask<float>(mTmp);
                    LoadDataWithT<T>(srcUb, srcUb, src0Reg0, src1Reg0, pregFull, preg,
                        j * rLengthWithPadding + (2 * i) * sregLower,
                        j * rLengthWithPadding + rHeadLength + (2 * i) * sregLower);
                    Muls(dstReg0, src1Reg0, k2Rec, preg);
                    Axpy(dstReg0, src0Reg0, k2Rec, pregFull);
                    preg = MicroAPI::UpdateMask<float>(mTmp);
                    LoadDataWithT<T>(srcUb, srcUb, src0Reg1, src1Reg1, pregFull, preg,
                        j * rLengthWithPadding + (2 * i + 1) * sregLower,
                        j * rLengthWithPadding + rHeadLength + (2 * i + 1) * sregLower);
                    Muls(dstReg1, src1Reg1, k2Rec, preg);
                    Axpy(dstReg1, src0Reg1, k2Rec, pregFull);
                    Add(dstReg, dstReg0, dstReg1, pregFull);
                    ReduceSum(dstReg, dstReg, pregFull);
                    DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((workUbOrigin + i), dstReg, pregOne);
                }
            }
            for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                __local_mem__ float* workUbOrigin = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);
                for (uint16_t i = 0; i < repeatTimes2; i++) {
                    LoadDataWithT<T>(srcUb, srcUb, src0Reg0, src1Reg0, pregFull, preg2,
                        j * rLengthWithPadding + repeatTimes1 * 2 * sregLower,
                        j * rLengthWithPadding + rHeadLength + repeatTimes1 * 2 * sregLower);
                    Muls(dstReg0, src1Reg0, k2Rec, preg2);
                    Axpy(dstReg0, src0Reg0, k2Rec, pregFull);

                    LoadDataWithT<T>(
                        srcUb, src0Reg1, pregFull, j * rLengthWithPadding + repeatTimes1 * 2 * sregLower + sregLower);
                    Muls(dstReg1, src0Reg1, k2Rec, pregFull);

                    Add(dstReg, dstReg0, dstReg1, pregFull);
                    ReduceSum(dstReg, dstReg, pregFull);
                    DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                        (workUbOrigin + repeatTimes1 + i), dstReg, pregOne);
                }
            }
            for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                __local_mem__ float* workUbOrigin = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);
                // Processes the remaining data of the entire block.
                for (uint16_t i = 0; i < repeatTimes3; i++) {
                    LoadDataWithT<T>(srcUb, src0Reg0, pregFull, j * rLengthWithPadding + mVL + 2 * i * sregLower);
                    LoadDataWithT<T>(srcUb, src0Reg1, pregFull, j * rLengthWithPadding + mVL + (2 * i + 1) * sregLower);
                    Muls(dstReg0, src0Reg0, k2Rec, pregFull);
                    Muls(dstReg1, src0Reg1, k2Rec, pregFull);
                    Add(dstReg, dstReg0, dstReg1, pregFull);
                    ReduceSum(dstReg, dstReg, pregFull);
                    DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                        (workUbOrigin + repeatTimes1 + repeatTimes2 + i), dstReg, pregOne);
                }
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                __local_mem__ float* workUbOrigin = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);
                DataCopy(v0, workUbOrigin);
                DataCopy(v1, workUbOrigin + sregLower * 1);
                DataCopy(v2, workUbOrigin + sregLower * 2);
                DataCopy(v3, workUbOrigin + sregLower * 3);
                // 0~4=>0~1
                Add(v0, v0, v2, pregFull);
                Add(v1, v1, v3, pregFull);
                // 0~1=>0
                Add(v0, v0, v1, pregFull);
                // reduce
                ReduceSum(dstReg, v0, pregLastCount);
                Muls(meanReg, dstReg, k2RRec, pregOne);
                // save mean
                DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((meanUb + j), meanReg, pregOne);
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            if constexpr (isOutputVariance) {
                for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                    uint32_t mTmp = m;
                    __local_mem__ float* workUbOrigin = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);
                    DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanReg, meanUb + j);
                    // tail block add to main block
                    for (uint16_t i = 0; i < repeatTimes1; i++) {
                        MicroAPI::MaskReg preg = MicroAPI::UpdateMask<float>(mTmp);
                        LoadDataWithT<T>(srcUb, srcUb, src0Reg0, src1Reg0, pregFull, preg,
                            j * rLengthWithPadding + (2 * i) * sregLower,
                            j * rLengthWithPadding + rHeadLength + (2 * i) * sregLower);

                        Sub(src0Reg0, src0Reg0, meanReg, pregFull);
                        Sub(src1Reg0, src1Reg0, meanReg, preg);

                        Mul(src0Reg0, src0Reg0, src0Reg0, pregFull);
                        Mul(src1Reg0, src1Reg0, src1Reg0, pregFull);
                        Muls(dstReg0, src1Reg0, k2Rec, pregFull);
                        Axpy(dstReg0, src0Reg0, k2Rec, pregFull);

                        preg = MicroAPI::UpdateMask<float>(mTmp);
                        LoadDataWithT<T>(srcUb, srcUb, src0Reg1, src1Reg1, pregFull, preg,
                            j * rLengthWithPadding + (2 * i + 1) * sregLower,
                            j * rLengthWithPadding + rHeadLength + (2 * i + 1) * sregLower);

                        Sub(src0Reg1, src0Reg1, meanReg, pregFull);
                        Sub(src1Reg1, src1Reg1, meanReg, preg);

                        Mul(src0Reg1, src0Reg1, src0Reg1, pregFull);
                        Mul(src1Reg1, src1Reg1, src1Reg1, pregFull);
                        Muls(dstReg1, src1Reg1, k2Rec, pregFull);
                        Axpy(dstReg1, src0Reg1, k2Rec, pregFull);

                        Add(dstReg, dstReg0, dstReg1, pregFull);
                        ReduceSum(dstReg, dstReg, pregFull);
                        DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                            (workUbOrigin + i), dstReg, pregOne);
                    }
                }
                for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                    __local_mem__ float* workUbOrigin = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);
                    DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanReg, meanUb + j);
                    for (uint16_t i = 0; i < repeatTimes2; i++) {
                        LoadDataWithT<T>(srcUb, srcUb, src0Reg0, src1Reg0, pregFull, preg2,
                            j * rLengthWithPadding + repeatTimes1 * 2 * sregLower,
                            j * rLengthWithPadding + rHeadLength + repeatTimes1 * 2 * sregLower);

                        Sub(src0Reg0, src0Reg0, meanReg, pregFull);
                        Sub(src1Reg0, src1Reg0, meanReg, pregFull);

                        Mul(src0Reg0, src0Reg0, src0Reg0, pregFull);
                        Mul(src1Reg0, src1Reg0, src1Reg0, pregFull);
                        Muls(dstReg0, src1Reg0, k2Rec, preg2);
                        Axpy(dstReg0, src0Reg0, k2Rec, pregFull);

                        LoadDataWithT<T>(srcUb, src0Reg1, pregFull,
                            j * rLengthWithPadding + repeatTimes1 * 2 * sregLower + sregLower);

                        Sub(src0Reg1, src0Reg1, meanReg, pregFull);

                        Mul(src0Reg1, src0Reg1, src0Reg1, pregFull);

                        Muls(dstReg1, src0Reg1, k2Rec, pregFull);

                        Add(dstReg, dstReg0, dstReg1, pregFull);
                        ReduceSum(dstReg, dstReg, pregFull);
                        DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                            (workUbOrigin + repeatTimes1 + i), dstReg, pregOne);
                    }
                }
                for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                    __local_mem__ float* workUbOrigin = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);
                    DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanReg, meanUb + j);
                    // Processes the remaining data of the entire block.
                    for (uint16_t i = 0; i < repeatTimes3; i++) {
                        LoadDataWithT<T>(srcUb, src0Reg0, pregFull, j * rLengthWithPadding + mVL + (2 * i) * sregLower);
                        LoadDataWithT<T>(
                            srcUb, src0Reg1, pregFull, j * rLengthWithPadding + mVL + (2 * i + 1) * sregLower);

                        Sub(src0Reg0, src0Reg0, meanReg, pregFull);
                        Mul(src0Reg0, src0Reg0, src0Reg0, pregFull);
                        Muls(dstReg0, src0Reg0, k2Rec, pregFull);
                        Sub(src0Reg1, src0Reg1, meanReg, pregFull);
                        Mul(src0Reg1, src0Reg1, src0Reg1, pregFull);
                        Muls(dstReg1, src0Reg1, k2Rec, pregFull);

                        Add(dstReg, dstReg0, dstReg1, pregFull);

                        ReduceSum(dstReg, dstReg, pregFull);
                        DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                            (workUbOrigin + repeatTimes1 + repeatTimes2 + i), dstReg, pregOne);
                    }
                }
                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
                for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                    __local_mem__ float* workUbOrigin = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);
                    DataCopy(v0, workUbOrigin);
                    DataCopy(v1, workUbOrigin + sregLower * 1);
                    DataCopy(v2, workUbOrigin + sregLower * 2);
                    DataCopy(v3, workUbOrigin + sregLower * 3);
                    // 0~4=>0~1
                    Add(v0, v0, v2, pregFull);
                    Add(v1, v1, v3, pregFull);
                    // 0~1=>0
                    Add(v0, v0, v1, pregFull);
                    // reduce
                    ReduceSum(dstReg, v0, pregLastCount);
                    Muls(dstReg, dstReg, k2RRec, pregOne);
                    // save variance
                    DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((varianceUb + j), dstReg, pregOne);
                }
            }
        }
    } else {
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<float> v0, v1, v2, v3, v4, v5, v6, v7;
            MicroAPI::RegTensor<float> src0Reg0;
            MicroAPI::RegTensor<float> src1Reg0;
            MicroAPI::RegTensor<float> dstReg0;
            MicroAPI::RegTensor<float> src0Reg1;
            MicroAPI::RegTensor<float> src1Reg1;
            MicroAPI::RegTensor<float> dstReg1;
            MicroAPI::RegTensor<float> dstReg;
            MicroAPI::RegTensor<float> meanReg;

            MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg pregOne = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::VL1>();
            MicroAPI::MaskReg pregLastCount = MicroAPI::UpdateMask<float>(lastCountTmp);
            count = count2;
            MicroAPI::MaskReg preg2 = MicroAPI::UpdateMask<float>(count);
            for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                uint32_t mTmp = m;
                __local_mem__ float* workUbOrigin = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);
                // tail block add to main block
                for (uint16_t i = 0; i < repeatTimes1; i++) {
                    MicroAPI::MaskReg preg = MicroAPI::UpdateMask<float>(mTmp);
                    LoadDataWithT<T>(srcUb, srcUb, src0Reg0, src1Reg0, pregFull, preg,
                        j * rLengthWithPadding + (2 * i) * sregLower,
                        j * rLengthWithPadding + rHeadLength + (2 * i) * sregLower);
                    Muls(dstReg0, src1Reg0, k2Rec, preg);
                    Axpy(dstReg0, src0Reg0, k2Rec, pregFull);
                    preg = MicroAPI::UpdateMask<float>(mTmp);
                    LoadDataWithT<T>(srcUb, srcUb, src0Reg1, src1Reg1, pregFull, preg,
                        j * rLengthWithPadding + (2 * i + 1) * sregLower,
                        j * rLengthWithPadding + rHeadLength + (2 * i + 1) * sregLower);
                    Muls(dstReg1, src1Reg1, k2Rec, preg);
                    Axpy(dstReg1, src0Reg1, k2Rec, pregFull);
                    Add(dstReg, dstReg0, dstReg1, pregFull);
                    ReduceSum(dstReg, dstReg, pregFull);
                    DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((workUbOrigin + i), dstReg, pregOne);
                }
            }
            for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                __local_mem__ float* workUbOrigin = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);
                for (uint16_t i = 0; i < repeatTimes2; i++) {
                    LoadDataWithT<T>(srcUb, srcUb, src0Reg0, src1Reg0, pregFull, preg2,
                        j * rLengthWithPadding + repeatTimes1 * 2 * sregLower,
                        j * rLengthWithPadding + rHeadLength + repeatTimes1 * 2 * sregLower);
                    Muls(dstReg0, src1Reg0, k2Rec, preg2);
                    Axpy(dstReg0, src0Reg0, k2Rec, pregFull);

                    LoadDataWithT<T>(
                        srcUb, src0Reg1, pregFull, j * rLengthWithPadding + repeatTimes1 * 2 * sregLower + sregLower);
                    Muls(dstReg1, src0Reg1, k2Rec, pregFull);

                    Add(dstReg, dstReg0, dstReg1, pregFull);
                    ReduceSum(dstReg, dstReg, pregFull);
                    DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                        (workUbOrigin + repeatTimes1 + i), dstReg, pregOne);
                }
            }
            for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                __local_mem__ float* workUbOrigin = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);
                // Processes the remaining data of the entire block.
                for (uint16_t i = 0; i < repeatTimes3; i++) {
                    LoadDataWithT<T>(srcUb, src0Reg0, pregFull, j * rLengthWithPadding + mVL + 2 * i * sregLower);
                    LoadDataWithT<T>(srcUb, src0Reg1, pregFull, j * rLengthWithPadding + mVL + (2 * i + 1) * sregLower);
                    Muls(dstReg0, src0Reg0, k2Rec, pregFull);
                    Muls(dstReg1, src0Reg1, k2Rec, pregFull);
                    Add(dstReg, dstReg0, dstReg1, pregFull);
                    ReduceSum(dstReg, dstReg, pregFull);
                    DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                        (workUbOrigin + repeatTimes1 + repeatTimes2 + i), dstReg, pregOne);
                }
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                // Fold the tmpbuffer in half.
                uint16_t currentHalfAddTimes = halfAddTimes;
                __local_mem__ float* workUbOrigin = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);
                for (uint16_t k = 0; k < halfAddRepeatTimes; k++) {
                    currentHalfAddTimes = currentHalfAddTimes / Internal::kLayernormFoldNum; // Fold
                    for (uint16_t i = 0; i < currentHalfAddTimes; i++) {
                        DataCopy(src0Reg0, workUbOrigin + i * sregLower);
                        DataCopy(src1Reg0, workUbOrigin + (currentHalfAddTimes + i) * sregLower);
                        Add(dstReg, src0Reg0, src1Reg0, pregFull);
                        DataCopy(workUbOrigin + i * sregLower, dstReg, pregFull);
                    }
                    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
                }
                // the last reducesum
                DataCopy(src0Reg0, workUbOrigin);
                ReduceSum(dstReg, src0Reg0, pregLastCount);
                Muls(meanReg, dstReg, k2RRec, pregOne);
                // save mean
                DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((meanUb + j), meanReg, pregOne);
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            if constexpr (isOutputVariance) {
                for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                    uint32_t mTmp = m;
                    __local_mem__ float* workUbOrigin = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);
                    DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanReg, meanUb + j);
                    // tail block add to main block
                    for (uint16_t i = 0; i < repeatTimes1; i++) {
                        MicroAPI::MaskReg preg = MicroAPI::UpdateMask<float>(mTmp);
                        LoadDataWithT<T>(srcUb, srcUb, src0Reg0, src1Reg0, pregFull, preg,
                            j * rLengthWithPadding + (2 * i) * sregLower,
                            j * rLengthWithPadding + rHeadLength + (2 * i) * sregLower);

                        Sub(src0Reg0, src0Reg0, meanReg, pregFull);
                        Sub(src1Reg0, src1Reg0, meanReg, preg);

                        Mul(src0Reg0, src0Reg0, src0Reg0, pregFull);
                        Mul(src1Reg0, src1Reg0, src1Reg0, pregFull);
                        Muls(dstReg0, src1Reg0, k2Rec, pregFull);
                        Axpy(dstReg0, src0Reg0, k2Rec, pregFull);

                        preg = MicroAPI::UpdateMask<float>(mTmp);
                        LoadDataWithT<T>(srcUb, srcUb, src0Reg1, src1Reg1, pregFull, preg,
                            j * rLengthWithPadding + (2 * i + 1) * sregLower,
                            j * rLengthWithPadding + rHeadLength + (2 * i + 1) * sregLower);

                        Sub(src0Reg1, src0Reg1, meanReg, pregFull);
                        Sub(src1Reg1, src1Reg1, meanReg, preg);

                        Mul(src0Reg1, src0Reg1, src0Reg1, pregFull);
                        Mul(src1Reg1, src1Reg1, src1Reg1, pregFull);
                        Muls(dstReg1, src1Reg1, k2Rec, pregFull);
                        Axpy(dstReg1, src0Reg1, k2Rec, pregFull);

                        Add(dstReg, dstReg0, dstReg1, pregFull);
                        ReduceSum(dstReg, dstReg, pregFull);
                        DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                            (workUbOrigin + i), dstReg, pregOne);
                    }
                }
                for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                    __local_mem__ float* workUbOrigin = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);
                    DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanReg, meanUb + j);
                    for (uint16_t i = 0; i < repeatTimes2; i++) {
                        LoadDataWithT<T>(srcUb, srcUb, src0Reg0, src1Reg0, pregFull, preg2,
                            j * rLengthWithPadding + repeatTimes1 * 2 * sregLower,
                            j * rLengthWithPadding + rHeadLength + repeatTimes1 * 2 * sregLower);

                        Sub(src0Reg0, src0Reg0, meanReg, pregFull);
                        Sub(src1Reg0, src1Reg0, meanReg, pregFull);

                        Mul(src0Reg0, src0Reg0, src0Reg0, pregFull);
                        Mul(src1Reg0, src1Reg0, src1Reg0, pregFull);
                        Muls(dstReg0, src1Reg0, k2Rec, preg2);
                        Axpy(dstReg0, src0Reg0, k2Rec, pregFull);

                        LoadDataWithT<T>(srcUb, src0Reg1, pregFull,
                            j * rLengthWithPadding + repeatTimes1 * 2 * sregLower + sregLower);

                        Sub(src0Reg1, src0Reg1, meanReg, pregFull);

                        Mul(src0Reg1, src0Reg1, src0Reg1, pregFull);

                        Muls(dstReg1, src0Reg1, k2Rec, pregFull);

                        Add(dstReg, dstReg0, dstReg1, pregFull);
                        ReduceSum(dstReg, dstReg, pregFull);
                        DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                            (workUbOrigin + repeatTimes1 + i), dstReg, pregOne);
                    }
                }
                for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                    __local_mem__ float* workUbOrigin = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);
                    DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanReg, meanUb + j);
                    // Processes the remaining data of the entire block.
                    for (uint16_t i = 0; i < repeatTimes3; i++) {
                        LoadDataWithT<T>(srcUb, src0Reg0, pregFull, j * rLengthWithPadding + mVL + (2 * i) * sregLower);
                        LoadDataWithT<T>(
                            srcUb, src0Reg1, pregFull, j * rLengthWithPadding + mVL + (2 * i + 1) * sregLower);

                        Sub(src0Reg0, src0Reg0, meanReg, pregFull);
                        Mul(src0Reg0, src0Reg0, src0Reg0, pregFull);
                        Muls(dstReg0, src0Reg0, k2Rec, pregFull);
                        Sub(src0Reg1, src0Reg1, meanReg, pregFull);
                        Mul(src0Reg1, src0Reg1, src0Reg1, pregFull);
                        Muls(dstReg1, src0Reg1, k2Rec, pregFull);

                        Add(dstReg, dstReg0, dstReg1, pregFull);

                        ReduceSum(dstReg, dstReg, pregFull);
                        DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                            (workUbOrigin + repeatTimes1 + repeatTimes2 + i), dstReg, pregOne);
                    }
                }
                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
                for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                    // Fold the tmpbuffer in half.
                    uint16_t currentHalfAddTimes = halfAddTimes;
                    __local_mem__ float* workUbOrigin = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);
                    for (uint16_t k = 0; k < halfAddRepeatTimes; k++) {
                        currentHalfAddTimes = currentHalfAddTimes / Internal::kLayernormFoldNum; // Fold
                        for (uint16_t i = 0; i < currentHalfAddTimes; i++) {
                            DataCopy(src0Reg0, workUbOrigin + i * sregLower);
                            DataCopy(src1Reg0, workUbOrigin + (currentHalfAddTimes + i) * sregLower);
                            Add(dstReg, src0Reg0, src1Reg0, pregFull);
                            DataCopy(workUbOrigin + i * sregLower, dstReg, pregFull);
                        }
                        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
                    }
                    // the last reducesum
                    DataCopy(v0, workUbOrigin);
                    ReduceSum(dstReg, v0, pregLastCount);
                    Muls(dstReg, dstReg, k2RRec, pregOne);
                    // save variance
                    DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((varianceUb + j), dstReg, pregOne);
                }
            }
        }
    }
}

template <typename U, typename T, bool isReuseSource = false, const LayerNormConfig& config = LNCFG_NORM>
__aicore__ inline void LayerNormImpl(const LocalTensor<T>& output, const LocalTensor<float>& outputMean,
    const LocalTensor<float>& outputRstd, const LocalTensor<T>& inputX, const LocalTensor<U>& gamma,
    const LocalTensor<U>& beta, const float epsilon, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const LayerNormPara& para, const LayerNormSeparateTiling& tiling)
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

    LocalTensor<float> workLocalBegin = sharedTmpBuffer.ReinterpretCast<float>();
    LocalTensor<float> varianceLocal = workLocalBegin;

    uint32_t sregLower = (uint32_t)(GetVecLen() / sizeof(float)); // 64;
    uint32_t tempOffset = CeilDivision(para.aLength, 8) * 8;
    LocalTensor<float> workLocal = workLocalBegin[tempOffset];

    if (tiling.rLength <= sregLower) {
        ComputeMeanVariance64((__local_mem__ float*)outputMean.GetPhyAddr(),
            (__local_mem__ float*)varianceLocal.GetPhyAddr(), (__local_mem__ T*)inputX.GetPhyAddr(), para.aLength,
            tiling.rLength, para.rLengthWithPadding, tiling.k2Rec, tiling.k2RRec, tiling.k2RRec);
    } else if (tiling.rLength <= sregLower * 2) {
        ComputeMeanVariance128((__local_mem__ float*)outputMean.GetPhyAddr(),
            (__local_mem__ float*)varianceLocal.GetPhyAddr(), (__local_mem__ T*)inputX.GetPhyAddr(), para.aLength,
            tiling.rLength, para.rLengthWithPadding, tiling.k2Rec, tiling.k2RRec);
    } else {
        ComputeMeanVarianceUseY((__local_mem__ float*)outputMean.GetPhyAddr(),
            (__local_mem__ float*)varianceLocal.GetPhyAddr(), (__local_mem__ T*)inputX.GetPhyAddr(),
            (__local_mem__ T*)output.GetPhyAddr(), tiling.oneTmpSize, para.aLength, tiling.rLength,
            para.rLengthWithPadding, tiling.rHeadLength, tiling.k2Rec, tiling.k2RRec);
    }
    NormalizePara nlPara{para.aLength, tiling.rLength, para.rLengthWithPadding};
    if constexpr (!config.isNoBeta && !config.isNoGamma) {
        NormalizeImpl<U, T, false, NLCFG_NORM>(
            output, outputRstd, outputMean, varianceLocal, inputX, gamma, beta, sharedTmpBuffer, epsilon, nlPara);
    } else if constexpr (!config.isNoBeta && config.isNoGamma) {
        NormalizeImpl<U, T, false, NLCFG_NOGAMMA>(
            output, outputRstd, outputMean, varianceLocal, inputX, gamma, beta, sharedTmpBuffer, epsilon, nlPara);
    } else if constexpr (config.isNoBeta && !config.isNoGamma) {
        NormalizeImpl<U, T, false, NLCFG_NOBETA>(
            output, outputRstd, outputMean, varianceLocal, inputX, gamma, beta, sharedTmpBuffer, epsilon, nlPara);
    } else if constexpr (config.isNoBeta && config.isNoGamma) {
        NormalizeImpl<U, T, false, NLCFG_NOOPT>(
            output, outputRstd, outputMean, varianceLocal, inputX, gamma, beta, sharedTmpBuffer, epsilon, nlPara);
    }
}

template <typename U, typename T, bool isReuseSource = false, const LayerNormConfig& config = LNCFG_NORM>
__aicore__ inline void LayerNormImpl(const LocalTensor<T>& output, const LocalTensor<float>& outputMean,
    const LocalTensor<float>& outputRstd, const LocalTensor<T>& inputX, const LocalTensor<U>& gamma,
    const LocalTensor<U>& beta, const float epsilon, const LayerNormPara& para, const LayerNormSeparateTiling& tiling)
{
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });

    LayerNormImpl<U, T, isReuseSource, config>(
        output, outputMean, outputRstd, inputX, gamma, beta, epsilon, sharedTmpBuffer, para, tiling);
}

template <typename T, const WelfordUpdateConfig& config = WFUPDATE_DEFAULT_CFG>
__aicore__ inline void WelfordUpdateImplForB16(__local_mem__ float* outMean, __local_mem__ float* outVar,
    __local_mem__ T* src, __local_mem__ float* inMean, __local_mem__ float* inVar, const WelfordUpdateParam& para)
{
    constexpr uint16_t sregLowerB32 = static_cast<uint16_t>(GetVecLen() / sizeof(float)); // 64
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg preg;

        MicroAPI::RegTensor<T> b16vreg;
        MicroAPI::RegTensor<T> vreg1;
        MicroAPI::RegTensor<T> vreg2;
        MicroAPI::RegTensor<float> f32vreg;
        MicroAPI::RegTensor<float> tmpVreg;
        MicroAPI::RegTensor<float> srcVreg;
        MicroAPI::RegTensor<float> meanVreg;
        MicroAPI::RegTensor<float> varVreg;
        MicroAPI::RegTensor<float> outMeanreg;
        MicroAPI::RegTensor<float> outVarreg;

        MicroAPI::RegTensor<uint16_t> zeroReg;
        uint32_t K = para.abComputeLength;
        uint32_t sregLower = static_cast<uint32_t>(Internal::LAYERNORM_B16_VF_LEN);

        if constexpr (config.isInplace) {
            uint32_t inPlaceLength = AlignUp(para.abLength - para.abComputeLength, 8);
            uint16_t repeatInplace = static_cast<uint16_t>(CeilDivision(inPlaceLength, Internal::LAYERNORM_B32_VF_LEN));
            uint32_t dstOffset = para.abLength - inPlaceLength;
            for (uint16_t i = 0; i < 1; ++i) {
                uint32_t sreg = inPlaceLength;
                uint32_t rowOffset = i * para.abLength;
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

        MicroAPI::Duplicate(zeroReg, (uint16_t)0x0000);
        uint16_t repeat = static_cast<uint16_t>(CeilDivision(K, sregLower));
        for (uint16_t i = 0; i < 1; ++i) {
            uint32_t rowOffset = i * para.abLength;
            uint32_t sreg = static_cast<uint32_t>(K);
            for (uint16_t j = 0; j < static_cast<uint16_t>(repeat); ++j) {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(b16vreg, src + rowOffset + j * sregLower);
                MicroAPI::Interleave<uint16_t>((MicroAPI::RegTensor<uint16_t>&)vreg1,
                    (MicroAPI::RegTensor<uint16_t>&)vreg2, (MicroAPI::RegTensor<uint16_t>&)b16vreg,
                    (MicroAPI::RegTensor<uint16_t>&)zeroReg);

                // B16 is calculated in two parts. Each part is 64 elements.
                preg = MicroAPI::UpdateMask<uint32_t>(sreg);
                uint32_t firstOffset = rowOffset + (2 * j) * sregLowerB32;
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(meanVreg, inMean + firstOffset);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(varVreg, inVar + firstOffset);
                MicroAPI::Cast<float, T, layoutZMrgZ>(srcVreg, vreg1, preg);
                MicroAPI::Sub(tmpVreg, srcVreg, meanVreg, preg);
                MicroAPI::Muls(outMeanreg, tmpVreg, static_cast<float>(para.nRec), preg);
                MicroAPI::Add(outMeanreg, outMeanreg, meanVreg, preg);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(outMean + firstOffset, outMeanreg, preg);

                MicroAPI::Sub(f32vreg, srcVreg, outMeanreg, preg);
                MicroAPI::Mul(f32vreg, tmpVreg, f32vreg, preg);
                MicroAPI::Add(outVarreg, f32vreg, varVreg, preg);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(outVar + firstOffset, outVarreg, preg);

                // The back half
                preg = MicroAPI::UpdateMask<uint32_t>(sreg);
                uint32_t secondOffset = rowOffset + (2 * j + 1) * sregLowerB32;
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(meanVreg, inMean + secondOffset);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(varVreg, inVar + secondOffset);
                MicroAPI::Cast<float, T, layoutZMrgZ>(srcVreg, vreg2, preg);
                MicroAPI::Sub(tmpVreg, srcVreg, meanVreg, preg);
                MicroAPI::Muls(outMeanreg, tmpVreg, static_cast<float>(para.nRec), preg);
                MicroAPI::Add(outMeanreg, outMeanreg, meanVreg, preg);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(outMean + secondOffset, outMeanreg, preg);

                MicroAPI::Sub(f32vreg, srcVreg, outMeanreg, preg);
                MicroAPI::Mul(f32vreg, tmpVreg, f32vreg, preg);
                MicroAPI::Add(outVarreg, f32vreg, varVreg, preg);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(outVar + secondOffset, outVarreg, preg);
            }
        }
    }
}

template <typename T, const WelfordUpdateConfig& config = WFUPDATE_DEFAULT_CFG>
__aicore__ inline void WelfordUpdateImplForB32(__local_mem__ float* outMean, __local_mem__ float* outVar,
    __local_mem__ T* src, __local_mem__ float* inMean, __local_mem__ float* inVar, const WelfordUpdateParam& para)
{
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg preg;
        MicroAPI::RegTensor<T> srcVreg;
        MicroAPI::RegTensor<float> f32vreg;
        MicroAPI::RegTensor<float> tmpVreg;

        MicroAPI::RegTensor<float> meanVreg;
        MicroAPI::RegTensor<float> varVreg;
        MicroAPI::RegTensor<float> outMeanreg;
        MicroAPI::RegTensor<float> outVarreg;

        uint32_t K = para.abComputeLength;
        uint32_t sregLower = (uint32_t)Internal::LAYERNORM_B32_VF_LEN;
        if constexpr (config.isInplace) {
            uint32_t inPlaceLength = AlignUp(para.abLength - para.abComputeLength, 8);
            uint16_t repeatInplace = static_cast<uint16_t>(CeilDivision(inPlaceLength, Internal::LAYERNORM_B32_VF_LEN));
            uint32_t dstOffset = para.abLength - inPlaceLength;
            for (uint16_t i = 0; i < 1; ++i) {
                uint32_t rowOffset = i * para.abLength;
                uint32_t sreg = inPlaceLength;
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

        uint16_t repeat = static_cast<uint16_t>(CeilDivision(K, sregLower));
        for (uint16_t i = 0; i < 1; ++i) {
            uint32_t rowOffset = i * para.abLength;
            uint32_t sreg = static_cast<uint32_t>(K);
            for (uint16_t j = 0; j < static_cast<uint16_t>(repeat); ++j) {
                preg = MicroAPI::UpdateMask<uint32_t>(sreg);
                uint32_t offset = rowOffset + j * sregLower;
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_NORM>(srcVreg, src + offset);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(meanVreg, inMean + offset);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(varVreg, inVar + offset);

                MicroAPI::Sub(tmpVreg, srcVreg, meanVreg, preg);
                MicroAPI::Muls(outMeanreg, tmpVreg, static_cast<float>(para.nRec), preg);
                MicroAPI::Add(outMeanreg, outMeanreg, meanVreg, preg);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(outMean + offset, outMeanreg, preg);

                MicroAPI::Sub(f32vreg, srcVreg, outMeanreg, preg);
                MicroAPI::Mul(f32vreg, tmpVreg, f32vreg, preg);
                MicroAPI::Add(outVarreg, f32vreg, varVreg, preg);
                MicroAPI::DataCopy<float, MicroAPI::StoreDist::DIST_NORM_B32>(outVar + offset, outVarreg, preg);
            }
        }
    }
}

template <typename T, typename U = float, bool isReuseSource = false,
    const WelfordUpdateConfig& config = WFUPDATE_DEFAULT_CFG>
__aicore__ inline void WelfordUpdateImpl(const LocalTensor<U>& outputMean, const LocalTensor<U>& outputVariance,
    const LocalTensor<U>& inputMean, const LocalTensor<U>& inputVariance, const LocalTensor<T>& inputX,
    const WelfordUpdateParam& para)
{
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
#endif // AICORE_ADV_API_DETAIL_NORMALIZATION_LAYERNORM_LAYERNORM_C310_IMPL_H
