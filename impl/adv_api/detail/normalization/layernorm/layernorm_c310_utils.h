/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file layernorm_c310_utils.h
 * \brief
 */
#ifndef IMPL_NORMALIZATION_LAYERNORM_C310_UTILS_H
#define IMPL_NORMALIZATION_LAYERNORM_C310_UTILS_H

#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"
#include "kernel_tiling/kernel_tiling.h"

namespace AscendC {
namespace Internal {
const uint16_t kLayernormFoldNum = 2;
constexpr uint32_t LAYERNORM_B16_VF_LEN = GetVecLen() / sizeof(uint16_t);
constexpr uint32_t LAYERNORM_B32_VF_LEN = GetVecLen() / sizeof(uint32_t);
} // namespace Internal

template <typename T>
__simd_callee__ inline void LoadDataWithT(__local_mem__ T* src0, __local_mem__ T* src1, MicroAPI::RegTensor<float>& dstReg0,
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
__simd_callee__ inline void LoadDataWithT(
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

// Helper function for the first loop in ComputeMeanUseY
template <typename T>
__aicore__ inline void ComputeMeanLoop1(__local_mem__ T* const srcUb, __local_mem__ T* const workUbYOrigin,
    MicroAPI::MaskReg& pregFull, MicroAPI::MaskReg& pregOne, const uint32_t aLength, const uint32_t rLengthWithPadding,
    const uint32_t rHeadLength, const uint32_t m, const uint16_t repeatTimes1, const float k2Rec,
    const uint16_t sregLower, MicroAPI::RegTensor<float>& src0Reg0, MicroAPI::RegTensor<float>& src1Reg0,
    MicroAPI::RegTensor<float>& src0Reg1, MicroAPI::RegTensor<float>& src1Reg1, MicroAPI::RegTensor<float>& dstReg0,
    MicroAPI::RegTensor<float>& dstReg1, MicroAPI::RegTensor<float>& dstReg)
{
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
}

// Helper function for the second loop in ComputeMeanUseY
template <typename T>
__aicore__ inline void ComputeMeanLoop2(__local_mem__ T* const srcUb, __local_mem__ T* const workUbYOrigin,
    MicroAPI::MaskReg& pregFull, MicroAPI::MaskReg& pregOne, MicroAPI::MaskReg& preg2, const uint32_t aLength,
    const uint32_t rLengthWithPadding, const uint32_t rHeadLength, const uint16_t repeatTimes1,
    const uint16_t repeatTimes2, const float k2Rec, const uint16_t sregLower, MicroAPI::RegTensor<float>& src0Reg0,
    MicroAPI::RegTensor<float>& src1Reg0, MicroAPI::RegTensor<float>& src0Reg1, MicroAPI::RegTensor<float>& dstReg0,
    MicroAPI::RegTensor<float>& dstReg1, MicroAPI::RegTensor<float>& dstReg)
{
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
}

// Helper function for the third loop in ComputeMeanUseY
template <typename T>
__aicore__ inline void ComputeMeanLoop3(__local_mem__ T* const srcUb, __local_mem__ T* const workUbYOrigin,
    MicroAPI::MaskReg& pregFull, MicroAPI::MaskReg& pregOne, const uint32_t aLength, const uint32_t rLengthWithPadding,
    const uint16_t repeatTimes1, const uint16_t repeatTimes2, const uint16_t repeatTimes3, const uint32_t mVL,
    const float k2Rec, const uint16_t sregLower, MicroAPI::RegTensor<float>& src0Reg0,
    MicroAPI::RegTensor<float>& src0Reg1, MicroAPI::RegTensor<float>& dstReg0, MicroAPI::RegTensor<float>& dstReg1,
    MicroAPI::RegTensor<float>& dstReg)
{
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
}

template <typename T>
__aicore__ inline void ComputeMeanUseY(__local_mem__ T* const srcUb, __local_mem__ T* const workUbYOrigin,
    MicroAPI::MaskReg& pregFull, MicroAPI::MaskReg& pregOne, MicroAPI::MaskReg& pregLastCount, MicroAPI::MaskReg& preg2,
    const uint32_t aLength, const uint32_t rLengthWithPadding, const uint32_t rHeadLength, const uint32_t m,
    const uint16_t repeatTimes1, const uint16_t repeatTimes2, const uint16_t repeatTimes3, const uint32_t mVL,
    const float k2Rec, const uint16_t sregLower)
{
    MicroAPI::RegTensor<float> src0Reg0;
    MicroAPI::RegTensor<float> src1Reg0;
    MicroAPI::RegTensor<float> src0Reg1;
    MicroAPI::RegTensor<float> src1Reg1;
    MicroAPI::RegTensor<float> dstReg0;
    MicroAPI::RegTensor<float> dstReg1;
    MicroAPI::RegTensor<float> dstReg;

    ComputeMeanLoop1<T>(srcUb, workUbYOrigin, pregFull, pregOne, aLength, rLengthWithPadding, rHeadLength, m,
        repeatTimes1, k2Rec, sregLower, src0Reg0, src1Reg0, src0Reg1, src1Reg1, dstReg0, dstReg1, dstReg);
    ComputeMeanLoop2<T>(srcUb, workUbYOrigin, pregFull, pregOne, preg2, aLength, rLengthWithPadding, rHeadLength,
        repeatTimes1, repeatTimes2, k2Rec, sregLower, src0Reg0, src1Reg0, src0Reg1, dstReg0, dstReg1, dstReg);
    ComputeMeanLoop3<T>(srcUb, workUbYOrigin, pregFull, pregOne, aLength, rLengthWithPadding, repeatTimes1,
        repeatTimes2, repeatTimes3, mVL, k2Rec, sregLower, src0Reg0, src0Reg1, dstReg0, dstReg1, dstReg);
}

// Helper function for the first loop in ComputeVarianceUseY
template <typename T>
__aicore__ inline void ComputeVarianceLoop1(__local_mem__ T* const srcUb, __local_mem__ T* const workUbYOrigin,
    __local_mem__ float* const meanUb, MicroAPI::MaskReg& pregFull, MicroAPI::MaskReg& pregOne, const uint32_t aLength,
    const uint32_t rLengthWithPadding, const uint32_t rHeadLength, const uint32_t m, const uint16_t repeatTimes1,
    const float k2Rec, const uint16_t sregLower, MicroAPI::RegTensor<float>& meanReg,
    MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& src0Reg0, MicroAPI::RegTensor<float>& src1Reg0,
    MicroAPI::RegTensor<float>& src0Reg1, MicroAPI::RegTensor<float>& src1Reg1, MicroAPI::RegTensor<float>& dstReg0,
    MicroAPI::RegTensor<float>& dstReg1)
{
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
            DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((workUbOrigin + i), dstReg, pregOne);
        }
    }
}

// Helper function for the second loop in ComputeVarianceUseY
template <typename T>
__aicore__ inline void ComputeVarianceLoop2(__local_mem__ T* const srcUb, __local_mem__ T* const workUbYOrigin,
    __local_mem__ float* const meanUb, MicroAPI::MaskReg& pregFull, MicroAPI::MaskReg& pregOne,
    MicroAPI::MaskReg& preg2, const uint32_t aLength, const uint32_t rLengthWithPadding, const uint32_t rHeadLength,
    const uint16_t repeatTimes1, const uint16_t repeatTimes2, const float k2Rec, const uint16_t sregLower,
    MicroAPI::RegTensor<float>& meanReg, MicroAPI::RegTensor<float>& src0Reg0, MicroAPI::RegTensor<float>& src1Reg0,
    MicroAPI::RegTensor<float>& src0Reg1, MicroAPI::RegTensor<float>& dstReg0, MicroAPI::RegTensor<float>& dstReg1,
    MicroAPI::RegTensor<float>& dstReg)
{
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

            LoadDataWithT<T>(
                srcUb, src0Reg1, pregFull, j * rLengthWithPadding + repeatTimes1 * 2 * sregLower + sregLower);

            Sub(src0Reg1, src0Reg1, meanReg, pregFull);

            Mul(src0Reg1, src0Reg1, src0Reg1, pregFull);

            Muls(dstReg1, src0Reg1, k2Rec, pregFull);

            Add(dstReg, dstReg0, dstReg1, pregFull);
            ReduceSum(dstReg, dstReg, pregFull);
            DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                (workUbOrigin + repeatTimes1 + i), dstReg, pregOne);
        }
    }
}

// Helper function for the third loop in ComputeVarianceUseY
template <typename T>
__aicore__ inline void ComputeVarianceLoop3(__local_mem__ T* const srcUb, __local_mem__ T* const workUbYOrigin,
    __local_mem__ float* const meanUb, MicroAPI::MaskReg& pregFull, MicroAPI::MaskReg& pregOne, const uint32_t aLength,
    const uint32_t rLengthWithPadding, const uint16_t repeatTimes1, const uint16_t repeatTimes2,
    const uint16_t repeatTimes3, const uint32_t mVL, const float k2Rec, const uint16_t sregLower,
    MicroAPI::RegTensor<float>& meanReg, MicroAPI::RegTensor<float>& src0Reg0, MicroAPI::RegTensor<float>& src0Reg1,
    MicroAPI::RegTensor<float>& dstReg0, MicroAPI::RegTensor<float>& dstReg1, MicroAPI::RegTensor<float>& dstReg)
{
    for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
        __local_mem__ float* workUbOrigin = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);
        DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(meanReg, meanUb + j);
        // Processes the remaining data of the entire block.
        for (uint16_t i = 0; i < repeatTimes3; i++) {
            LoadDataWithT<T>(srcUb, src0Reg0, pregFull, j * rLengthWithPadding + mVL + (2 * i) * sregLower);
            LoadDataWithT<T>(srcUb, src0Reg1, pregFull, j * rLengthWithPadding + mVL + (2 * i + 1) * sregLower);

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
}

template <typename T>
__aicore__ inline void ComputeVarianceUseY(__local_mem__ T* const srcUb, __local_mem__ T* const workUbYOrigin,
    __local_mem__ float* const meanUb, MicroAPI::MaskReg& pregFull, MicroAPI::MaskReg& pregOne,
    MicroAPI::MaskReg& pregLastCount, MicroAPI::MaskReg& preg2, const uint32_t aLength,
    const uint32_t rLengthWithPadding, const uint32_t rHeadLength, const uint32_t m, const uint16_t repeatTimes1,
    const uint16_t repeatTimes2, const uint16_t repeatTimes3, const uint32_t mVL, const float k2Rec,
    const uint16_t sregLower)
{
    MicroAPI::RegTensor<float> meanReg;
    MicroAPI::RegTensor<float> dstReg;
    MicroAPI::RegTensor<float> src0Reg0;
    MicroAPI::RegTensor<float> src1Reg0;
    MicroAPI::RegTensor<float> src0Reg1;
    MicroAPI::RegTensor<float> src1Reg1;
    MicroAPI::RegTensor<float> dstReg0;
    MicroAPI::RegTensor<float> dstReg1;

    ComputeVarianceLoop1<T>(srcUb, workUbYOrigin, meanUb, pregFull, pregOne, aLength, rLengthWithPadding, rHeadLength,
        m, repeatTimes1, k2Rec, sregLower, meanReg, dstReg, src0Reg0, src1Reg0, src0Reg1, src1Reg1, dstReg0, dstReg1);
    ComputeVarianceLoop2<T>(srcUb, workUbYOrigin, meanUb, pregFull, pregOne, preg2, aLength, rLengthWithPadding,
        rHeadLength, repeatTimes1, repeatTimes2, k2Rec, sregLower, meanReg, src0Reg0, src1Reg0, src0Reg1, dstReg0,
        dstReg1, dstReg);
    ComputeVarianceLoop3<T>(srcUb, workUbYOrigin, meanUb, pregFull, pregOne, aLength, rLengthWithPadding, repeatTimes1,
        repeatTimes2, repeatTimes3, mVL, k2Rec, sregLower, meanReg, src0Reg0, src0Reg1, dstReg0, dstReg1, dstReg);
}

// Helper: reduce temporary work buffer into a scalar per row and store to dstUb
template <typename T, uint16_t HalfAddTimes>
__aicore__ inline void ReduceWorkBufferAndStore(__local_mem__ T* const workUbYOrigin, __local_mem__ float* const dstUb,
    MicroAPI::MaskReg& pregFull, MicroAPI::MaskReg& pregOne, MicroAPI::MaskReg& pregLastCount, const uint32_t aLength,
    const uint32_t rLengthWithPadding, const uint16_t halfAddRepeatTimes, const uint32_t lastCount, const float k2RRec,
    const uint16_t sregLower, const uint16_t dynamicHalfAddTimes = 0)
{
    MicroAPI::RegTensor<float> v0, v1, v2, v3;
    MicroAPI::RegTensor<float> s0, s1;
    MicroAPI::RegTensor<float> tmp, outReg;

    for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); ++j) {
        __local_mem__ float* workUb = (__local_mem__ float*)(workUbYOrigin + j * rLengthWithPadding);

        if constexpr (HalfAddTimes == 1) {
            DataCopy(v0, workUb);
            ReduceSum(tmp, v0, pregLastCount);
        } else if constexpr (HalfAddTimes == 2) {
            DataCopy(v0, workUb);
            DataCopy(v1, workUb + sregLower);
            Add(v0, v0, v1, pregFull);
            ReduceSum(tmp, v0, pregLastCount);
        } else if constexpr (HalfAddTimes == 4) {
            DataCopy(v0, workUb);
            DataCopy(v1, workUb + sregLower * 1);
            DataCopy(v2, workUb + sregLower * 2);
            DataCopy(v3, workUb + sregLower * 3);
            Add(v0, v0, v2, pregFull);
            Add(v1, v1, v3, pregFull);
            Add(v0, v0, v1, pregFull);
            ReduceSum(tmp, v0, pregLastCount);
        } else { // general case
            uint16_t cur = dynamicHalfAddTimes;
            for (uint16_t k = 0; k < halfAddRepeatTimes; ++k) {
                cur = cur / Internal::kLayernormFoldNum; // Fold
                for (uint16_t i = 0; i < cur; ++i) {
                    DataCopy(s0, workUb + i * sregLower);
                    DataCopy(s1, workUb + (cur + i) * sregLower);
                    Add(tmp, s0, s1, pregFull);
                    DataCopy(workUb + i * sregLower, tmp, pregFull);
                }
                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            }
            DataCopy(s0, workUb);
            ReduceSum(tmp, s0, pregLastCount);
        }

        Muls(outReg, tmp, k2RRec, pregOne);
        DataCopy<float, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((dstUb + j), outReg, pregOne);
    }
}

} // namespace AscendC
#endif // IMPL_NORMALIZATION_LAYERNORM_C310_UTILS_H