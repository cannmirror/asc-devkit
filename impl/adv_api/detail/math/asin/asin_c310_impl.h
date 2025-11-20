/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file asin_c310_impl.h
 * \brief
 */
#ifndef IMPL_MATH_ASIN_ASIN_C310_IMPL_H
#define IMPL_MATH_ASIN_ASIN_C310_IMPL_H
#include "kernel_tensor.h"
#include "../math_constant_util.h"
#include "../../common/check.h"

namespace AscendC {
namespace Internal {
constexpr MicroAPI::CastTrait ASIN_CAST_TRAIT_NONE = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_NONE};

constexpr MicroAPI::CastTrait ASIN_CAST_TRAIT_FLOOR = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_FLOOR};

constexpr MicroAPI::CastTrait ASIN_CAST_TRAIT_RINT = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

// Calculate Taylor Expansion according to (((k_nx^2 + k_n) * x^2 + k_(n-1)) * x^2 +k_(n-2) ……)*x^2 +k_0)*x.
template <typename T, typename RegT>
__simd_callee__ inline void AsinTaylorComputeInner(RegT& dstReg, RegT& srcReg, MicroAPI::MaskReg& mask)
{
    MicroAPI::Muls(dstReg, dstReg, static_cast<T>(kCOEF[ASIN_TAYLOR_EXPAND_COUNT]), mask);
    MicroAPI::Adds(dstReg, dstReg, static_cast<T>(kCOEF[6]), mask);
    MicroAPI::Mul(dstReg, dstReg, srcReg, mask);
    MicroAPI::Adds(dstReg, dstReg, static_cast<T>(kCOEF[5]), mask);
    MicroAPI::Mul(dstReg, dstReg, srcReg, mask);
    MicroAPI::Adds(dstReg, dstReg, static_cast<T>(kCOEF[4]), mask);
    MicroAPI::Mul(dstReg, dstReg, srcReg, mask);
    MicroAPI::Adds(dstReg, dstReg, static_cast<T>(kCOEF[3]), mask);
    MicroAPI::Mul(dstReg, dstReg, srcReg, mask);
    MicroAPI::Adds(dstReg, dstReg, static_cast<T>(kCOEF[2]), mask);
    MicroAPI::Mul(dstReg, dstReg, srcReg, mask);
    MicroAPI::Adds(dstReg, dstReg, static_cast<T>(kCOEF[1]), mask);
    MicroAPI::Mul(dstReg, dstReg, srcReg, mask);
    MicroAPI::Adds(dstReg, dstReg, static_cast<T>(kCOEF[0]), mask);
}

template <typename T, typename RegT>
__simd_callee__ inline void AsinTaylorCompute(RegT& dstReg, RegT& srcReg, MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<T> tmpReg;
    MicroAPI::Mul(dstReg, srcReg, srcReg, mask);
    MicroAPI::Mul(tmpReg, srcReg, srcReg, mask);
    AsinTaylorComputeInner<T>(dstReg, tmpReg, mask);
    MicroAPI::Mul(dstReg, dstReg, srcReg, mask);
}

// Calculate Taylor Expansion of Asin based on its square value, and set the source to be sqrt(x).
template <typename T, typename RegT>
__simd_callee__ inline void AsinTaylorComputeBySquareValue(RegT& dstReg, RegT& srcReg, MicroAPI::MaskReg& mask)
{
    MicroAPI::Muls(dstReg, srcReg, static_cast<T>(NUM_ONE), mask);
    AsinTaylorComputeInner<T>(dstReg, srcReg, mask);
    // Update src to be sqrt(x).
    MicroAPI::Sqrt(srcReg, srcReg, mask);
    MicroAPI::Mul(dstReg, dstReg, srcReg, mask);
}

template <typename T, typename RegT>
__simd_callee__ inline void CalRes2(RegT& resReg, RegT& srcReg, MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<T> tmpReg;
    MicroAPI::Mul(tmpReg, srcReg, srcReg, mask);
    MicroAPI::Muls(tmpReg, tmpReg, NEG_ONE, mask);
    MicroAPI::Adds(tmpReg, tmpReg, NUM_ONE, mask);
    MicroAPI::Sqrt(tmpReg, tmpReg, mask);
    AsinTaylorCompute<T>(resReg, tmpReg, mask);
    MicroAPI::Muls(resReg, resReg, NEG_ONE, mask);
    MicroAPI::Adds(resReg, resReg, HALF_PI, mask);
}

template <typename T, typename RegT>
__simd_callee__ inline void ProcessBranch(RegT& resReg1, RegT& resReg2, RegT& tmpReg, MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<int32_t> s32Reg;
    MicroAPI::Mins(tmpReg, tmpReg, BOUNDARY, mask);
    MicroAPI::Adds(tmpReg, tmpReg, -BOUNDARY, mask);
    MicroAPI::Cast<int32_t, T, ASIN_CAST_TRAIT_FLOOR>(s32Reg, tmpReg, mask);
    MicroAPI::Cast<T, int32_t, ASIN_CAST_TRAIT_RINT>(tmpReg, s32Reg, mask);
    MicroAPI::Muls(tmpReg, tmpReg, NEG_ONE, mask);
    MicroAPI::Mul(resReg1, resReg1, tmpReg, mask);
    MicroAPI::Muls(tmpReg, tmpReg, NEG_ONE, mask);
    MicroAPI::Adds(tmpReg, tmpReg, NUM_ONE, mask);
    MicroAPI::Mul(resReg2, resReg2, tmpReg, mask);
    MicroAPI::Add(resReg1, resReg1, resReg2, mask);
}

// Calculate the sign of given values.
// Algorithm:
// FP16: sign(x) = 2^(15) * x /(2^(-15) + 2^(15) *|x|)
// FP32: sign(x) = 2^(62) * x /(2^(-62) + 2^(62) *|x|)
template <typename T, typename RegT>
__simd_callee__ inline void GetSign(RegT& dstReg, RegT& srcReg, MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<T> denominatorReg;
    constexpr float FP16_MAX = 32768;                 // 2^15
    constexpr float FP16_MIN = 3.0517578125e-05;      // 2^-15
    constexpr float FP32_MAX = 4611686018427387904;   // 2^62
    constexpr float FP32_MIN = 2.168404344971009e-19; // 2^-62
    constexpr float kFpMax = sizeof(T) == sizeof(float) ? FP32_MAX : FP16_MAX;
    constexpr float kFpMin = sizeof(T) == sizeof(float) ? FP32_MIN : FP16_MIN;
    MicroAPI::Muls(dstReg, srcReg, static_cast<T>(kFpMax), mask);
    MicroAPI::Abs(denominatorReg, dstReg, mask);
    MicroAPI::Adds(denominatorReg, denominatorReg, static_cast<T>(kFpMin), mask);
    MicroAPI::Div(dstReg, dstReg, denominatorReg, mask);
}

// Compute asin values based on input types.
// asin(x) = arcsin(sqrt(1-x^2)) - PI*0.5 when x belongs to (-1, -2^(-0.5))
// asin(x) = the 15th order taylor expansion when x belongs to (-2^(-0.5), 2^(-0.5))
// asin(x) = PI*0.5 - arcsin(sqrt(1-x^2)) when x belongs to (2^(-0.5), 1)
template <typename T, bool convertToAcos = false>
__simd_vf__ inline void AsinComputeVFF32(__local_mem__ T* dstUb, __local_mem__ T* srcUb, uint32_t calSize,
    uint16_t repeatTimes, uint16_t stride)
{
    MicroAPI::MaskReg mask;
    MicroAPI::RegTensor<T> srcReg;
    MicroAPI::RegTensor<T> dstReg;
    MicroAPI::RegTensor<T> resReg1;
    MicroAPI::RegTensor<T> resReg2;
    MicroAPI::RegTensor<T> signReg;
    MicroAPI::RegTensor<T> tmpReg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = MicroAPI::UpdateMask<T>(calSize);
        MicroAPI::DataCopy(srcReg, srcUb + i * stride);
        // Calculate res2 = PI*0.5 - taylor_compute(sqrt(1 - x^2)) -> resReg2.
        CalRes2<T>(resReg2, srcReg, mask);
        // Calculate res1 = taylor_compute(abs(x)) -> dst, abs(x) -> resReg1.
        MicroAPI::Mul(tmpReg, srcReg, srcReg, mask);
        AsinTaylorComputeBySquareValue<T>(resReg1, tmpReg, mask);
        // As NPU are not good at scalar process like CPU for if-else statement, the solution here used for handling above
        // 3 scenarios is to calculate 0/1 choices combining the results on both options.
        // e.g.
        // Step1: Calculate both option results of x, no matter which range it's at.
        // result1(x), result2(x)
        // Step2: Calculate 0/1 choices of both option results of x, no matter which range it's at.
        // choice1(x), choice2(x)
        // Step3: Combine choice result and options results, since at least one choice should be zero.
        // Result = choice1(x) * result1(x) + choice2(x) * result2(x)
        // choice1 = -Floor(min(abs(x), BOUNDARY) - BOUNDARY).
        // choice2 = 1 - choice1
        // res = res1 * choice1 + res2 * choice2
        ProcessBranch<T>(resReg1, resReg2, tmpReg, mask);
        GetSign<T>(signReg, srcReg, mask);
        MicroAPI::Mul(dstReg, resReg1, signReg, mask);
        if constexpr (convertToAcos) {
            // Compute acos values according to formula: arccos(x) = PI*0.5 - arcsin(x).
            MicroAPI::Adds(dstReg, dstReg, static_cast<T>(-HALF_PI), mask);
            MicroAPI::Muls(dstReg, dstReg, static_cast<T>(NEG_ONE), mask);
        }
        MicroAPI::DataCopy(dstUb + i * stride, dstReg, mask);
    }
}

template <typename T, bool convertToAcos = false>
__simd_vf__ inline void AsinComputeVFF16(__local_mem__ T* dstUb, __local_mem__ T* srcUb, uint32_t calSize,
    uint16_t repeatTimes, uint16_t stride)
{
    MicroAPI::MaskReg mask;
    MicroAPI::RegTensor<half> srcReg;
    MicroAPI::RegTensor<half> dstReg;
    MicroAPI::RegTensor<half> halfReg1;
    MicroAPI::RegTensor<half> halfReg2;
    MicroAPI::RegTensor<half> tmpReg;
    MicroAPI::RegTensor<float> floatReg1;
    MicroAPI::RegTensor<float> floatReg2;
    MicroAPI::RegTensor<int8_t> s8Reg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = MicroAPI::UpdateMask<float>(calSize);
        // Cast src from half to float type for getting more precise results, but only computes by finishing
        // taylor expansion computation as it's the majority reason of precision loss.
        MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcReg, srcUb + i * stride);
        MicroAPI::Cast<float, half, ASIN_CAST_TRAIT_NONE>(floatReg2, srcReg, mask);
        // Calculate res2 = PI*0.5 - taylor_compute(sqrt(1 - x^2)).
        MicroAPI::Mul(floatReg2, floatReg2, floatReg2, mask);
        MicroAPI::Muls(floatReg2, floatReg2, NEG_ONE, mask);
        MicroAPI::Adds(floatReg2, floatReg2, NUM_ONE, mask);
        AsinTaylorComputeBySquareValue<half>(floatReg1, floatReg2, mask);
        MicroAPI::Muls(floatReg1, floatReg1, NEG_ONE, mask);
        MicroAPI::Adds(floatReg1, floatReg1, HALF_PI, mask);

        // Calculate res1 = taylor_compute(abs(x)).
        MicroAPI::Abs(halfReg2, srcReg, mask);
        AsinTaylorCompute<half>(dstReg, halfReg2, mask);

        // As NPU are not good at scalar process like CPU for if-else statement, the solution here used for handling above
        // 3 scenarios is to calculate 0/1 choices combining the results on both options.
        // e.g.
        // Step1: Calculate both option results of x, no matter which range it's at.
        // result1(x), result2(x)
        // Step2: Calculate 0/1 choices of both option results of x, no matter which range it's at.
        // choice1(x), choice2(x)
        // Step3: Combine choice result and optional result, since at least one choice should be zero.
        // Result = choice1(x) * result1(x) + choice2(x) * result2(x)
        // choice1 = -Floor(min(abs(x), BOUNDARY) - BOUNDARY).
        // choice2 = 1 - choice1
        // res = res1 * choice1 + res2 * choice2
        MicroAPI::Mins(halfReg2, halfReg2, static_cast<half>(BOUNDARY), mask);
        MicroAPI::Adds(halfReg2, halfReg2, static_cast<half>(-BOUNDARY), mask);
        MicroAPI::Cast<int8_t, half, ASIN_CAST_TRAIT_FLOOR>(s8Reg, halfReg2, mask);
        MicroAPI::Cast<half, int8_t, ASIN_CAST_TRAIT_NONE>(halfReg2, s8Reg, mask);
        MicroAPI::Muls(halfReg2, halfReg2, static_cast<half>(NEG_ONE), mask);
        MicroAPI::Mul(dstReg, dstReg, halfReg2, mask);
        MicroAPI::Muls(halfReg2, halfReg2, static_cast<half>(NEG_ONE), mask);
        MicroAPI::Adds(halfReg2, halfReg2, static_cast<half>(NUM_ONE), mask);
        MicroAPI::Cast<float, half, ASIN_CAST_TRAIT_NONE>(floatReg2, halfReg2, mask);
        MicroAPI::Mul(floatReg1, floatReg1, floatReg2, mask);
        MicroAPI::Cast<float, half, ASIN_CAST_TRAIT_NONE>(floatReg2, dstReg, mask);
        MicroAPI::Add(floatReg1, floatReg2, floatReg1, mask);
        GetSign<half>(halfReg1, srcReg, mask);
        MicroAPI::Cast<float, half, ASIN_CAST_TRAIT_NONE>(floatReg2, halfReg1, mask);
        MicroAPI::Mul(floatReg1, floatReg1, floatReg2, mask);
        if constexpr (convertToAcos) {
            // Compute acos values according to formula: arccos(x) = PI*0.5 - arcsin(x).
            MicroAPI::Adds(floatReg1, floatReg1, -HALF_PI, mask);
            MicroAPI::Muls(floatReg1, floatReg1, NEG_ONE, mask);
        }
        MicroAPI::Cast<half, float, ASIN_CAST_TRAIT_RINT>(dstReg, floatReg1, mask);
        MicroAPI::DataCopy<half, MicroAPI::StoreDist::DIST_PACK_B32>(dstUb + i * stride, dstReg, mask);
    }
}
} // namespace Internal

template <typename T, bool convertToAcos = false>
__aicore__ inline void AsinCompute(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint32_t calSize)
{
    __local_mem__ T *dstUb = (__local_mem__ T *)dst.GetPhyAddr();
    __local_mem__ T *srcUb = (__local_mem__ T *)src.GetPhyAddr();

    // half dtype will be converted to float to improve precision;
    constexpr uint16_t stride = GetVecLen() / sizeof(float);
    uint16_t repeatTimes = CeilDivision(calSize, stride);
    if constexpr (IsSameType<T, half>::value) {
        Internal::AsinComputeVFF16<T, convertToAcos>(dstUb, srcUb, calSize, repeatTimes, stride);
    } else if (IsSameType<T, float>::value) {
        Internal::AsinComputeVFF32<T, convertToAcos>(dstUb, srcUb, calSize, repeatTimes, stride);
    }
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void AsinImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    static_assert(SupportType<T, half, float>(), "Asin only support half/float data type on current device!");

    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Asin");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Asin");
    AsinCompute(dstTensor, srcTensor, calCount);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void AsinImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");
    AsinImpl<T, isReuseSource>(dstTensor, srcTensor, calCount);
}
} // namespace AscendC

#endif // IMPL_MATH_ASIN_ASIN_C310_IMPL_H
