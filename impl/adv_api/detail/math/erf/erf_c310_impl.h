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
 * \file erf_c310_impl.h
 * \brief
 */
#ifndef IMPL_MATH_ERF_ERF_C310_IMPL_H
#define IMPL_MATH_ERF_ERF_C310_IMPL_H

#include "kernel_basic_intf.h"
#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"
#include "include/adv_api/math/erf_utils.h"
#include "../../common/check.h"

namespace AscendC {
namespace ErfAPI {

constexpr MicroAPI::CastTrait castTraitF162F32 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr MicroAPI::CastTrait castTraitF322F16 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

constexpr uint32_t ERF_C0 = 0x3F8060FE;
constexpr uint32_t ERF_P1[] = {
    0x38EB4C3A, 0xBAAE005B, 0x3C09919F, 0xBD24D99A,
    0x3E235519, 0x3F69B4F9, 0x3F210A14
};
constexpr uint32_t ERF_P2[] = {
    0x38B1E96A, 0xBA574D20, 0x3BAAD5EA, 0xBCDC1BE7,
    0x3DE718AF, 0xBEC093AC, 0x3E0375D3
};

// Clip x to [-3.92, 3.92]
__simd_callee__ inline void ErfClip(MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& srcReg, MicroAPI::MaskReg& mask)
{
    constexpr float ERF_BOUNDARY_MAX = 3.92;
    MicroAPI::Mins(dstReg, srcReg,  ERF_BOUNDARY_MAX, mask);
    MicroAPI::Maxs(dstReg, dstReg, -ERF_BOUNDARY_MAX, mask);
}

// P(x) = (((((0.053443748819x^2+0.75517016694e1)x^2+0.10162808918e3)x^2
//          +0.13938061484e4)x^2+0.50637915060e4)x^2+0.29639384698e5)x
__simd_callee__ inline void ErfComputeP(MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& srcReg, MicroAPI::MaskReg& mask)
{
    constexpr float SCALAR_P0 = 0.29639384698e5;
    constexpr float SCALAR_P1 = 0.50637915060e4;
    constexpr float SCALAR_P2 = 0.13938061484e4;
    constexpr float SCALAR_P3 = 0.10162808918e3;
    constexpr float SCALAR_P4 = 0.75517016694e1;
    constexpr float SCALAR_P5 = 0.053443748819;

    MicroAPI::RegTensor<float> tmpReg;
    MicroAPI::Mul(tmpReg, srcReg, srcReg, mask);
    MicroAPI::Muls(dstReg, tmpReg, SCALAR_P5, mask);
    MicroAPI::Adds(dstReg, dstReg, SCALAR_P4, mask);
    MicroAPI::Mul(dstReg, dstReg, tmpReg, mask);
    MicroAPI::Adds(dstReg, dstReg, SCALAR_P3, mask);
    MicroAPI::Mul(dstReg, dstReg, tmpReg, mask);
    MicroAPI::Adds(dstReg, dstReg, SCALAR_P2, mask);
    MicroAPI::Mul(dstReg, dstReg, tmpReg, mask);
    MicroAPI::Adds(dstReg, dstReg, SCALAR_P1, mask);
    MicroAPI::Mul(dstReg, dstReg, tmpReg, mask);
    MicroAPI::Adds(dstReg, dstReg, SCALAR_P0, mask);
    MicroAPI::Mul(dstReg, dstReg, srcReg, mask);
}

// Q(x) = ((((x^2+0.31212858877e2)x^2+0.39856963806e3)x^2+0.30231248150e4)x^2+0.13243365831e5)x^2+0.26267224157e5
__simd_callee__ inline void ErfComputeQ(MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& srcReg,
     MicroAPI::MaskReg& mask)
{
    constexpr float SCALAR_Q0 = 0.26267224157e5;
    constexpr float SCALAR_Q1 = 0.13243365831e5;
    constexpr float SCALAR_Q2 = 0.30231248150e4;
    constexpr float SCALAR_Q3 = 0.39856963806e3;
    constexpr float SCALAR_Q4 = 0.31212858877e2;

    MicroAPI::RegTensor<float> tmpReg;
    MicroAPI::Mul(tmpReg, srcReg, srcReg, mask);
    MicroAPI::Adds(dstReg, tmpReg, SCALAR_Q4, mask);
    MicroAPI::Mul(dstReg, dstReg, tmpReg, mask);
    MicroAPI::Adds(dstReg, dstReg, SCALAR_Q3, mask);
    MicroAPI::Mul(dstReg, dstReg, tmpReg, mask);
    MicroAPI::Adds(dstReg, dstReg, SCALAR_Q2, mask);
    MicroAPI::Mul(dstReg, dstReg, tmpReg, mask);
    MicroAPI::Adds(dstReg, dstReg, SCALAR_Q1, mask);
    MicroAPI::Mul(dstReg, dstReg, tmpReg, mask);
    MicroAPI::Adds(dstReg, dstReg, SCALAR_Q0, mask);
}

__simd_callee__ inline void ErfPadeCompute(MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& srcReg,
    MicroAPI::MaskReg& mask)
{
    // x = Clip(x), Erf(x) = P(x) / Q(x)
    MicroAPI::RegTensor<float> tmpReg;
    ErfClip(dstReg, srcReg, mask);
    ErfComputeP(tmpReg, dstReg, mask);
    ErfComputeQ(dstReg, dstReg, mask);

    MicroAPI::Div(dstReg, tmpReg, dstReg, mask);
}

__simd_callee__ inline void FMaf(MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& srcReg1,
    MicroAPI::RegTensor<float>& srcReg2, MicroAPI::RegTensor<float>& srcReg3, MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<float> tmpReg = srcReg1;
    MicroAPI::FusedMulDstAdd(tmpReg, srcReg2, srcReg3, mask);
    dstReg = tmpReg;
}

__simd_callee__ inline void ErfSpecialCaseCompute(MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& srcReg,
    MicroAPI::RegTensor<float>& tmpReg, MicroAPI::MaskReg& mask)
{
    /*
     * if (f5 < int32_as_float(0x3F8060FE)) {
     *    *y = f26;
     * } else {
     *    float f23 = exp(f26 * log(2.0));
     *    float f25 = int32_as_float(0x3F800000) - f23;
     *    unsigned int r3 = float_as_int32(f4) & 0x80000000;
     *    unsigned int r4 = r3 | float_as_int32(f25);
     *    *y = int32_as_float(r4);
     * }
     */
    constexpr uint32_t ERF_R0 = 0x3F8060FE;
    constexpr uint32_t ERF_R1 = 0x3F800000;
    constexpr uint32_t ERF_R2 = 0x80000000;
    constexpr float LOG2_VALUE = 2.0f;

    MicroAPI::RegTensor<float> tmpF5Reg, tmpF32Reg, tmpF32Reg1;
    MicroAPI::RegTensor<uint32_t> tmpU32Reg;
    MicroAPI::MaskReg cmpMask;
    MicroAPI::Abs(tmpF5Reg, srcReg, mask);
    MicroAPI::Duplicate(tmpU32Reg, ERF_R0, mask);
    MicroAPI::Compare<float, CMPMODE::LT>(cmpMask, tmpF5Reg, (MicroAPI::RegTensor<float> &)tmpU32Reg, mask);

    MicroAPI::Duplicate(tmpF32Reg, LOG2_VALUE, mask);
    MicroAPI::Log(tmpF32Reg, tmpF32Reg, mask);
    MicroAPI::Mul(tmpF32Reg, tmpReg, tmpF32Reg, mask);
    MicroAPI::Exp(tmpF32Reg, tmpF32Reg, mask);  // tmpF32Reg: f23
    MicroAPI::Duplicate(tmpU32Reg, ERF_R1, mask);
    MicroAPI::Sub(tmpF32Reg1, (MicroAPI::RegTensor<float> &)tmpU32Reg, tmpF32Reg, mask); //tmpF32Reg1: f25

    MicroAPI::Duplicate(tmpU32Reg, ERF_R2, mask);
    MicroAPI::And(tmpU32Reg, (MicroAPI::RegTensor<uint32_t> &)srcReg, tmpU32Reg, mask);
    MicroAPI::Or(tmpU32Reg, tmpU32Reg, (MicroAPI::RegTensor<uint32_t> &)tmpF32Reg1, mask); // tmpU32Reg: r4
    
    MicroAPI::Select(dstReg, tmpReg, (MicroAPI::RegTensor<float> &)tmpU32Reg, cmpMask);
}

__simd_callee__ inline void ErfSubsectionCompute(MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& srcReg,
     MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<float> tmpF5Reg, tmpF32Reg, tmpF32Reg1, tmpF32Reg2;
    MicroAPI::RegTensor<uint32_t> tmpU32Reg, tmpU32Reg1;
    MicroAPI::MaskReg cmpMask;
    /*
     * float f4 = x;
     * float f5 = fabsf(x);
     * bool p1 = f5 < int32_as_float(0x3F8060FE);
     * bool p2 = f5 >= int32_as_float(0x3F8060FE);
     */
    MicroAPI::Abs(tmpF5Reg, srcReg, mask);
    MicroAPI::Duplicate(tmpU32Reg, ERF_C0, mask);
    MicroAPI::Compare<float, CMPMODE::GE>(cmpMask, tmpF5Reg, (MicroAPI::RegTensor<float> &)tmpU32Reg, mask);
    /*
     * float f6 = f4 * f4;
     * float f7 = p2 ? f5 : f6;
     * float f8 = p2 ? int32_as_float(0x38EB4C3A) : int32_as_float(0x28B1E96A);
     * float f9 = p2 ? int32_as_float(0xBAAE005B) : int32_as_float(0xBA574D20);
     * float f10 = fmaf(f8, f7, f9);
     */
    MicroAPI::Mul(tmpF32Reg1, srcReg, srcReg, mask);
    MicroAPI::Select(tmpF32Reg, tmpF5Reg, tmpF32Reg1, cmpMask);  // tmpF32Reg: f7
    MicroAPI::Duplicate(tmpU32Reg, ERF_P1[0], mask);
    MicroAPI::Duplicate(tmpU32Reg1, ERF_P2[0], mask);
    MicroAPI::Select(tmpF32Reg1, 
        (MicroAPI::RegTensor<float> &)tmpU32Reg, (MicroAPI::RegTensor<float> &)tmpU32Reg1, cmpMask);
    MicroAPI::Duplicate(tmpU32Reg, ERF_P1[1], mask);
    MicroAPI::Duplicate(tmpU32Reg1, ERF_P2[1], mask);
    MicroAPI::Select(tmpF32Reg2, 
        (MicroAPI::RegTensor<float> &)tmpU32Reg, (MicroAPI::RegTensor<float> &)tmpU32Reg1, cmpMask);
    FMaf(tmpF32Reg1, tmpF32Reg1, tmpF32Reg, tmpF32Reg2, mask);  // tmpF32Reg1: f10
    /*
     * float f11 = p2 ? int32_as_float(0x3C09919F) : int32_as_float(0x3BAAD5EA);
     * float f12 = fmaf(f10, f7, f11);
     */
    MicroAPI::Duplicate(tmpU32Reg, ERF_P1[2], mask);  // int32_as_float(0x3C09919F)
    MicroAPI::Duplicate(tmpU32Reg1, ERF_P2[2], mask); // int32_as_float(0x3BAAD5EA)
    MicroAPI::Select(tmpF32Reg2, 
        (MicroAPI::RegTensor<float> &)tmpU32Reg, (MicroAPI::RegTensor<float> &)tmpU32Reg1, cmpMask); // tmpF32Reg2: f11
    FMaf(tmpF32Reg1, tmpF32Reg1, tmpF32Reg, tmpF32Reg2, mask); // tmpF32Reg1: f12
    /*
     * float f13 = p2 ? int32_as_float(0xBD24D99A) : int32_as_float(0xBCDC1BE7);
     * float f14 = fmaf(f12, f7, f13);
     */
    MicroAPI::Duplicate(tmpU32Reg, ERF_P1[3], mask); // int32_as_float(0xBD24D99A)
    MicroAPI::Duplicate(tmpU32Reg1, ERF_P2[3], mask); // int32_as_float(0xBCDC1BE7)
    MicroAPI::Select(tmpF32Reg2, 
        (MicroAPI::RegTensor<float> &)tmpU32Reg, (MicroAPI::RegTensor<float> &)tmpU32Reg1, cmpMask); // tmpF32Reg2: f13
    FMaf(tmpF32Reg1, tmpF32Reg1, tmpF32Reg, tmpF32Reg2, mask); // tmpF32Reg1: f14
    /*
     * float f15 = p2 ? int32_as_float(0x3E235519) : int32_as_float(0x3DE718AF);
     * float f16 = fmaf(f14, f7, f15);
     */
    MicroAPI::Duplicate(tmpU32Reg, ERF_P1[4], mask); // int32_as_float(0x3E235519)
    MicroAPI::Duplicate(tmpU32Reg1, ERF_P2[4], mask); // int32_as_float(0x3DE718AF)
    MicroAPI::Select(tmpF32Reg2, 
        (MicroAPI::RegTensor<float> &)tmpU32Reg, (MicroAPI::RegTensor<float> &)tmpU32Reg1, cmpMask); // tmpF32Reg2: f13
    FMaf(tmpF32Reg1, tmpF32Reg1, tmpF32Reg, tmpF32Reg2, mask); // tmpF32Reg1: f16
    /*
     * float f17 = p2 ? int32_as_float(0x3F69B4F9) : int32_as_float(0xBEC093AC);
     * float f18 = fmaf(f16, f7, f17);
     */
    MicroAPI::Duplicate(tmpU32Reg, ERF_P1[5], mask); // int32_as_float(0x3F69B4F9)
    MicroAPI::Duplicate(tmpU32Reg1, ERF_P2[5], mask); // int32_as_float(0xBEC093AC)
    MicroAPI::Select(tmpF32Reg2, 
        (MicroAPI::RegTensor<float> &)tmpU32Reg, (MicroAPI::RegTensor<float> &)tmpU32Reg1, cmpMask); // tmpF32Reg2: f13
    FMaf(tmpF32Reg1, tmpF32Reg1, tmpF32Reg, tmpF32Reg2, mask); // tmpF32Reg1: f18
    /*
     * float f19 = p2 ? int32_as_float(0x3F210A14) : int32_as_float(0x3E0375D3);
     * float f20 = fmaf(f18, f7, f19);
     */
    MicroAPI::Duplicate(tmpU32Reg, ERF_P1[6], mask); // int32_as_float(0x3F210A14)
    MicroAPI::Duplicate(tmpU32Reg1, ERF_P2[6], mask); // int32_as_float(0x3E0375D3)
    MicroAPI::Select(tmpF32Reg2, 
        (MicroAPI::RegTensor<float> &)tmpU32Reg, (MicroAPI::RegTensor<float> &)tmpU32Reg1, cmpMask); // tmpF32Reg2: f19
    FMaf(tmpF32Reg1, tmpF32Reg1, tmpF32Reg, tmpF32Reg2, mask); // tmpF32Reg1: f20
    /*
     * float f21 = -f5;
     * float f22 = p2 ? f21 : f4
     * float f26 = fmaf(f20, f22, f22);
     */
    MicroAPI::Neg(tmpF32Reg, tmpF5Reg, mask);
    MicroAPI::Select(tmpF32Reg2, tmpF32Reg, srcReg, cmpMask);
    FMaf(tmpF32Reg1, tmpF32Reg1, tmpF32Reg2, tmpF32Reg2, mask); // tmpF32Reg1: f26
    ErfSpecialCaseCompute(dstReg, srcReg, tmpF32Reg1, mask);
}

template <typename T, bool isReuseSource = false, const ErfConfig &config = defaultErfConfig>
__simd_vf__ inline void ErfCoreImpl(__ubuf__ T* dstUb, __ubuf__ T* srcUb, uint32_t calCount, uint16_t repeatTimes)
{
    MicroAPI::MaskReg mask;
    MicroAPI::RegTensor<T> srcReg;
    MicroAPI::RegTensor<float> castReg;
    MicroAPI::RegTensor<float> tmpReg;
    MicroAPI::RegTensor<float> dstReg;

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = MicroAPI::UpdateMask<float>(calCount);
        if constexpr (sizeof(T) == sizeof(half)) {
            MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcReg, srcUb + i * B32_DATA_NUM_PER_REPEAT);
            MicroAPI::Cast<float, T, castTraitF162F32>(castReg, srcReg, mask);
        } else {
            MicroAPI::LoadAlign(castReg, srcUb + i * B32_DATA_NUM_PER_REPEAT);
        }
        if constexpr (config.algo == ErfAlgo::PADE_APPROXIMATION) {
            ErfPadeCompute(dstReg, castReg, mask);
        } else {
            ErfSubsectionCompute(dstReg, castReg, mask);
        }
        if constexpr (sizeof(T) == sizeof(half)) {
            MicroAPI::Cast<T, float, castTraitF322F16>(srcReg, dstReg, mask);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(dstUb + i * B32_DATA_NUM_PER_REPEAT, srcReg, mask);
        } else {
            MicroAPI::StoreAlign(dstUb + i * B32_DATA_NUM_PER_REPEAT, dstReg, mask);
        }
    }
}
} // namespace ErfAPI

template <typename T, bool isReuseSource = false, const ErfConfig &config = defaultErfConfig>
__aicore__ inline void ErfCheckParams(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    static_assert(SupportType<T, half, float>(), "current data type is not supported on current device!");
    CheckTensorPos<T>(dstTensor, Hardware::UB, "dstTensor", "VECIN / VECCALC / VECOUT", "Erf");
    CheckTensorPos<T>(srcTensor, Hardware::UB, "srcTensor", "VECIN / VECCALC / VECOUT", "Erf");
    CheckTensorPos<uint8_t>(sharedTmpBuffer, Hardware::UB, "sharedTmpBuffer", "VECIN / VECCALC / VECOUT", "Erf");
    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Erf");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Erf");
}

template <typename T, bool isReuseSource = false, const ErfConfig &config = defaultErfConfig>
__aicore__ inline void ErfImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    ErfCheckParams<T, isReuseSource, config>(dstTensor, srcTensor, sharedTmpBuffer, calCount);
    __ubuf__ T *dstUb = (__ubuf__ T *)dstTensor.GetPhyAddr();
    __ubuf__ T *srcUb = (__ubuf__ T *)srcTensor.GetPhyAddr();
    uint16_t repeatTimes = CeilDivision(calCount, B32_DATA_NUM_PER_REPEAT);
    ErfAPI::ErfCoreImpl<T, isReuseSource, config>(dstUb, srcUb, calCount, repeatTimes);
}

template <typename T, bool isReuseSource = false, const ErfConfig &config = defaultErfConfig>
__aicore__ inline void ErfImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    // Using the Stack Space to Allocate tmpBuffer
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    ErfImpl<T, isReuseSource, config>(dstTensor, srcTensor, sharedTmpBuffer, calCount);
}

} // namespace AscendC

#endif // IMPL_MATH_ERF_ERF_C310_IMPL_H
