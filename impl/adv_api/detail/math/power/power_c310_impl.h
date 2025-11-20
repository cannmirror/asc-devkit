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
 * \file power_c310_impl.h
 * \brief
 */

#ifndef IMPL_MATH_POWER_POWER_C310_IMPL_H
#define IMPL_MATH_POWER_POWER_C310_IMPL_H
#include "kernel_tensor.h"
#include "kernel_tiling/kernel_tiling.h"

#include "power_common_utils.h"

namespace AscendC {
namespace PowerC310Impl {

constexpr MicroAPI::CastTrait castTraitF16F32 = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                  MicroAPI::MaskMergeMode::ZEROING };
constexpr MicroAPI::CastTrait castTraitF32F16 = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                  MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT };
constexpr MicroAPI::CastTrait castTraitF32I32 = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                  MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND };
constexpr MicroAPI::CastTrait castTraitI32F32 = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                  MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC }; 
constexpr MicroAPI::CastTrait castTraitI8I16 =  { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                  MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND };
constexpr MicroAPI::CastTrait castTraitI16I8 =  { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                  MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND };

namespace PowF {
constexpr float LOG2_LOWEST_VALUE = 1.175494351e-38f;
constexpr float LOG2_LOWEST_VALUE_MULS = 8388608.0f;

constexpr float LOG2_REDUCE_COEFF1 = 0.70710678f;
constexpr int32_t LOG2_REDUCE_COEFF2 = 0xff800000;
constexpr float LOG2_REDUCE_FMAF_COEFF1 = 1.19209290e-7f;

constexpr float LOG2_BEST_FMAF_COEFF1 = 0.37282535f;
constexpr float LOG2_BEST_FMAF_COEFF2 = 0.4097556f;
constexpr float LOG2_BEST_FMAF_COEFF3 = 0.57711965f;
constexpr float LOG2_BEST_FMAF_COEFF4 = 0.96179646f;
constexpr float LOG2_BEST_FMAF_COEFF5 = 2.88539f;
constexpr float LOG2_BEST_FMAF_COEFF6 = 0.00000003851926f;

constexpr float EXPF_BEST_FMAF_COEFF1 = 0.000152392517f;
constexpr float EXPF_BEST_FMAF_COEFF2 = 0.00133913534f;
constexpr float EXPF_BEST_FMAF_COEFF3 = 0.00961883925f;
constexpr float EXPF_BEST_FMAF_COEFF4 = 0.0555035882f;
constexpr float EXPF_BEST_FMAF_COEFF5 = 0.240226448f;
constexpr float EXPF_BEST_FMAF_COEFF6 = 0.693147182f;


constexpr int32_t EXPF_INTERVAL_CMP = -2097152000; 
constexpr int32_t EXPF_INTERVAL_CAST = 2130706432; 

constexpr float EXP_OVFL_UNFL_F = 152.0f; 
constexpr int32_t INF = 0x7f800000; 
constexpr int32_t NEG_INF = 0xff800000;
constexpr int32_t I32_NAN = 0x7f7fffff;
constexpr int32_t F32_NAN = 0x7fc00000;

constexpr int32_t R10_COEFF = 0x7F800000;
constexpr int32_t R12_COEFF = 0x7FFFFFFF;


constexpr int16_t COMPARE_ZERO_OFFSET = 31;
constexpr int16_t SHITF_OFFSET = 23;
constexpr float F32_FRACTIONS = -23.0f;


__simd_callee__ inline void IsInfNum(MicroAPI::MaskReg &infMask, MicroAPI::RegTensor<float> &srcReg, MicroAPI::MaskReg& mask)
{
    MicroAPI::MaskReg tmpMask;
    MicroAPI::CompareScalar<int32_t, CMPMODE::EQ>(tmpMask, (MicroAPI::RegTensor<int32_t>&)srcReg, INF, mask);
    MicroAPI::CompareScalar<int32_t, CMPMODE::EQ>(infMask, (MicroAPI::RegTensor<int32_t>&)srcReg, NEG_INF, mask);
    MicroAPI::MaskOr(infMask, infMask, tmpMask, mask);
}

__simd_callee__ inline void IsNanNum(MicroAPI::MaskReg &nanMask, MicroAPI::RegTensor<float> &srcReg, MicroAPI::MaskReg& mask)
{
    MicroAPI::Compare<float, CMPMODE::NE>(nanMask, srcReg, srcReg, mask);
}

__simd_callee__ inline void RFloor(MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& srcReg, MicroAPI::MaskReg& mask) 
{
    MicroAPI::Truncate<float, RoundMode::CAST_FLOOR, MicroAPI::MaskMergeMode::ZEROING>(dstReg, srcReg, mask);
}

__aicore__ inline void CompareNegZero(MicroAPI::MaskReg &filterMask, MicroAPI::RegTensor<float>& srcReg, MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<uint32_t> tmpReg;
    MicroAPI::ShiftRights(tmpReg, (MicroAPI::RegTensor<uint32_t>&)srcReg, COMPARE_ZERO_OFFSET, mask);
    MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(filterMask, tmpReg, 0.0f, mask);
}

__aicore__ inline void CopySignF(MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& srcReg1,
    MicroAPI::RegTensor<float>& srcReg2, MicroAPI::MaskReg& mask, MicroAPI::MaskReg& cmpMask1, MicroAPI::MaskReg &cmpMask2)
{
    MicroAPI::RegTensor<float> tmpFloatReg, tmpFloatReg2;
    MicroAPI::CompareScalar<float, CMPMODE::GE>(cmpMask1, srcReg2, 0.0f, mask);
    CompareNegZero(cmpMask2, srcReg2, mask);
    MicroAPI::MaskAnd(cmpMask1, cmpMask1, cmpMask2, mask);
    MicroAPI::Abs(tmpFloatReg, srcReg1, mask);
    MicroAPI::Neg(tmpFloatReg2, tmpFloatReg, mask);
    MicroAPI::Select(dstReg, tmpFloatReg, tmpFloatReg2, cmpMask1);
}

__simd_callee__ inline void FMaf(MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& srcReg1,
    MicroAPI::RegTensor<float>& srcReg2, MicroAPI::RegTensor<float>& srcReg3, MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<float> tmpReg = srcReg1;
    MicroAPI::FusedMulDstAdd(tmpReg, srcReg2, srcReg3, mask);
    dstReg = tmpReg;
}

__simd_callee__ inline void FMaf(MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& srcReg1,
    float scalarValue, MicroAPI::RegTensor<float>& srcReg2, MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<float> tmpReg;
    MicroAPI::Duplicate(tmpReg, scalarValue, mask);
    FMaf(dstReg, srcReg1, tmpReg, srcReg2, mask);
}

__simd_callee__ inline void GetLogFExt(MicroAPI::RegTensor<float>& logHigh, MicroAPI::RegTensor<float>& logLow,
    MicroAPI::RegTensor<float>& srcReg, MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<float> tmpIReg, tmpAReg;
    MicroAPI::RegTensor<int32_t> tmpEReg, tmpIntReg;
    MicroAPI::RegTensor<float> tmpFloatReg, tmpFloatReg2;
    MicroAPI::Duplicate(tmpIReg, 0.0f, mask);
    /* init varaiable a and i:
     *  if (a < 1.175494351e-38f){ // 0x1.0p-126
     *      a = a * 8388608.0f; // 0x1.0p+23
     *      i = -23.0f;
     *   }
     */
    MicroAPI::MaskReg cmpMask;
    MicroAPI::CompareScalar<float, CMPMODE::LT>(cmpMask, srcReg, LOG2_LOWEST_VALUE, mask);
    MicroAPI::Muls(tmpAReg, srcReg, LOG2_LOWEST_VALUE_MULS, mask);
    MicroAPI::Duplicate(tmpFloatReg, F32_FRACTIONS, mask);
    MicroAPI::Select(tmpIReg, tmpFloatReg, tmpIReg, cmpMask);
    // step 1: e = (__float_as_int (a) - __float_as_int (0.70710678f)) & 0xff800000;
    MicroAPI::Duplicate(tmpFloatReg, LOG2_REDUCE_COEFF1, mask);
    MicroAPI::Sub(tmpEReg, (MicroAPI::RegTensor<int32_t> &)srcReg, (MicroAPI::RegTensor<int32_t> &)tmpFloatReg, mask);
    MicroAPI::Duplicate(tmpIntReg, LOG2_REDUCE_COEFF2, mask);
    MicroAPI::And(tmpEReg, tmpEReg, tmpIntReg, mask);
    // step 2: m = __int_as_float (__float_as_int (a) - e);
    MicroAPI::RegTensor<float> tmpMReg;
    MicroAPI::Sub((MicroAPI::RegTensor<int32_t> &)tmpMReg, (MicroAPI::RegTensor<int32_t> &)srcReg, tmpEReg, mask);
    // step 3: i = fmaf ((float)e, 1.19209290e-7f, i);
    MicroAPI::Cast<float, int32_t, castTraitF32I32>(tmpFloatReg, tmpEReg, mask);
    MicroAPI::Axpy(tmpIReg, tmpFloatReg, LOG2_REDUCE_FMAF_COEFF1, mask);
    // step 4：p = m + 1.0f; m = m - 1.0f;
    MicroAPI::RegTensor<float> tmpPReg;
    MicroAPI::Adds(tmpPReg, tmpMReg, 1.0f, mask);
    MicroAPI::Adds(tmpMReg, tmpMReg, -1.0f, mask);
    // step 5：r = 1.0f / p
    MicroAPI::RegTensor<float> tmpRReg;
    MicroAPI::Duplicate(tmpFloatReg, 1.0f, mask);
    MicroAPI::Div(tmpRReg, tmpFloatReg, tmpPReg, mask);
    // step 6：qhi = m * r;
    MicroAPI::RegTensor<float> tmpQHIReg, tmpQLOReg;
    MicroAPI::Mul(tmpQHIReg, tmpMReg, tmpRReg, mask);
    // step 7：qhi1 = fmaf (qhi, -m, fmaf (qhi, -2.0f, m))
    MicroAPI::Muls(tmpFloatReg, tmpQHIReg, -2.0f, mask);
    MicroAPI::Add(tmpFloatReg, tmpFloatReg, tmpMReg, mask);
    MicroAPI::Neg(tmpFloatReg2, tmpMReg, mask);

    MicroAPI::Mul(tmpFloatReg2, tmpQHIReg, tmpFloatReg2, mask);
    MicroAPI::Add(tmpFloatReg, tmpFloatReg2, tmpFloatReg, mask);

    // step 8：qlo = r * qhi1
    MicroAPI::Mul(tmpQLOReg, tmpRReg, tmpFloatReg, mask);
    // step 9：s = qhi * qhi;
    MicroAPI::RegTensor<float> tmpSReg;
    MicroAPI::Mul(tmpSReg, tmpQHIReg, tmpQHIReg, mask);
    /* 
     * step 10：
     * r =  0.37282535f;
     * r = fmaf (r, s, 0.4097556f)
     * r = fmaf (r, s, 0.57711965f)
     * r = fmaf (r, s, 0.96179646f)
     */
    MicroAPI::Duplicate(tmpRReg, LOG2_BEST_FMAF_COEFF2, mask);
    MicroAPI::Axpy<float>(tmpRReg, tmpSReg, LOG2_BEST_FMAF_COEFF1, mask);
    MicroAPI::Duplicate(tmpFloatReg, LOG2_BEST_FMAF_COEFF3, mask);
    MicroAPI::FusedMulDstAdd<float>(tmpRReg, tmpSReg, tmpFloatReg, mask);
    MicroAPI::Duplicate(tmpFloatReg, LOG2_BEST_FMAF_COEFF4, mask);
    MicroAPI::FusedMulDstAdd<float>(tmpRReg, tmpSReg, tmpFloatReg, mask);
    // step 11：r = r * s
    MicroAPI::Mul(tmpRReg, tmpRReg, tmpSReg, mask);
    /*
     * step 12:
     * first_hi = fmaf(2.88539f, qhi, i)
     * first_lo = fmaf(2.88539f, qhi, i-first_hi)
    */
    MicroAPI::RegTensor<float> tmpFHIReg, tmpFLOReg;
    FMaf(tmpFHIReg, tmpQHIReg, LOG2_BEST_FMAF_COEFF5, tmpIReg, mask);
    MicroAPI::Sub(tmpFloatReg2, tmpIReg, tmpFHIReg, mask);
    FMaf(tmpFLOReg, tmpQHIReg, LOG2_BEST_FMAF_COEFF5, tmpFloatReg2, mask);
    /*
     * step 13:
     * GOOD:
     * last_lo= fmaf(r,qhi,fmaf(2.88539f,qlo,first_lo));
     * sum_lo=fmaf(0.00000003851926f,qhi,last_lo);
     * BETTER:
     * last_lo= fmaf(r,qhi,fmaf(2.88539f,qlo,r*qlo));
     * sum_lo=fmaf(0.00000003851926f,qhi,last_lo)+first_lo;
     * BEST:
     * last_lo= fmaf(r,qhi,fmaf(2.88539f,qlo,3.0f*r*qlo));
     * sum_lo=fmaf(0.00000003851926f,qhi,last_lo)+first_lo;
    */
    MicroAPI::RegTensor<float> tmpLLOReg, tmpSLOReg;
    MicroAPI::Mul(tmpLLOReg, tmpRReg, tmpQLOReg, mask);
    MicroAPI::Axpy(tmpLLOReg, tmpQLOReg, LOG2_BEST_FMAF_COEFF5, mask);
    MicroAPI::MulAddDst(tmpLLOReg, tmpRReg, tmpQHIReg, mask);
    MicroAPI::Axpy(tmpLLOReg, tmpQHIReg, LOG2_BEST_FMAF_COEFF6, mask);
    MicroAPI::Add(tmpSLOReg, tmpLLOReg, tmpFLOReg, mask);
    /*
     * step 14:
     * loghi = first_hi+sum_lo;
     * loglo = (first_hi - *loghi) + sum_lo;    
    */
    MicroAPI::Add(logHigh, tmpFHIReg, tmpSLOReg, mask);
    MicroAPI::Sub(tmpFloatReg2, tmpFHIReg, logHigh, mask);
    MicroAPI::Add(logLow, tmpFloatReg2, tmpSLOReg, mask);
}

__simd_callee__ inline void GetExpfUnchecked(MicroAPI::RegTensor<float>& dstReg,MicroAPI::RegTensor<float>& tmPHIReg,
    MicroAPI::RegTensor<float>& tmPLOReg, MicroAPI::MaskReg& mask)
{
    /*
     * step 1:
     * r = fmaf (0.000152392517f, plo, 0.00133913534f);
     * r = fmaf (r, plo, 0.00961883925f);
     * r = fmaf (r, plo, 0.0555035882f);
     * r = fmaf (r, plo, 0.240226448f);
     * r = fmaf (r, plo, 0.693147182f);
     * r = fmaf (r, plo, 1.0f);
     */
    MicroAPI::RegTensor<float> tmpFloatReg, tmpFloatReg2, tmpRReg;
    MicroAPI::Duplicate(tmpRReg, EXPF_BEST_FMAF_COEFF2, mask);
    MicroAPI::Axpy(tmpRReg, tmPLOReg, EXPF_BEST_FMAF_COEFF1, mask);
    MicroAPI::Duplicate(tmpFloatReg, EXPF_BEST_FMAF_COEFF3, mask);
    MicroAPI::FusedMulDstAdd(tmpRReg, tmPLOReg, tmpFloatReg, mask);
    MicroAPI::Duplicate(tmpFloatReg, EXPF_BEST_FMAF_COEFF4, mask);
    MicroAPI::FusedMulDstAdd(tmpRReg, tmPLOReg, tmpFloatReg, mask);
    MicroAPI::Duplicate(tmpFloatReg, EXPF_BEST_FMAF_COEFF5, mask);
    MicroAPI::FusedMulDstAdd(tmpRReg, tmPLOReg, tmpFloatReg, mask);
    MicroAPI::Duplicate(tmpFloatReg, EXPF_BEST_FMAF_COEFF6, mask);
    MicroAPI::FusedMulDstAdd(tmpRReg, tmPLOReg, tmpFloatReg, mask);
    MicroAPI::Duplicate(tmpFloatReg, 1.0f, mask);
    MicroAPI::FusedMulDstAdd(tmpRReg, tmPLOReg, tmpFloatReg, mask);

    // step2: r1 = (phi>0.0f) ? 0 : -2097152000;
    MicroAPI::RegTensor<float> tmpF1Reg, tmpF2Reg;
    MicroAPI::RegTensor<int32_t> tmpR1Reg, tmpR2Reg;
    MicroAPI::MaskReg cmpMask;
    MicroAPI::CompareScalar<float,CMPMODE::LE>(cmpMask,tmPHIReg, 0, mask);
    MicroAPI::Duplicate(tmpR1Reg, EXPF_INTERVAL_CMP, cmpMask);
    // step3: r2 = r1 + 2130706432;
    MicroAPI::Adds(tmpR2Reg, tmpR1Reg, EXPF_INTERVAL_CAST, mask);
    // step4: f1 = r * s32_to_f32(r2);
    MicroAPI::Mul(tmpF1Reg, tmpRReg, (MicroAPI::RegTensor<float>&)tmpR2Reg, mask);
    /*
     * step5: int32_t r5 = cvt_rzi(phi);
     * if (phi < 0.0f) return (int32_t)ceilf(phi);
     * else return (int32_t)floorf(phi);
    */
    MicroAPI::RegTensor<int32_t> tmpR5Reg;
    MicroAPI::MaskReg condMask;
    MicroAPI::Cast<int32_t, float, castTraitI32F32>(tmpR5Reg, tmPHIReg, mask);
    // step6: f2 = s32_to_f32((r5 << 23) - r1);
    MicroAPI::RegTensor<int32_t> tmpTReg;
    MicroAPI::ShiftLefts(tmpR5Reg, tmpR5Reg, SHITF_OFFSET, mask);
    MicroAPI::Sub(tmpTReg, tmpR5Reg, tmpR1Reg, mask);
    tmpF2Reg = (MicroAPI::RegTensor<float>&)tmpTReg;
    // step7: f1 * f2
    MicroAPI::Mul(dstReg, tmpF1Reg, tmpF2Reg, mask);
}

__simd_callee__ inline void ComputeExpoOddInt(MicroAPI::MaskReg& oddMask, MicroAPI::RegTensor<float>& expReg, MicroAPI::MaskReg& mask)
{
    // calculate exp is odd or not: expo_odd_int = fmaf (-2.0f, floorf (0.5f * b), b) == 1.0f;
    MicroAPI::RegTensor<float> tmpFloatReg;
    MicroAPI::Muls(tmpFloatReg, expReg, 0.5f, mask);
    RFloor(tmpFloatReg, tmpFloatReg, mask);
    FMaf(tmpFloatReg, tmpFloatReg, -2.0f, expReg, mask);
    MicroAPI::CompareScalar<float, CMPMODE::EQ>(oddMask, tmpFloatReg, 1.0f, mask);
}


__simd_callee__ inline void ProcessSpecialCaseForPowF(MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& baseReg,
    MicroAPI::RegTensor<float>& expReg, MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<float> tmpFloatReg, tmpFloatReg2;
    MicroAPI::RegTensor<int32_t> tmpINFReg, tmpNANReg;
    MicroAPI::MaskReg cmpMask1, cmpMask2, curMask;
    /*
     * bool p3_b_eq_0 = (b==0.0f);
     * bool p4_a_eq_1 = (a==1.0f);
     * if (p3_b_eq_0 || p4_a_eq_1)
     *      return 1.0f;
     * if (isnan(a) || isnan(b))
     *      return NAN;
     */
    MicroAPI::CompareScalar<float, CMPMODE::EQ>(cmpMask1, expReg, 0.0f, mask);
    MicroAPI::CompareScalar<float, CMPMODE::EQ>(cmpMask2, baseReg, 1.0f, mask);
    MicroAPI::MaskOr(cmpMask2, cmpMask1, cmpMask2, mask);
    MicroAPI::Duplicate<float, MicroAPI::MaskMergeMode::MERGING>(dstReg, 1.0f, cmpMask2);
    MicroAPI::MaskNot(curMask, cmpMask2, mask);
    IsNanNum(cmpMask1, baseReg, mask);
    IsNanNum(cmpMask2, expReg, mask);
    MicroAPI::MaskOr(cmpMask2, cmpMask1, cmpMask2, curMask);
    MicroAPI::Duplicate<int32_t, MicroAPI::MaskMergeMode::MERGING>((MicroAPI::RegTensor<int32_t>&)dstReg, F32_NAN, cmpMask2);
    MicroAPI::MaskXor(curMask, cmpMask2, curMask, mask);
    /*
     * if (isinf(a) || (a==0.0f))
     *     int32_t r10=f32_to_s32(a);
     *     int32_t r11=r10^0x7F800000;
     *     bool p8_a_lower_0 = (b < 0.0f);
     *     int32_t r12 = (p8_a_lower_0 ? r11 : r10);
     *     int32_t r13= r12 & 0x7FFFFFFF;
     *     return s32_to_f32(p1_expo_odd_int?r12:r13);
     *
     */
    IsInfNum(cmpMask1, baseReg, curMask);
    MicroAPI::CompareScalar<float, CMPMODE::EQ>(cmpMask2, baseReg, 0.0f, curMask);
    MicroAPI::MaskOr(cmpMask1, cmpMask1, cmpMask2, mask);
    MicroAPI::CompareScalar<float, CMPMODE::LT>(cmpMask2, expReg, 0.0f, cmpMask1);
    MicroAPI::Duplicate((MicroAPI::RegTensor<int32_t>&)tmpFloatReg, R10_COEFF, mask);
    MicroAPI::Xor((MicroAPI::RegTensor<int32_t>&)tmpFloatReg, (MicroAPI::RegTensor<int32_t>&)baseReg,\
                  (MicroAPI::RegTensor<int32_t>&)tmpFloatReg, curMask);
    MicroAPI::Select(tmpFloatReg, tmpFloatReg, baseReg, cmpMask2);
    MicroAPI::Duplicate((MicroAPI::RegTensor<int32_t>&)tmpFloatReg2, R12_COEFF, mask);
    MicroAPI::And((MicroAPI::RegTensor<int32_t>&)tmpFloatReg2, (MicroAPI::RegTensor<int32_t>&)tmpFloatReg,\
                  (MicroAPI::RegTensor<int32_t>&)tmpFloatReg2, curMask);
    ComputeExpoOddInt(cmpMask2, expReg, mask);
    MicroAPI::Select(tmpFloatReg, tmpFloatReg, tmpFloatReg2, cmpMask2);
    MicroAPI::Select(dstReg, tmpFloatReg, dstReg, cmpMask1);
    MicroAPI::MaskXor(curMask, cmpMask1, curMask, mask);
    /*
     * if (a < 0.0f)
     *    float tmp_r=p1_expo_odd_int?(-r):r;
     *     r = (b != floorf(b)) ? NAN : tmp_r;
     * 
     */
    MicroAPI::Neg(tmpFloatReg, dstReg, curMask);
    MicroAPI::Select(tmpFloatReg, tmpFloatReg, dstReg, cmpMask2);
    RFloor(tmpFloatReg2, expReg, curMask);
    MicroAPI::Compare<float, CMPMODE::NE>(cmpMask1, expReg, tmpFloatReg2, curMask);
    MicroAPI::Duplicate<int32_t, MicroAPI::MaskMergeMode::MERGING>(
        (MicroAPI::RegTensor<int32_t>&)tmpFloatReg,F32_NAN, cmpMask1);
    MicroAPI::CompareScalar<float, CMPMODE::LT>(cmpMask2, baseReg, 0.0f, curMask);
    MicroAPI::Select(dstReg, tmpFloatReg, dstReg, cmpMask2);
    /*
     *  if ((a == -1.0f) && isinf(b))
     *      r = 1.0f;
     * 
     */
    MicroAPI::CompareScalar<int32_t, CMPMODE::EQ>(cmpMask1, (MicroAPI::RegTensor<int32_t>&)expReg, INF, curMask);
    MicroAPI::CompareScalar<int32_t, CMPMODE::EQ>(cmpMask2, (MicroAPI::RegTensor<int32_t>&)expReg, NEG_INF, curMask);
    MicroAPI::MaskOr(cmpMask1, cmpMask1, cmpMask2, mask);
    MicroAPI::CompareScalar<float, CMPMODE::EQ>(cmpMask2, baseReg, -1.0f, cmpMask1);
    MicroAPI::Duplicate<float, MicroAPI::MaskMergeMode::MERGING>(dstReg, 1.0f, cmpMask2);
}

__simd_callee__ inline void GetExpCore(MicroAPI::RegTensor<float>& dstReg, 
    MicroAPI::RegTensor<float>& tmpLHIReg, MicroAPI::RegTensor<float>& tmpLLOReg,
    MicroAPI::RegTensor<float>& expReg, MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<float> tmpTHIReg, tmPHIReg, tmPLOReg, tmpRReg;
    // step 1: thi = lhi * b;
    MicroAPI::Mul(tmpTHIReg, tmpLHIReg, expReg, mask);
    // step 2: phi = roundf(thi);
    MicroAPI::Truncate<float, RoundMode::CAST_ROUND>(tmPHIReg, tmpTHIReg, mask);
    // step 3: plo = fmaf(lhi, b, -phi)+llo*b;
    MicroAPI::RegTensor<float> tmpFloatReg, tmpFloatReg2;
    MicroAPI::Neg(tmPLOReg, tmPHIReg, mask);
    MicroAPI::MulAddDst(tmPLOReg, tmpLHIReg, expReg, mask);
    MicroAPI::MulAddDst(tmPLOReg, tmpLLOReg, expReg, mask);
    // step 4: my_expf__improved(phi, plo, &r);
    GetExpfUnchecked(tmpRReg, tmPHIReg, tmPLOReg, mask);
    /*
     * step 5:
     * tmp_r =(thi < 0.0f) ?0.0f : MY_INF_F;
     * r = (fabsf(thi) > EXP_OVFL_UNFL_F) ? tmp_r : r;
     */
    MicroAPI::RegTensor<int32_t> tmpINFReg;
    MicroAPI::MaskReg cmpMask1, cmpMask2;
    MicroAPI::CompareScalar<float, CMPMODE::GE>(cmpMask1, tmpTHIReg, 0.0f, mask);
    // mode zeroing dup inf/zero reg.
    MicroAPI::Duplicate((MicroAPI::RegTensor<int32_t>&)tmpFloatReg, INF, cmpMask1);
    MicroAPI::Abs(tmpFloatReg2, tmpTHIReg, mask);
    MicroAPI::CompareScalar<float, CMPMODE::GT>(cmpMask2, tmpFloatReg2, EXP_OVFL_UNFL_F, mask);
    MicroAPI::Select(dstReg, tmpFloatReg, tmpRReg, cmpMask2);
}

template<typename T>
__simd_callee__ inline void LoadSrcData(MicroAPI::RegTensor<float>& srcReg, __ubuf__ T* src0, uint16_t index, MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<T> srcTmpReg;
    if constexpr (std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value) {
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcTmpReg, src0 + index * B32_DATA_NUM_PER_REPEAT);
        MicroAPI::Cast<float, T, castTraitF16F32>(srcReg, srcTmpReg, mask);
    } else {
        MicroAPI::DataCopy(srcReg, src0 + index * B32_DATA_NUM_PER_REPEAT);
    }
}

template<typename T>
__simd_callee__ inline void LoadSrcScalarData(MicroAPI::RegTensor<float>& srcReg, const T scalarValue)
{
    MicroAPI::RegTensor<T> srcTmpReg;
    MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<T>();
    MicroAPI::Duplicate(srcTmpReg, scalarValue, fullMask);
    if constexpr (std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value) {
        MicroAPI::Cast<float, T, castTraitF16F32>(srcReg, srcTmpReg, fullMask);
    } else {
        srcReg = srcTmpReg;
    }
}

template<typename T>
__simd_callee__ inline void StoreDstData(__ubuf__ T* dst, MicroAPI::RegTensor<float>& dstReg, uint16_t index, MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<T> dstTmpReg;
    if constexpr (std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value) {
        MicroAPI::Cast<T, float, castTraitF32F16>(dstTmpReg, dstReg, mask);
        MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
            (MicroAPI::RegTensor<uint16_t>&)dstTmpReg, (MicroAPI::RegTensor<uint32_t>&)dstTmpReg);
        MicroAPI::MaskPack(mask, mask);
        MicroAPI::DataCopy(dst + index * B32_DATA_NUM_PER_REPEAT, dstTmpReg, mask);
    } else {
        MicroAPI::DataCopy(dst + index * B32_DATA_NUM_PER_REPEAT, dstReg, mask);
    }
}

template<typename T>
__simd_vf__ inline void ComputePowFBaseLogImpl(__ubuf__ float* tmpLHIBuffer, __ubuf__ float* tmpLLOBuffer,
    __ubuf__ T* src0, uint32_t calCount, uint16_t repeatTime)
{
    MicroAPI::MaskReg mask;
    MicroAPI::RegTensor<float> tmpBaseReg, tmpDstReg;
    MicroAPI::RegTensor<float> tmpLHIReg, tmpLLOReg;

    for(uint16_t i = 0; i < repeatTime; i++) {
        mask = MicroAPI::UpdateMask<float>(calCount);
        LoadSrcData(tmpBaseReg, src0, i, mask);

        MicroAPI::Abs(tmpDstReg, tmpBaseReg, mask);
        GetLogFExt(tmpLHIReg, tmpLLOReg, tmpDstReg, mask);

        MicroAPI::DataCopy(tmpLHIBuffer + i * B32_DATA_NUM_PER_REPEAT, tmpLHIReg, mask);
        MicroAPI::DataCopy(tmpLLOBuffer + i * B32_DATA_NUM_PER_REPEAT, tmpLLOReg, mask);
    }
}

template<typename T>
__simd_vf__ inline void ComputePowFBaseLogImpl(__ubuf__ float* tmpLHIBuffer, __ubuf__ float* tmpLLOBuffer,
    const T scalarValue, uint32_t calCount, uint16_t repeatTime)
{
    MicroAPI::MaskReg mask;
    MicroAPI::RegTensor<float> tmpBaseReg, tmpDstReg;
    MicroAPI::RegTensor<float> tmpLHIReg, tmpLLOReg;
    LoadSrcScalarData(tmpBaseReg, scalarValue);

    for (uint16_t i = 0; i < repeatTime; i++) {
        mask = MicroAPI::UpdateMask<float>(calCount);
        MicroAPI::Abs(tmpDstReg, tmpBaseReg, mask);
        GetLogFExt(tmpLHIReg, tmpLLOReg, tmpDstReg, mask);
        MicroAPI::DataCopy(tmpLHIBuffer + i * B32_DATA_NUM_PER_REPEAT, tmpLHIReg, mask);
        MicroAPI::DataCopy(tmpLLOBuffer + i * B32_DATA_NUM_PER_REPEAT, tmpLLOReg, mask);
    }
}

template<typename T>
__simd_vf__ inline void ComputePowFExpImpl(__ubuf__ float* tmpExpBuffer, __ubuf__ float* tmpLogHighBuffer,
    __ubuf__ float* tmpLogLowBuffer, __ubuf__ T* src1, uint32_t calCount, uint16_t repeatTime)
{
    MicroAPI::MaskReg mask;
    MicroAPI::RegTensor<float> tmpLHIReg, tmpLLOReg, tmpExpReg, tmpDstReg;

    for(uint16_t i = 0; i < repeatTime; i++) {
        mask = MicroAPI::UpdateMask<float>(calCount);
        LoadSrcData(tmpLHIReg, tmpLogHighBuffer, i, mask);
        LoadSrcData(tmpLLOReg, tmpLogLowBuffer, i, mask);
        LoadSrcData(tmpExpReg, src1, i, mask);
        GetExpCore(tmpDstReg, tmpLHIReg, tmpLLOReg, tmpExpReg, mask);
        MicroAPI::DataCopy(tmpExpBuffer + i * B32_DATA_NUM_PER_REPEAT, tmpDstReg, mask);
    }
}

template<typename T>
__simd_vf__ inline void ComputePowFExpImpl(__ubuf__ float* tmpExpBuffer, __ubuf__ float* tmpLogHighBuffer,
    __ubuf__ float* tmpLogLowBuffer, const T scalarValue, uint32_t calCount, uint16_t repeatTime)
{
    MicroAPI::MaskReg mask;
    MicroAPI::RegTensor<float> tmpLHIReg, tmpLLOReg, tmpExpReg, tmpDstReg;
    LoadSrcScalarData(tmpExpReg, scalarValue);
    for(uint16_t i = 0; i < repeatTime; i++) {
        mask = MicroAPI::UpdateMask<float>(calCount);
        LoadSrcData(tmpLHIReg, tmpLogHighBuffer, i, mask);
        LoadSrcData(tmpLLOReg, tmpLogLowBuffer, i, mask);
        GetExpCore(tmpDstReg, tmpLHIReg, tmpLLOReg, tmpExpReg, mask);
        MicroAPI::DataCopy(tmpExpBuffer + i * B32_DATA_NUM_PER_REPEAT, tmpDstReg, mask);
    }
}

template<typename T>
__simd_vf__ inline void ComputePowFSpecialCaseImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1,
    __ubuf__ float* tmpExpBuffer, uint32_t calCount, uint16_t repeatTime)
{
    MicroAPI::MaskReg mask;
    MicroAPI::RegTensor<float> tmpBaseReg, tmpExpReg, castDstReg;

    for(uint16_t i = 0; i < repeatTime; i++) {
        mask = MicroAPI::UpdateMask<float>(calCount);
        LoadSrcData(tmpBaseReg, src0, i, mask);
        LoadSrcData(tmpExpReg, src1, i, mask);
        MicroAPI::DataCopy(castDstReg, tmpExpBuffer + i * B32_DATA_NUM_PER_REPEAT);
        ProcessSpecialCaseForPowF(castDstReg, tmpBaseReg, tmpExpReg, mask);
        StoreDstData(dst, castDstReg, i, mask);
    }
}

template<typename T>
__simd_vf__ inline void ComputePowFSpecialCaseImpl(__ubuf__ T* dst, __ubuf__ T* src0, const T scalarValue,
    __ubuf__ float* tmpExpBuffer, uint32_t calCount, uint16_t repeatTime)
{
    MicroAPI::MaskReg mask;
    MicroAPI::RegTensor<float> tmpBaseReg, tmpExpReg, castDstReg;
    LoadSrcScalarData(tmpExpReg, scalarValue);
    for(uint16_t i = 0; i < repeatTime; i++) {
        mask = MicroAPI::UpdateMask<float>(calCount);
        LoadSrcData(tmpBaseReg, src0, i, mask);
        MicroAPI::DataCopy(castDstReg, tmpExpBuffer + i * B32_DATA_NUM_PER_REPEAT);
        ProcessSpecialCaseForPowF(castDstReg, tmpBaseReg, tmpExpReg, mask);
        StoreDstData(dst, castDstReg, i, mask);
    }
}

template<typename T>
__simd_vf__ inline void ComputePowFSpecialCaseImpl(__ubuf__ T* dst, const T scalarValue, __ubuf__ T* src1,
    __ubuf__ float* tmpExpBuffer, uint32_t calCount, uint16_t repeatTime)
{
    MicroAPI::MaskReg mask;
    MicroAPI::RegTensor<float> tmpBaseReg, tmpExpReg, castDstReg;
    LoadSrcScalarData(tmpBaseReg, scalarValue);
    for(uint16_t i = 0; i < repeatTime; i++) {
        mask = MicroAPI::UpdateMask<float>(calCount);
        LoadSrcData(tmpExpReg, src1, i, mask);
        MicroAPI::DataCopy(castDstReg, tmpExpBuffer + i * B32_DATA_NUM_PER_REPEAT);
        ProcessSpecialCaseForPowF(castDstReg, tmpBaseReg, tmpExpReg, mask);
        StoreDstData(dst, castDstReg, i, mask);
    }
}

__aicore__ inline void InitTmpBuffer(__ubuf__ uint32_t*& tmpBuffer, __ubuf__ float*& tmpLogBuffer,
    __ubuf__ float*& tmpExpBuffer, const uint32_t alignCount)
{
    tmpLogBuffer = (__local_mem__ float *)tmpBuffer;
    tmpExpBuffer = (__local_mem__ float *)((__local_mem__ uint8_t*)tmpLogBuffer + sizeof(float) * alignCount);
}

template<typename T>
__aicore__ inline void PowFComputeImpl(__local_mem__ T* dst, __local_mem__ T* src0, __local_mem__ T* src1,
    __ubuf__ uint32_t* tmpBuffer, uint32_t calCount)
{
    constexpr uint16_t eleCountPerVL = GetVecLen() / sizeof(float);
    uint16_t repeatTime = DivCeil(calCount, eleCountPerVL);
    __local_mem__ float* tmpLowBuffer;
    __local_mem__ float* tmpHighBuffer;

    uint32_t alignCount = (calCount + 31) / 32 * 32;

    InitTmpBuffer(tmpBuffer, tmpHighBuffer, tmpLowBuffer, alignCount);
    __local_mem__ float* tmpExpBuffer = tmpHighBuffer;

    ComputePowFBaseLogImpl<T>(tmpHighBuffer, tmpLowBuffer, src0, calCount, repeatTime);
    ComputePowFExpImpl<T>(tmpExpBuffer, tmpHighBuffer, tmpLowBuffer, src1,  calCount, repeatTime);
    ComputePowFSpecialCaseImpl<T>(dst, src0, src1, tmpExpBuffer, calCount, repeatTime);
}

template<typename T>
__aicore__ inline void PowFComputeImpl(__local_mem__ T* dst, __local_mem__ T* src0, const T& scalarValue,
    __ubuf__ uint32_t* tmpBuffer, uint32_t calCount)
{
    constexpr uint16_t eleCountPerVL = GetVecLen() / sizeof(float);
    uint16_t repeatTime = DivCeil(calCount, eleCountPerVL);
    __local_mem__ float* tmpLowBuffer;
    __local_mem__ float* tmpHighBuffer;

    uint32_t alignCount = (calCount + 31) / 32 * 32;

    InitTmpBuffer(tmpBuffer, tmpHighBuffer, tmpLowBuffer, alignCount);

    __local_mem__ float* tmpExpBuffer = tmpHighBuffer;

    ComputePowFBaseLogImpl<T>(tmpHighBuffer, tmpLowBuffer, src0, calCount, repeatTime);
    ComputePowFExpImpl<T>(tmpExpBuffer, tmpHighBuffer, tmpLowBuffer, scalarValue,  calCount, repeatTime);
    ComputePowFSpecialCaseImpl<T>(dst, src0, scalarValue, tmpExpBuffer, calCount, repeatTime);
}

template<typename T>
__aicore__ inline void PowFComputeImpl(__local_mem__ T* dst, const T& scalarValue, __local_mem__ T* src1,
    __ubuf__ uint32_t* tmpBuffer, uint32_t calCount)
{
    constexpr uint16_t eleCountPerVL = GetVecLen() / sizeof(float);
    uint16_t repeatTime = DivCeil(calCount, eleCountPerVL);
    __local_mem__ float* tmpLowBuffer;
    __local_mem__ float* tmpHighBuffer;

    uint32_t alignCount = (calCount + 31) / 32 * 32;

    InitTmpBuffer(tmpBuffer, tmpHighBuffer, tmpLowBuffer, alignCount);
    __local_mem__ float* tmpExpBuffer = tmpHighBuffer;

    ComputePowFBaseLogImpl<T>(tmpHighBuffer, tmpLowBuffer, scalarValue, calCount, repeatTime);
    ComputePowFExpImpl<T>(tmpExpBuffer, tmpHighBuffer, tmpLowBuffer, src1, calCount, repeatTime);
    ComputePowFSpecialCaseImpl<T>(dst, scalarValue, src1, tmpExpBuffer, calCount, repeatTime);
}

/*********** PowF Intrinsic Impl **********/
__simd_callee__ inline void GetPowFInstrinsicCore(MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& baseReg,
    MicroAPI::RegTensor<float>& expReg, MicroAPI::MaskReg& mask)
{
    // Compute dst = exp(exp * ln(|base|))
    MicroAPI::RegTensor<float> tmpReg;
    MicroAPI::Abs(tmpReg, baseReg, mask);
    MicroAPI::Ln(tmpReg, tmpReg, mask);
    MicroAPI::Mul(dstReg, expReg, tmpReg, mask);
    MicroAPI::Exp(dstReg, dstReg, mask);
}

template<typename T>
__simd_vf__ inline void PowFInstrinsicTensorTensorImpl(__local_mem__ T* dst, __local_mem__ T* src0, __local_mem__ T* src1,
    uint32_t calCount, uint16_t repeatTime)
{
    MicroAPI::MaskReg mask, tmpMask;
    MicroAPI::RegTensor<float> tmpBaseReg, tmpExpReg, castDstReg;

    for(uint16_t i = 0; i < repeatTime; i++) {
        mask = MicroAPI::UpdateMask<float>(calCount);
        tmpMask = mask;
        LoadSrcData(tmpBaseReg, src0, i, mask);
        LoadSrcData(tmpExpReg, src1, i, mask);
        GetPowFInstrinsicCore(castDstReg, tmpBaseReg, tmpExpReg, mask);
        ProcessSpecialCaseForPowF(castDstReg, tmpBaseReg, tmpExpReg, tmpMask);
        StoreDstData(dst, castDstReg, i, mask);
    }
}

template<typename T>
__simd_vf__ inline void PowFInstrinsicTensorScalarImpl(__local_mem__ T* dst, __local_mem__ T* src0, const T scalarValue,
    uint32_t calCount, uint16_t repeatTime)
{
    MicroAPI::MaskReg mask, tmpMask;
    MicroAPI::RegTensor<float> tmpBaseReg, tmpExpReg, castDstReg;
    LoadSrcScalarData(tmpExpReg, scalarValue);

    for(uint16_t i = 0; i < repeatTime; i++) {
        mask = MicroAPI::UpdateMask<float>(calCount);
        tmpMask = mask;
        LoadSrcData(tmpBaseReg, src0, i, mask);
        GetPowFInstrinsicCore(castDstReg, tmpBaseReg, tmpExpReg, mask);
        ProcessSpecialCaseForPowF(castDstReg, tmpBaseReg, tmpExpReg, tmpMask);
        StoreDstData(dst, castDstReg, i, mask);
    }
}

template<typename T>
__simd_vf__ inline void PowFInstrinsicScalarTensorImpl(__local_mem__ T* dst, const T scalarValue, __local_mem__ T* src1,
    uint32_t calCount, uint16_t repeatTime)
{
    MicroAPI::MaskReg mask, tmpMask;
    MicroAPI::RegTensor<float> tmpBaseReg, tmpExpReg, castDstReg;
    LoadSrcScalarData(tmpBaseReg, scalarValue);

    for(uint16_t i = 0; i < repeatTime; i++) {
        mask = MicroAPI::UpdateMask<float>(calCount);
        tmpMask = mask;
        LoadSrcData(tmpExpReg, src1, i, mask);
        GetPowFInstrinsicCore(castDstReg, tmpBaseReg, tmpExpReg, mask);
        ProcessSpecialCaseForPowF(castDstReg, tmpBaseReg, tmpExpReg, tmpMask);
        StoreDstData(dst, castDstReg, i, mask);
    }
}

template<typename T>
__aicore__ inline void PowFInstrinsicImpl(__local_mem__ T* dst, __local_mem__ T* src0, __local_mem__ T* src1,
    uint32_t calCount)
{
    constexpr uint16_t eleCountPerVL = GetVecLen() / sizeof(float);
    uint16_t repeatTimes = DivCeil(calCount, eleCountPerVL);
    PowFInstrinsicTensorTensorImpl<T>(dst, src0, src1, calCount, repeatTimes);
}

template<typename T>
__aicore__ inline void PowFInstrinsicImpl(__local_mem__ T* dst, __local_mem__ T* src0, const T& scalarValue,
    uint32_t calCount)
{
    constexpr uint16_t eleCountPerVL = GetVecLen() / sizeof(float);
    uint16_t repeatTimes = DivCeil(calCount, eleCountPerVL);
    PowFInstrinsicTensorScalarImpl<T>(dst, src0, scalarValue, calCount, repeatTimes);
}

template<typename T>
__aicore__ inline void PowFInstrinsicImpl(__local_mem__ T* dst, const T& scalarValue, __local_mem__ T* src1,
    uint32_t calCount)
{
    constexpr uint16_t eleCountPerVL = GetVecLen() / sizeof(float);
    uint16_t repeatTimes = DivCeil(calCount, eleCountPerVL);
    PowFInstrinsicScalarTensorImpl<T>(dst, scalarValue, src1, calCount, repeatTimes);
}
} // namespace PowF

namespace PowI {
constexpr int16_t SHIFT_ONE_BIT = 1;
constexpr int16_t BITS_PER_BYTE = 8;

template<typename T>
__simd_callee__ inline void GetPowI(MicroAPI::RegTensor<T>& dstReg, MicroAPI::RegTensor<T>& baseReg,
    MicroAPI::RegTensor<T>& expReg, MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<T> tmpReg, tmpReg2;
    MicroAPI::MaskReg expMask;
    // step 1: result1 = result * a;
    MicroAPI::Mul(tmpReg, dstReg, baseReg, mask);
    MicroAPI::Duplicate(tmpReg2, 1, mask);
    // step 2: mask = b & 1;
    MicroAPI::And(tmpReg2, expReg, tmpReg2, mask);
    // step 3: result = select(mask, result1, result);
    MicroAPI::CompareScalar<T, CMPMODE::EQ>(expMask, tmpReg2, 1, mask);
    MicroAPI::Select(dstReg, tmpReg, dstReg, expMask);
    // step4: b /= 2;
    MicroAPI::ShiftRights(expReg, expReg, SHIFT_ONE_BIT, mask);
    // step5: a *= a;
    MicroAPI::Mul(baseReg, baseReg, baseReg, mask);
}

template<typename T>
__simd_callee__ inline void ProcessSpecialCaseForPowI(MicroAPI::RegTensor<T>& dstReg, MicroAPI::RegTensor<T>& baseReg,
    MicroAPI::RegTensor<T>& expReg, MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<T> tmpRReg;
    MicroAPI::MaskReg cmpMask1, cmpMask2, condMask;
    /*
     * special case 1:
     * if (exp == 0) || (base == 1) {
     *    r = 1;
     * }
     */
    MicroAPI::CompareScalar<T, CMPMODE::EQ>(cmpMask1, expReg, 0, mask);
    MicroAPI::CompareScalar<T, CMPMODE::EQ>(cmpMask2, baseReg, 1, mask);
    MicroAPI::MaskOr(condMask, cmpMask1, cmpMask2, mask);
    MicroAPI::Duplicate(tmpRReg, 1, mask);
    MicroAPI::Select(dstReg, tmpRReg, dstReg, condMask);
    MicroAPI::MaskXor(mask, mask, condMask, mask);

    if constexpr (SupportType<T, int8_t, int16_t, int32_t>()) {
        /*
        * special case 2:
        * else if (base != -1 && exp < 0) {
        *    r = 0;
        * }
        */
        MicroAPI::CompareScalar<T, CMPMODE::NE>(cmpMask1, baseReg, -1, mask);
        MicroAPI::CompareScalar<T, CMPMODE::LT>(cmpMask2, expReg, 0, mask);
        MicroAPI::MaskAnd(condMask, cmpMask1, cmpMask2, mask);
        MicroAPI::Duplicate(tmpRReg, 0, mask);
        MicroAPI::Select(dstReg, tmpRReg, dstReg, condMask);
    }
}

template<typename T>
__simd_callee__ inline void GetPowICompute(MicroAPI::RegTensor<T>& dstReg, MicroAPI::RegTensor<T>& baseReg,
    MicroAPI::RegTensor<T>& expReg, MicroAPI::MaskReg& mask, const uint16_t maxLoop)
{
    MicroAPI::RegTensor<T> tmpBaseReg = baseReg;
    MicroAPI::RegTensor<T> tmpExpReg = expReg;
    MicroAPI::MaskReg tmpMask = mask;
    for (uint16_t j = 0; j < maxLoop; j++) {
        GetPowI(dstReg, tmpBaseReg, tmpExpReg, mask);
    }
    ProcessSpecialCaseForPowI(dstReg, baseReg, expReg, tmpMask);
}

template<typename T>
struct PowICastType {
    using type = T;
};

template<>
struct PowICastType<int8_t> {
    using type = int16_t;
};

template<>
struct PowICastType<uint8_t> {
    using type = uint16_t;
};

template<typename T>
__aicore__ inline uint16_t CountLeadingZeros(T x) {
    if (x == 0) return sizeof(T) * 8;
    uint16_t count = 0;
    T mask = static_cast<T>(1 << (sizeof(T) * 8 - 1));
    while((x & mask) == 0) {
        count ++;
        x <<= 1;
    }
    return count;
}

template<typename T, typename ConvType>
__simd_callee__ inline void LoadSrcData(MicroAPI::RegTensor<ConvType>& dstReg, __ubuf__ T* src, uint16_t index, MicroAPI::MaskReg& mask)
{
    constexpr uint16_t eleCountPerVL = GetVecLen() / sizeof(ConvType);
    MicroAPI::RegTensor<T> srcTmpReg;
    if constexpr (sizeof(T) == 1) {
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B8>(srcTmpReg, src + index * eleCountPerVL);
        MicroAPI::Cast<ConvType, T, castTraitI8I16>(dstReg, srcTmpReg, mask);
    } else {
        MicroAPI::DataCopy(dstReg, src + index * eleCountPerVL);
    }
}

template<typename T, typename ConvType>
__simd_callee__ inline void StoreDstData(__ubuf__ T* dst, MicroAPI::RegTensor<ConvType>& dstReg, uint16_t index, MicroAPI::MaskReg& mask)
{
    constexpr uint16_t eleCountPerVL = GetVecLen() / sizeof(ConvType);
    MicroAPI::RegTensor<T> dstTmpReg;

    if constexpr (sizeof(T) == 1) {
        MicroAPI::Pack<uint8_t, uint16_t, MicroAPI::HighLowPart::LOWEST>(
            (MicroAPI::RegTensor<uint8_t>&)dstTmpReg, (MicroAPI::RegTensor<uint16_t>&)dstReg);
        MicroAPI::MaskPack(mask, mask);
        MicroAPI::DataCopy(dst + index * eleCountPerVL, dstTmpReg, mask);
    } else {
        MicroAPI::DataCopy(dst + index * eleCountPerVL, dstReg, mask);
    }
}

template<typename T>
__aicore__ inline uint16_t GetMaxLoop(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, uint32_t calCount)
{
    if constexpr(sizeof(T) == 1) {
        return sizeof(T) * BITS_PER_BYTE;
    } else {
        AscendC::ReduceMax(dstTensor, srcTensor, srcTensor, calCount);
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        T maxNum = dstTensor.GetValue(0);
        return sizeof(T) * BITS_PER_BYTE - CountLeadingZeros(maxNum);
    }
}

template<typename T>
__simd_vf__ inline void PowIComputeImpl(__local_mem__ T* dst, __local_mem__ T* src0, __local_mem__ T* src1, uint32_t calCount, uint16_t maxLoop)
{
    using ConvType = typename PowICastType<T>::type;
    constexpr uint16_t eleCountPerVL = GetVecLen() / sizeof(ConvType);
    uint16_t repeatTime = DivCeil(calCount, eleCountPerVL);

    MicroAPI::RegTensor<ConvType> baseReg, expReg;
    MicroAPI::RegTensor<ConvType> initRetReg, dstReg;
    MicroAPI::MaskReg mask;

    MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<T>();
    MicroAPI::Duplicate(initRetReg, 1, fullMask);
    for (uint16_t i = 0; i < repeatTime; i++) {
        mask = MicroAPI::UpdateMask<ConvType>(calCount);

        LoadSrcData(baseReg, src0, i, mask);
        LoadSrcData(expReg, src1, i, mask);
        dstReg = initRetReg;
        GetPowICompute(dstReg, baseReg, expReg, mask, maxLoop);
        StoreDstData(dst, dstReg, i, mask);
    }
}

template<typename T>
__simd_vf__ inline void PowIComputeImpl(__local_mem__ T* dst, __local_mem__ T* src0, const T scalarValue, uint32_t calCount, uint16_t maxLoop)
{
    using ConvType = typename PowICastType<T>::type;
    constexpr uint16_t eleCountPerVL = GetVecLen() / sizeof(ConvType);
    uint16_t repeatTime = DivCeil(calCount, eleCountPerVL);

    MicroAPI::RegTensor<ConvType> baseReg, expReg;
    MicroAPI::RegTensor<ConvType> initRetReg, dstReg;
    MicroAPI::MaskReg mask;

    MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<T>();
    MicroAPI::Duplicate(initRetReg, 1, fullMask);
    MicroAPI::Duplicate(expReg, scalarValue, fullMask);

    for (uint16_t i = 0; i < repeatTime; i++) {
        mask = MicroAPI::UpdateMask<ConvType>(calCount);
        LoadSrcData(baseReg, src0, i, mask);
        dstReg = initRetReg;
        GetPowICompute(dstReg, baseReg, expReg, mask, maxLoop);
        StoreDstData(dst, dstReg, i, mask);
    }
}

template<typename T>
__simd_vf__ inline void PowIComputeImpl(__local_mem__ T* dst, const T scalarValue, __local_mem__ T* src1,  uint32_t calCount, uint16_t maxLoop)
{
    using ConvType = typename PowICastType<T>::type;
    constexpr uint16_t eleCountPerVL = GetVecLen() / sizeof(ConvType);
    uint16_t repeatTime = DivCeil(calCount, eleCountPerVL);

    MicroAPI::RegTensor<ConvType> baseReg, expReg;
    MicroAPI::RegTensor<ConvType> initRetReg, dstReg;
    MicroAPI::MaskReg mask;
    MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<T>();
    MicroAPI::Duplicate(initRetReg, 1, fullMask);
    MicroAPI::Duplicate(baseReg, scalarValue, fullMask);

    for (uint16_t i = 0; i < repeatTime; i++) {
        mask = MicroAPI::UpdateMask<ConvType>(calCount);
        LoadSrcData(expReg, src1, i, mask);
        dstReg = initRetReg;
        GetPowICompute(dstReg, baseReg, expReg, mask, maxLoop);
        StoreDstData(dst, dstReg, i, mask);
    }
}

} // namespace PowI
} // namespace PowerC310Impl

template <typename T, const PowerConfig& config = defaultPowerConfig>
__aicore__ inline void PowCheckType()
{
    if constexpr (config.algo == AscendC::PowerAlgo::DOUBLE_FLOAT_TECH) {
        static_assert(SupportType<T, half, float, bfloat16_t>(),
            "Type must be half/float/bfloat16 in double float tech algorithm."
        );
    }

    if constexpr (config.algo == AscendC::PowerAlgo::INTRINSIC) {
        static_assert(SupportType<T, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float>(),
            "Type must be uint8_t/int8_t/uint16_t/int16_t/uint32_t/int32_t/half/float in intrinsic tech algorithm."
        );
    }
}

template <typename T>
__aicore__ inline constexpr bool IsFloatNum()
{
    return SupportType<T, float, half, bfloat16_t>();
}

template <typename T>
__aicore__ inline constexpr bool IsIntegerNum()
{
    return SupportType<T, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t>();
}

__aicore__ inline constexpr uint32_t GetPowerTmpBufferLiveNode() {
    constexpr uint32_t tmpBufferLiveNode = sizeof(float) * 2;
    return tmpBufferLiveNode;
}

template<typename T>
__aicore__ inline uint32_t GetPowTmpBufferSize(const LocalTensor<uint8_t>& sharedTmpBuffer) {
    uint32_t sharedTmpBufferSize = sharedTmpBuffer.GetSize() / GetPowerTmpBufferLiveNode();
    return AlignUp(sharedTmpBufferSize, GetDataBlockSizeInBytes()) / sizeof(T);
}

// PowImpl(tensor, tensor) float/half input
template<typename T, const PowerConfig& config>
__aicore__ inline void PowImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor,
    const LocalTensor<T>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer, uint32_t calCount)
{
    __local_mem__ T* src0 = (__local_mem__ T *)src0Tensor.GetPhyAddr();
    __local_mem__ T* src1 = (__local_mem__ T *)src1Tensor.GetPhyAddr();
    __local_mem__ T* dst = (__local_mem__ T *)dstTensor.GetPhyAddr();

    if constexpr (IsFloatNum<T>()) {
        if constexpr (config.algo == PowerAlgo::INTRINSIC) {
            PowerC310Impl::PowF::PowFInstrinsicImpl(dst, src0, src1, calCount);
        } else {
            __local_mem__ uint32_t* tmpBuffer =  (__local_mem__ uint32_t *)sharedTmpBuffer.GetPhyAddr();
            uint32_t sharedTmpBufferSize = GetPowTmpBufferSize<T>(sharedTmpBuffer);
            uint32_t count = calCount;
            uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, sharedTmpBufferSize));
            for (uint16_t i = 0; i < repeatTimes; i++) {
                uint32_t remainCount = count - sharedTmpBufferSize * i;
                uint32_t oneRepSize = remainCount < sharedTmpBufferSize ? remainCount : sharedTmpBufferSize;
                PowerC310Impl::PowF::PowFComputeImpl(dst + i * sharedTmpBufferSize, src0 + i * sharedTmpBufferSize,
                    src1 + i * sharedTmpBufferSize, tmpBuffer, oneRepSize);
            }
        }
    } else if constexpr (IsIntegerNum<T>()) {
        uint16_t maxLoop = PowerC310Impl::PowI::GetMaxLoop(dstTensor, src1Tensor, calCount);
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        PowerC310Impl::PowI::PowIComputeImpl<T>(dst, src0, src1, calCount, maxLoop);
    }
}

// PowImpl(tensor, scalar) float input
template<typename T, const PowerConfig& config>
__aicore__ inline void PowImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor,
    const T& scalarValue, const LocalTensor<uint8_t>& sharedTmpBuffer, uint32_t calCount)
{
    __local_mem__ T *base = (__local_mem__ T *)src0Tensor.GetPhyAddr();
    __local_mem__ T *dst = (__local_mem__ T *)dstTensor.GetPhyAddr();

    if constexpr (IsFloatNum<T>()) {
        if constexpr (config.algo == PowerAlgo::INTRINSIC) {
            PowerC310Impl::PowF::PowFInstrinsicImpl(dst, base, scalarValue, calCount);
        } else {
            __local_mem__ uint32_t* tmpBuffer =  (__local_mem__ uint32_t *)sharedTmpBuffer.GetPhyAddr();
            uint32_t sharedTmpBufferSize = GetPowTmpBufferSize<T>(sharedTmpBuffer);
            uint32_t count = calCount;
            uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, sharedTmpBufferSize));
            for (uint16_t i = 0; i < repeatTimes; i++) {
                uint32_t remainCount = count - sharedTmpBufferSize * i;
                uint32_t oneRepSize = remainCount < sharedTmpBufferSize ? remainCount : sharedTmpBufferSize;
                PowerC310Impl::PowF::PowFComputeImpl(
                    dst + i * sharedTmpBufferSize, base + i * sharedTmpBufferSize, scalarValue, tmpBuffer, oneRepSize);
            }
        }
    } else if constexpr (IsIntegerNum<T>()) {
        uint16_t maxLoop = sizeof(T) * PowerC310Impl::PowI::BITS_PER_BYTE - PowerC310Impl::PowI::CountLeadingZeros(scalarValue);
        PowerC310Impl::PowI::PowIComputeImpl<T>(dst, base, scalarValue, calCount, maxLoop);
    }
}

// PowImpl(scalar, tensor) float input
template<typename T, const PowerConfig& config>
__aicore__ inline void PowImpl(const LocalTensor<T>& dstTensor, const T& scalarValue,
    const LocalTensor<T>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer, uint32_t calCount)
{
    __local_mem__ T *exp = (__local_mem__ T *)src1Tensor.GetPhyAddr();
    __local_mem__ T *dst = (__local_mem__ T *)dstTensor.GetPhyAddr();

    if constexpr (IsFloatNum<T>()) {
        if constexpr (config.algo == PowerAlgo::INTRINSIC) {
            PowerC310Impl::PowF::PowFInstrinsicImpl(dst, scalarValue, exp, calCount);
        } else {
            __local_mem__ uint32_t* tmpBuffer =  (__local_mem__ uint32_t *)sharedTmpBuffer.GetPhyAddr();
            uint32_t sharedTmpBufferSize = GetPowTmpBufferSize<T>(sharedTmpBuffer);
            uint32_t count = calCount;
            uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, sharedTmpBufferSize));
            for (uint16_t i = 0; i < repeatTimes; i++) {
                uint32_t remainCount = count - sharedTmpBufferSize * i;
                uint32_t oneRepSize = remainCount < sharedTmpBufferSize ? remainCount : sharedTmpBufferSize;
                PowerC310Impl::PowF::PowFComputeImpl(dst + i * sharedTmpBufferSize, scalarValue,
                    exp + i * sharedTmpBufferSize, tmpBuffer, oneRepSize);
            }
        }
    } else if constexpr (IsIntegerNum<T>()) {
        uint16_t maxLoop = PowerC310Impl::PowI::GetMaxLoop(dstTensor, src1Tensor, calCount);
        event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIdSToV);
        WaitFlag<HardEvent::S_V>(eventIdSToV);
        PowerC310Impl::PowI::PowIComputeImpl<T>(dst, scalarValue, exp, calCount, maxLoop);
    }
}

template<typename T, bool isReuseSource = false, const PowerConfig& config = defaultPowerConfig>
__aicore__ inline void PowerCommonImpl(const LocalTensor<T>& dstTensor, const T& scalarValue,
    const LocalTensor<T>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer, uint32_t calCount)
{
    if ASCEND_IS_AIC {
        return;
    }
    PowCheckType<T, config>();
    CheckTensorPos<T>(dstTensor, Hardware::UB, "dstTensor", "VECIN / VECOUT / VECCALC", "Power");
    CheckTensorPos<T>(src1Tensor, Hardware::UB, "src1Tensor", "VECIN / VECOUT / VECCALC", "Power");
    CheckTensorPos<uint8_t>(sharedTmpBuffer, Hardware::UB, "sharedTmpBuffer", "VECIN / VECOUT / VECCALC", "Power");
    CheckCalCount(calCount, "calCount", src1Tensor, "src1Tensor", "Power");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Power");

    PowImpl<T, config>(dstTensor, scalarValue, src1Tensor, sharedTmpBuffer, calCount);
}

template<typename T, bool isReuseSource = false, const PowerConfig& config = defaultPowerConfig>
__aicore__ inline void PowerCommonImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor,
    const T& scalarValue, const LocalTensor<uint8_t>& sharedTmpBuffer, uint32_t calCount)
{
    if ASCEND_IS_AIC {
        return;
    }
    PowCheckType<T, config>();
    CheckTensorPos<T>(dstTensor, Hardware::UB, "dstTensor", "VECIN / VECOUT / VECCALC", "Power");
    CheckTensorPos<T>(src0Tensor, Hardware::UB, "src0Tensor", "VECIN / VECOUT / VECCALC", "Power");
    CheckTensorPos<uint8_t>(sharedTmpBuffer, Hardware::UB, "sharedTmpBuffer", "VECIN / VECOUT / VECCALC", "Power");
    CheckCalCount(calCount, "calCount", src0Tensor, "src0Tensor", "Power");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Power");

    PowImpl<T, config>(dstTensor, src0Tensor, scalarValue, sharedTmpBuffer, calCount);
}

template<typename T, bool isReuseSource = false, const PowerConfig& config = defaultPowerConfig>
__aicore__ inline void PowerCommonImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor,
    const LocalTensor<T>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer, uint32_t calCount)
{
    if ASCEND_IS_AIC {
        return;
    }
    PowCheckType<T, config>();
    CheckTensorPos<T>(dstTensor, Hardware::UB, "dstTensor", "VECIN / VECOUT / VECCALC", "Power");
    CheckTensorPos<T>(src0Tensor, Hardware::UB, "src0Tensor", "VECIN / VECOUT / VECCALC", "Power");
    CheckTensorPos<T>(src1Tensor, Hardware::UB, "src1Tensor", "VECIN / VECOUT / VECCALC", "Power");
    CheckTensorPos<uint8_t>(sharedTmpBuffer, Hardware::UB, "sharedTmpBuffer", "VECIN / VECOUT / VECCALC", "Power");
    CheckCalCount(calCount, "calCount", src0Tensor, "src0Tensor", "Power");
    CheckCalCount(calCount, "calCount", src1Tensor, "src1Tensor", "Power");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Power");

    PowImpl<T, config>(dstTensor, src0Tensor, src1Tensor, sharedTmpBuffer, calCount);
}

template<typename T, bool isReuseSource = false, const PowerConfig& config = defaultPowerConfig>
__aicore__ inline void PowerCommonImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor,
    const LocalTensor<T>& src1Tensor, uint32_t calCount)
{
    LocalTensor<uint8_t> stackTensor;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(stackTensor);
    ASCENDC_ASSERT((ans),
                   { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    PowerCommonImpl<T, isReuseSource, config>(dstTensor, src0Tensor, src1Tensor, stackTensor, calCount);
}

template<typename T, bool isReuseSource = false, const PowerConfig& config = defaultPowerConfig>
__aicore__ inline void PowerCommonImpl(const LocalTensor<T>& dstTensor, const T& src0Scalar,
    const LocalTensor<T>& src1Tensor, uint32_t calCount)
{
    LocalTensor<uint8_t> stackTensor;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(stackTensor);
    ASCENDC_ASSERT((ans),
                   { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    PowerCommonImpl<T, isReuseSource, config>(dstTensor, src0Scalar, src1Tensor, stackTensor, calCount);
}

template<typename T, bool isReuseSource = false, const PowerConfig& config = defaultPowerConfig>
__aicore__ inline void PowerCommonImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor,
    const T& src1Scalar, uint32_t calCount)
{
    LocalTensor<uint8_t> stackTensor;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(stackTensor);
    ASCENDC_ASSERT((ans),
                   { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    PowerCommonImpl<T, isReuseSource, config>(dstTensor, src0Tensor, src1Scalar, stackTensor, calCount);
}
} //namesapce AscendC
#endif
