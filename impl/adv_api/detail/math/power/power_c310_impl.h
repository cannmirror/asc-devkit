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
#include "kernel_basic_intf.h"
#include "kernel_tensor.h"
#include "kernel_tiling/kernel_tiling.h"
#include "power_common_utils.h"
#include "../../common/check.h"

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

constexpr float LOG2_BEST_FMAF_COEFF1 = 0.129394531f;
constexpr float LOG2_BEST_FMAF_COEFF2 = 0.141957462f;
constexpr float LOG2_BEST_FMAF_COEFF3 = 0.200015724f;
constexpr float LOG2_BEST_FMAF_COEFF4 = 0.333333254f;
constexpr float LOG2_HI1 = 6.93147182e-1f;
constexpr float LOG2_HI2 = -6.93147182e-1f;
constexpr float LOG2_LO = -1.90465421e-9f;

constexpr int32_t EXPF_INTERVAL_CMP = -2097152000; 
constexpr int32_t EXPF_INTERVAL_CAST = 2130706432; 

constexpr float EXP_OVFL_UNFL_F = -104.0f;
constexpr float EXP_MIN_F = 88.7228390f;
constexpr int32_t INF = 0x7f800000;
constexpr int32_t NEG_INF = 0xff800000;
constexpr int32_t I32_NAN = 0x7f7fffff;
constexpr int32_t F32_NAN = 0x7fc00000;

constexpr int32_t R10_COEFF = 0x7F800000;
constexpr int32_t R12_COEFF = 0x7FFFFFFF;

constexpr int16_t COMPARE_ZERO_OFFSET = 31;
constexpr int16_t SHIFT_OFFSET = 23;
constexpr float F32_FRACTIONS = -23.0f;

struct PowerLogParams {
    MicroAPI::RegTensor<float> zeroReg;
    MicroAPI::RegTensor<float> oneReg;
    MicroAPI::RegTensor<float> fractionReg;
    MicroAPI::RegTensor<float> subReg;
    MicroAPI::RegTensor<int32_t> intReg;
    MicroAPI::RegTensor<float> rReg;
    MicroAPI::RegTensor<float> addReg1;
    MicroAPI::RegTensor<float> addReg2;
};

__simd_callee__ inline void PowerLogParamsInit(PowerLogParams& params) {
    MicroAPI::Duplicate(params.zeroReg, 0.0f);
    MicroAPI::Duplicate(params.oneReg, 1.0f);
    MicroAPI::Duplicate(params.fractionReg, F32_FRACTIONS);
    MicroAPI::Duplicate(params.subReg, LOG2_REDUCE_COEFF1);
    MicroAPI::Duplicate(params.intReg, LOG2_REDUCE_COEFF2);
    MicroAPI::Duplicate(params.rReg, LOG2_BEST_FMAF_COEFF2);
    MicroAPI::Duplicate(params.addReg1, LOG2_BEST_FMAF_COEFF3);
    MicroAPI::Duplicate(params.addReg2, LOG2_BEST_FMAF_COEFF4);
}

__simd_callee__ inline void IsInfNum(MicroAPI::MaskReg &infMask, MicroAPI::RegTensor<float> &srcReg,
    MicroAPI::RegTensor<int32_t>& tmpR12Reg, MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<float> tmpFloatReg;
    MicroAPI::And((MicroAPI::RegTensor<int32_t>&)tmpFloatReg, (MicroAPI::RegTensor<int32_t>&)srcReg, tmpR12Reg, mask);
    MicroAPI::CompareScalar<int32_t, CMPMODE::EQ>(infMask, (MicroAPI::RegTensor<int32_t>&)tmpFloatReg, INF, mask);
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
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ ==3003 || __NPU_ARCH__ ==3113))
    MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(filterMask, tmpReg, 0, mask);
#else
    MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(filterMask, tmpReg, 0.0f, mask);
#endif
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

__simd_callee__ inline void GetLogFExtStepOne(MicroAPI::RegTensor<float>& tmpIReg, MicroAPI::RegTensor<float>& tmpMReg,
    MicroAPI::RegTensor<float>& tmpRReg, MicroAPI::RegTensor<float>& srcReg, PowerLogParams& params, MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<float> tmpAReg, tmpFloatReg;
    MicroAPI::RegTensor<int32_t> tmpEReg;
    /* init variable a and i:
     *  if (a < 1.175494351e-38f){ // 0x1.0p-126
     *      a = a * 8388608.0f; // 0x1.0p+23
     *      i = -23.0f;
     *   }
     */
    MicroAPI::MaskReg cmpMask;
    MicroAPI::CompareScalar<float, CMPMODE::LT>(cmpMask, srcReg, LOG2_LOWEST_VALUE, mask);
    MicroAPI::Muls(tmpAReg, srcReg, LOG2_LOWEST_VALUE_MULS, mask);
    MicroAPI::Select(tmpIReg, params.fractionReg, params.zeroReg, cmpMask);
    // step 1: e = (__float_as_int (a) - __float_as_int (0.70710678f)) & 0xff800000;
    tmpFloatReg = params.subReg;
    MicroAPI::Sub(tmpEReg, (MicroAPI::RegTensor<int32_t> &)srcReg, (MicroAPI::RegTensor<int32_t> &)tmpFloatReg, mask);
    MicroAPI::And(tmpEReg, tmpEReg, params.intReg, mask);
    // step 2: m = __int_as_float (__float_as_int (a) - e);
    MicroAPI::Sub((MicroAPI::RegTensor<int32_t> &)tmpMReg, (MicroAPI::RegTensor<int32_t> &)srcReg, tmpEReg, mask);
    // step 3: i = fmaf ((float)e, 1.19209290e-7f, i);
    MicroAPI::Cast<float, int32_t, castTraitF32I32>(tmpFloatReg, tmpEReg, mask);
    MicroAPI::Axpy(tmpIReg, tmpFloatReg, LOG2_REDUCE_FMAF_COEFF1, mask);
    // step 4：p = m + 1.0f; m = m - 1.0f;
    MicroAPI::RegTensor<float> tmpPReg;
    MicroAPI::Adds(tmpPReg, tmpMReg, 1.0f, mask);
    MicroAPI::Adds(tmpMReg, tmpMReg, -1.0f, mask);
    // step 5：r = 1.0f / p
    MicroAPI::Div(tmpRReg, params.oneReg, tmpPReg, mask);
}

__simd_callee__ inline void GetLogFExtStepTwo(MicroAPI::RegTensor<float>& logHigh, MicroAPI::RegTensor<float>& logLow,
    MicroAPI::RegTensor<float>& tmpIReg, MicroAPI::RegTensor<float>& tmpMReg, MicroAPI::RegTensor<float>& tmpRReg,
    PowerLogParams& params, MicroAPI::MaskReg& mask)
{
    // step 6：qhi = m * r;
    MicroAPI::RegTensor<float> tmpQHIReg, tmpQLOReg;
    MicroAPI::RegTensor<float> tmpFloatReg, tmpFloatReg2;
    MicroAPI::Mul(tmpQHIReg, tmpMReg, tmpRReg, mask);
    // step 7：qhi1 = fmaf (qhi, -m, fmaf (qhi, -2.0f, m))
    MicroAPI::Muls(tmpFloatReg, tmpQHIReg, -2.0f, mask);
    MicroAPI::Add(tmpFloatReg, tmpFloatReg, tmpMReg, mask);
    MicroAPI::Neg(tmpFloatReg2, tmpMReg, mask);
    MicroAPI::FusedMulDstAdd(tmpFloatReg2, tmpQHIReg, tmpFloatReg, mask);
    // step 8：qlo = r * qhi1
    MicroAPI::Mul(tmpQLOReg, tmpRReg, tmpFloatReg2, mask);
    // step 9：s = qhi * qhi;
    MicroAPI::RegTensor<float> tmpSReg;
    MicroAPI::Mul(tmpSReg, tmpQHIReg, tmpQHIReg, mask);
    /* 
     * step 10：
     * r =  0.129394531f;
     * r = fmaf (r, s, 0.141957462f)
     * r = fmaf (r, s, 0.200015724f)
     * r = fmaf (r, s, 0.333333254f)
     */
    tmpRReg = params.rReg;
    MicroAPI::Axpy<float>(tmpRReg, tmpSReg, LOG2_BEST_FMAF_COEFF1, mask);
    MicroAPI::FusedMulDstAdd<float>(tmpRReg, tmpSReg, params.addReg1, mask);
    MicroAPI::FusedMulDstAdd<float>(tmpRReg, tmpSReg, params.addReg2, mask);
    // step 11：r = r * s
    MicroAPI::Mul(tmpRReg, tmpRReg, tmpSReg, mask);
    /* 
     * step 12:
     * qhi = qhi + qhi
     * qlo = qlo + qlo
    */
    MicroAPI::Add(tmpQHIReg, tmpQHIReg, tmpQHIReg, mask);
    MicroAPI::Add(tmpQLOReg, tmpQLOReg, tmpQLOReg, mask);
    // step 13: first_hi = fmaf(LOG2_HI, i, qhi)
    MicroAPI::RegTensor<float> tmpFHIReg, tmpFLOReg;
    MicroAPI::Muls(tmpFHIReg, tmpIReg, LOG2_HI1, mask);
    MicroAPI::Add(tmpFHIReg, tmpFHIReg, tmpQHIReg, mask);
    // step 14: first_lo = fmaf(-LOG2_HI, i, first_hi)
    tmpFloatReg2 = tmpFHIReg;
    MicroAPI::Axpy(tmpFloatReg2, tmpIReg, LOG2_HI2, mask);
    MicroAPI::Sub(tmpFLOReg, tmpQHIReg, tmpFloatReg2, mask);
    // step 15: second_lo = fmaf(r, qhi, first_lo)
    MicroAPI::FusedMulDstAdd(tmpQHIReg, tmpRReg, tmpFLOReg, mask);
    // step 16: last_lo = fmaf(3*qlo, r, qlo)
    MicroAPI::Muls(tmpFloatReg, tmpQLOReg, 3.0f, mask);
    MicroAPI::MulAddDst(tmpQLOReg, tmpFloatReg, tmpRReg, mask);
    // step 17: sum_lo = fmaf(LOG2_LO, i, last_lo) + second_lo
    MicroAPI::Axpy(tmpQLOReg, tmpIReg, LOG2_LO, mask);
    MicroAPI::Add(tmpQLOReg, tmpQLOReg, tmpQHIReg, mask);
    /*
     * step 18:
     * loghi = first_hi + sum_lo;
     * loglo = (first_hi - *loghi) + sum_lo;    
    */
    MicroAPI::Add(logHigh, tmpFHIReg, tmpQLOReg, mask);
    MicroAPI::Sub(tmpFloatReg, tmpFHIReg, logHigh, mask);
    MicroAPI::Add(logLow, tmpFloatReg, tmpQLOReg, mask);
}

__simd_callee__ inline void ComputeExpoOddInt(MicroAPI::MaskReg& oddMask, MicroAPI::RegTensor<float>& expReg,
    MicroAPI::RegTensor<float>& twoReg, MicroAPI::MaskReg& mask)
{
    // calculate exp is odd or not: expo_odd_int = fmaf (-2.0f, floorf (0.5f * b), b) == 1.0f;
    MicroAPI::RegTensor<float> tmpFloatReg;
    MicroAPI::Muls(tmpFloatReg, expReg, 0.5f, mask);
    RFloor(tmpFloatReg, tmpFloatReg, mask);
    MicroAPI::FusedMulDstAdd(tmpFloatReg, twoReg, expReg, mask);
    MicroAPI::CompareScalar<float, CMPMODE::EQ>(oddMask, tmpFloatReg, 1.0f, mask);
}


__simd_callee__ inline void ProcessSpecialCaseForPowF(MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& baseReg,
    MicroAPI::RegTensor<float>& expReg, MicroAPI::RegTensor<int32_t>& tmpR10Reg, MicroAPI::RegTensor<int32_t>& tmpR12Reg,
    MicroAPI::RegTensor<float>& twoReg, MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<float> tmpFloatReg, tmpFloatReg2;
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
     *     int32_t r10 = f32_to_s32(a);
     *     int32_t r11 = r10 ^ 0x7F800000;
     *     bool p8_a_lower_0 = (b < 0.0f);
     *     int32_t r12 = (p8_a_lower_0 ? r11 : r10);
     *     int32_t r13 = r12 & 0x7FFFFFFF;
     *     return s32_to_f32(p1_expo_odd_int ? r12 : r13);
     *
     */
    IsInfNum(cmpMask1, baseReg, tmpR12Reg, curMask);
    MicroAPI::CompareScalar<float, CMPMODE::EQ>(cmpMask2, baseReg, 0.0f, curMask);
    MicroAPI::MaskOr(cmpMask1, cmpMask1, cmpMask2, mask);
    MicroAPI::CompareScalar<float, CMPMODE::LT>(cmpMask2, expReg, 0.0f, cmpMask1);
    MicroAPI::Xor((MicroAPI::RegTensor<int32_t>&)tmpFloatReg, (MicroAPI::RegTensor<int32_t>&)baseReg, tmpR10Reg, curMask);
    MicroAPI::Select(tmpFloatReg, tmpFloatReg, baseReg, cmpMask2);
    MicroAPI::And((MicroAPI::RegTensor<int32_t>&)tmpFloatReg2, (MicroAPI::RegTensor<int32_t>&)tmpFloatReg, tmpR12Reg, curMask);
    ComputeExpoOddInt(cmpMask2, expReg, twoReg, mask);
    MicroAPI::Select(tmpFloatReg, tmpFloatReg, tmpFloatReg2, cmpMask2);
    MicroAPI::Select(dstReg, tmpFloatReg, dstReg, cmpMask1);
    MicroAPI::MaskXor(curMask, cmpMask1, curMask, mask);
    /*
     * if (a < 0.0f)
     *    float tmp_r = p1_expo_odd_int ? (-r) : r;
     *     r = (b != floorf(b)) ? NAN : tmp_r;
     * 
     */
    MicroAPI::Neg(tmpFloatReg, dstReg, curMask);
    MicroAPI::Select(tmpFloatReg, tmpFloatReg, dstReg, cmpMask2);
    RFloor(tmpFloatReg2, expReg, curMask);
    MicroAPI::Compare<float, CMPMODE::NE>(cmpMask1, expReg, tmpFloatReg2, curMask);
    MicroAPI::Duplicate<int32_t, MicroAPI::MaskMergeMode::MERGING>((MicroAPI::RegTensor<int32_t>&)tmpFloatReg, F32_NAN, cmpMask1);
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

__simd_callee__ inline void GetExpCore(MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& tmpLHIReg,
    MicroAPI::RegTensor<float>& tmpLLOReg, MicroAPI::RegTensor<float>& expReg, MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<float> tmPHIReg, tmPLOReg, tmpRReg;
    // step 1: thi = lhi * b;
    MicroAPI::Mul(tmPHIReg, tmpLHIReg, expReg, mask);
    // step 2: plo = fmaf(lhi, b, -phi) + llo*b;
    MicroAPI::RegTensor<float> tmpFloatReg, tmpFloatReg2;
    MicroAPI::Neg(tmPLOReg, tmPHIReg, mask);
    MicroAPI::MulAddDst(tmPLOReg, tmpLHIReg, expReg, mask);
    MicroAPI::MulAddDst(tmPLOReg, tmpLLOReg, expReg, mask);
    // step 3: r = exp(phi)
    MicroAPI::Exp(tmpRReg, tmPHIReg, mask);
    // step 4: r = plo*r + r
    MicroAPI::MulAddDst(tmpRReg, tmPLOReg, tmpRReg, mask);
    /*
     * step 5:
     * tmp_r =(phi < 0.0f) ?0.0f : MY_INF_F;
     * r = (phi > EXP_MIN_F or phi < EXP_OVFL_UNFL_F) ? tmp_r : r;
     */
    MicroAPI::MaskReg cmpMask1, cmpMask2;
    MicroAPI::CompareScalar<float, CMPMODE::GE>(cmpMask1, tmPHIReg, 0.0f, mask);
    // mode zeroing dup inf/zero reg.
    MicroAPI::Duplicate((MicroAPI::RegTensor<int32_t>&)tmpFloatReg, INF, cmpMask1);
    MicroAPI::CompareScalar<float, CMPMODE::GE>(cmpMask2, tmPHIReg, EXP_MIN_F, mask);
    MicroAPI::CompareScalar<float, CMPMODE::LT>(cmpMask1, tmPHIReg, EXP_OVFL_UNFL_F, mask);
    MicroAPI::MaskOr(cmpMask2, cmpMask1, cmpMask2, mask);
    MicroAPI::Select(dstReg, tmpFloatReg, tmpRReg, cmpMask2);
}

template<typename T>
__simd_callee__ inline void LoadSrcData(MicroAPI::RegTensor<float>& srcReg, __ubuf__ T* src0, uint16_t index, MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<T> srcTmpReg;
    if constexpr (std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value) {
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcTmpReg, src0 + index * B32_DATA_NUM_PER_REPEAT);
        MicroAPI::Cast<float, T, castTraitF16F32>(srcReg, srcTmpReg, mask);
    } else {
        MicroAPI::LoadAlign(srcReg, src0 + index * B32_DATA_NUM_PER_REPEAT);
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
        MicroAPI::StoreAlign(dst + index * B32_DATA_NUM_PER_REPEAT, dstTmpReg, mask);
    } else {
        MicroAPI::StoreAlign(dst + index * B32_DATA_NUM_PER_REPEAT, dstReg, mask);
    }
}

template<typename T>
__simd_vf__ inline void ComputePowFBaseLogStepOneImpl(__ubuf__ float* tmpLHIBuffer, __ubuf__ float* tmpLLOBuffer,
    __ubuf__ float* tmpLogBuffer, __ubuf__ T* src0, uint32_t calCount, uint16_t repeatTime)
{
    MicroAPI::MaskReg mask;
    MicroAPI::RegTensor<float> tmpBaseReg, tmpDstReg;
    MicroAPI::RegTensor<float> tmpLHIReg, tmpLLOReg, tmpRReg;

    PowerLogParams params;
    PowerLogParamsInit(params);

    for(uint16_t i = 0; i < repeatTime; i++) {
        mask = MicroAPI::UpdateMask<float>(calCount);
        LoadSrcData(tmpBaseReg, src0, i, mask);

        MicroAPI::Abs(tmpDstReg, tmpBaseReg, mask);
        GetLogFExtStepOne(tmpLHIReg, tmpLLOReg, tmpRReg, tmpDstReg, params, mask);

        MicroAPI::StoreAlign(tmpLHIBuffer + i * B32_DATA_NUM_PER_REPEAT, tmpLHIReg, mask);
        MicroAPI::StoreAlign(tmpLLOBuffer + i * B32_DATA_NUM_PER_REPEAT, tmpLLOReg, mask);
        MicroAPI::StoreAlign(tmpLogBuffer + i * B32_DATA_NUM_PER_REPEAT, tmpRReg, mask);
    }
}

template<typename T>
__simd_vf__ inline void ComputePowFBaseLogStepOneImpl(__ubuf__ float* tmpLHIBuffer, __ubuf__ float* tmpLLOBuffer,
    __ubuf__ float* tmpLogBuffer, const T scalarValue, uint32_t calCount, uint16_t repeatTime)
{
    MicroAPI::MaskReg mask;
    MicroAPI::RegTensor<float> tmpBaseReg, tmpDstReg;
    MicroAPI::RegTensor<float> tmpLHIReg, tmpLLOReg, tmpRReg;
    LoadSrcScalarData(tmpBaseReg, scalarValue);

    PowerLogParams params;
    PowerLogParamsInit(params);

    MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Abs(tmpDstReg, tmpBaseReg, fullMask);
    GetLogFExtStepOne(tmpLHIReg, tmpLLOReg, tmpRReg, tmpDstReg, params, fullMask);

    for(uint16_t i = 0; i < repeatTime; i++) {
        mask = MicroAPI::UpdateMask<float>(calCount);
        MicroAPI::StoreAlign(tmpLHIBuffer + i * B32_DATA_NUM_PER_REPEAT, tmpLHIReg, mask);
        MicroAPI::StoreAlign(tmpLLOBuffer + i * B32_DATA_NUM_PER_REPEAT, tmpLLOReg, mask);
        MicroAPI::StoreAlign(tmpLogBuffer + i * B32_DATA_NUM_PER_REPEAT, tmpRReg, mask);
    }
}

template<typename T>
__simd_vf__ inline void ComputePowFBaseLogStepTwoImpl(__ubuf__ float* tmpLHIBuffer, __ubuf__ float* tmpLLOBuffer,
   __ubuf__ float* tmpLogBuffer, __ubuf__ T* src0, uint32_t calCount, uint16_t repeatTime)
{
    MicroAPI::MaskReg mask;
    MicroAPI::RegTensor<float> tmpLHIReg, tmpLLOReg;
    MicroAPI::RegTensor<float> tmpIReg, tmpMReg, tmpRReg;

    PowerLogParams params;
    PowerLogParamsInit(params);

    for(uint16_t i = 0; i < repeatTime; i++) {
        mask = MicroAPI::UpdateMask<float>(calCount);
        LoadSrcData(tmpIReg, tmpLHIBuffer, i, mask);
        LoadSrcData(tmpMReg, tmpLLOBuffer, i, mask);
        LoadSrcData(tmpRReg, tmpLogBuffer, i, mask);

        GetLogFExtStepTwo(tmpLHIReg, tmpLLOReg, tmpIReg, tmpMReg, tmpRReg, params, mask);

        MicroAPI::StoreAlign(tmpLHIBuffer + i * B32_DATA_NUM_PER_REPEAT, tmpLHIReg, mask);
        MicroAPI::StoreAlign(tmpLLOBuffer + i * B32_DATA_NUM_PER_REPEAT, tmpLLOReg, mask);
        MicroAPI::StoreAlign(tmpLogBuffer + i * B32_DATA_NUM_PER_REPEAT, tmpRReg, mask);
    }
}

template<typename T>
__simd_vf__ inline void ComputePowFBaseLogStepTwoImpl(__ubuf__ float* tmpLHIBuffer, __ubuf__ float* tmpLLOBuffer,
    __ubuf__ float* tmpLogBuffer, const T scalarValue, uint32_t calCount, uint16_t repeatTime)
{
    MicroAPI::MaskReg mask;
    MicroAPI::RegTensor<float> tmpLHIReg, tmpLLOReg;
    MicroAPI::RegTensor<float> tmpIReg, tmpMReg, tmpRReg;

    PowerLogParams params;
    PowerLogParamsInit(params);

    MicroAPI::MaskReg fullMask = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
    LoadSrcData(tmpIReg, tmpLHIBuffer, 0, fullMask);
    LoadSrcData(tmpMReg, tmpLLOBuffer, 0, fullMask);
    LoadSrcData(tmpRReg, tmpLogBuffer, 0, fullMask);

    GetLogFExtStepTwo(tmpLHIReg, tmpLLOReg, tmpIReg, tmpMReg, tmpRReg, params, fullMask);

    for(uint16_t i = 0; i < repeatTime; i++) {
        mask = MicroAPI::UpdateMask<float>(calCount);
        MicroAPI::StoreAlign(tmpLHIBuffer + i * B32_DATA_NUM_PER_REPEAT, tmpLHIReg, mask);
        MicroAPI::StoreAlign(tmpLLOBuffer + i * B32_DATA_NUM_PER_REPEAT, tmpLLOReg, mask);
        MicroAPI::StoreAlign(tmpLogBuffer + i * B32_DATA_NUM_PER_REPEAT, tmpRReg, mask);
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
        MicroAPI::StoreAlign(tmpExpBuffer + i * B32_DATA_NUM_PER_REPEAT, tmpDstReg, mask);
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
        MicroAPI::StoreAlign(tmpExpBuffer + i * B32_DATA_NUM_PER_REPEAT, tmpDstReg, mask);
    }
}

template<typename T>
__simd_vf__ inline void ComputePowFSpecialCaseImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1,
    __ubuf__ float* tmpExpBuffer, uint32_t calCount, uint16_t repeatTime)
{
    MicroAPI::MaskReg mask;
    MicroAPI::RegTensor<float> tmpBaseReg, tmpExpReg, castDstReg, twoReg;
    MicroAPI::RegTensor<int32_t> tmpR10Reg, tmpR12Reg;
    MicroAPI::Duplicate(tmpR10Reg, R10_COEFF);
    MicroAPI::Duplicate(tmpR12Reg, R12_COEFF);
    MicroAPI::Duplicate(twoReg, -2.0f);
    for(uint16_t i = 0; i < repeatTime; i++) {
        mask = MicroAPI::UpdateMask<float>(calCount);
        LoadSrcData(tmpBaseReg, src0, i, mask);
        LoadSrcData(tmpExpReg, src1, i, mask);
        MicroAPI::LoadAlign(castDstReg, tmpExpBuffer + i * B32_DATA_NUM_PER_REPEAT);
        ProcessSpecialCaseForPowF(castDstReg, tmpBaseReg, tmpExpReg, tmpR10Reg, tmpR12Reg, twoReg, mask);
        StoreDstData(dst, castDstReg, i, mask);
    }
}

template<typename T>
__simd_vf__ inline void ComputePowFSpecialCaseImpl(__ubuf__ T* dst, __ubuf__ T* src0, const T scalarValue,
    __ubuf__ float* tmpExpBuffer, uint32_t calCount, uint16_t repeatTime)
{
    MicroAPI::MaskReg mask;
    MicroAPI::RegTensor<float> tmpBaseReg, tmpExpReg, castDstReg, twoReg;
    MicroAPI::RegTensor<int32_t> tmpR10Reg, tmpR12Reg;
    MicroAPI::Duplicate(tmpR10Reg, R10_COEFF);
    MicroAPI::Duplicate(tmpR12Reg, R12_COEFF);
    MicroAPI::Duplicate(twoReg, -2.0f);
    LoadSrcScalarData(tmpExpReg, scalarValue);
    for(uint16_t i = 0; i < repeatTime; i++) {
        mask = MicroAPI::UpdateMask<float>(calCount);
        LoadSrcData(tmpBaseReg, src0, i, mask);
        MicroAPI::LoadAlign(castDstReg, tmpExpBuffer + i * B32_DATA_NUM_PER_REPEAT);
        ProcessSpecialCaseForPowF(castDstReg, tmpBaseReg, tmpExpReg, tmpR10Reg, tmpR12Reg, twoReg, mask);
        StoreDstData(dst, castDstReg, i, mask);
    }
}

template<typename T>
__simd_vf__ inline void ComputePowFSpecialCaseImpl(__ubuf__ T* dst, const T scalarValue, __ubuf__ T* src1,
    __ubuf__ float* tmpExpBuffer, uint32_t calCount, uint16_t repeatTime)
{
    MicroAPI::MaskReg mask;
    MicroAPI::RegTensor<float> tmpBaseReg, tmpExpReg, castDstReg, twoReg;
    MicroAPI::RegTensor<int32_t> tmpR10Reg, tmpR12Reg;
    MicroAPI::Duplicate(tmpR10Reg, R10_COEFF);
    MicroAPI::Duplicate(tmpR12Reg, R12_COEFF);
    MicroAPI::Duplicate(twoReg, -2.0f);
    LoadSrcScalarData(tmpBaseReg, scalarValue);
    for(uint16_t i = 0; i < repeatTime; i++) {
        mask = MicroAPI::UpdateMask<float>(calCount);
        LoadSrcData(tmpExpReg, src1, i, mask);
        MicroAPI::LoadAlign(castDstReg, tmpExpBuffer + i * B32_DATA_NUM_PER_REPEAT);
        ProcessSpecialCaseForPowF(castDstReg, tmpBaseReg, tmpExpReg, tmpR10Reg, tmpR12Reg, twoReg, mask);
        StoreDstData(dst, castDstReg, i, mask);
    }
}

__aicore__ inline void InitTmpBuffer(__ubuf__ uint32_t*& tmpBuffer, __ubuf__ float*& tmpHighBuffer,
    __ubuf__ float*& tmpLowBuffer, __ubuf__ float*& tmpLogBuffer, const uint32_t alignCount)
{
    tmpHighBuffer = (__ubuf__ float *)tmpBuffer;
    tmpLowBuffer = (__ubuf__ float *)((__ubuf__ uint8_t*)tmpHighBuffer + sizeof(float) * alignCount);
    tmpLogBuffer = (__ubuf__ float *)((__ubuf__ uint8_t*)tmpLowBuffer + sizeof(float) * alignCount);
}

template<typename T>
__aicore__ inline void PowFComputeImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1,
    __ubuf__ uint32_t* tmpBuffer, uint32_t calCount)
{
    constexpr uint16_t eleCountPerVL = GetVecLen() / sizeof(float);
    uint16_t repeatTime = DivCeil(calCount, eleCountPerVL);
    __ubuf__ float* tmpLowBuffer;
    __ubuf__ float* tmpHighBuffer;
    __ubuf__ float* tmpLogBuffer;

    uint32_t alignCount = (calCount + 31) / 32 * 32;

    InitTmpBuffer(tmpBuffer, tmpHighBuffer, tmpLowBuffer, tmpLogBuffer, alignCount);
    __ubuf__ float* tmpExpBuffer = tmpHighBuffer;

    ComputePowFBaseLogStepOneImpl<T>(tmpHighBuffer, tmpLowBuffer, tmpLogBuffer, src0, calCount, repeatTime);
    ComputePowFBaseLogStepTwoImpl<T>(tmpHighBuffer, tmpLowBuffer, tmpLogBuffer, src0, calCount, repeatTime);
    ComputePowFExpImpl<T>(tmpExpBuffer, tmpHighBuffer, tmpLowBuffer, src1,  calCount, repeatTime);
    ComputePowFSpecialCaseImpl<T>(dst, src0, src1, tmpExpBuffer, calCount, repeatTime);
}

template<typename T>
__aicore__ inline void PowFComputeImpl(__ubuf__ T* dst, __ubuf__ T* src0, const T& scalarValue,
    __ubuf__ uint32_t* tmpBuffer, uint32_t calCount)
{
    constexpr uint16_t eleCountPerVL = GetVecLen() / sizeof(float);
    uint16_t repeatTime = DivCeil(calCount, eleCountPerVL);
    __ubuf__ float* tmpLowBuffer;
    __ubuf__ float* tmpHighBuffer;
    __ubuf__ float* tmpLogBuffer;

    uint32_t alignCount = (calCount + 31) / 32 * 32;

    InitTmpBuffer(tmpBuffer, tmpHighBuffer, tmpLowBuffer, tmpLogBuffer, alignCount);
    __ubuf__ float* tmpExpBuffer = tmpHighBuffer;

    ComputePowFBaseLogStepOneImpl<T>(tmpHighBuffer, tmpLowBuffer, tmpLogBuffer, src0, calCount, repeatTime);
    ComputePowFBaseLogStepTwoImpl<T>(tmpHighBuffer, tmpLowBuffer, tmpLogBuffer, src0, calCount, repeatTime);
    ComputePowFExpImpl<T>(tmpExpBuffer, tmpHighBuffer, tmpLowBuffer, scalarValue,  calCount, repeatTime);
    ComputePowFSpecialCaseImpl<T>(dst, src0, scalarValue, tmpExpBuffer, calCount, repeatTime);
}

template<typename T>
__aicore__ inline void PowFComputeImpl(__ubuf__ T* dst, const T& scalarValue, __ubuf__ T* src1,
    __ubuf__ uint32_t* tmpBuffer, uint32_t calCount)
{
    constexpr uint16_t eleCountPerVL = GetVecLen() / sizeof(float);
    uint16_t repeatTime = DivCeil(calCount, eleCountPerVL);
    __ubuf__ float* tmpLowBuffer;
    __ubuf__ float* tmpHighBuffer;
    __ubuf__ float* tmpLogBuffer;

    uint32_t alignCount = (calCount + 31) / 32 * 32;

    InitTmpBuffer(tmpBuffer, tmpHighBuffer, tmpLowBuffer, tmpLogBuffer, alignCount);
    __ubuf__ float* tmpExpBuffer = tmpHighBuffer;

    ComputePowFBaseLogStepOneImpl<T>(tmpHighBuffer, tmpLowBuffer, tmpLogBuffer, scalarValue, calCount, repeatTime);
    ComputePowFBaseLogStepTwoImpl<T>(tmpHighBuffer, tmpLowBuffer, tmpLogBuffer, scalarValue, calCount, repeatTime);
    ComputePowFExpImpl<T>(tmpExpBuffer, tmpHighBuffer, tmpLowBuffer, src1, calCount, repeatTime);
    ComputePowFSpecialCaseImpl<T>(dst, scalarValue, src1, tmpExpBuffer, calCount, repeatTime);
}

/*********** PowF Intrinsic Impl **********/
__simd_callee__ inline void GetPowFIntrinsicCore(MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& baseReg,
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
__simd_vf__ inline void PowFIntrinsicTensorTensorImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1,
    uint32_t calCount, uint16_t repeatTime)
{
    MicroAPI::MaskReg mask, tmpMask;
    MicroAPI::RegTensor<float> tmpBaseReg, tmpExpReg, castDstReg, twoReg;
    MicroAPI::RegTensor<int32_t> tmpR10Reg, tmpR12Reg;
    MicroAPI::Duplicate(tmpR10Reg, R10_COEFF);
    MicroAPI::Duplicate(tmpR12Reg, R12_COEFF);
    MicroAPI::Duplicate(twoReg, -2.0f);
    for(uint16_t i = 0; i < repeatTime; i++) {
        mask = MicroAPI::UpdateMask<float>(calCount);
        tmpMask = mask;
        LoadSrcData(tmpBaseReg, src0, i, mask);
        LoadSrcData(tmpExpReg, src1, i, mask);
        GetPowFIntrinsicCore(castDstReg, tmpBaseReg, tmpExpReg, mask);
        ProcessSpecialCaseForPowF(castDstReg, tmpBaseReg, tmpExpReg, tmpR10Reg, tmpR12Reg, twoReg, tmpMask);
        StoreDstData(dst, castDstReg, i, mask);
    }
}

template<typename T>
__simd_vf__ inline void PowFIntrinsicTensorScalarImpl(__ubuf__ T* dst, __ubuf__ T* src0, const T scalarValue,
    uint32_t calCount, uint16_t repeatTime)
{
    MicroAPI::MaskReg mask, tmpMask;
    MicroAPI::RegTensor<float> tmpBaseReg, tmpExpReg, castDstReg, twoReg;
    MicroAPI::RegTensor<int32_t> tmpR10Reg, tmpR12Reg;
    MicroAPI::Duplicate(tmpR10Reg, R10_COEFF);
    MicroAPI::Duplicate(tmpR12Reg, R12_COEFF);
    MicroAPI::Duplicate(twoReg, -2.0f);
    LoadSrcScalarData(tmpExpReg, scalarValue);
    for(uint16_t i = 0; i < repeatTime; i++) {
        mask = MicroAPI::UpdateMask<float>(calCount);
        tmpMask = mask;
        LoadSrcData(tmpBaseReg, src0, i, mask); 
        GetPowFIntrinsicCore(castDstReg, tmpBaseReg, tmpExpReg, mask);
        ProcessSpecialCaseForPowF(castDstReg, tmpBaseReg, tmpExpReg, tmpR10Reg, tmpR12Reg, twoReg, tmpMask);
        StoreDstData(dst, castDstReg, i, mask);
    }
}

template<typename T>
__simd_vf__ inline void PowFIntrinsicScalarTensorImpl(__ubuf__ T* dst, const T scalarValue, __ubuf__ T* src1,
    uint32_t calCount, uint16_t repeatTime)
{
    MicroAPI::MaskReg mask, tmpMask;
    MicroAPI::RegTensor<float> tmpBaseReg, tmpExpReg, castDstReg, twoReg;
    MicroAPI::RegTensor<int32_t> tmpR10Reg, tmpR12Reg;
    MicroAPI::Duplicate(tmpR10Reg, R10_COEFF);
    MicroAPI::Duplicate(tmpR12Reg, R12_COEFF);
    MicroAPI::Duplicate(twoReg, -2.0f);
    LoadSrcScalarData(tmpBaseReg, scalarValue);
    for(uint16_t i = 0; i < repeatTime; i++) {
        mask = MicroAPI::UpdateMask<float>(calCount);
        tmpMask = mask;
        LoadSrcData(tmpExpReg, src1, i, mask);
        GetPowFIntrinsicCore(castDstReg, tmpBaseReg, tmpExpReg, mask);
        ProcessSpecialCaseForPowF(castDstReg, tmpBaseReg, tmpExpReg, tmpR10Reg, tmpR12Reg, twoReg, tmpMask);
        StoreDstData(dst, castDstReg, i, mask);
    }
}

template<typename T>
__aicore__ inline void PowFIntrinsicImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1,
    uint32_t calCount)
{
    constexpr uint16_t eleCountPerVL = GetVecLen() / sizeof(float);
    uint16_t repeatTimes = DivCeil(calCount, eleCountPerVL);
    PowFIntrinsicTensorTensorImpl<T>(dst, src0, src1, calCount, repeatTimes);
}

template<typename T>
__aicore__ inline void PowFIntrinsicImpl(__ubuf__ T* dst, __ubuf__ T* src0, const T& scalarValue,
    uint32_t calCount)
{
    constexpr uint16_t eleCountPerVL = GetVecLen() / sizeof(float);
    uint16_t repeatTimes = DivCeil(calCount, eleCountPerVL);
    PowFIntrinsicTensorScalarImpl<T>(dst, src0, scalarValue, calCount, repeatTimes);
}

template<typename T>
__aicore__ inline void PowFIntrinsicImpl(__ubuf__ T* dst, const T& scalarValue, __ubuf__ T* src1,
    uint32_t calCount)
{
    constexpr uint16_t eleCountPerVL = GetVecLen() / sizeof(float);
    uint16_t repeatTimes = DivCeil(calCount, eleCountPerVL);
    PowFIntrinsicScalarTensorImpl<T>(dst, scalarValue, src1, calCount, repeatTimes);
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
    // step 4: b /= 2;
    MicroAPI::ShiftRights(expReg, expReg, SHIFT_ONE_BIT, mask);
    // step 5: a *= a;
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
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B8>(srcTmpReg, src + index * eleCountPerVL);
        MicroAPI::Cast<ConvType, T, castTraitI8I16>(dstReg, srcTmpReg, mask);
    } else {
        MicroAPI::LoadAlign(dstReg, src + index * eleCountPerVL);
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
        MicroAPI::StoreAlign(dst + index * eleCountPerVL, dstTmpReg, mask);
    } else {
        MicroAPI::StoreAlign(dst + index * eleCountPerVL, dstReg, mask);
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
__simd_vf__ inline void PowIComputeImpl(__ubuf__ T* dst, __ubuf__ T* src0, __ubuf__ T* src1, uint32_t calCount, uint16_t maxLoop)
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
__simd_vf__ inline void PowIComputeImpl(__ubuf__ T* dst, __ubuf__ T* src0, const T scalarValue, uint32_t calCount, uint16_t maxLoop)
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
__simd_vf__ inline void PowIComputeImpl(__ubuf__ T* dst, const T scalarValue, __ubuf__ T* src1,  uint32_t calCount, uint16_t maxLoop)
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
    constexpr uint32_t tmpBufferLiveNode = sizeof(float) * 3;
    return tmpBufferLiveNode;
}

template<typename T>
__aicore__ inline uint32_t GetPowTmpBufferSize(const LocalTensor<uint8_t>& sharedTmpBuffer) {
    uint32_t sharedTmpBufferSize = sharedTmpBuffer.GetSize() / GetPowerTmpBufferLiveNode();
    return AlignUp(sharedTmpBufferSize, GetDataBlockSizeInBytes());
}

// PowImpl(tensor, tensor) float/half input
template<typename T, const PowerConfig& config>
__aicore__ inline void PowImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor,
    const LocalTensor<T>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer, uint32_t calCount)
{
    __ubuf__ T* src0 = (__ubuf__ T *)src0Tensor.GetPhyAddr();
    __ubuf__ T* src1 = (__ubuf__ T *)src1Tensor.GetPhyAddr();
    __ubuf__ T* dst = (__ubuf__ T *)dstTensor.GetPhyAddr();

    if constexpr (IsFloatNum<T>()) {
        if constexpr (config.algo == PowerAlgo::INTRINSIC) {
            PowerC310Impl::PowF::PowFIntrinsicImpl(dst, src0, src1, calCount);
        } else {
            __ubuf__ uint32_t* tmpBuffer =  (__ubuf__ uint32_t *)sharedTmpBuffer.GetPhyAddr();
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
    __ubuf__ T *base = (__ubuf__ T *)src0Tensor.GetPhyAddr();
    __ubuf__ T *dst = (__ubuf__ T *)dstTensor.GetPhyAddr();

    if constexpr (IsFloatNum<T>()) {
        if constexpr (config.algo == PowerAlgo::INTRINSIC) {
            PowerC310Impl::PowF::PowFIntrinsicImpl(dst, base, scalarValue, calCount);
        } else {
            __ubuf__ uint32_t* tmpBuffer =  (__ubuf__ uint32_t *)sharedTmpBuffer.GetPhyAddr();
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
    __ubuf__ T *exp = (__ubuf__ T *)src1Tensor.GetPhyAddr();
    __ubuf__ T *dst = (__ubuf__ T *)dstTensor.GetPhyAddr();

    if constexpr (IsFloatNum<T>()) {
        if constexpr (config.algo == PowerAlgo::INTRINSIC) {
            PowerC310Impl::PowF::PowFIntrinsicImpl(dst, scalarValue, exp, calCount);
        } else {
            __ubuf__ uint32_t* tmpBuffer =  (__ubuf__ uint32_t *)sharedTmpBuffer.GetPhyAddr();
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
} //namespace AscendC
#endif
