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
 * \file sincos_c310_impl.h
 * \brief
 */
#ifndef LIB_MATH_SINCOS_SINCOS_C310_IMPL_H
#define LIB_MATH_SINCOS_SINCOS_C310_IMPL_H
#include "kernel_basic_intf.h"
#include "kernel_tensor.h"
#ifdef ASCENDC_CPU_DEBUG
#include "../../api_check/kernel_check/math/sincos/sincos_check.h"
#endif // ASCENDC_CPU_DEBUG
#include "../../api_check/kernel_api_check.h"

namespace AscendC {
struct SinCosConfig {
    bool isReuseSource;
};
constexpr SinCosConfig DEFAULT_SINCOS_CONFIG = { false };
namespace SinCosImpl {

constexpr MicroAPI::CastTrait castTraitF16F32 = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, 
                                                  MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN };
constexpr MicroAPI::CastTrait castTraitF32F16 = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, 
                                                  MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT };
constexpr MicroAPI::CastTrait castTraitI64F32 = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                  MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND };
constexpr MicroAPI::CastTrait castTraitF32I64 = { MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::NO_SAT,
                                                  MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND };
constexpr MicroAPI::CastTrait castTraitI32F32 = { MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::NO_SAT,
                                                  MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND };

template <typename T, typename U>
__simd_callee__ inline void AndScalar(MicroAPI::RegTensor<T> &dstReg, MicroAPI::RegTensor<U> &srcReg, 
    T val, MicroAPI::MaskReg& mask)
{
    MicroAPI::RegTensor<T> tmpReg;
    MicroAPI::Duplicate(tmpReg, val, mask);
    MicroAPI::And(dstReg, (MicroAPI::RegTensor<T>&)srcReg, tmpReg, mask);
}

__simd_callee__ inline void FMaf(MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& srcReg,
    float scalarValue, MicroAPI::MaskReg& mask)
{
    // dst = dst * src + scalarValue
    MicroAPI::RegTensor<float> tmpReg;
    MicroAPI::Duplicate(tmpReg, scalarValue);
    MicroAPI::FusedMulDstAdd(dstReg, srcReg, tmpReg, mask);
}

__simd_callee__ inline void FMaf(MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& srcReg1,
    MicroAPI::RegTensor<float>& srcReg2, MicroAPI::RegTensor<float>& srcReg3, MicroAPI::MaskReg& mask)
{
    // dst = src1 * src2 + src3
    MicroAPI::RegTensor<float> tmpReg = srcReg1;
    MicroAPI::FusedMulDstAdd(tmpReg, srcReg2, srcReg3, mask);
    dstReg = tmpReg;
}

__simd_callee__ inline void FMaf(MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& srcReg1,
    MicroAPI::RegTensor<float>& srcReg2, float scalarValue, MicroAPI::MaskReg& mask)
{
    // dst = src1 * src2 + scalerValue
    MicroAPI::RegTensor<float> tmpReg;
    MicroAPI::Duplicate(tmpReg, scalarValue, mask);
    FMaf(dstReg, srcReg1, srcReg2, tmpReg, mask);
}

__simd_callee__ inline void FMaf(MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& srcReg1,
    float scalarValue, MicroAPI::RegTensor<float>& srcReg2, MicroAPI::MaskReg& mask)
{
    // dst = src1 * scalarValue + src2
    MicroAPI::RegTensor<float> tmpReg;
    MicroAPI::Duplicate(tmpReg, scalarValue, mask);
    FMaf(dstReg, srcReg1, tmpReg, srcReg2, mask);
}

__simd_callee__ inline void FMaf(MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& srcReg1,
    float scalarValue, float scalarValue2, MicroAPI::MaskReg& mask)
{
    // dst = src1 * scalarValue + scalarValue2
    MicroAPI::RegTensor<float> tmpReg, tmpReg2;
    MicroAPI::Duplicate(tmpReg, scalarValue, mask);
    MicroAPI::Duplicate(tmpReg2, scalarValue2, mask);
    FMaf(dstReg, srcReg1, tmpReg, tmpReg2, mask);
}

__simd_callee__ inline void BitShiftCombine(MicroAPI::RegTensor<uint32_t> &dstReg, MicroAPI::RegTensor<uint32_t> &srcReg1,
    MicroAPI::RegTensor<uint32_t> &srcReg2, MicroAPI::RegTensor<int32_t> &srcRegE, MicroAPI::MaskReg& mask)
{
    // dst = (src1  << e) | (src2 >> (32 - e));
    constexpr uint32_t BITSHIFTS = 32;

    MicroAPI::RegTensor<uint32_t> tmpU32Reg1, tmpU32Reg2;
    MicroAPI::ShiftLeft(tmpU32Reg1, srcReg1, (MicroAPI::RegTensor<int32_t>&)srcRegE, mask);
    MicroAPI::Duplicate(tmpU32Reg2, BITSHIFTS, mask);
    MicroAPI::Sub(tmpU32Reg2, tmpU32Reg2, (MicroAPI::RegTensor<uint32_t>&)srcRegE, mask);
    MicroAPI::ShiftRight(tmpU32Reg2, srcReg2, (MicroAPI::RegTensor<int32_t>&)tmpU32Reg2, mask);
    MicroAPI::Or(dstReg, tmpU32Reg1, tmpU32Reg2, mask);
}

__aicore__ inline void GenerateZeroVreg(MicroAPI::RegTensor<uint32_t>& zeroReg)
{
    MicroAPI::MaskReg b32FullMask = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Duplicate(zeroReg, 0, b32FullMask);
}

__simd_callee__ inline void ReinterpretedU32ToFloatAndCastToU32(MicroAPI::RegTensor<uint32_t> &dstReg,
    MicroAPI::RegTensor<uint32_t> &srcReg, MicroAPI::MaskReg &mask)
{
    // dst = (unsigned int) reinterpret_cast<float &>(src);
    MicroAPI::RegTensor<float> tmpF32Reg;
    MicroAPI::RegTensor<int64_t, MicroAPI::RegTraitNumTwo> tmpI64Reg;

    tmpF32Reg = (MicroAPI::RegTensor<float>&)srcReg;
    MicroAPI::Cast<int64_t, float, castTraitF32I64>(tmpI64Reg, tmpF32Reg, mask);
    dstReg = (MicroAPI::RegTensor<uint32_t>&)tmpI64Reg.reg[0];
}

__simd_callee__ inline void TrigComputeP(MicroAPI::RegTensor<uint32_t> &regPHigh, MicroAPI::RegTensor<uint32_t> &regPLow,
    MicroAPI::RegTensor<uint32_t> &regIa, MicroAPI::RegTensor<uint32_t> &regMid, MicroAPI::RegTensor<uint32_t> &regLo,
    MicroAPI::RegTensor<uint32_t> &regHi, MicroAPI::MaskReg &mask)
{
    MicroAPI::RegTensor<uint64_t, MicroAPI::RegTraitNumTwo> tmpU64Reg;
    MicroAPI::RegTensor<uint32_t> tmpU32Reg, zeroReg;
    MicroAPI::MaskReg carrypMask;

    // step 12: p = (unsigned long long int)ia * lo;
    MicroAPI::Mull((MicroAPI::RegTensor<uint32_t>&)regPLow,
        (MicroAPI::RegTensor<uint32_t>&)regPHigh, regIa, regLo, mask);

    // step 13: p = (unsigned long long int)ia * mid + (p >> 32);
    MicroAPI::Mull((MicroAPI::RegTensor<uint32_t>&)tmpU64Reg.reg[0],
        (MicroAPI::RegTensor<uint32_t>&)tmpU64Reg.reg[1], regIa, regMid, mask);

    MicroAPI::AddCarryOut(carrypMask, (MicroAPI::RegTensor<uint32_t>&)regPLow,
        (MicroAPI::RegTensor<uint32_t>&)tmpU64Reg.reg[0], (MicroAPI::RegTensor<uint32_t>&)regPHigh, mask);
    MicroAPI::Duplicate(zeroReg, 0, mask);
    MicroAPI::AddCarryOuts(carrypMask, (MicroAPI::RegTensor<uint32_t>&)regPHigh,
        (MicroAPI::RegTensor<uint32_t>&)tmpU64Reg.reg[1], zeroReg, carrypMask, mask);

    // step 14: p = ((unsigned long long int)(ia * hi) << 32) + p;
    MicroAPI::Mul(tmpU32Reg, regIa, regHi, mask);
    MicroAPI::AddCarryOut(carrypMask, (MicroAPI::RegTensor<uint32_t>&)regPHigh,
        (MicroAPI::RegTensor<uint32_t>&)regPHigh, tmpU32Reg, mask);
}

__simd_callee__ inline void TrigComputeHLQ(MicroAPI::RegTensor<float> &regDh, MicroAPI::RegTensor<float> &regDl,
    MicroAPI::RegTensor<int32_t> &regQ, MicroAPI::RegTensor<uint32_t> &regPHigh, MicroAPI::RegTensor<uint32_t> &regPLow,
    MicroAPI::MaskReg &mask)
{
    constexpr int16_t Q_SHIFT_BITS = 62;
    constexpr int16_t B32_BITS = 32;
    constexpr uint64_t P_AND_COEFF1 = 0x3fffffffffffffffULL;
    constexpr uint64_t P_AND_COEFF2 = 0x2000000000000000ULL;
    constexpr uint64_t P_SUBS_COEFF = 0x4000000000000000ULL;
    constexpr float P_MULS = 2.0f;
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ ==3003 || __NPU_ARCH__ ==3113))
    constexpr float B64_SHIFT_BITS = static_cast<float>(1ULL << 32);
#else
    constexpr uint64_t B64_SHIFT_BITS = 1ULL << 32;
#endif

    MicroAPI::RegTensor<int64_t, MicroAPI::RegTraitNumTwo> tmpI64Reg;
    MicroAPI::RegTensor<int32_t> tmpRegQ;
    MicroAPI::RegTensor<uint32_t> tmpU32Reg;
    MicroAPI::MaskReg tmpMask;

    // step 15: q = (int)(p >> 62);
    MicroAPI::ShiftRights((MicroAPI::RegTensor<uint32_t>&)regQ, 
        (MicroAPI::RegTensor<uint32_t>&)regPHigh, (int16_t)(Q_SHIFT_BITS - B32_BITS), mask);

    // step 16: p = p & 0x3fffffffffffffffULL;
    MicroAPI::Duplicate(tmpU32Reg, P_AND_COEFF1 >> B32_BITS, mask);
    MicroAPI::And((MicroAPI::RegTensor<uint32_t>&)regPHigh,
        (MicroAPI::RegTensor<uint32_t>&)regPHigh, tmpU32Reg, mask);

    /* step 17:
     * if (p & 0x2000000000000000ULL) {   // fraction >= 0.5
     *    p = p - 0x4000000000000000ULL; // fraction - 1.0
     *    q = q + 1;
     * }
     */
    MicroAPI::Duplicate(tmpU32Reg, P_AND_COEFF2 >> B32_BITS, mask);
    MicroAPI::And(tmpU32Reg, (MicroAPI::RegTensor<uint32_t>&)regPHigh, tmpU32Reg, mask);
    MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(tmpMask, tmpU32Reg, 0, mask);
    MicroAPI::Duplicate(tmpU32Reg, P_SUBS_COEFF >> B32_BITS, mask);
    MicroAPI::Sub(tmpU32Reg, (MicroAPI::RegTensor<uint32_t>&)regPHigh, tmpU32Reg, mask);
    MicroAPI::Select((MicroAPI::RegTensor<uint32_t>&)regPHigh, tmpU32Reg,
        (MicroAPI::RegTensor<uint32_t>&)regPHigh, tmpMask);
    MicroAPI::Adds(tmpRegQ, regQ, 1, mask);
    MicroAPI::Select(regQ, tmpRegQ, regQ, tmpMask);

    /* compute remainder of x / (pi/2) */
    // step 18: float d_h, d_l;
    // step 19: long long int P = (long long int)p;
    MicroAPI::RegTensor<float> tmpRegDH, tmpRegDL, tmpF32Reg;
    MicroAPI::RegTensor<int32_t> tmpI32Reg, tmpRegPHigh, tmpRegPLow;
    MicroAPI::Copy((MicroAPI::RegTensor<uint32_t>&)tmpRegPHigh, regPHigh);
    /*
     * d_h' = (float)P_high;
     * d_l' = (float)(P_low >> 1) * 2;
     */
    MicroAPI::Cast<float, int32_t, castTraitI32F32>(tmpRegDH, tmpRegPHigh, mask);
    MicroAPI::ShiftRights(tmpU32Reg, regPLow, (int16_t)1, mask);
    MicroAPI::Cast<float, int32_t, castTraitI32F32>(tmpRegDL, (MicroAPI::RegTensor<int32_t>&)tmpU32Reg, mask);
    MicroAPI::Muls(tmpRegDL, tmpRegDL, P_MULS, mask);
    // next: d_l = (float)(P_high - (int)d_h') *(2**32) + (float)P_low
    MicroAPI::Cast<int32_t, float, castTraitI32F32>(tmpI32Reg, tmpRegDH, mask);
    MicroAPI::Sub(tmpI32Reg, tmpRegPHigh, tmpI32Reg, mask);
    MicroAPI::Cast<float, int32_t, castTraitI32F32>(tmpF32Reg, tmpI32Reg, mask);
    MicroAPI::Muls(tmpF32Reg, tmpF32Reg, B64_SHIFT_BITS, mask);
    MicroAPI::Add(regDl, tmpF32Reg, tmpRegDL, mask);
    // then: d_h = d_h' * (2**32)
    MicroAPI::Muls(regDh, tmpRegDH, B64_SHIFT_BITS, mask);
}

__simd_callee__ inline void TrigRedSlowpathFComputeP(MicroAPI::RegTensor<uint32_t> &regPHigh, MicroAPI::RegTensor<uint32_t> &regPLow,
    MicroAPI::RegTensor<float> &srcReg, MicroAPI::RegTensor<uint32_t>& oneOverPiFReg, MicroAPI::MaskReg& mask)
{
    constexpr uint32_t TA_AND_COEFF = 0x007fffff;
    constexpr uint32_t IA_ADD_COEFF = 0x4f000000;
    constexpr int16_t TA_SHIFT_BITS = 23;
    constexpr int32_t TA_SHIFT_AND_COEFF = 0x000000ff;
    constexpr int16_t I_SHIFT_BITS = 5;
    constexpr int32_t E_SUB_COEFF = 126;
    constexpr int32_t E_AND_COEFF = 31;
    constexpr uint32_t LO_SELECT = 1;
    constexpr uint32_t TMP_SELECT = 2;

    MicroAPI::RegTensor<uint32_t> regIa, regHi, regMid, regLo, regTmp, regI;
    MicroAPI::RegTensor<int32_t> regE;
    MicroAPI::RegTensor<int32_t> tmpI32Reg;
    MicroAPI::RegTensor<uint32_t> tmpU32Reg;

    // step 1: unsigned int ta = reinterpret_cast<unsigned int &>(a);
    // ta can be obtained by (RegTensor<uint32_t>&)srcReg;
    // step 2: ia = (ta&0x007fffff) + 0x4f000000;
    AndScalar(regIa, (MicroAPI::RegTensor<uint32_t> &)srcReg, TA_AND_COEFF, mask);
    MicroAPI::Adds(regIa, regIa, IA_ADD_COEFF, mask);

    // step 3: ia = (unsigned int) reinterpret_cast<float &>(ia);
    ReinterpretedU32ToFloatAndCastToU32(regIa, regIa, mask);

    // step 4: e = ((ta >> 23) & 0x000000ff) - 127;
    MicroAPI::ShiftRights(regE, (MicroAPI::RegTensor<int32_t> &)srcReg, TA_SHIFT_BITS, mask);
    AndScalar(regE, regE, TA_SHIFT_AND_COEFF, mask);
    MicroAPI::Adds(regE, regE, -E_SUB_COEFF, mask);

    // step 5: i = (unsigned int)e >> 5;
    MicroAPI::ShiftRights(regI, (MicroAPI::RegTensor<uint32_t>&)regE, I_SHIFT_BITS, mask);
    // step 6: e = (unsigned int)e & 31;
    MicroAPI::Duplicate(tmpI32Reg, E_AND_COEFF, mask);
    MicroAPI::And(regE, (MicroAPI::RegTensor<int32_t>&)regE, tmpI32Reg, mask);

    // step 7:hi  = i ? one_over_pi_f [i-1] : 0;
    MicroAPI::MaskReg tmpMask;
    MicroAPI::RegTensor<uint32_t> tmpRegSelect;
    MicroAPI::CompareScalar<uint32_t, CMPMODE::GT>(tmpMask, regI, 0, mask);
    MicroAPI::Adds(tmpU32Reg, regI, -1, mask);
    MicroAPI::Gather(tmpRegSelect, oneOverPiFReg, tmpU32Reg);
    MicroAPI::Duplicate(regHi, 0, mask);
    MicroAPI::Select(regHi, tmpRegSelect, regHi, tmpMask);

    // step 8: mid = one_over_pi_f [i+0];
    MicroAPI::Gather(regMid, oneOverPiFReg, regI);
    // // step 9: lo  = one_over_pi_f [i+1];
    MicroAPI::Adds(tmpU32Reg, regI, LO_SELECT, mask);
    MicroAPI::Gather(regLo, oneOverPiFReg, tmpU32Reg);
    // step 10: tmp = one_over_pi_f [i+2];
    MicroAPI::Adds(tmpU32Reg, regI, TMP_SELECT, mask);
    MicroAPI::Gather(regTmp, oneOverPiFReg, tmpU32Reg);

    /* step 11:
     * if(e) {
     *    hi  = (hi  << e) | (mid >> (32 - e));
     *    mid = (mid << e) | (lo  >> (32 - e));
     *    lo  = (lo  << e) | (tmp >> (32 - e));
     * }
     */
    MicroAPI::CompareScalar<int32_t, CMPMODE::GT>(tmpMask, regE, 0, mask);
    BitShiftCombine(tmpRegSelect, regHi, regMid, regE, mask);
    MicroAPI::Select(regHi, tmpRegSelect, regHi, tmpMask);
    BitShiftCombine(tmpRegSelect, regMid, regLo, regE, mask);
    MicroAPI::Select(regMid, tmpRegSelect, regMid, tmpMask);
    BitShiftCombine(tmpRegSelect, regLo, regTmp, regE, mask);
    MicroAPI::Select(regLo, tmpRegSelect, regLo, tmpMask);

    TrigComputeP(regPHigh, regPLow, regIa, regMid, regLo, regHi, mask);
}

__simd_callee__ inline void TrigRedSlowpathFComputeRI(MicroAPI::RegTensor<float> &dstRegR, MicroAPI::RegTensor<int32_t> &dstRegI,
    MicroAPI::RegTensor<uint32_t> &regPHigh, MicroAPI::RegTensor<uint32_t> &regPLow, MicroAPI::RegTensor<float> &srcReg,
    MicroAPI::MaskReg& mask)
{
    constexpr float R_MUL_COEFF = 3.4061215800865545e-19;

    MicroAPI::RegTensor<int32_t> regQ;
    MicroAPI::RegTensor<float> regR, regDh, regDl;
    MicroAPI::RegTensor<float> tmpF32Reg;
    MicroAPI::RegTensor<int32_t> tmpRegQ;
    MicroAPI::MaskReg tmpMask;

    TrigComputeHLQ(regDh, regDl, regQ, regPHigh, regPLow, mask);
    
    // step 23: r = d_l*3.4061215800865545e-19;
    MicroAPI::Muls(regR, regDl, R_MUL_COEFF, mask);
    // step 24: r = r + d_h*3.4061215800865545e-19;
    MicroAPI::Duplicate(tmpF32Reg, R_MUL_COEFF, mask);
    MicroAPI::MulAddDst(regR, regDh, tmpF32Reg, mask);

    /* step 25:
     * if (a < 0.0f) {
     *    r = -r;
     *    q = -q;
     * }
     */
    MicroAPI::CompareScalar<float, CMPMODE::LT>(tmpMask, srcReg, 0.0f, mask);
    MicroAPI::Neg(tmpF32Reg, regR, mask);
    MicroAPI::Select(regR, tmpF32Reg, regR, tmpMask);
    MicroAPI::Neg(tmpRegQ, regQ, mask);
    MicroAPI::Select(regQ, tmpRegQ, regQ, tmpMask);

    // step 26: *quadrant = q;
    dstRegR = regR;
    dstRegI = regQ;
}

__simd_callee__ inline void SinfPoly(MicroAPI::RegTensor<float> &dstReg, MicroAPI::RegTensor<float> &srcRegA,
    MicroAPI::RegTensor<float> &srcRegS, MicroAPI::MaskReg& mask)
{
    constexpr float SIN_POLY_COEFF0 = 2.86567956e-6f;
    constexpr float SIN_POLY_COEFF1 = -1.98559923e-4f;
    constexpr float SIN_POLY_COEFF2 = 8.33338592e-3f;
    constexpr float SIN_POLY_COEFF3 = -1.66666672e-1f;
    constexpr float SIN_POLY_COEFF5 = 0.0f;

    MicroAPI::RegTensor<float> tmpRegT;
    // step 1: r = 2.86567956e-6f;
    MicroAPI::Duplicate(dstReg, SIN_POLY_COEFF0, mask);
    // step 2: r = r* s+ -1.98559923e-4f;
    FMaf(dstReg, srcRegS, SIN_POLY_COEFF1, mask);
    // step 3: r = r* s+  8.33338592e-3f;
    FMaf(dstReg, srcRegS, SIN_POLY_COEFF2, mask);
    // step 4: r = r* s+ -1.66666672e-1f;
    FMaf(dstReg, srcRegS, SIN_POLY_COEFF3, mask);
    // step 5: t = a* s+ 0.0f;
    FMaf(tmpRegT, srcRegA, srcRegS, SIN_POLY_COEFF5, mask);
    // step 6: r = r* t+ a;
    MicroAPI::FusedMulDstAdd(dstReg, tmpRegT, srcRegA, mask);
}

__simd_callee__ inline void CosfPoly(MicroAPI::RegTensor<float> &dstReg, MicroAPI::RegTensor<float> &srcRegS,
    MicroAPI::MaskReg& mask)
{
    constexpr float COS_POLY_COEFF0 = 2.44677067e-5f;
    constexpr float COS_POLY_COEFF1 = -1.38877297e-3f;
    constexpr float COS_POLY_COEFF2 = 4.16666567e-2f;
    constexpr float COS_POLY_COEFF3 = -5.00000000e-1f;
    constexpr float COS_POLY_COEFF4 = 1.00000000e+0f;

    // step 1: r = 2.44677067e-5f; 
    MicroAPI::Duplicate(dstReg, COS_POLY_COEFF0, mask);
    // step 2: r = r* s+ -1.38877297e-3f;
    FMaf(dstReg, srcRegS, COS_POLY_COEFF1, mask);
    // step 3: r = r* s+  4.16666567e-2f;
    FMaf(dstReg, srcRegS, COS_POLY_COEFF2, mask);
    // step 4: r = r* s+ -5.00000000e-1f;
    FMaf(dstReg, srcRegS, COS_POLY_COEFF3, mask);
    // step 5: r = r* s+  1.00000000e+0f;
    FMaf(dstReg, srcRegS, COS_POLY_COEFF4, mask);
}

__simd_callee__ inline void TrigRedFPreporcessForHalf(MicroAPI::RegTensor<float> &regR, MicroAPI::RegTensor<int32_t> &regI,
    MicroAPI::RegTensor<float> &srcRegA, MicroAPI::MaskReg& mask)
{
    constexpr float J_MUL_COEFF = 0.636619747f;
    constexpr float J_ADD_COEFF = 12582912.0f;
    constexpr float J_MUL_COEFF1 = -1.57079601e+00f;
    constexpr float J_MUL_COEFF2 = -3.13916473e-07f;
    constexpr float J_MUL_COEFF3 = -5.39030253e-15f;

    MicroAPI::RegTensor<float> regJ;
    MicroAPI::RegTensor<float> tmpF32Reg;
    MicroAPI::RegTensor<int32_t> tmpI32Reg;

    // step 1: a = a * 0.0f + a; convert inf to NAN
    MicroAPI::Duplicate(tmpF32Reg, 0.0f, mask);
    MicroAPI::FusedMulDstAdd(srcRegA, tmpF32Reg, srcRegA, mask);

    // step 2: j = a*0.636619747f + 12582912.0f;
    FMaf(regJ, srcRegA, J_MUL_COEFF, J_ADD_COEFF, mask);
    // step 3: i = reinterpret_cast<int&> (j);
    regI = (MicroAPI::RegTensor<int32_t>&)regJ;
    // step 4: j = j - 12582912.0f;
    MicroAPI::Adds(regJ, regJ, -J_ADD_COEFF, mask);
    // step 5: r = j* -1.57079601e+00f+ a; // -0x1.921fb0p+00 // pio2_high
    FMaf(regR, regJ, J_MUL_COEFF1, srcRegA, mask);
    // step 6: r = j* -3.13916473e-07f+ r; // -0x1.5110b4p-22 // pio2_mid
    MicroAPI::Duplicate(tmpF32Reg, J_MUL_COEFF2, mask);
    MicroAPI::MulAddDst(regR, regJ, tmpF32Reg, mask);
    // step 7: r = j* -5.39030253e-15f+ r; // -0x1.846988p-48 // pio2_low
    MicroAPI::Duplicate(tmpF32Reg, J_MUL_COEFF3, mask);
    MicroAPI::MulAddDst(regR, regJ, tmpF32Reg, mask);
}

__simd_callee__ inline void TrigRedFComputeP(MicroAPI::MaskReg& tmpMask, MicroAPI::RegTensor<uint32_t> &regPHigh,
    MicroAPI::RegTensor<uint32_t> &regPLow, MicroAPI::RegTensor<float> &srcRegA,
    MicroAPI::RegTensor<uint32_t> &oneOverPiFReg, MicroAPI::MaskReg& mask)
{
    constexpr float A_ABS_COEFF = 3.1415926535f*0.25f;
    MicroAPI::RegTensor<float> tmpF32Reg;
    MicroAPI::RegTensor<int32_t> tmpI32Reg;
    /* step 8:
     * if (std::abs(a) > 3.1415926535f*0.25f) {
     *     r = trig_red_slowpath_f (a, &i);
     * }
     */
    MicroAPI::Abs(tmpF32Reg, srcRegA, mask);
    MicroAPI::CompareScalar<float, CMPMODE::GT>(tmpMask, tmpF32Reg, A_ABS_COEFF, mask);
    TrigRedSlowpathFComputeP(regPHigh, regPLow, srcRegA, oneOverPiFReg, mask);
}

__simd_callee__ inline void TrigRedFComputeRI(MicroAPI::MaskReg& tmpMask, MicroAPI::RegTensor<float> &dstRegR,
    MicroAPI::RegTensor<int32_t> &dstRegI, MicroAPI::RegTensor<uint32_t> &regPHigh, MicroAPI::RegTensor<uint32_t> &regPLow,
    MicroAPI::RegTensor<float> &srcRegA, MicroAPI::MaskReg& mask)
{
    constexpr float A_ABS_COEFF = 3.1415926535f*0.25f;
    MicroAPI::RegTensor<float> tmpF32Reg;
    MicroAPI::RegTensor<int32_t> tmpI32Reg;
    /* step 8:
     * if (std::abs(a) > 3.1415926535f*0.25f) {
     *     r = trig_red_slowpath_f (a, &i);
     * }
     */
    MicroAPI::Abs(tmpF32Reg, srcRegA, mask);
    MicroAPI::CompareScalar<float, CMPMODE::GT>(tmpMask, tmpF32Reg, A_ABS_COEFF, mask);
    TrigRedSlowpathFComputeRI(dstRegR, dstRegI, regPHigh, regPLow, srcRegA, mask);
}

/* Compute sine and cosine simultaneously, based on quadrant */
__simd_callee__ inline void SCFCore(MicroAPI::RegTensor<float> &dstRegSin, MicroAPI::RegTensor<float> &dstRegCos,
    MicroAPI::RegTensor<int32_t> &regI, MicroAPI::RegTensor<float> &regR, MicroAPI::MaskReg& mask)
{
    constexpr int32_t I_AND_CONDITION = 2;

    // step 9: float c, s, t;
    MicroAPI::RegTensor<float> regC, regS, regT;
    MicroAPI::RegTensor<float> tmpF32Reg, tmpF32Reg1;
    MicroAPI::RegTensor<int32_t> tmpI32Reg;
    // step 10: s = r * r;
    MicroAPI::Mul(regS, regR, regR, mask);
    // step 11: c = cosf_poly (s);
    CosfPoly(regC, regS, mask);
    // step 12: s = sinf_poly (r, s);
    SinfPoly(tmpF32Reg, regR, regS, mask);
    regS = tmpF32Reg;

    /* step 13:
     * if (i & 2) {
     *    s = 0.0f - s; // don't change "sign" of NaNs or create negative zeros
     *    c = 0.0f - c; // don't change "sign" of NaNs or create negative zeros
     * }
     */
    MicroAPI::MaskReg tmpMask;
    MicroAPI::Duplicate(tmpI32Reg, I_AND_CONDITION, mask);
    MicroAPI::And(tmpI32Reg, regI, tmpI32Reg, mask);
    MicroAPI::CompareScalar<int32_t, CMPMODE::GT>(tmpMask, tmpI32Reg, 0, mask);
    MicroAPI::Duplicate(tmpF32Reg1, 0.0f, mask);
    MicroAPI::Sub(tmpF32Reg, tmpF32Reg1, regS, mask);
    MicroAPI::Select(regS, tmpF32Reg, regS, tmpMask);
    MicroAPI::Sub(tmpF32Reg, tmpF32Reg1, regC, mask);
    MicroAPI::Select(regC, tmpF32Reg, regC, tmpMask);

    /* step 14:
     * if (i & 1) {
     *    t = 0.0f - s; // don't change "sign" of NaNs or create negative zeros
     *    s = c;
     *    c = t;
     * }
     */
    MicroAPI::Duplicate(tmpI32Reg, 1, mask);
    MicroAPI::And(tmpI32Reg, regI, tmpI32Reg, mask);
    MicroAPI::CompareScalar<int32_t, CMPMODE::GT>(tmpMask, tmpI32Reg, 0, mask);
    MicroAPI::Duplicate(tmpF32Reg, 0.0f, mask);
    MicroAPI::Sub(tmpF32Reg, tmpF32Reg, regS, mask);
    MicroAPI::Select(regT, tmpF32Reg, regT, tmpMask);
    MicroAPI::Select(regS, regC, regS, tmpMask);
    MicroAPI::Select(regC, regT, regC, tmpMask);

    // step 15: *sp = s;  //sp is the sin result
    dstRegSin = regS;
    // step 16: *cp = c;  //cp is the cos result
    dstRegCos = regC;
}

__aicore__ inline void InitializeFloatTempBuffer(__ubuf__ uint32_t *&tmpBuffer, __ubuf__ float *&tmpBufferR,
    __ubuf__ int32_t *&tmpBufferI, const uint32_t alignCount)
{
    constexpr uint32_t oneOverPiFAlignedLength = 8;
    static unsigned int oneOverPiF[6] =
    {
        0x28be60db, 0x9391054a, 0x7f09d5f4,
        0x7d4d3770, 0x36d8a566, 0x4f10e410
    };

    for (uint16_t i = 0; i < 6; ++i) {
        tmpBuffer[i] = oneOverPiF[i];
    }

    tmpBufferR = (__ubuf__ float *)((__ubuf__ uint8_t *)tmpBuffer + sizeof(uint32_t) * oneOverPiFAlignedLength);
    tmpBufferI = (__ubuf__ int32_t *)((__ubuf__ uint8_t *)tmpBufferR + sizeof(float) * alignCount);
}

__aicore__ inline void InitializeHalfTempBuffer(__ubuf__ uint32_t *&tmpBuffer, __ubuf__ float *&tmpBufferR,
    __ubuf__ int32_t *&tmpBufferI, const uint32_t alignCount)
{
    tmpBufferR = (__ubuf__ float *)((__ubuf__ uint8_t *)tmpBuffer);
    tmpBufferI = (__ubuf__ int32_t *)((__ubuf__ uint8_t *)tmpBufferR + sizeof(float) * alignCount);
}

template <typename T>
__simd_vf__ inline void TrigRedFPreProcessImpl(__ubuf__ float *tmpBufferR, __ubuf__ int32_t *tmpBufferI,
    __ubuf__ T *src,  uint32_t calCount, uint16_t repeatTimes)
{
    MicroAPI::RegTensor<T> srcReg;
    MicroAPI::RegTensor<int32_t> regI;
    MicroAPI::RegTensor<float> castReg, regR;

    for (uint16_t i = 0; i < repeatTimes; i++) {
        MicroAPI::MaskReg mask = MicroAPI::UpdateMask<float>(calCount);
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcReg, src + i * B32_DATA_NUM_PER_REPEAT);
        MicroAPI::Cast<float, T, SinCosImpl::castTraitF16F32>(castReg, srcReg, mask);
        TrigRedFPreporcessForHalf(regR, regI, castReg, mask);

        MicroAPI::StoreAlign(tmpBufferR + i * B32_DATA_NUM_PER_REPEAT, regR, mask);
        MicroAPI::StoreAlign(tmpBufferI + i * B32_DATA_NUM_PER_REPEAT, regI, mask);
    }
}

template <typename T>
__simd_vf__ inline void TrigRedFComputePImpl(__ubuf__ uint32_t *tmpBufferRegPHigh, __ubuf__ uint32_t *tmpBufferRegPLow,
    __ubuf__ T *src, __ubuf__ uint32_t *tmpBuffer, uint32_t calCount, uint16_t repeatTimes)
{
    MicroAPI::RegTensor<T> srcReg;
    MicroAPI::RegTensor<uint32_t> oneOverPiFReg, regPHigh, regPLow, regI;
    MicroAPI::RegTensor<float> castReg, regR, tmpF32Reg;
    MicroAPI::MaskReg selectMask;

    // Load the array of one_over_pi_f
    MicroAPI::LoadAlign(oneOverPiFReg, tmpBuffer);

    for (uint16_t i = 0; i < repeatTimes; i++) {
        MicroAPI::MaskReg mask = MicroAPI::UpdateMask<float>(calCount);
        MicroAPI::LoadAlign(castReg, src + i * B32_DATA_NUM_PER_REPEAT);
        // a = a * 0.0f + a
        MicroAPI::Duplicate(tmpF32Reg, 0.0f, mask);
        MicroAPI::FusedMulDstAdd(castReg, tmpF32Reg, castReg, mask);
        // initilize q and r: *q = 0; r = a;
        MicroAPI::Duplicate(regI, 0, mask);
        regR = castReg;
        // store the origin q and r in ub
        MicroAPI::StoreAlign((__ubuf__ float *)tmpBufferRegPHigh + i * B32_DATA_NUM_PER_REPEAT, regR, mask);
        MicroAPI::StoreAlign(tmpBufferRegPLow + i * B32_DATA_NUM_PER_REPEAT, regI, mask);

        TrigRedFComputeP(selectMask, regPHigh, regPLow, castReg, oneOverPiFReg, mask);

        MicroAPI::StoreAlign(tmpBufferRegPHigh + i * B32_DATA_NUM_PER_REPEAT, regPHigh, selectMask);
        MicroAPI::StoreAlign(tmpBufferRegPLow + i * B32_DATA_NUM_PER_REPEAT, regPLow, selectMask);
    }
}

template <typename T>
__simd_vf__ inline void TrigRedFComputeRIImpl(__ubuf__ uint32_t *tmpBufferRegPHigh, __ubuf__ uint32_t *tmpBufferRegPLow,
    __ubuf__ T *src, __ubuf__ uint32_t *tmpBuffer, uint32_t calCount, uint16_t repeatTimes)
{
    MicroAPI::RegTensor<T> srcReg;
    MicroAPI::RegTensor<uint32_t> regPHigh, regPLow;
    MicroAPI::RegTensor<int32_t> regI;
    MicroAPI::RegTensor<float> castReg, regR, tmpF32Reg;
    MicroAPI::MaskReg selectMask;

    for (uint16_t i = 0; i < repeatTimes; i++) {
        MicroAPI::MaskReg mask = MicroAPI::UpdateMask<float>(calCount);
        MicroAPI::LoadAlign(castReg, src + i * B32_DATA_NUM_PER_REPEAT);
        MicroAPI::LoadAlign(regPHigh, tmpBufferRegPHigh + i * B32_DATA_NUM_PER_REPEAT);
        MicroAPI::LoadAlign(regPLow, tmpBufferRegPLow + i * B32_DATA_NUM_PER_REPEAT);
        // a = a * 0.0f + a
        MicroAPI::Duplicate(tmpF32Reg, 0.0f, mask);
        MicroAPI::FusedMulDstAdd(castReg, tmpF32Reg, castReg, mask);

        TrigRedFComputeRI(selectMask, regR, regI, regPHigh, regPLow, castReg, mask);

        MicroAPI::StoreAlign((__ubuf__ float *)tmpBufferRegPHigh + i * B32_DATA_NUM_PER_REPEAT, regR, selectMask);
        MicroAPI::StoreAlign((__ubuf__ int32_t *)tmpBufferRegPLow + i * B32_DATA_NUM_PER_REPEAT, regI, selectMask);
    }
}

template <typename T, int mode = 0>
__simd_vf__ inline void SCFCoreImpl(__ubuf__ T *dst, __ubuf__ float *tmpBufferR, __ubuf__ int32_t *tmpBufferI,
    uint32_t calCount, uint16_t repeatTimes)
{
    MicroAPI::RegTensor<T> srcReg;
    MicroAPI::RegTensor<int32_t> regI;
    MicroAPI::RegTensor<float> regR;
    MicroAPI::RegTensor<float> dstRegCos, dstRegSin, dstReg;

    for (uint16_t i = 0; i < repeatTimes; i++) {
        MicroAPI::MaskReg mask = MicroAPI::UpdateMask<float>(calCount);
        MicroAPI::LoadAlign(regR, tmpBufferR + i * B32_DATA_NUM_PER_REPEAT);
        MicroAPI::LoadAlign(regI, tmpBufferI + i * B32_DATA_NUM_PER_REPEAT);

        SCFCore(dstRegSin, dstRegCos, regI, regR, mask);

        if constexpr (mode == 0) {
            dstReg = dstRegSin;
        } else {
            dstReg = dstRegCos;
        }

        if constexpr (sizeof(T) == sizeof(half)) {
            MicroAPI::Cast<T, float, SinCosImpl::castTraitF32F16>(srcReg, dstReg, mask);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(dst + i * B32_DATA_NUM_PER_REPEAT, srcReg, mask);
        } else {
            MicroAPI::StoreAlign(dst + i * B32_DATA_NUM_PER_REPEAT, dstReg, mask);
        }
    }
}

template <typename T>
__simd_vf__ inline void BSCFCoreImpl(__ubuf__ T *dstSin, __ubuf__ T *dstCos, __ubuf__ float *tmpBufferR, __ubuf__ int32_t *tmpBufferI,
    uint32_t calCount, uint16_t repeatTimes)
{
    MicroAPI::RegTensor<T> srcReg;
    MicroAPI::RegTensor<int32_t> regI;
    MicroAPI::RegTensor<float> regR;
    MicroAPI::RegTensor<float> dstRegCos, dstRegSin;

    for (uint16_t i = 0; i < repeatTimes; i++) {
        MicroAPI::MaskReg mask = MicroAPI::UpdateMask<float>(calCount);
        MicroAPI::LoadAlign(regR, tmpBufferR + i * B32_DATA_NUM_PER_REPEAT);
        MicroAPI::LoadAlign(regI, tmpBufferI + i * B32_DATA_NUM_PER_REPEAT);

        SCFCore(dstRegSin, dstRegCos, regI, regR, mask);

        if constexpr (sizeof(T) == sizeof(half)) {
            MicroAPI::Cast<T, float, SinCosImpl::castTraitF32F16>(srcReg, dstRegSin, mask);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(dstSin + i * B32_DATA_NUM_PER_REPEAT, srcReg, mask);
            MicroAPI::Cast<T, float, SinCosImpl::castTraitF32F16>(srcReg, dstRegCos, mask);
            MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_PACK_B32>(dstCos + i * B32_DATA_NUM_PER_REPEAT, srcReg, mask);
        } else {
            MicroAPI::StoreAlign(dstSin + i * B32_DATA_NUM_PER_REPEAT, dstRegSin, mask);
            MicroAPI::StoreAlign(dstCos + i * B32_DATA_NUM_PER_REPEAT, dstRegCos, mask);
        }
    }
}
} // namespace SinCosImpl

template <typename T> 
__aicore__ inline void SinRadianReductionImpl(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ uint32_t *tmpBuffer, uint32_t calCount)
{
    static_assert((std::is_same_v<T, half> || std::is_same_v<T, float>),
        "current data type is not supported on current device!");
    constexpr uint32_t oneRepSize = GetVecLen() / sizeof(float);
    uint16_t repeatTimes = CeilDivision(calCount, oneRepSize);
    __ubuf__ float *tmpBufferR;
    __ubuf__ int32_t *tmpBufferI;
    uint32_t alignCount = (calCount + 31) / 32 * 32;

    if constexpr (std::is_same_v<T, float>) {
        SinCosImpl::InitializeFloatTempBuffer(tmpBuffer, tmpBufferR, tmpBufferI, alignCount);
        SinCosImpl::TrigRedFComputePImpl<T>(
            (__ubuf__ uint32_t *)tmpBufferR, (__ubuf__ uint32_t *)tmpBufferI, src, tmpBuffer, calCount, repeatTimes);
        SinCosImpl::TrigRedFComputeRIImpl<T>(
            (__ubuf__ uint32_t *)tmpBufferR, (__ubuf__ uint32_t *)tmpBufferI, src, tmpBuffer, calCount, repeatTimes);
    } else if constexpr (std::is_same_v<T, half>) {
        SinCosImpl::InitializeHalfTempBuffer(tmpBuffer, tmpBufferR, tmpBufferI, alignCount);
        SinCosImpl::TrigRedFPreProcessImpl<T>(tmpBufferR, tmpBufferI, src, calCount, repeatTimes);
    }
    SinCosImpl::SCFCoreImpl<T, 0>(dst, tmpBufferR, tmpBufferI, calCount, repeatTimes);
}

template <typename T> 
__aicore__ inline void CosRadianReductionImpl(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ uint32_t *tmpBuffer, uint32_t calCount)
{
    static_assert((std::is_same_v<T, half> || std::is_same_v<T, float>),
        "current data type is not supported on current device!");
    constexpr uint32_t oneRepSize = GetVecLen() / sizeof(float);
    uint16_t repeatTimes = CeilDivision(calCount, oneRepSize);
    __ubuf__ float *tmpBufferR;
    __ubuf__ int32_t *tmpBufferI;
    uint32_t alignCount = (calCount + 31) / 32 * 32;

    if constexpr (std::is_same_v<T, float>) {
        SinCosImpl::InitializeFloatTempBuffer(tmpBuffer, tmpBufferR, tmpBufferI, alignCount);
        SinCosImpl::TrigRedFComputePImpl<T>(
            (__ubuf__ uint32_t *)tmpBufferR, (__ubuf__ uint32_t *)tmpBufferI, src, tmpBuffer, calCount, repeatTimes);
        SinCosImpl::TrigRedFComputeRIImpl<T>(
            (__ubuf__ uint32_t *)tmpBufferR, (__ubuf__ uint32_t *)tmpBufferI, src, tmpBuffer, calCount, repeatTimes);
    } else if constexpr (std::is_same_v<T, half>) {
        SinCosImpl::InitializeHalfTempBuffer(tmpBuffer, tmpBufferR, tmpBufferI, alignCount);
        SinCosImpl::TrigRedFPreProcessImpl<T>(tmpBufferR, tmpBufferI, src, calCount, repeatTimes);
    }
    SinCosImpl::SCFCoreImpl<T, 1>(dst, tmpBufferR, tmpBufferI, calCount, repeatTimes);
}

template <const SinCosConfig& config, typename T>
__aicore__ inline void SinCosRadianReductionImpl(const LocalTensor<T>& dst0, const LocalTensor<T>& dst1, 
    const LocalTensor<T>& src, const LocalTensor<uint8_t>& sharedTmpBuffer, uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    CHECK_FUNC_HIGHLEVEL_API(SinCos, (T, config.isReuseSource), (dst0, dst1, src, sharedTmpBuffer, calCount));
    __ubuf__ T* dstSinAddr = (__ubuf__ T*)dst0.GetPhyAddr();
    __ubuf__ T* dstCosAddr = (__ubuf__ T*)dst1.GetPhyAddr();
    __ubuf__ T* srcAddr = (__ubuf__ T*)src.GetPhyAddr();
    __ubuf__ uint32_t* tmpBuffer = (__ubuf__ uint32_t*)sharedTmpBuffer.GetPhyAddr();
    static_assert((std::is_same_v<T, half> || std::is_same_v<T, float>),
        "current data type is not supported on current device!");
    constexpr uint32_t oneRepSize = GetVecLen() / sizeof(float);
    uint16_t repeatTimes = CeilDivision(calCount, oneRepSize);
    __ubuf__ float *tmpBufferR;
    __ubuf__ int32_t *tmpBufferI;
    uint32_t alignCount = (calCount + 31) / 32 * 32;

    if constexpr (std::is_same_v<T, float>) {
        SinCosImpl::InitializeFloatTempBuffer(tmpBuffer, tmpBufferR, tmpBufferI, alignCount);
        SinCosImpl::TrigRedFComputePImpl<T>(
            (__ubuf__ uint32_t *)tmpBufferR, (__ubuf__ uint32_t *)tmpBufferI, srcAddr, tmpBuffer, calCount, repeatTimes);
        SinCosImpl::TrigRedFComputeRIImpl<T>(
            (__ubuf__ uint32_t *)tmpBufferR, (__ubuf__ uint32_t *)tmpBufferI, srcAddr, tmpBuffer, calCount, repeatTimes);
    } else if constexpr (std::is_same_v<T, half>) {
        SinCosImpl::InitializeHalfTempBuffer(tmpBuffer, tmpBufferR, tmpBufferI, alignCount);
        SinCosImpl::TrigRedFPreProcessImpl<T>(tmpBufferR, tmpBufferI, srcAddr, calCount, repeatTimes);
    }
    SinCosImpl::BSCFCoreImpl<T>(dstSinAddr, dstCosAddr, tmpBufferR, tmpBufferI, calCount, repeatTimes);
}

template <const SinCosConfig& config, typename T>
__aicore__ inline void SinCosRadianReductionImpl(const LocalTensor<T>& dst0, const LocalTensor<T>& dst1, 
    const LocalTensor<T>& src, uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    LocalTensor<uint8_t> sharedTmpBuffer;
    bool ans = PopStackBuffer<uint8_t, TPosition::LCM>(sharedTmpBuffer);
    ASCENDC_ASSERT((ans), { KERNEL_LOG(KERNEL_ERROR, "PopStackBuffer Error!"); });
    SinCosRadianReductionImpl<config, T>(dst0, dst1, src, sharedTmpBuffer, calCount);
}
} // namespace AscendC

#endif // LIB_MATH_SINCOS_SINCOS_C310_IMPL_H
