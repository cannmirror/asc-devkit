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

/*!
 * \file lgamma_c310_impl.h
 * \brief
 */
#ifndef IMPL_MATH_LGAMMA_LGAMMA_C310_IMPL_H
#define IMPL_MATH_LGAMMA_LGAMMA_C310_IMPL_H
#include "kernel_tensor.h"
#include "include/adv_api/math/sin.h"
#include "lgamma_common_utils.h"
#include "../../common/check.h"

namespace AscendC {
namespace LgammaInternal {
constexpr MicroAPI::CastTrait LGAMMA_CAST_TRAIT_F322S32 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
constexpr MicroAPI::CastTrait LGAMMA_CAST_TRAIT_F162F32 = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr MicroAPI::CastTrait LGAMMA_CAST_TRAIT_F322F16 = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
struct MulAddsParams {
    float r0;
    float r1;
    float r2;
    float r3;
    float r4;
    float r5;
    float r6;
    float r7;
    float r8;
    float r9;
    __aicore__ constexpr MulAddsParams(const float r0In, const float r1In, const float r2In, const float r3In,
                                       const float r4In, const float r5In, const float r6In, const float r7In,
                                       const float r8In, const float r9In)
    : r0(r0In), r1(r1In), r2(r2In), r3(r3In), r4(r4In), r5(r5In), r6(r6In), r7(r7In), r8(r8In), r9(r9In)
    {}
};

__aicore__ inline constexpr MulAddsParams GetConstants0715()
{
    constexpr float r0 = 0.0458826646209;
    constexpr float r1 = 0.103739671409;
    constexpr float r2 = 0.122803635895;
    constexpr float r3 = 0.127524212003;
    constexpr float r4 = 0.143216684461;
    constexpr float r5 = 0.169343575835;
    constexpr float r6 = 0.207407936454;
    constexpr float r7 = 0.27058750391;
    constexpr float r8 = 0.400685429573;
    constexpr float r9 = 0.82246696949;

    constexpr MulAddsParams params(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9);
    return params;
}

__aicore__ inline constexpr MulAddsParams GetConstants153()
{
    constexpr float r0 = 4.95984932058e-05;
    constexpr float r1 = -0.000220894842641;
    constexpr float r2 = 0.000541314249858;
    constexpr float r3 = -0.00120451697148;
    constexpr float r4 = 0.00288425176404;
    constexpr float r5 = -0.00738275796175;
    constexpr float r6 = 0.0205813199282;
    constexpr float r7 = -0.067352488637;
    constexpr float r8 = 0.322467029095;
    constexpr float r9 = 0.422784328461;

    constexpr MulAddsParams params(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9);
    return params;
}

__aicore__ inline void LGammaCalcMulAdd(MicroAPI::RegTensor<float>& resReg, MicroAPI::RegTensor<float>& tmpReg,
    MicroAPI::MaskReg mask, const MulAddsParams& params)
{
    MicroAPI::Muls(resReg, tmpReg, params.r0, mask);
    MicroAPI::Adds(resReg, resReg, params.r1, mask);
    MicroAPI::Mul(resReg, resReg, tmpReg, mask);
    MicroAPI::Adds(resReg, resReg, params.r2, mask);
    MicroAPI::Mul(resReg, resReg, tmpReg, mask);
    MicroAPI::Adds(resReg, resReg, params.r3, mask);
    MicroAPI::Mul(resReg, resReg, tmpReg, mask);
    MicroAPI::Adds(resReg, resReg, params.r4, mask);
    MicroAPI::Mul(resReg, resReg, tmpReg, mask);
    MicroAPI::Adds(resReg, resReg, params.r5, mask);
    MicroAPI::Mul(resReg, resReg, tmpReg, mask);
    MicroAPI::Adds(resReg, resReg, params.r6, mask);
    MicroAPI::Mul(resReg, resReg, tmpReg, mask);
    MicroAPI::Adds(resReg, resReg, params.r7, mask);
    MicroAPI::Mul(resReg, resReg, tmpReg, mask);
    MicroAPI::Adds(resReg, resReg, params.r8, mask);
    MicroAPI::Mul(resReg, resReg, tmpReg, mask);
    MicroAPI::Adds(resReg, resReg, params.r9, mask);
    MicroAPI::Mul(resReg, resReg, tmpReg, mask);
}

__aicore__ inline void Lgamma1Compute(MicroAPI::RegTensor<float>& resReg, MicroAPI::RegTensor<float>& tmpReg,
    MicroAPI::RegTensor<float>& aReg, MicroAPI::RegTensor<float>& bReg, MicroAPI::RegTensor<float>& oneReg,
    MicroAPI::MaskReg mask)
{
    /*
    inv_x = 1/x
    y = 0.5 * log(2 * pi * inv_x)
    y += x * (log(x + 1 / (12 * x - 0.1 * inv_x)) - 1)
    */

    // inv_x = 1 / x
    MicroAPI::Div(bReg, oneReg, tmpReg, mask);

    // y = 0.5 * log(2 * pi * inv_x)
    MicroAPI::Muls(resReg, bReg, PI, mask);
    MicroAPI::Muls(resReg, resReg, f2, mask);
    MicroAPI::Ln(resReg, resReg, mask);
    MicroAPI::Muls(resReg, resReg, f05, mask);

    // a = -0.1 * inv_x
    MicroAPI::Muls(aReg, bReg, N01, mask);
    // b = x * 12
    MicroAPI::Muls(bReg, tmpReg, t12, mask);
    // b += a -> 12 * x - 0.1 * inv_x
    MicroAPI::Add(bReg, bReg, aReg, mask);
    // b = 1/b -> 1/(12 * x - 0.1 * inv_x)
    MicroAPI::Div(bReg, oneReg, bReg, mask);
    // b = x + b -> x + 1/(12 * x - 0.1 * inv_x)
    MicroAPI::Add(bReg, bReg, tmpReg, mask);
    // b = ln(b) -> log(x + 1/(12 * x - 0.1 * inv_x))
    MicroAPI::Ln(bReg, bReg, mask);
    // b -= 1 -> log(x + 1/(12 * x - 0.1 * inv_x)) - 1
    MicroAPI::Adds(bReg, bReg, fn1, mask);
    // b = bx -> x * (log(x + 1 / (12 * x - 0.1 * inv_x)) - 1)
    MicroAPI::Mul(bReg, bReg, tmpReg, mask);
    // y += x * (log(x + 1 / (12 * x - 0.1 * inv_x)) - 1)
    MicroAPI::Add(resReg, resReg, bReg, mask);
}

__aicore__ inline void LgammaComputePosHalf(MicroAPI::RegTensor<float>& resReg, MicroAPI::RegTensor<float>& srcReg,
    MicroAPI::RegTensor<float>& tmpReg, MicroAPI::RegTensor<float>& aReg, MicroAPI::RegTensor<float>& bReg,
    MicroAPI::RegTensor<float>& oneReg, MicroAPI::MaskReg mask)
{
    // y = lgamma1(x+4) - log(x) - log(x+1) - log(x+2) - log(x+3)

    // lgamma1(x + 4)
    MicroAPI::Adds(tmpReg, srcReg, t4, mask);
    Lgamma1Compute(resReg, tmpReg, aReg, bReg, oneReg, mask);

    // a = log(x)
    MicroAPI::Ln(aReg, srcReg, mask);
    // b = log(x + 1)
    MicroAPI::Adds(bReg, srcReg, f1, mask);
    MicroAPI::Ln(bReg, bReg, mask);
    // a = a + b
    MicroAPI::Add(aReg, aReg, bReg, mask);

    // b = log(x + 2)
    MicroAPI::Adds(bReg, srcReg, f2, mask);
    MicroAPI::Ln(bReg, bReg, mask);
    // a = a + b
    MicroAPI::Add(aReg, aReg, bReg, mask);

    // b = log(x + 3)
    MicroAPI::Adds(bReg, srcReg, f3, mask);
    MicroAPI::Ln(bReg, bReg, mask);
    // a = a + b
    MicroAPI::Add(aReg, aReg, bReg, mask);

    // y = lgamma1(x + 4) - a
    MicroAPI::Sub(resReg, resReg, aReg, mask);
}

__aicore__ inline void LgammaComputeNegHalf(MicroAPI::RegTensor<float>& resReg, MicroAPI::RegTensor<float>& srcReg,
    MicroAPI::RegTensor<float>& tmpReg, MicroAPI::RegTensor<float>& aReg, MicroAPI::RegTensor<float>& bReg,
    MicroAPI::RegTensor<float>& cReg, MicroAPI::RegTensor<float>& oneReg, MicroAPI::RegTensor<float>& piReg,
    MicroAPI::MaskReg mask)
{
    // y = log(pi / |sin(pi * (x - floor(x)|) - LgammaComputePosHalf(1 - x)

    // LgammaComputePosHalf(1 - x)
    MicroAPI::Sub(tmpReg, oneReg, srcReg, mask);
    LgammaComputePosHalf(resReg, tmpReg, aReg, bReg, cReg, oneReg, mask);

    // log(pi / abs((sin(pi * (x - floor(x))))))
    // floor(x)
    MicroAPI::Truncate<float, RoundMode::CAST_FLOOR, MicroAPI::MaskMergeMode::ZEROING>(tmpReg, srcReg, mask);
    // a = x - floor(x)
    MicroAPI::Sub(aReg, srcReg, tmpReg, mask);
    // a = pi * a -> pi * (x - floor(x))
    MicroAPI::Muls(aReg, aReg, PI, mask);
    // b = sin(pi * (x - floor(x)))
    MicroAPI::Sin::SinPolynomialApproximation(bReg, aReg, aReg, tmpReg, cReg, mask);
    MicroAPI::Abs(bReg, bReg, mask);
    // b = pi / |sin(pi * (x - floor(x)))|
    MicroAPI::Div(bReg, piReg, bReg, mask);
    // b = ln(pi / |sin(pi * (x - floor(x)))|)
    MicroAPI::Ln(bReg, bReg, mask);

    // log(pi / |sin(pi * (x - floor(x)|) - LgammaComputePosHalf(1 - x)
    MicroAPI::Sub(resReg, bReg, resReg, mask);
}

// cal result of 0 <= x < 0.7, Ln return -inf when x is 0
__aicore__ inline void LGamma007(MicroAPI::RegTensor<float>& resReg, MicroAPI::RegTensor<float>& srcReg,
    MicroAPI::RegTensor<float>& tmpReg, MicroAPI::MaskReg mask)
{
    constexpr float r0 = 0.00358751555905;
    constexpr float r1 = -0.00547128543258;
    constexpr float r2 = -0.0446271263063;
    constexpr float r3 = 0.167317703366;
    constexpr float r4 = -0.0421359799802;
    constexpr float r5 = -0.655867278576;
    constexpr float r6 = 0.577215373516;

    // y = ((((((r0 * x + r1) * x + r2) * x + r3) * x + r4) * x + r5) * x + r6) * x * x + x
    MicroAPI::Muls(tmpReg, srcReg, r0, mask);
    MicroAPI::Adds(tmpReg, tmpReg, r1, mask);
    MicroAPI::Mul(tmpReg, tmpReg, srcReg, mask);
    MicroAPI::Adds(tmpReg, tmpReg, r2, mask);
    MicroAPI::Mul(tmpReg, tmpReg, srcReg, mask);
    MicroAPI::Adds(tmpReg, tmpReg, r3, mask);
    MicroAPI::Mul(tmpReg, tmpReg, srcReg, mask);
    MicroAPI::Adds(tmpReg, tmpReg, r4, mask);
    MicroAPI::Mul(tmpReg, tmpReg, srcReg, mask);
    MicroAPI::Adds(tmpReg, tmpReg, r5, mask);
    MicroAPI::Mul(tmpReg, tmpReg, srcReg, mask);
    MicroAPI::Adds(tmpReg, tmpReg, r6, mask);
    MicroAPI::Mul(resReg, srcReg, srcReg, mask);
    MicroAPI::Mul(resReg, resReg, tmpReg, mask);
    MicroAPI::Add(tmpReg, resReg, srcReg, mask);
    // y = -ln(y)
    MicroAPI::Ln(resReg, tmpReg, mask);
    MicroAPI::Muls(resReg, resReg, fn1, mask);
}

// cal result of 0.7 <= x < 1.5
__aicore__ inline void LGamma0715(MicroAPI::RegTensor<float>& resReg, MicroAPI::RegTensor<float>& tmpReg,
    MicroAPI::MaskReg mask)
{
    constexpr float r10 = 0.577215671539;
    constexpr MulAddsParams params = GetConstants0715();
    // y = ((((((((((r0 * x + r1) * x + r2) * x + r3) * x + r4) * x + r5) * x
    //                     + r6) * x + r7) * x + r8) * x + r9) * x + r10) * x
    LGammaCalcMulAdd(resReg, tmpReg, mask, params);
    MicroAPI::Adds(resReg, resReg, r10, mask);
    MicroAPI::Mul(resReg, resReg, tmpReg, mask);
}

// cal result of 1.5 <= x < 3
__aicore__ inline void LGamma153(MicroAPI::RegTensor<float>& resReg, MicroAPI::RegTensor<float>& tmpReg,
    MicroAPI::MaskReg mask)
{
    constexpr MulAddsParams params = GetConstants153();
    // y = (((((((((r0 * x + r1) * x + r2) * x + r3) * x + r4) * x
    //           + r5) * x + r6) * x + r7) * x + r8) * x + r9) * x
    LGammaCalcMulAdd(resReg, tmpReg, mask, params);
}

// cal result of 3 <= x < 5.8
__aicore__ inline void LGamma358(MicroAPI::RegTensor<float>& resReg, MicroAPI::RegTensor<float>& tmpReg,
    MicroAPI::RegTensor<float>& aReg, MicroAPI::RegTensor<float>& bReg, MicroAPI::MaskReg mask)
{
    constexpr float r0 = -748.890319824;
    constexpr float r1 = -12349.7421875;
    constexpr float r2 = -41061.375;
    constexpr float r3 = -48310.6640625;
    constexpr float ftmp2 = -143033.40625;
    constexpr float r4 = -259.250976562;
    constexpr float r5 = -10777.1796875;
    constexpr float r6 = -92685.046875;
    constexpr float ftmp3 = -206353.578125;

    // a = (((r0 * x + r1) * x + r2) * x + r3) * x + ftmp2
    MicroAPI::Muls(aReg, tmpReg, r0, mask);
    MicroAPI::Adds(aReg, aReg, r1, mask);
    MicroAPI::Mul(aReg, aReg, tmpReg, mask);
    MicroAPI::Adds(aReg, aReg, r2, mask);
    MicroAPI::Mul(aReg, aReg, tmpReg, mask);
    MicroAPI::Adds(aReg, aReg, r3, mask);
    MicroAPI::Mul(aReg, aReg, tmpReg, mask);
    MicroAPI::Adds(aReg, aReg, ftmp2, mask);

    // b = ((r4 * x + r5) * x + r6) * x + ftmp3
    MicroAPI::Muls(bReg, tmpReg, r4, mask);
    MicroAPI::Adds(bReg, bReg, r5, mask);
    MicroAPI::Mul(bReg, bReg, tmpReg, mask);
    MicroAPI::Adds(bReg, bReg, r6, mask);
    MicroAPI::Mul(bReg, bReg, tmpReg, mask);
    MicroAPI::Adds(bReg, bReg, ftmp3, mask);

    // y = a / b + x
    MicroAPI::Div(resReg, aReg, bReg, mask);
    MicroAPI::Add(resReg, resReg, tmpReg, mask);
}

// cal result of x >= 5.8
__aicore__ inline void LGamma58(MicroAPI::RegTensor<float>& resReg, MicroAPI::RegTensor<float>& srcReg,
    MicroAPI::RegTensor<float>& tmpReg, MicroAPI::RegTensor<float>& aReg, MicroAPI::RegTensor<float>& bReg,
    MicroAPI::RegTensor<float>& cReg, MicroAPI::RegTensor<float>& oneReg, MicroAPI::MaskReg mask)
{
    constexpr float r0 = 0.000777830660809;
    constexpr float r1 = -0.00277765537612;
    constexpr float ftmp1 = 0.0833332762122;
    constexpr float ftmp2 = 0.91893851757;

    // tmp = ln(x) * 0.5
    MicroAPI::Ln(tmpReg, srcReg, mask);
    MicroAPI::Muls(tmpReg, tmpReg, f05, mask);

    // a = tmp * (x - 0.5)
    MicroAPI::Adds(aReg, srcReg, fn05, mask);
    MicroAPI::Mul(aReg, tmpReg, aReg, mask);

    // b = 1/x
    MicroAPI::Div(bReg, oneReg, srcReg, mask);

    // c = b * b
    MicroAPI::Mul(cReg, bReg, bReg, mask);

    // res = ((r0 * c + r1) * c + 0.0833332762122) * b + 0.91893851757
    MicroAPI::Muls(resReg, cReg, r0, mask);
    MicroAPI::Adds(resReg, resReg, r1, mask);
    MicroAPI::Mul(resReg, resReg, cReg, mask);
    MicroAPI::Adds(resReg, resReg, ftmp1, mask);
    MicroAPI::Mul(resReg, resReg, bReg, mask);
    MicroAPI::Adds(resReg, resReg, ftmp2, mask);

    // res = res + a + a - x
    MicroAPI::Add(resReg, resReg, aReg, mask);
    MicroAPI::Add(resReg, resReg, aReg, mask);
    MicroAPI::Sub(resReg, resReg, srcReg, mask);
}

// cal for x >= 0 when data type is float
__aicore__ inline void LGammaPositive(MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& srcReg,
    MicroAPI::RegTensor<float>& resReg, MicroAPI::RegTensor<float>& tmpReg, MicroAPI::RegTensor<float>& aReg,
    MicroAPI::RegTensor<float>& bReg, MicroAPI::RegTensor<float>& cReg, MicroAPI::RegTensor<float>& oneReg,
    MicroAPI::MaskReg cmpMaskReg1, MicroAPI::MaskReg cmpMaskReg2, MicroAPI::MaskReg cmpMaskReg,
    MicroAPI::MaskReg mask)
{
    NotNumUnion notNum;
    notNum.i = F32_INF;
    // cal and select 0 <= x < 0.7 res
    MicroAPI::CompareScalar<float, CMPMODE::LT>(cmpMaskReg, srcReg, f07, mask);
    LGamma007(resReg, srcReg, tmpReg, mask);
    MicroAPI::Select(dstReg, resReg, srcReg, cmpMaskReg);

    // cal and select 0.7 <= x < 1.5 res
    MicroAPI::CompareScalar<float, CMPMODE::GE>(cmpMaskReg2, srcReg, f07, mask);
    MicroAPI::CompareScalar<float, CMPMODE::LT>(cmpMaskReg1, srcReg, f15, mask);
    MicroAPI::MaskAnd(cmpMaskReg, cmpMaskReg1, cmpMaskReg2, mask);
    MicroAPI::Sub(tmpReg, oneReg, srcReg, mask);
    LGamma0715(resReg, tmpReg, mask);
    MicroAPI::Select(dstReg, resReg, dstReg, cmpMaskReg);

    // cal and select 1.5 <= x < 3 res
    MicroAPI::CompareScalar<float, CMPMODE::GE>(cmpMaskReg2, srcReg, f15, mask);
    MicroAPI::CompareScalar<float, CMPMODE::LT>(cmpMaskReg1, srcReg, f3, mask);
    MicroAPI::MaskAnd(cmpMaskReg, cmpMaskReg1, cmpMaskReg2, mask);
    MicroAPI::Adds(tmpReg, srcReg, fn2, mask);
    LGamma153(resReg, tmpReg, mask);
    MicroAPI::Select(dstReg, resReg, dstReg, cmpMaskReg);

    // cal and select 3 <= x < 5.8 res
    MicroAPI::CompareScalar<float, CMPMODE::GE>(cmpMaskReg2, srcReg, f3, mask);
    MicroAPI::CompareScalar<float, CMPMODE::LT>(cmpMaskReg1, srcReg, f58, mask);
    MicroAPI::MaskAnd(cmpMaskReg, cmpMaskReg1, cmpMaskReg2, mask);
    MicroAPI::Adds(tmpReg, srcReg, fn3, mask);
    LGamma358(resReg, tmpReg, aReg, bReg, mask);
    MicroAPI::Select(dstReg, resReg, dstReg, cmpMaskReg);

    // cal and select 5.8 <= x < inf res
    MicroAPI::CompareScalar<float, CMPMODE::GE>(cmpMaskReg2, srcReg, f58, mask);
    MicroAPI::CompareScalar<float, CMPMODE::LT>(cmpMaskReg1, srcReg, notNum.f, mask);
    MicroAPI::MaskAnd(cmpMaskReg, cmpMaskReg1, cmpMaskReg2, mask);
    LGamma58(resReg, srcReg, tmpReg, aReg, bReg, cReg, oneReg, mask);
    MicroAPI::Select(dstReg, resReg, dstReg, cmpMaskReg);
}

// cal res on cReg and select it on tmpReg, cmpMaskReg for even
__aicore__ inline void LGammaCalNegTmp1(MicroAPI::RegTensor<float>& resReg, MicroAPI::RegTensor<float>& srcReg,
    MicroAPI::RegTensor<float>& tmpReg, MicroAPI::RegTensor<float>& aReg, MicroAPI::RegTensor<float>& bReg,
    MicroAPI::RegTensor<float>& cReg, MicroAPI::RegTensor<int32_t>& oneS32Reg, MicroAPI::MaskReg cmpMaskReg,
    MicroAPI::MaskReg mask)
{
    MicroAPI::RegTensor<int32_t> castReg;
    MicroAPI::RegTensor<int32_t> s32Reg;

    // a = floor(x + x + 0.5)
    MicroAPI::Add(aReg, srcReg, srcReg, mask);
    MicroAPI::Adds(aReg, aReg, f05, mask);
    MicroAPI::Truncate<float, RoundMode::CAST_FLOOR, MicroAPI::MaskMergeMode::ZEROING>(aReg, aReg, mask);

    // see if a is even, res = (int32)a & 1
    MicroAPI::Cast<int32_t, float, LGAMMA_CAST_TRAIT_F322S32>(castReg, aReg, mask);
    MicroAPI::And(s32Reg, castReg, oneS32Reg, mask);

    // res = 0 for even
    MicroAPI::CompareScalar(cmpMaskReg, s32Reg, 0, mask);

    // a = (a * (-0.5) + x) * pi
    MicroAPI::Muls(aReg, aReg, fn05, mask);
    MicroAPI::Add(aReg, aReg, srcReg, mask);
    MicroAPI::Muls(aReg, aReg, fPi, mask);

    // even coefficients
    constexpr float r0 = -0.000195746586541645228862762451171875;
    constexpr float r2 = 0.0083327032625675201416015625;
    constexpr float r4 = -0.16666662693023681640625;

    // odd coefficients
    constexpr float r1 = 0.00002427957952022552490234375;
    constexpr float r3 = -0.001388786011375486850738525390625;
    constexpr float r5 = 0.0416667275130748748779296875;
    constexpr float r7 = -0.4999999701976776123046875;

    // b = a * a
    MicroAPI::Mul(bReg, aReg, aReg, mask);

    // c = ((r0 * b + r2) * b + r4) * b * a + a, for even
    MicroAPI::Muls(cReg, bReg, r0, mask);
    MicroAPI::Adds(cReg, cReg, r2, mask);
    MicroAPI::Mul(cReg, bReg, cReg, mask);
    MicroAPI::Adds(cReg, cReg, r4, mask);
    MicroAPI::Mul(cReg, bReg, cReg, mask);
    MicroAPI::Mul(cReg, cReg, aReg, mask);
    MicroAPI::Add(cReg, cReg, aReg, mask);
    MicroAPI::Select(tmpReg, cReg, resReg, cmpMaskReg);

    // c = (((r1 * b + r3) * b + r5) * b + r7) * b + 1, for odd
    MicroAPI::Muls(cReg, bReg, r1, mask);
    MicroAPI::Adds(cReg, cReg, r3, mask);
    MicroAPI::Mul(cReg, bReg, cReg, mask);
    MicroAPI::Adds(cReg, cReg, r5, mask);
    MicroAPI::Mul(cReg, bReg, cReg, mask);
    MicroAPI::Adds(cReg, cReg, r7, mask);
    MicroAPI::Mul(cReg, bReg, cReg, mask);
    MicroAPI::Adds(cReg, cReg, f1, mask);
    MicroAPI::Select(tmpReg, tmpReg, cReg, cmpMaskReg);
}

// Get final result
__aicore__ inline void LGammaCalNegTmp2(MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& srcReg,
    MicroAPI::RegTensor<float>& tmpReg, MicroAPI::RegTensor<float>& cReg, MicroAPI::MaskReg mask)
{
    constexpr float ftmp = 1.1447298526763916015625;
    // c = 1.1447298526763916015625 - ln(|c| * x) - LGammaPositive(x)
    MicroAPI::Abs(tmpReg, tmpReg, mask);
    MicroAPI::Mul(cReg, tmpReg, srcReg, mask);
    MicroAPI::Ln(cReg, cReg, mask);
    MicroAPI::Muls(cReg, cReg, fn1, mask);
    MicroAPI::Adds(cReg, cReg, ftmp, mask);
    MicroAPI::Sub(cReg, cReg, dstReg, mask);
}

// cal for x < 0 when data type is float
__aicore__ inline void LGammaNegative(MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& srcReg,
    MicroAPI::RegTensor<float>& resReg, MicroAPI::RegTensor<float>& tmpReg, MicroAPI::RegTensor<float>& aReg,
    MicroAPI::RegTensor<float>& bReg, MicroAPI::RegTensor<float>& cReg, MicroAPI::RegTensor<float>& infReg,
    MicroAPI::RegTensor<int32_t>& oneS32Reg, MicroAPI::MaskReg cmpMaskReg1, MicroAPI::MaskReg cmpMaskReg2,
    MicroAPI::MaskReg cmpMaskReg, MicroAPI::MaskReg mask)
{
    constexpr float minf = 9.99999968266e-20;
    NotNumUnion notNum;
    notNum.i = F32_INF;

    // cal res for |x| < 9.99999968266e-20
    MicroAPI::CompareScalar<float, CMPMODE::LT>(cmpMaskReg1, srcReg, minf, mask);
    // tmp = -ln(src)
    MicroAPI::Ln(tmpReg, srcReg, mask);
    MicroAPI::Muls(resReg, tmpReg, fn1, mask);

    // cal result of odd and even and select it on tmpReg
    LGammaCalNegTmp1(resReg, srcReg, tmpReg, aReg, bReg, cReg, oneS32Reg, cmpMaskReg, mask);
    // get final result on cReg
    LGammaCalNegTmp2(dstReg, srcReg, tmpReg, cReg, mask);
    // select result for the range of |x| < 9.99999968266e-20
    MicroAPI::Select(resReg, resReg, cReg, cmpMaskReg1);

    // a = round(x)
    MicroAPI::Truncate<float, RoundMode::CAST_ROUND, MicroAPI::MaskMergeMode::ZEROING>(aReg, srcReg, mask);
    // a == x -> res = inf
    MicroAPI::Compare(cmpMaskReg2, aReg, srcReg, mask);
    MicroAPI::Select(resReg, infReg, resReg, cmpMaskReg2);
}

__aicore__ inline void LgammaComputeHalf(MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& srcReg,
    MicroAPI::RegTensor<float>& resReg, MicroAPI::RegTensor<float>& tmpReg, MicroAPI::RegTensor<float>& aReg,
    MicroAPI::RegTensor<float>& bReg, MicroAPI::RegTensor<float>& cReg, MicroAPI::RegTensor<float>& oneReg,
    MicroAPI::RegTensor<float>& piReg, MicroAPI::MaskReg cmpMaskReg, MicroAPI::MaskReg mask)
{
    // compute result x >= 0
    LgammaComputePosHalf(resReg, srcReg, tmpReg, aReg, bReg, oneReg, mask);
    // compute mask x >= 0
    MicroAPI::CompareScalar<float, CMPMODE::GE>(cmpMaskReg, srcReg, 0.0f, mask);
    MicroAPI::Select(dstReg, resReg, srcReg, cmpMaskReg);

    // compute result x < 0
    LgammaComputeNegHalf(resReg, srcReg, tmpReg, aReg, bReg, cReg, oneReg, piReg, mask);
    MicroAPI::Select(dstReg, dstReg, resReg, cmpMaskReg);
    MicroAPI::Abs(aReg, srcReg, mask);
    MicroAPI::CompareScalar<float, CMPMODE::GE>(cmpMaskReg, aReg, 65504.0f, mask);
    MicroAPI::Select(dstReg, aReg, dstReg, cmpMaskReg);
}

__aicore__ inline void LgammaComputeFloat(MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& srcReg,
    MicroAPI::RegTensor<float>& resReg, MicroAPI::RegTensor<float>& tmpReg, MicroAPI::RegTensor<float>& aReg,
    MicroAPI::RegTensor<float>& bReg, MicroAPI::RegTensor<float>& cReg, MicroAPI::RegTensor<float>& oneReg,
    MicroAPI::RegTensor<float>& infReg, MicroAPI::RegTensor<int32_t>& oneS32Reg, MicroAPI::MaskReg cmpMaskReg1,
    MicroAPI::MaskReg cmpMaskReg2, MicroAPI::MaskReg cmpMaskReg, MicroAPI::MaskReg posMaskReg, MicroAPI::MaskReg mask)
{
    // When ReuseSource is true, we will reuse src in tmpScalar and initialize it to 0 for the CmpMask

    // Gen masks with x >= 0 and < 0, which will not be overwritten in the future
    MicroAPI::CompareScalar<float, CMPMODE::GE>(posMaskReg, srcReg, 0.0f, mask);

    // src = |src|, will no longer use original src in the future.
    // When ReuseSource is true, we will reuse src in tmpScalar and initialize it to 0 for the CmpMask
    MicroAPI::Abs(srcReg, srcReg, mask);

    // Cal the result for x >= 0, select to dst
    LGammaPositive(dstReg, srcReg, resReg, tmpReg, aReg, bReg, cReg, oneReg,
                    cmpMaskReg1, cmpMaskReg2, cmpMaskReg, mask);

    // Cal the result for x < 0, select to dst
    LGammaNegative(dstReg, srcReg, resReg, tmpReg, aReg, bReg, cReg, infReg,
                    oneS32Reg, cmpMaskReg1, cmpMaskReg2, cmpMaskReg, mask);
    MicroAPI::Select(dstReg, dstReg, resReg, posMaskReg);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void LgammaComputeImpl(__local_mem__ T *dstUb, __local_mem__ T *srcUb,
    uint32_t sreg, uint16_t repeatTimes)
{
    constexpr uint32_t stride = GetVecLen() / sizeof(float);
    NotNumUnion notNum;
    notNum.i = F32_INF;

    MicroAPI::MaskReg cmpMaskReg1;
    MicroAPI::MaskReg cmpMaskReg2;
    MicroAPI::MaskReg cmpMaskReg;
    MicroAPI::MaskReg posMaskReg;
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<float>();
    MicroAPI::RegTensor<T> srcReg;
    MicroAPI::RegTensor<float> castReg;
    MicroAPI::RegTensor<float> tmpReg;
    MicroAPI::RegTensor<float> aReg;
    MicroAPI::RegTensor<float> bReg;
    MicroAPI::RegTensor<float> cReg;
    MicroAPI::RegTensor<float> resReg;
    MicroAPI::RegTensor<float> dstReg;
    MicroAPI::RegTensor<float> oneReg;
    MicroAPI::RegTensor<float> infReg;
    MicroAPI::RegTensor<float> piReg;
    MicroAPI::RegTensor<int32_t> oneS32Reg;

    MicroAPI::Duplicate(oneReg, f1, mask);
    if constexpr (IsSameType<T, float>::value) {
        MicroAPI::Duplicate(oneS32Reg, 1, mask);
        MicroAPI::Duplicate(infReg, notNum.f, mask);
    } else {
        MicroAPI::Duplicate(piReg, PI, mask);
    }
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = MicroAPI::UpdateMask<float>(sreg);
        if constexpr (IsSameType<T, half>::value) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcReg, srcUb + i * stride);
            MicroAPI::Cast<float, T, LGAMMA_CAST_TRAIT_F162F32>(castReg, srcReg, mask);
            LgammaComputeHalf(dstReg, castReg, resReg, tmpReg, aReg, bReg, cReg, oneReg, piReg, cmpMaskReg, mask);
            MicroAPI::Cast<T, float, LGAMMA_CAST_TRAIT_F322F16>(srcReg, dstReg, mask);
            MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(dstUb + i * stride, srcReg, mask);
        } else {
            MicroAPI::DataCopy(castReg, srcUb + i * stride);
            LgammaComputeFloat(dstReg, castReg, resReg, tmpReg, aReg, bReg, cReg, oneReg, infReg,
                            oneS32Reg, cmpMaskReg1, cmpMaskReg2, cmpMaskReg, posMaskReg, mask);
            MicroAPI::DataCopy(dstUb + i * stride, dstReg, mask);
        }
    }
}
} // namespace LgammaInternal

template <typename T, bool isReuseSource = false>
__aicore__ inline void LgammaImpl(const LocalTensor<T> &dst, const LocalTensor<T> &src,
    const LocalTensor<uint8_t> &tmp, const uint32_t calCount)
{
    CheckTensorPosition(dst, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(src, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(tmp, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");

    CheckCalCount(calCount, "calCount", src, "srcTensor", "Lgamma");
    CheckCalCount(calCount, "calCount", dst, "dstTensor", "Lgamma");

    static_assert((SupportType<T, half, float>(), "current data type is not supported on current device!"));
    constexpr uint32_t stride = GetVecLen() / sizeof(float);
    uint16_t repeatTimes = CeilDivision(calCount, stride);

    __local_mem__ T *dstUb = (__local_mem__ T *)dst.GetPhyAddr();
    __local_mem__ T *srcUb = (__local_mem__ T *)src.GetPhyAddr();
    VF_CALL<LgammaInternal::LgammaComputeImpl<T, isReuseSource>>(dstUb, srcUb, calCount, repeatTimes);
}
}  // namespace AscendC
#endif  // IMPL_MATH_LGAMMA_LGAMMA_C310_IMPL_H
