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
 * \file lgamma_c310_impl.h
 * \brief
 */
#ifndef IMPL_MATH_LGAMMA_LGAMMA_C310_IMPL_H
#define IMPL_MATH_LGAMMA_LGAMMA_C310_IMPL_H
#include "kernel_tensor.h"
#include "lgamma_common_utils.h"
#include "../../common/check.h"

namespace AscendC {
namespace LgammaInternal {
__simd_callee__ inline void LgammaCalPow70To023(MicroAPI::RegTensor<float>& resReg, MicroAPI::RegTensor<float>& tmpReg,
    MicroAPI::MaskReg mask)
{
    constexpr float r0 = -7.72156649015328655494e-02;
    MicroAPI::RegTensor<float> aReg;
    MicroAPI::RegTensor<float> bReg;

    constexpr MulAddsParams params0 = GetConstantsPow70To023();
    // y = (((((r5 * x + r4) * x + r3) * x + r2) * x + r1) * x + r0) * x
    LGammaCalcMulAdd(aReg, tmpReg, mask, params0);
    MicroAPI::Mul(aReg, aReg, tmpReg, mask);
    MicroAPI::Adds(aReg, aReg, r0, mask);
    MicroAPI::Mul(aReg, aReg, tmpReg, mask);

    constexpr MulAddsParams params1 = GetConstantsPow70To023V2();
    // y = (((((v5 * x + v4) * x + v3) * x + v2) * x + v1) * x + 1
    LGammaCalcMulAdd(bReg, tmpReg, mask, params1);
    MicroAPI::Mul(bReg, bReg, tmpReg, mask);
    MicroAPI::Adds(bReg, bReg, f1, mask);

    MicroAPI::Div(resReg, aReg, bReg, mask);
    MicroAPI::Muls(aReg, tmpReg, fn05, mask);
    MicroAPI::Add(resReg, resReg, aReg, mask);
}

__simd_callee__ inline void LgammaCal023To073(MicroAPI::RegTensor<float>& resReg, MicroAPI::RegTensor<float>& srcReg,
    MicroAPI::RegTensor<float>& tmpReg, MicroAPI::MaskReg mask)
{
    constexpr float r0 = -1.0348905219959115;
    constexpr float r1 = -0.3808607740435038;
    constexpr float r2 = 0.10589227895856956;
    constexpr float r3 = 0.20700451798849123;
    constexpr float r4 = -1.7736742652812825;
    constexpr float r5 = -0.577911391162728;
    constexpr float r6 = -0.3713263607336305;
    constexpr float r7 = 1.8253839945289898;

    MicroAPI::RegTensor<float> aReg;
    MicroAPI::RegTensor<float> bReg;
    MicroAPI::RegTensor<float> cReg;
    MicroAPI::RegTensor<float> dReg;
    MicroAPI::RegTensor<float> eReg;
    MicroAPI::RegTensor<float> fReg;

    // a = x + r0
    MicroAPI::Adds(aReg, srcReg, r0, mask);
    // b = r3 / x
    MicroAPI::Duplicate(tmpReg, r3, mask);
    MicroAPI::Div(bReg, tmpReg, srcReg, mask);
    // c = b + r1
    MicroAPI::Adds(cReg, bReg, r1, mask);
    // d = r6 - b
    MicroAPI::Duplicate(tmpReg, r6, mask);
    MicroAPI::Sub(dReg, tmpReg, bReg, mask);
    // e = b + r7
    MicroAPI::Adds(eReg, bReg, r7, mask);
    // f = r5 / d
    MicroAPI::Duplicate(tmpReg, r5, mask);
    MicroAPI::Div(fReg, tmpReg, dReg, mask);
    // res = f - e
    MicroAPI::Sub(resReg, fReg, eReg, mask);
    // res = r4 / res
    MicroAPI::Duplicate(tmpReg, r4, mask);
    MicroAPI::Div(resReg, tmpReg, resReg, mask);
    // res = res + f
    MicroAPI::Add(resReg, resReg, fReg, mask);
    // res = res * a
    MicroAPI::Mul(resReg, resReg, aReg, mask);
    // res = r2 - res
    MicroAPI::Duplicate(tmpReg, r2, mask);
    MicroAPI::Sub(resReg, tmpReg, resReg, mask);
    // res = res * c
    MicroAPI::Mul(resReg, resReg, cReg, mask);
    //res = res - a
    MicroAPI::Sub(resReg, resReg, aReg, mask);
}

__simd_callee__ inline void LgammaCal07To123(MicroAPI::RegTensor<float>& resReg, MicroAPI::RegTensor<float>& absReg,
    MicroAPI::RegTensor<float>& tmpReg, MicroAPI::MaskReg mask)
{
    MicroAPI::Adds(tmpReg, absReg, fn1, mask);
    LgammaCalPow70To023(resReg, tmpReg, mask);
}

__simd_callee__ inline void LgammaCal123o173(MicroAPI::RegTensor<float>& resReg, MicroAPI::RegTensor<float>& absReg,
    MicroAPI::RegTensor<float>& tmpReg, MicroAPI::MaskReg mask)
{
    constexpr float tc = -1.46163214496836224576e+00;
    MicroAPI::RegTensor<float> aReg;
    MicroAPI::RegTensor<float> bReg;
    MicroAPI::RegTensor<float> cReg;
    MicroAPI::RegTensor<float> dReg;
    MicroAPI::RegTensor<float> wReg;
    MicroAPI::RegTensor<float> zReg;

    constexpr MulAddsParams params0 = GetConstants123To173();
    // a = (((r4 * x + r3) * x + r2) * x + r1) * x + r0
    LGammaCalcMulAdd(aReg, absReg, mask, params0);

    constexpr float tt = -3.63867699703950536541e-18;
    constexpr float tf = -1.21486290535849611461e-01;
    // y = x + tc
    MicroAPI::Adds(tmpReg, absReg, tc, mask);
    // z = y * y
    MicroAPI::Mul(zReg, tmpReg, tmpReg, mask);
    // w = z * y
    MicroAPI::Mul(wReg, zReg, tmpReg, mask);
    constexpr MulAddsParams params1 = GetConstants123To173V2();
    // b = (((t4 * w + t3) * w + t2) * w + t1) * w + t0
    LGammaCalcMulAdd(bReg, wReg, mask, params1);
    constexpr MulAddsParams params2 = GetConstants123To173V3();
    // c = (((t9 * w + t8) * w + t7) * w + t6) * w + t5
    LGammaCalcMulAdd(cReg, wReg, mask, params2);
    constexpr MulAddsParams params3 = GetConstants123To173V4();
    // d = (((t14 * w + t13) * w + t12) * w + t11) * w + t10
    LGammaCalcMulAdd(dReg, wReg, mask, params3);
    // res = z * b - （tt - w * (c + y * d))
    MicroAPI::Mul(resReg, zReg, bReg, mask);
    MicroAPI::Mul(tmpReg, tmpReg, dReg, mask);
    MicroAPI::Add(tmpReg, tmpReg, cReg, mask);
    MicroAPI::Mul(tmpReg, tmpReg, wReg, mask);
    MicroAPI::Duplicate(bReg, tt, mask);
    MicroAPI::Sub(tmpReg, bReg, tmpReg, mask);
    MicroAPI::Sub(resReg, resReg, tmpReg, mask);
    // res = res + tf
    MicroAPI::Adds(resReg, resReg, tf, mask);
    // res = res + a
    MicroAPI::Add(resReg, resReg, aReg, mask);
}

__simd_callee__ inline void LgammaCal173To2(MicroAPI::RegTensor<float>& resReg, MicroAPI::RegTensor<float>& srcReg,
    MicroAPI::RegTensor<float>& tmpReg, MicroAPI::MaskReg mask)
{
    constexpr float r0 = 0.422784325282832;
    constexpr float r1 = -2.000000000156967;
    constexpr float r2 = 1.5704527182920263;
    constexpr float r3 = 4.56346777685429;
    constexpr float r4 = 11.072315459770081;
    constexpr float r5 = -2.351768784938553;

    MicroAPI::RegTensor<float> aReg;
    MicroAPI::RegTensor<float> bReg;
    MicroAPI::RegTensor<float> cReg;
    MicroAPI::RegTensor<float> dReg;

    // a = x + r1
    MicroAPI::Adds(aReg, srcReg, r1, mask);
    // b =  x + r2
    MicroAPI::Adds(bReg, srcReg, r2, mask);
    // c = x / r5
    MicroAPI::Duplicate(tmpReg, r5, mask);
    MicroAPI::Div(cReg, srcReg, tmpReg, mask);
    // d = r4 / a
    MicroAPI::Duplicate(tmpReg, r4, mask);
    MicroAPI::Div(dReg, tmpReg, aReg, mask);
    // res = r3 - c
    MicroAPI::Duplicate(tmpReg, r3, mask);
    MicroAPI::Sub(resReg, tmpReg, cReg, mask);
    // res = res + d
    MicroAPI::Add(resReg, resReg, dReg, mask);
    // res = b / res
    MicroAPI::Div(resReg, bReg, resReg, mask);
    // res = res + r0
    MicroAPI::Adds(resReg, resReg, r0, mask);
    // res = res * a
    MicroAPI::Mul(resReg, aReg, resReg, mask);
}

__simd_callee__ inline void LgammaCal2To22(MicroAPI::RegTensor<float>& resReg, MicroAPI::RegTensor<float>& srcReg,
    MicroAPI::RegTensor<float>& tmpReg, MicroAPI::MaskReg mask)
{
    constexpr float r0 = -1.0012922592272755;
    constexpr float r1 = 2.4477420497054303;
    constexpr float r2 = -0.9987077407790342;
    constexpr float r3 = 0.18634242386907246;
    constexpr float r4 = 1.7427676919841077;
    constexpr float r5 = 2.2595210499965988;

    MicroAPI::RegTensor<float> aReg;
    MicroAPI::RegTensor<float> bReg;
    MicroAPI::RegTensor<float> cReg;

    // a = x + r2
    MicroAPI::Adds(aReg, srcReg, r2, mask);
    // b =  a - r0
    MicroAPI::Adds(bReg, aReg, r0, mask);
    // c = a / b
    MicroAPI::Div(cReg, aReg, bReg, mask);
    // c = r4 / c
    MicroAPI::Duplicate(tmpReg, r4, mask);
    MicroAPI::Div(cReg, tmpReg, cReg, mask);
    // res = r5 - c
    MicroAPI::Duplicate(tmpReg, r5, mask);
    MicroAPI::Sub(resReg, tmpReg, cReg, mask);
    // res = r3 / res
    MicroAPI::Duplicate(tmpReg, r3, mask);
    MicroAPI::Div(resReg, tmpReg, resReg, mask);
    // res = res + c
    MicroAPI::Add(resReg, cReg, resReg, mask);
    // res = r1 - res
    MicroAPI::Duplicate(tmpReg, r1, mask);
    MicroAPI::Sub(resReg, tmpReg, resReg, mask);
    // res = b / res
    MicroAPI::Div(resReg, bReg, resReg, mask);
}

__simd_callee__ inline void LgammaCal22To25(MicroAPI::RegTensor<float>& resReg, MicroAPI::RegTensor<float>& srcReg,
    MicroAPI::RegTensor<float>& tmpReg, MicroAPI::MaskReg mask)
{
    constexpr float r0 = 0.5071375856257216;
    constexpr float r1 = -2.0000069063575214;
    constexpr float r2 = -0.08525477083158943;
    constexpr float r3 = -0.7681154946764155;
    constexpr float r4 = -1.7452430203554998;

    MicroAPI::RegTensor<float> aReg;
    MicroAPI::RegTensor<float> bReg;

    // a = x + r1
    MicroAPI::Adds(aReg, srcReg, r1, mask);
    // res =  r4 - x
    MicroAPI::Muls(tmpReg, srcReg, fn1, mask);
    MicroAPI::Adds(resReg, tmpReg, r4, mask);
    // res = r3 / res
    MicroAPI::Duplicate(tmpReg, r3, mask);
    MicroAPI::Div(resReg, tmpReg, resReg, mask);
    // b = r2 -res
    MicroAPI::Duplicate(tmpReg, r2, mask);
    MicroAPI::Sub(bReg, tmpReg, resReg, mask);
    // res = b + a
    MicroAPI::Add(resReg, bReg, aReg, mask);
    // res = res * b
    MicroAPI::Mul(resReg, bReg, resReg, mask);
    // res = r0 - res
    MicroAPI::Duplicate(tmpReg, r0, mask);
    MicroAPI::Sub(resReg, tmpReg, resReg, mask);
    // res = a * res
    MicroAPI::Mul(resReg, aReg, resReg, mask);
}

__simd_callee__ inline void LgammaCal25To3(MicroAPI::RegTensor<float>& resReg, MicroAPI::RegTensor<float>& srcReg,
    MicroAPI::RegTensor<float>& tmpReg, MicroAPI::MaskReg mask)
{
    constexpr float r0 = -2.225444571099429;
    constexpr float r1 = -2.7949471270267456;
    constexpr float r2 = -0.017787316457839795;
    constexpr float r3 = 0.14449865133660186;
    constexpr float r4 = 0.2566236573575009;

    MicroAPI::RegTensor<float> aReg;
    MicroAPI::RegTensor<float> bReg;
    MicroAPI::RegTensor<float> cReg;
    MicroAPI::RegTensor<float> dReg;
    MicroAPI::RegTensor<float> eReg;

    // a = x + r0
    MicroAPI::Adds(aReg, srcReg, r0, mask);
    // b =  r4 - x
    MicroAPI::Muls(tmpReg, srcReg, fn1, mask);
    MicroAPI::Adds(bReg, tmpReg, r4, mask);
    // c = a + r1
    MicroAPI::Adds(cReg, aReg, r1, mask);
    // d = r0 + b
    MicroAPI::Adds(dReg, bReg, r0, mask);
    // c = x + c
    MicroAPI::Add(cReg, srcReg, cReg, mask);
    // d = r3 / d
    MicroAPI::Duplicate(tmpReg, r3, mask);
    MicroAPI::Div(dReg, tmpReg, dReg, mask);
    // d = d + r2
    MicroAPI::Adds(dReg, dReg, r2, mask);
    // c = d + c
    MicroAPI::Add(cReg, dReg, cReg, mask);
    // e = d * c
    MicroAPI::Mul(eReg, cReg, dReg, mask);
    // d = c + r1
    MicroAPI::Adds(dReg, cReg, r1, mask);
    // res = e * d
    MicroAPI::Mul(resReg, eReg, dReg, mask);
    // res = a - res
    MicroAPI::Sub(resReg, aReg, resReg, mask);
}

__simd_callee__ inline void LgammaCal3To8(MicroAPI::RegTensor<float>& resReg, MicroAPI::RegTensor<float>& absReg,
    MicroAPI::RegTensor<float>& tmpReg, MicroAPI::MaskReg mask)
{
    constexpr float r0 = -7.72156649015328655494e-02;
    constexpr float r1 = 2.14982415960608852501e-01;
    constexpr float v1 = 1.39200533467621045958e+00;
    MicroAPI::RegTensor<float> aReg;
    MicroAPI::RegTensor<float> bReg;
    MicroAPI::RegTensor<float> floorReg;
    MicroAPI::MaskReg maskReg;

    // i = floor(x)
    MicroAPI::Truncate<float, RoundMode::CAST_FLOOR, MicroAPI::MaskMergeMode::ZEROING>(floorReg, absReg, mask);
    // y = x - i
    MicroAPI::Sub(tmpReg, absReg, floorReg, mask);
    constexpr MulAddsParams params0 = GetConstants3To8();
    // a = ((((((r6 * y + r5) * y + r4) * y + r3) * y + r2) * y + r1) * y + r0) * y
    LGammaCalcMulAdd(aReg, tmpReg, mask, params0);
    MicroAPI::Mul(aReg, aReg, tmpReg, mask);
    MicroAPI::Adds(aReg, aReg, r1, mask);
    MicroAPI::Mul(aReg, aReg, tmpReg, mask);
    MicroAPI::Adds(aReg, aReg, r0, mask);
    MicroAPI::Mul(aReg, aReg, tmpReg, mask);

    constexpr MulAddsParams params1 = GetConstants3To8V2();
    // b = ((((((y * v6 + v5) + v4) * y + v3) * y + v2) * y + v1) * y + 1
    LGammaCalcMulAdd(bReg, tmpReg, mask, params1);
    MicroAPI::Mul(bReg, bReg, tmpReg, mask);
    MicroAPI::Adds(bReg, bReg, v1, mask);
    MicroAPI::Mul(bReg, bReg, tmpReg, mask);
    MicroAPI::Adds(bReg, bReg, f1, mask);

    // res = 0.5 * y + a / b
    MicroAPI::Div(aReg, aReg, bReg, mask);
    MicroAPI::Muls(bReg, tmpReg, f05, mask);
    MicroAPI::Add(resReg, bReg, aReg, mask);

    MicroAPI::Duplicate(aReg, f1, mask);
    // a[i >= 3] *= y[i >= 3] + 2
    MicroAPI::CompareScalar<float, CMPMODE::GE>(maskReg, floorReg, f3, mask);
    MicroAPI::Adds(bReg, tmpReg, f2, mask);
    MicroAPI::Mul(bReg, aReg, bReg, mask);
    MicroAPI::Select(aReg, bReg, aReg, maskReg);
    // a[i >= 4] *= y[i >= 4] + 3
    MicroAPI::CompareScalar<float, CMPMODE::GE>(maskReg, floorReg, F4, mask);
    MicroAPI::Adds(bReg, tmpReg, f3, mask);
    MicroAPI::Mul(bReg, aReg, bReg, mask);
    MicroAPI::Select(aReg, bReg, aReg, maskReg);
    // a[i >= 5] *= y[i >= 5] + 4
    MicroAPI::CompareScalar<float, CMPMODE::GE>(maskReg, floorReg, F5, mask);
    MicroAPI::Adds(bReg, tmpReg, F4, mask);
    MicroAPI::Mul(bReg, aReg, bReg, mask);
    MicroAPI::Select(aReg, bReg, aReg, maskReg);
    // a[i >= 6] *= y[i >= 6] + 5
    MicroAPI::CompareScalar<float, CMPMODE::GE>(maskReg, floorReg, F6, mask);
    MicroAPI::Adds(bReg, tmpReg, F5, mask);
    MicroAPI::Mul(bReg, aReg, bReg, mask);
    MicroAPI::Select(aReg, bReg, aReg, maskReg);
    // a[i >= 7] *= y[i >= 7] + 6
    MicroAPI::CompareScalar<float, CMPMODE::GE>(maskReg, floorReg, F7, mask);
    MicroAPI::Adds(bReg, tmpReg, F6, mask);
    MicroAPI::Mul(bReg, aReg, bReg, mask);
    MicroAPI::Select(aReg, bReg, aReg, maskReg);
    // res = res + ln(a)
    MicroAPI::Ln(aReg, aReg, mask);
    MicroAPI::Add(resReg, resReg, aReg, mask);
}

__simd_callee__ inline void LgammaCal8ToPow58(MicroAPI::RegTensor<float>& resReg, MicroAPI::RegTensor<float>& absReg,
    MicroAPI::RegTensor<float>& tmpReg, MicroAPI::MaskReg mask)
{
    constexpr float r0 = 4.18938533204672725052e-01;
    constexpr float r1 = 8.33333333333329678849e-02;
    MicroAPI::RegTensor<float> aReg;
    MicroAPI::RegTensor<float> bReg;

    // a = ln(x)
    MicroAPI::Ln(aReg, absReg, mask);
    // b = 1 / x
    MicroAPI::Duplicate(tmpReg, f1, mask);
    MicroAPI::Div(bReg, tmpReg, absReg, mask);
    // y = b * b
    MicroAPI::Mul(tmpReg, bReg, bReg, mask);

    constexpr MulAddsParams params0 = GetConstants8ToPow58();
    // res = (((((y * r6 + r5) * y + r4) * y + r3) * y + r2) * y + r1) * b + r0
    LGammaCalcMulAdd(resReg, tmpReg, mask, params0);
    MicroAPI::Mul(resReg, resReg, tmpReg, mask);
    MicroAPI::Adds(resReg, resReg, r1, mask);
    MicroAPI::Mul(resReg, resReg, bReg, mask);
    MicroAPI::Adds(resReg, resReg, r0, mask);

    // res = (x - 0.5) * (t - 1) + res
    MicroAPI::Adds(bReg, absReg, fn05, mask);
    MicroAPI::Adds(aReg, aReg, fn1, mask);
    MicroAPI::Mul(aReg, aReg, bReg, mask);
    MicroAPI::Add(resReg, resReg, aReg, mask);
}

__simd_callee__ inline void SearchSinPi(MicroAPI::RegTensor<float>& resReg, MicroAPI::RegTensor<float>& srcReg,
    MicroAPI::RegTensor<float>& tmpReg, MicroAPI::MaskReg mask)
{
    constexpr float r0 = 0.9452154240716536;
    constexpr float r1 = 1.323678940948241;
    constexpr float r2 = 0.27798021173736565;
    constexpr float r3 = -5.189778155221027;
    constexpr float r4 = -3.605483266634817;
    constexpr float r5 = -0.01134864151666764;
    constexpr float r6 = 0.96602443393936;

    MicroAPI::RegTensor<float> aReg;
    MicroAPI::RegTensor<float> bReg;
    MicroAPI::RegTensor<float> cReg;

    // a = x * x
    MicroAPI::Mul(aReg, srcReg, srcReg, mask);
    // b = r2 * a
    MicroAPI::Muls(bReg, aReg, r2, mask);
    // c = b - r0
    MicroAPI::Duplicate(tmpReg, r0, mask);
    MicroAPI::Sub(cReg, bReg, tmpReg, mask);
    // res = r6 + c
    MicroAPI::Adds(resReg, cReg, r6, mask);
    // res = res * c
    MicroAPI::Mul(resReg, resReg, cReg, mask);
    // res = res + r5
    MicroAPI::Adds(resReg, resReg, r5, mask);
    // res = res * r4
    MicroAPI::Muls(resReg, resReg, r4, mask);
    // res = res + b
    MicroAPI::Add(resReg, resReg, bReg, mask);
    // res = res + r3
    MicroAPI::Adds(resReg, resReg, r3, mask);
    // res = a * res
    MicroAPI::Mul(resReg, resReg, aReg, mask);
    // res = res + r1
    MicroAPI::Adds(resReg, resReg, r1, mask);
    // res = res * c
    MicroAPI::Mul(resReg, resReg, cReg, mask);
    // res = r0 - res
    MicroAPI::Duplicate(tmpReg, r0, mask);
    MicroAPI::Sub(resReg, tmpReg, resReg, mask);
    // res = res + ro
    MicroAPI::Adds(resReg, resReg, r0, mask);
    // res = res * x
    MicroAPI::Mul(resReg, resReg, srcReg, mask);
}

__simd_callee__ inline void SinPi(MicroAPI::RegTensor<float>& resReg, MicroAPI::RegTensor<float>& srcReg,
    MicroAPI::RegTensor<float>& tmpReg, MicroAPI::MaskReg mask)
{
    MicroAPI::RegTensor<float> aReg;
    MicroAPI::RegTensor<float> bReg;
    MicroAPI::RegTensor<float> cReg;
    MicroAPI::MaskReg maskReg;

    MicroAPI::Duplicate(resReg, 0.0f, mask);
    MicroAPI::Abs(aReg, srcReg, mask);
    // y = abs_x - floor(abs_x)
    MicroAPI::Truncate<float, RoundMode::CAST_FLOOR, MicroAPI::MaskMergeMode::ZEROING>(tmpReg, aReg, mask);
    MicroAPI::Sub(tmpReg, aReg, tmpReg, mask);

    MicroAPI::CompareScalar<float, CMPMODE::GE>(maskReg, tmpReg, f05, mask);
    MicroAPI::Duplicate(aReg, f1, mask);
    MicroAPI::Sub(aReg, aReg, tmpReg, mask);
    SearchSinPi(cReg, aReg, bReg, mask);
    MicroAPI::Select(resReg, cReg, resReg, maskReg);

    MicroAPI::CompareScalar<float, CMPMODE::LT>(maskReg, tmpReg, f05, mask);
    SearchSinPi(cReg, tmpReg, bReg, mask);
    MicroAPI::Select(resReg, cReg, resReg, maskReg);

    MicroAPI::Duplicate(bReg, 0.0f, mask);
    MicroAPI::Duplicate(aReg, f1, mask);
    MicroAPI::Duplicate(cReg, fn1, mask);
    MicroAPI::CompareScalar<float, CMPMODE::LT>(maskReg, srcReg, 0.0f, mask);
    MicroAPI::Select(bReg, cReg, bReg, maskReg);
    MicroAPI::CompareScalar<float, CMPMODE::GT>(maskReg, srcReg, 0.0f, mask);
    MicroAPI::Select(bReg, aReg, bReg, maskReg);
    MicroAPI::Mul(resReg, resReg, bReg, mask);
}

__simd_callee__ inline void LgammaCalNegPow70(MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& resReg,
    MicroAPI::RegTensor<float>& srcReg, MicroAPI::RegTensor<float>& tmpReg, MicroAPI::MaskReg mask)
{
    constexpr float logpi = 1.14472988584940016388e+00;
    MicroAPI::RegTensor<float> aReg;

    // res = logpi - log(abs(sinpi(x))) - log(-x) - dst
    MicroAPI::Duplicate(tmpReg, logpi, mask);
    SinPi(resReg, srcReg, aReg, mask);
    MicroAPI::Abs(resReg, resReg, mask);
    MicroAPI::Ln(resReg, resReg, mask);
    MicroAPI::Sub(resReg, tmpReg, resReg, mask);
    MicroAPI::Muls(tmpReg, srcReg, fn1, mask);
    MicroAPI::Ln(tmpReg, tmpReg, mask);
    MicroAPI::Sub(resReg, resReg, tmpReg, mask);
    MicroAPI::Sub(resReg, resReg, dstReg, mask);
}

__simd_callee__ inline void LgammaCompute1(MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& absReg,
    MicroAPI::RegTensor<float>& resReg, MicroAPI::RegTensor<float>& tmpReg, MicroAPI::MaskReg cmpMaskReg1,
    MicroAPI::MaskReg cmpMaskReg2, MicroAPI::MaskReg cmpMaskReg, MicroAPI::MaskReg mask)
{
    // abs_x <= 2^-70
    MicroAPI::CompareScalar<float, CMPMODE::LE>(cmpMaskReg, absReg, (float&)POW_70, mask);
    MicroAPI::Ln(tmpReg, absReg, mask);
    MicroAPI::Muls(resReg, tmpReg, fn1, mask);
    MicroAPI::Select(dstReg, resReg, dstReg, cmpMaskReg);

    // 2^-70 < abs_x < 0.23
    MicroAPI::CompareScalar<float, CMPMODE::LT>(cmpMaskReg1, absReg, (float&)F023, mask);
    MicroAPI::MaskXor(cmpMaskReg, cmpMaskReg1, cmpMaskReg, mask);
    LgammaCalPow70To023(resReg, absReg, mask);
    MicroAPI::Sub(resReg, resReg, tmpReg, mask);
    MicroAPI::Select(dstReg, resReg, dstReg, cmpMaskReg);

    // 0.23 <= abs_x < 0.7
    MicroAPI::CompareScalar<float, CMPMODE::LT>(cmpMaskReg2, absReg, (float&)F07, mask);
    MicroAPI::MaskXor(cmpMaskReg, cmpMaskReg1, cmpMaskReg2, mask);
    LgammaCal023To073(resReg, absReg, tmpReg, mask);
    MicroAPI::Select(dstReg, resReg, dstReg, cmpMaskReg);

    // 0.7 <= abs_x < 1.23
    MicroAPI::CompareScalar<float, CMPMODE::LT>(cmpMaskReg1, absReg, (float&)F123, mask);
    MicroAPI::MaskXor(cmpMaskReg, cmpMaskReg1, cmpMaskReg2, mask);
    MicroAPI::CompareScalar<float, CMPMODE::NE>(cmpMaskReg2, absReg, f1, mask);
    MicroAPI::MaskAnd(cmpMaskReg, cmpMaskReg2, cmpMaskReg, mask);
    LgammaCal07To123(resReg, absReg, tmpReg, mask);
    MicroAPI::Select(dstReg, resReg, dstReg, cmpMaskReg);

    // 1.23 <= abs_x < 1.73
    MicroAPI::CompareScalar<float, CMPMODE::LT>(cmpMaskReg2, absReg, (float&)F173, mask);
    MicroAPI::MaskXor(cmpMaskReg, cmpMaskReg1, cmpMaskReg2, mask);
    LgammaCal123o173(resReg, absReg, tmpReg, mask);
    MicroAPI::Select(dstReg, resReg, dstReg, cmpMaskReg);

    // 1.73 <= abs_x < 2
    MicroAPI::CompareScalar<float, CMPMODE::LT>(cmpMaskReg1, absReg, f2, mask);
    MicroAPI::MaskXor(cmpMaskReg, cmpMaskReg1, cmpMaskReg2, mask);
    LgammaCal173To2(resReg, absReg, tmpReg, mask);
    MicroAPI::Select(dstReg, resReg, dstReg, cmpMaskReg);

    // 2 <= abs_x < 2.2
    MicroAPI::CompareScalar<float, CMPMODE::LT>(cmpMaskReg2, absReg, (float&)F22, mask);
    MicroAPI::MaskXor(cmpMaskReg, cmpMaskReg1, cmpMaskReg2, mask);
    LgammaCal2To22(resReg, absReg, tmpReg, mask);
    MicroAPI::Select(dstReg, resReg, dstReg, cmpMaskReg);

    // 2.2 <= abs_x < 2.5
    MicroAPI::CompareScalar<float, CMPMODE::LT>(cmpMaskReg1, absReg, (float&)F25, mask);
    MicroAPI::MaskXor(cmpMaskReg, cmpMaskReg1, cmpMaskReg2, mask);
    LgammaCal22To25(resReg, absReg, tmpReg, mask);
    MicroAPI::Select(dstReg, resReg, dstReg, cmpMaskReg);

    // 2.5 <= abs_x < 3
    MicroAPI::CompareScalar<float, CMPMODE::LT>(cmpMaskReg2, absReg, f3, mask);
    MicroAPI::MaskXor(cmpMaskReg, cmpMaskReg1, cmpMaskReg2, mask);
    LgammaCal25To3(resReg, absReg, tmpReg, mask);
    MicroAPI::Select(dstReg, resReg, dstReg, cmpMaskReg);

    // 3 <= abs_x < 8
    MicroAPI::CompareScalar<float, CMPMODE::LT>(cmpMaskReg1, absReg, F8, mask);
    MicroAPI::MaskXor(cmpMaskReg, cmpMaskReg1, cmpMaskReg2, mask);
    LgammaCal3To8(resReg, absReg, tmpReg, mask);
    MicroAPI::Select(dstReg, resReg, dstReg, cmpMaskReg);

    // 8 <= abs_x < 2^58
    MicroAPI::CompareScalar<float, CMPMODE::LT>(cmpMaskReg2, absReg, (float&)POW_58, mask);
    MicroAPI::MaskXor(cmpMaskReg, cmpMaskReg1, cmpMaskReg2, mask);
    LgammaCal8ToPow58(resReg, absReg, tmpReg, mask);
    MicroAPI::Select(dstReg, resReg, dstReg, cmpMaskReg);
}

__simd_callee__ inline void LgammaCompute2(MicroAPI::RegTensor<float>& dstReg, MicroAPI::RegTensor<float>& srcReg,
    MicroAPI::RegTensor<float>& resReg, MicroAPI::RegTensor<float>& tmpReg, MicroAPI::MaskReg cmpMaskReg,
    MicroAPI::MaskReg mask)
{
    MicroAPI::Abs(tmpReg, srcReg, mask);
    // 2^58 <= abs_x
    MicroAPI::CompareScalar<float, CMPMODE::GE>(cmpMaskReg, tmpReg, (float&)POW_58, mask);
    // x * (log(abs_x) - 1)
    MicroAPI::Ln(tmpReg, tmpReg, mask);
    MicroAPI::Adds(tmpReg, tmpReg, fn1, mask);
    MicroAPI::Mul(resReg, tmpReg, srcReg, mask);
    MicroAPI::Select(dstReg, resReg, dstReg, cmpMaskReg);

    // x < 2^-70
    MicroAPI::CompareScalar<float, CMPMODE::LT>(cmpMaskReg, srcReg, (float&)NEG_POW_70, mask);
    LgammaCalNegPow70(dstReg, resReg, srcReg, tmpReg, mask);
    MicroAPI::Select(dstReg, resReg, dstReg, cmpMaskReg);
}

template <typename T, bool isReuseSource = false>
__simd_vf__ inline void LgammaComputeImpl(__local_mem__ T *dstUb, __local_mem__ T *srcUb,
    __local_mem__ float *workUb, uint32_t calCount, uint16_t repeatTime)
{
    constexpr uint32_t stride = GetVecLen() / sizeof(float);
    uint32_t sreg0 = calCount;
    uint32_t sreg1 = calCount;

    MicroAPI::MaskReg cmpMaskReg1;
    MicroAPI::MaskReg cmpMaskReg2;
    MicroAPI::MaskReg cmpMaskReg;
    MicroAPI::MaskReg mask = MicroAPI::CreateMask<float>();
    MicroAPI::RegTensor<T> srcReg;
    MicroAPI::RegTensor<float> castReg;
    MicroAPI::RegTensor<float> tmpReg;
    MicroAPI::RegTensor<float> resReg;
    MicroAPI::RegTensor<float> dstReg;
    MicroAPI::RegTensor<float> absReg;

    for (uint16_t i = 0; i < repeatTime; ++i) {
        mask = MicroAPI::UpdateMask<float>(sreg0);
        MicroAPI::Duplicate(dstReg, static_cast<float>(0));
        if constexpr (IsSameType<T, half>::value) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcReg, srcUb + i * stride);
            MicroAPI::Cast<float, T, LGAMMA_CAST_TRAIT_F162F32>(castReg, srcReg, mask);
        } else {
            MicroAPI::DataCopy(castReg, srcUb + i * stride);
        }
        MicroAPI::Abs(absReg, castReg, mask);
        LgammaCompute1(dstReg, absReg, resReg, tmpReg, cmpMaskReg1, cmpMaskReg2, cmpMaskReg, mask);
        MicroAPI::DataCopy(workUb + i * stride, dstReg, mask);
    }

    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

    for (uint16_t i = 0; i < repeatTime; ++i) {
        mask = MicroAPI::UpdateMask<float>(sreg1);
        if constexpr (IsSameType<T, half>::value) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcReg, srcUb + i * stride);
            MicroAPI::Cast<float, T, LGAMMA_CAST_TRAIT_F162F32>(castReg, srcReg, mask);
        } else {
            MicroAPI::DataCopy(castReg, srcUb + i * stride);
        }
        MicroAPI::DataCopy(dstReg, workUb + i * stride);
        LgammaCompute2(dstReg, castReg, resReg, tmpReg, cmpMaskReg, mask);
        if constexpr (IsSameType<T, half>::value) {
            MicroAPI::Cast<T, float, LGAMMA_CAST_TRAIT_F322F16>(srcReg, dstReg, mask);
            MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(dstUb + i * stride, srcReg, mask);
        } else {
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

    static_assert(SupportType<T, half, float>(), "current data type is not supported on current device!");
    constexpr uint32_t stride = GetVecLen() / sizeof(float);
    uint16_t repeatTime = CeilDivision(calCount, stride);
    auto workLocal = tmp.ReinterpretCast<float>();

    __local_mem__ T *dstUb = (__local_mem__ T *)dst.GetPhyAddr();
    __local_mem__ T *srcUb = (__local_mem__ T *)src.GetPhyAddr();
    __local_mem__ float *workUb = (__local_mem__ float *)workLocal.GetPhyAddr();
    LgammaInternal::LgammaComputeImpl<T, isReuseSource>(dstUb, srcUb, workUb, calCount, repeatTime);
}
}  // namespace AscendC
#endif  // IMPL_MATH_LGAMMA_LGAMMA_C310_IMPL_H
