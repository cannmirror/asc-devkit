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

/* !
 * \file tan_c310_impl.h
 * \brief
 */
#ifndef IMPL_MATH_TAN_TAN_C310_IMPL_H
#define IMPL_MATH_TAN_TAN_C310_IMPL_H

#include "kernel_tensor.h"
#include "../../common/check.h"

namespace AscendC {
namespace TanInternal {
constexpr MicroAPI::CastTrait TAN_CAST_TRAIT_F162F32 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr MicroAPI::CastTrait TAN_CAST_TRAIT_F322F16 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
// define the number of x div pi
constexpr float PI_FOR_X_TODIV = 0.3183098733425140380859375;
constexpr float KPI_FIRS_PI_MULS = 0.0009670257568359375;
// define the PI for compute
constexpr float PI_V2 = 3.140625;
// define the number of down of pi_div
constexpr float PI_DOWN = 1.57079637050628662109375;
constexpr float PI_DOWN_NEG = -1.57079637050628662109375;
// kpi_2
constexpr float KPI_TWI_PI_MULS = 6.2771141529083251953125e-7;
constexpr float PI_RESDOWN_ADDS = 0.00000004371139000189375;
constexpr float PI_RESDOWN_ADDS_NEG = -0.00000004371139000189375;
// kpi_3
constexpr float KPI_THIR_PI_MULS = 1.21644916362129151821136474609375e-10;
// kpi_4
constexpr float KPI_FOR_PI_MULS = -1.0291767438275201129727065563201904296875e-13;
// define the number of tan_compute
constexpr float TAN_RES_MULIT_SCA = 0.0698520831551998762793;
constexpr float TAN_RES_ADDICT_UP = -6.8711573651634203789;
constexpr float TAN_2ADDS = 61.20362572811089435388;
constexpr float TAN_3ADDS = -24.8048928861126769186219;

// normalized x to (-pi/2,pi/2) using x = x-round(x/π)*π
__simd_callee__ inline void TanRound(MicroAPI::RegTensor<float> &srcReg, MicroAPI::RegTensor<float> &tmpReg,
    MicroAPI::RegTensor<float> &roundReg, MicroAPI::RegTensor<float> &resReg, MicroAPI::RegTensor<float> &downReg1,
    MicroAPI::RegTensor<float> &downReg2, MicroAPI::MaskReg mask)
{
    /*
    k=round(x/π), x0=x-kπ, x0∈(-π/2, π/2)
    π=π_0+π_1+π_2+π_3+π_4 achieve final precision compensation.
    Final solution：
    k = round(x * invpi)
    x -= k * pi_0
    x -= k * pi_1
    down1 = x + pio2_high // pi/2 + x
    down2 = x - pio2_high // x - pi/2
    x -= k * pi_2
    down1 -= k * pi_2
    down2 -= k * pi_2
    down1 -= down_adds
    down2 += down_adds
    x -= k * pi_3
    down1 -= k * pi_3
    down2 -= k * pi_3
    x -= k * pi_4
    down1 -= k * pi_4
    down2 -= k * pi_4
    */

    // round_pi_div= round(x*0.3183098733425140380859375)
    MicroAPI::Muls(roundReg, srcReg, PI_FOR_X_TODIV, mask);
    // tie to even
    MicroAPI::Truncate<float, RoundMode::CAST_RINT, MicroAPI::MaskMergeMode::ZEROING>(roundReg, roundReg, mask);

    // kpi_0 = round_pi_div*3.140625
    MicroAPI::Muls(tmpReg, roundReg, PI_V2, mask);
    // input_x = (x-kpi_0)
    MicroAPI::Sub(resReg, srcReg, tmpReg, mask);

    // kpi_1 = muls(round_pi_div, 0.0009670257568359375)
    MicroAPI::Muls(tmpReg, roundReg, KPI_FIRS_PI_MULS, mask);
    // input_x = sub(input_x, kpi_1)
    MicroAPI::Sub(resReg, resReg, tmpReg, mask);
    // res_down1 = adds(input_x, 1.57079637050628662109375)
    MicroAPI::Adds(downReg1, resReg, PI_DOWN, mask);
    // res_down2 = adds(input_x, -1.57079637050628662109375)
    MicroAPI::Adds(downReg2, resReg, PI_DOWN_NEG, mask);

    // kpi_2 = muls(round_pi_div, 6.2771141529083251953125e-7)
    MicroAPI::Muls(tmpReg, roundReg, KPI_TWI_PI_MULS, mask);
    // input_x = sub(input_x, kpi_2)
    MicroAPI::Sub(resReg, resReg, tmpReg, mask);
    // res_down1 = sub(res_down1, kpi_2)
    MicroAPI::Sub(downReg1, downReg1, tmpReg, mask);
    // res_down2 = sub(res_down2, kpi_2)
    MicroAPI::Sub(downReg2, downReg2, tmpReg, mask);
    // res_down1 = adds(res_down1, -0.00000004371139000189375)
    MicroAPI::Adds(downReg1, downReg1, PI_RESDOWN_ADDS_NEG, mask);
    // res_down2 = adds(res_down2, 0.00000004371139000189375)
    MicroAPI::Adds(downReg2, downReg2, PI_RESDOWN_ADDS, mask);

    // kpi_3 = muls(round_pi_div, 1.21644916362129151821136474609375e-10)
    MicroAPI::Muls(tmpReg, roundReg, KPI_THIR_PI_MULS, mask);
    // input_x =sub(input_x, kpi_3)
    MicroAPI::Sub(resReg, resReg, tmpReg, mask);
    // res_down1 = sub(res_down1, kpi_3)
    MicroAPI::Sub(downReg1, downReg1, tmpReg, mask);
    // res_down2 = sub(res_down2, kpi_3)
    MicroAPI::Sub(downReg2, downReg2, tmpReg, mask);

    // kpi_4 = muls(round_pi_div, -1.0291767438275201129727065563201904296875e-13)
    MicroAPI::Muls(tmpReg, roundReg, KPI_FOR_PI_MULS, mask);
    // input_x =sub(input_x, kpi_4)
    MicroAPI::Sub(resReg, resReg, tmpReg, mask);
    // res_down1 = sub(res_down1, kpi_4)
    MicroAPI::Sub(downReg1, downReg1, tmpReg, mask);
    // res_down2 = sub(res_down2, kpi_4)
    MicroAPI::Sub(downReg2, downReg2, tmpReg, mask);
}

__simd_callee__ inline void TanPolynomialApproximation(MicroAPI::RegTensor<float> &dstReg, MicroAPI::RegTensor<float> &tmpReg,
    MicroAPI::RegTensor<float> &roundReg, MicroAPI::RegTensor<float> &resReg, MicroAPI::RegTensor<float> &downReg1,
    MicroAPI::RegTensor<float> &downReg2, MicroAPI::MaskReg mask)
{
    /*
    tan(x) = xP(x) / ((π/2 - x)(π/2 + x)Q(x))
    P(x) = (x^2 * R0 + R1) * x^2 + R2
    Q(x) = x^2 * R3
    R0 = 0.0698520831551998762793
    R1 = -6.8711573651634203789
    R2 = 61.20362572811089435388
    R3 = -24.8048928861126769186219
    */

    // x^2 = mul(input_x, input_x)
    MicroAPI::Mul(roundReg, resReg, resReg, mask);
    // res_up = muls(x^2, 0.0698520831551998762793)
    MicroAPI::Muls(tmpReg, roundReg, TAN_RES_MULIT_SCA, mask);
    // res_up = adds(res_up, -6.8711573651634203789)
    MicroAPI::Adds(tmpReg, tmpReg, TAN_RES_ADDICT_UP, mask);
    // res_up = mul(res_up, x^2)
    MicroAPI::Mul(tmpReg, tmpReg, roundReg, mask);
    // res_up = adds(res_up, 61.20362572811089435388)
    MicroAPI::Adds(tmpReg, tmpReg, TAN_2ADDS, mask);
    // res_up = mul(res_up, input_x)
    MicroAPI::Mul(tmpReg, tmpReg, resReg, mask);
    // res_down = adds(x^2, -24.8048928861126769186219)
    MicroAPI::Adds(roundReg, roundReg, TAN_3ADDS, mask);
    // res_down = mul(res_down, res_down1)
    MicroAPI::Mul(roundReg, roundReg, downReg1, mask);
    // res_down = mul(res_down, res_down2)
    MicroAPI::Mul(roundReg, roundReg, downReg2, mask);
    // res = div(res_up, res_down)
    MicroAPI::Div(dstReg, tmpReg, roundReg, mask);
}

template <typename T>
__simd_vf__ inline void TanCompute(__local_mem__ T *dstUb, __local_mem__ T *srcUb, uint32_t sreg, uint16_t repeatTimes)
{
    constexpr uint32_t stride = GetVecLen() / sizeof(float);

    MicroAPI::MaskReg mask;
    MicroAPI::RegTensor<T> srcReg;
    MicroAPI::RegTensor<float> castReg;
    MicroAPI::RegTensor<float> roundReg;
    MicroAPI::RegTensor<float> tmpReg;
    MicroAPI::RegTensor<float> downReg1;
    MicroAPI::RegTensor<float> downReg2;
    MicroAPI::RegTensor<float> resReg;
    MicroAPI::RegTensor<float> dstReg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = MicroAPI::UpdateMask<float>(sreg);
        if constexpr (sizeof(T) == sizeof(half)) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcReg, srcUb + i * stride);
            MicroAPI::Cast<float, T, TAN_CAST_TRAIT_F162F32>(castReg, srcReg, mask);
        } else {
            MicroAPI::DataCopy(castReg, srcUb + i * stride);
        }
        // the input is normalized to (-pi/2,pi/2)
        TanRound(castReg, tmpReg, roundReg, resReg, downReg1, downReg2, mask);
        TanPolynomialApproximation(dstReg, tmpReg, roundReg, resReg, downReg1, downReg2, mask);

        if constexpr (sizeof(T) == sizeof(half)) {
            MicroAPI::Cast<T, float, TAN_CAST_TRAIT_F322F16>(srcReg, dstReg, mask);
            MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(dstUb + i * stride, srcReg, mask);
        } else {
            MicroAPI::DataCopy(dstUb + i * stride, dstReg, mask);
        }
    }
}
} // namespace TanInternal

template <typename T, bool isReuseSource = false>
__aicore__ inline void TanImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");

    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Tan");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Tan");

    static_assert(SupportType<T, half, float>(), "current data type is not supported on current device!");
    constexpr uint32_t stride = GetVecLen() / sizeof(float);
    uint16_t repeatTimes = CeilDivision(calCount, stride);

    __local_mem__ T *dstUb = (__local_mem__ T *)dstTensor.GetPhyAddr();
    __local_mem__ T *srcUb = (__local_mem__ T *)srcTensor.GetPhyAddr();

    TanInternal::TanCompute<T>(dstUb, srcUb, calCount, repeatTimes);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void TanImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");

    TanImpl(dstTensor, srcTensor, calCount);
}
} // namespace AscendC

#endif // IMPL_MATH_TAN_TAN_C310_IMPL_H