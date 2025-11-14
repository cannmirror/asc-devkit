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
 * \file tanh_c310_impl.h
 * \brief
 */
#ifndef IMPL_MATH_TANH_TANH_C310_IMPL_H
#define IMPL_MATH_TANH_TANH_C310_IMPL_H

#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"

namespace AscendC {
namespace TanhInternal {
constexpr float FP32_ZERO_015 = 0.0157296831;
constexpr float FP32_ZERO_NEG_052 = -0.0523029624;
constexpr float FP32_ZERO_133 = 0.133152977;
constexpr float FP32_ZERO_NEG_333 = -0.333327681;
constexpr float FP32_TWENTY = 20.0;
constexpr float FP32_TWO = 2.0;
constexpr float FP32_ZERO_55 = 0.55;
constexpr float FP32_MIN_EXP = -8.8;
constexpr float FP32_MAX_EXP = 8.8;

constexpr MicroAPI::CastTrait tanhCastTraitF162F32 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr MicroAPI::CastTrait tanhCastTraitF322F16 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
}

template <typename T>
__simd_vf__ inline void TanhIntrinsicImpl(__local_mem__ T *dstUb, __local_mem__ T *srcUb,
    const uint32_t calCount, const uint16_t repeatTimes)
{
    uint32_t sreg = calCount;
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<T> srcReg;
    MicroAPI::RegTensor<float> castReg;
    MicroAPI::RegTensor<float> tmpReg;
    MicroAPI::RegTensor<float> dstReg;

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        preg = MicroAPI::UpdateMask<float>(sreg);
        if constexpr (sizeof(T) == sizeof(half)) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcReg, srcUb + i * B32_DATA_NUM_PER_REPEAT);
            MicroAPI::Cast<float, T, TanhInternal::tanhCastTraitF162F32>(castReg, srcReg, preg);
        } else {
            MicroAPI::DataCopy(castReg, srcUb + i * B32_DATA_NUM_PER_REPEAT);
        }
        MicroAPI::Mins(castReg, castReg, TanhInternal::FP32_MAX_EXP, preg);
        MicroAPI::Maxs(castReg, castReg, TanhInternal::FP32_MIN_EXP, preg);
        MicroAPI::Muls(tmpReg, castReg, TanhInternal::FP32_TWO, preg);
        MicroAPI::Exp(castReg, tmpReg, preg);

        MicroAPI::Adds(dstReg, castReg, -1.0f, preg);
        MicroAPI::Adds(tmpReg, castReg, 1.0f, preg);
        MicroAPI::Div(dstReg, dstReg, tmpReg, preg);
        if constexpr (sizeof(T) == sizeof(half)) {
            MicroAPI::Cast<T, float, TanhInternal::tanhCastTraitF322F16>(srcReg, dstReg, preg);
            MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(dstUb + i * B32_DATA_NUM_PER_REPEAT, srcReg, preg);
        } else {
            MicroAPI::DataCopy(dstUb + i * B32_DATA_NUM_PER_REPEAT, dstReg, preg);
        }
    }
}

template <typename T>
__simd_vf__ inline void TanhCompensationImpl(__local_mem__ T *dstUb, __local_mem__ T *srcUb,
    const uint32_t calCount, const uint16_t repeatTimes)
{
    uint32_t sreg = calCount;
    MicroAPI::MaskReg preg, cmpMaskReg;
    MicroAPI::RegTensor<T> srcReg;
    MicroAPI::RegTensor<float> vregInput, vregInputAbs;
    MicroAPI::RegTensor<float> vregInputSqr, vregInputMid;
    MicroAPI::RegTensor<float> vregOutput;
    MicroAPI::RegTensor<float> vregScalar1, vregScalar2;

    MicroAPI::Duplicate(vregScalar1, TanhInternal::FP32_ZERO_133);
    MicroAPI::Duplicate(vregScalar2, TanhInternal::FP32_ZERO_NEG_333);
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        preg = MicroAPI::UpdateMask<float>(sreg);
        if constexpr (sizeof(T) == sizeof(half)) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcReg, srcUb + i * B32_DATA_NUM_PER_REPEAT);
            MicroAPI::Cast<float, T, TanhInternal::tanhCastTraitF162F32>(vregInput, srcReg, preg);
        } else {
            MicroAPI::DataCopy(vregInput, srcUb + i * B32_DATA_NUM_PER_REPEAT);
        }
        MicroAPI::Mul(vregInputSqr, vregInput, vregInput, preg);
        MicroAPI::Muls(vregOutput, vregInputSqr, TanhInternal::FP32_ZERO_015, preg);
        MicroAPI::Adds(vregOutput, vregOutput, TanhInternal::FP32_ZERO_NEG_052, preg);
        MicroAPI::FusedMulDstAdd(vregOutput, vregInputSqr, vregScalar1, preg);
        MicroAPI::FusedMulDstAdd(vregOutput, vregInputSqr, vregScalar2, preg);
        MicroAPI::Mul(vregOutput, vregOutput, vregInputSqr, preg);
        MicroAPI::FusedMulDstAdd(vregOutput, vregInput, vregInput, preg);

        MicroAPI::Abs(vregInputAbs, vregInput, preg);
        MicroAPI::Mins(vregInput, vregInput, TanhInternal::FP32_TWENTY, preg);
        MicroAPI::Muls(vregInput, vregInput, TanhInternal::FP32_TWO, preg);
        MicroAPI::Exp(vregInput, vregInput, preg);
        MicroAPI::Adds(vregInputMid, vregInput, -1.0f, preg);
        MicroAPI::Adds(vregInputSqr, vregInput, 1.0f, preg);
        MicroAPI::Div(vregInputMid, vregInputMid, vregInputSqr, preg);

        MicroAPI::CompareScalar<float, CMPMODE::LT>(cmpMaskReg, vregInputAbs, TanhInternal::FP32_ZERO_55, preg);
        MicroAPI::Select(vregOutput, vregOutput, vregInputMid, cmpMaskReg);

        if constexpr (sizeof(T) == sizeof(half)) {
            MicroAPI::Cast<T, float, TanhInternal::tanhCastTraitF322F16>(srcReg, vregOutput, preg);
            MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(dstUb + i * B32_DATA_NUM_PER_REPEAT, srcReg, preg);
        } else {
            MicroAPI::DataCopy(dstUb + i * B32_DATA_NUM_PER_REPEAT, vregOutput, preg);
        }
    }
}

/*
 * Formula is y= (e^(2x)-1)/(e^(2x)+1)
 */
template <typename T, bool isReuseSource = false, const TanhConfig &config = DEFAULT_TANH_CONFIG>
__aicore__ inline void TanhImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const uint32_t calCount)
{
    static_assert(SupportType<T, half, float>(), "current data type is not supported on current device!");
    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");

    CheckCalCount(calCount, "calCount", srcTensor, "srcTensor", "Tanh");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Tanh");

    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    __local_mem__ T *dstUb = (__local_mem__ T *)dstTensor.GetPhyAddr();
    __local_mem__ T *srcUb = (__local_mem__ T *)srcTensor.GetPhyAddr();
    uint16_t repeatTimes = CeilDivision(calCount, B32_DATA_NUM_PER_REPEAT);
    if constexpr (config.algo == TanhAlgo::INTRINSIC) {
        TanhIntrinsicImpl<T>(dstUb, srcUb, calCount, repeatTimes);
    } else {
        TanhCompensationImpl<T>(dstUb, srcUb, calCount, repeatTimes);
    }
}

template <typename T, bool isReuseSource = false, const TanhConfig &config = DEFAULT_TANH_CONFIG>
__aicore__ inline void TanhImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");
    TanhImpl<T, isReuseSource, config>(dstTensor, srcTensor, calCount);
}
} // namespace AscendC

#endif // IMPL_MATH_TANH_TANH_C310_IMPL_H