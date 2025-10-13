/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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
constexpr MicroAPI::CastTrait tanhCastTraitF162F32 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr MicroAPI::CastTrait tanhCastTraitF322F16 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
}

template <typename T, bool isReuseSource = false>
__simd_vf__ inline void TanhVFImpl(__local_mem__ T *dstUb, __local_mem__ T *srcUb,
    const uint32_t calCount)
{
    constexpr float DOUBLE_X = 2;
    constexpr float FP32_MIN_EXP = -8.8;
    constexpr float FP32_MAX_EXP = 8.8;
    uint16_t repeatTimes = CeilDivision(calCount, B32_DATA_NUM_PER_REPEAT);
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
        MicroAPI::Mins(castReg, castReg, FP32_MAX_EXP, preg);
        MicroAPI::Maxs(castReg, castReg, FP32_MIN_EXP, preg);
        MicroAPI::Muls(tmpReg, castReg, DOUBLE_X, preg);
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

/*
 * Formula is y= (e^(2x)-1)/(e^(2x)+1)
 */
template <typename T, bool isReuseSource = false>
__aicore__ inline void TanhImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const uint32_t calCount)
{
    static_assert((SupportType<T, half, float>(), "current data type is not supported on current device!"));
    bool ret = (calCount <= srcTensor.GetSize()) && (calCount <= dstTensor.GetSize()) && (calCount >= 0);
    ASCENDC_ASSERT(
        ret, { KERNEL_LOG(KERNEL_ERROR, "calCount must be no less than 0 and smaller than or equal to src & dst tensor."); });
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    __local_mem__ T *dstUb = (__local_mem__ T *)dstTensor.GetPhyAddr();
    __local_mem__ T *srcUb = (__local_mem__ T *)srcTensor.GetPhyAddr();
    
    TanhVFImpl<T, isReuseSource>(dstUb, srcUb, calCount);

}

template <typename T, bool isReuseSource = false>
__aicore__ inline void TanhImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    TanhImpl(dstTensor, srcTensor, calCount);
}
} // namespace AscendC

#endif // IMPL_MATH_TANH_TANH_C310_IMPL_H