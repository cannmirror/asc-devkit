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
 * \file log_c310_impl.h
 * \brief
 */
#ifndef IMPL_MATH_LOG_LOG_C310_IMPL_H
#define IMPL_MATH_LOG_LOG_C310_IMPL_H
#include "kernel_tensor.h"
#include "kernel_pop_stack_buffer.h"
#include "kernel_tiling/kernel_tiling.h"

namespace AscendC {
template <typename T> __simd_vf__ inline void LogImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint32_t calCount)
{
    static_assert((std::is_same_v<T, half> || std::is_same_v<T, float>),
        "current data type is not supported on current device!");
    constexpr uint32_t sregLower = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(T));
    uint16_t repeatTimes = CeilDivision(calCount, sregLower);
    MicroAPI::RegTensor<T> vreg0;
    MicroAPI::RegTensor<T> vreg1;
    uint32_t sreg = calCount;
    MicroAPI::MaskReg preg;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        preg = MicroAPI::UpdateMask<T>(sreg);
        MicroAPI::DataCopy(vreg0, src + i * sregLower);
        MicroAPI::Log(vreg1, vreg0, preg);
        MicroAPI::DataCopy(dst + i * sregLower, vreg1, preg);
    }
}

template <typename T>
__simd_vf__ inline void LogXImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint32_t calCount, const float LnXRec)
{
    static_assert((std::is_same_v<T, half> || std::is_same_v<T, float>),
        "current data type is not supported on current device!");
    constexpr uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(float));
    uint16_t repeatTimes = CeilDivision(calCount, sregLower);
    static constexpr MicroAPI::CastTrait castTraitB16ToB32 = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr MicroAPI::CastTrait castTraitB32ToB16 = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    if constexpr (std::is_same_v<T, float>) {
        MicroAPI::RegTensor<float> vreg0;
        MicroAPI::RegTensor<float> vreg1;
        uint32_t sreg = calCount;
        MicroAPI::MaskReg mask;
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            mask = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::DataCopy(vreg0, src + i * sregLower);
            MicroAPI::Log(vreg1, vreg0, mask);
            MicroAPI::Muls(vreg1, vreg1, LnXRec, mask);
            MicroAPI::DataCopy(dst + i * sregLower, vreg1, mask);
        }
    } else if constexpr (std::is_same_v<T, half>) {
        MicroAPI::RegTensor<T> vreg0;
        MicroAPI::RegTensor<T> dst0;
        MicroAPI::RegTensor<T> dst1;
        MicroAPI::RegTensor<float> vreg1;
        uint32_t sreg = calCount;
        MicroAPI::MaskReg mask;
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            mask = MicroAPI::UpdateMask<float>(sreg);
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg0, src + i * sregLower);
            MicroAPI::Cast<float, half, castTraitB16ToB32>(vreg1, vreg0, mask);
            MicroAPI::Log(vreg1, vreg1, mask);
            MicroAPI::Muls(vreg1, vreg1, LnXRec, mask);
            MicroAPI::Cast<half, float, castTraitB32ToB16>(dst0, vreg1, mask);
            MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(dst + i * sregLower, dst0, mask);
        }
    }
}
} // namespace AscendC
#endif // IMPL_MATH_LOG_LOG_C310_IMPL_H
