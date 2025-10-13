/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
 /* !
 * \file if_finite_common_impl.h
 * \brief
 */

#ifndef LIB_MATH_IS_FINITE_IMPL_H
#define LIB_MATH_IS_FINITE_IMPL_H
#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
#include "kernel_tensor.h"

// Implementation Process
// 1. Use vcmp_ne for comparison. If the value is nan, the out should be false.
// 2. Use vcmps_eq for comparison. If the value is ±inf, the out should be false.
// 3. Use MaskOr to get final mask
// 4. Based on the output data type, use the corresponding 1 and 0 of maskreg select.

namespace AscendC {
template<typename T, typename U>
__simd_vf__ inline void IsFiniteVFImpl(__local_mem__ U *dstUb, __local_mem__ T *srcUb, uint32_t calCount)
{
    constexpr float ONE = 1.0;
    constexpr uint16_t BF16_ONE = 0x3f80;
    constexpr float ZERO = 0.0;
    constexpr uint32_t INF = 0x7f800000;
    constexpr uint32_t NEG_INF = 0xff800000;
    constexpr uint32_t HALF_INF = 0x7c00;
    constexpr uint32_t HALF_NEG_INF = 0xfc00;
    constexpr uint32_t B_HALF_INF = 0x7f80;
    constexpr uint32_t B_HALF_NEG_INF = 0xff80;

    MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vSrcReg0;
    MicroAPI::RegTensor<U, MicroAPI::RegTraitNumOne> vDstReg0, vReg0, vReg1;

    if constexpr (IsSameType<U, bfloat16_t>::value) {
        MicroAPI::Duplicate((MicroAPI::RegTensor<uint16_t, MicroAPI::RegTraitNumOne> &)vReg0, ZERO);
        MicroAPI::Duplicate((MicroAPI::RegTensor<uint16_t, MicroAPI::RegTraitNumOne> &)vReg1, BF16_ONE);
    } else if constexpr (IsSameType<U, bool>::value) {
        MicroAPI::Duplicate((MicroAPI::RegTensor<int8_t, MicroAPI::RegTraitNumOne> &)vReg0, ZERO);
        MicroAPI::Duplicate((MicroAPI::RegTensor<int8_t, MicroAPI::RegTraitNumOne> &)vReg1, ONE);
    } else {
        MicroAPI::Duplicate(vReg0, ZERO);
        MicroAPI::Duplicate(vReg1, ONE);
    }

    uint32_t sreg = static_cast<uint32_t>(calCount);
    MicroAPI::MaskReg preg;
    MicroAPI::MaskReg cmpMaskNAN, cmpMaskPINF, cmpMaskNINF, cmpMaskINF, cmpMaskReg;

    uint32_t sregLower = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(T));
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, sregLower));
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        preg = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumOne>(sreg);
        MicroAPI::DataCopy(vSrcReg0, srcUb + i * sregLower);
        MicroAPI::Compare<T, CMPMODE::NE>(cmpMaskNAN, vSrcReg0, vSrcReg0, preg);
        if constexpr (IsSameType<T, float>::value) {
            MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(cmpMaskPINF, (MicroAPI::RegTensor<uint32_t> &)vSrcReg0, INF, preg);
            MicroAPI::CompareScalar<uint32_t, CMPMODE::EQ>(cmpMaskNINF, (MicroAPI::RegTensor<uint32_t> &)vSrcReg0, NEG_INF, preg);
        } else if constexpr (IsSameType<T, half>::value) {
            MicroAPI::CompareScalar<uint16_t, CMPMODE::EQ>(cmpMaskPINF, (MicroAPI::RegTensor<uint16_t> &)vSrcReg0, HALF_INF, preg);
            MicroAPI::CompareScalar<uint16_t, CMPMODE::EQ>(cmpMaskNINF, (MicroAPI::RegTensor<uint16_t> &)vSrcReg0, HALF_NEG_INF, preg);
        } else if constexpr (IsSameType<T, bfloat16_t>::value) {
            MicroAPI::CompareScalar<uint16_t, CMPMODE::EQ>(cmpMaskPINF, (MicroAPI::RegTensor<uint16_t> &)vSrcReg0, B_HALF_INF, preg);
            MicroAPI::CompareScalar<uint16_t, CMPMODE::EQ>(cmpMaskNINF, (MicroAPI::RegTensor<uint16_t> &)vSrcReg0, B_HALF_NEG_INF, preg);
        }

        MicroAPI::MaskOr(cmpMaskINF, cmpMaskPINF, cmpMaskNINF, preg);
        MicroAPI::MaskOr(cmpMaskReg, cmpMaskNAN, cmpMaskINF, preg);
        if constexpr (IsSameType<U, bool>::value) {
            if constexpr (IsSameType<T, float>::value) {
                MicroAPI::MaskPack(cmpMaskReg, cmpMaskReg);
                MicroAPI::MaskPack(cmpMaskReg, cmpMaskReg);
                MicroAPI::Select((MicroAPI::RegTensor<int8_t, MicroAPI::RegTraitNumOne> &)vDstReg0, 
                    (MicroAPI::RegTensor<int8_t, MicroAPI::RegTraitNumOne> &)vReg0, (MicroAPI::RegTensor<int8_t, MicroAPI::RegTraitNumOne> &)vReg1, cmpMaskReg);
                MicroAPI::MaskPack(preg, preg);
                MicroAPI::MaskPack(preg, preg);
            } else {
                MicroAPI::MaskPack(cmpMaskReg, cmpMaskReg);
                MicroAPI::Select((MicroAPI::RegTensor<int8_t, MicroAPI::RegTraitNumOne> &)vDstReg0, 
                    (MicroAPI::RegTensor<int8_t, MicroAPI::RegTraitNumOne> &)vReg0, (MicroAPI::RegTensor<int8_t, MicroAPI::RegTraitNumOne> &)vReg1, cmpMaskReg);
                MicroAPI::MaskPack(preg, preg);
            }
        } else {
            MicroAPI::Select(vDstReg0, vReg0, vReg1, cmpMaskReg);
        }
        MicroAPI::DataCopy(dstUb + i * sregLower, vDstReg0, preg);
    }
}

template<typename T, typename U>
__aicore__ inline void IsFiniteImpl(const LocalTensor<U>& dst, const LocalTensor<T>& src, uint32_t calCount)
{
    if ASCEND_IS_AIC {
        return;
    }
    ASCENDC_ASSERT(calCount <= src.GetSize(),
        {KERNEL_LOG(KERNEL_ERROR, "CalCount should <= %d", src.GetSize());
        return;});

    static_assert(SupportType<T, float, half, bfloat16_t>(),
            "IsFinite do not support this type on current device");
 
    static_assert(SupportType<U, bool, float, half, bfloat16_t>(),
            "IsFinite do not support this output type on current device");
 
    __local_mem__ T *srcUb = (__local_mem__ T *)src.GetPhyAddr();
    __local_mem__ U *dstUb = (__local_mem__ U *)dst.GetPhyAddr();

    IsFiniteVFImpl<T, U>(dstUb, srcUb, calCount);
}

}  // namespace AscendC
#endif
#endif  // IMPL_MATH_ISFINITE_ISFINITE_COMMON_IMPL_H