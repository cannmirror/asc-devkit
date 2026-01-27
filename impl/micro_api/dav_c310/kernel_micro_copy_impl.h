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
 * \file kernel_micro_copy_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_COPY_IMPL_H
#define ASCENDC_MODULE_MICRO_COPY_IMPL_H

namespace AscendC {
namespace MicroAPI {
template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::MERGING, typename U>
__simd_callee__ inline void CopyImpl(U& dstReg, U& srcReg, MaskReg mask)
{
    using ActualT = typename U::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, bool, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t,
                  int64_t, half, float, bfloat16_t>(), "current data type is not supported on current device!");
    static_assert(SupportEnum<mode, MaskMergeMode::MERGING>(),
                  "current Move api only supported MaskMergeMode MERGING on current device!");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    if constexpr (IsSameType<ActualT, bool>::value) {
        vmov((RegTensor<int8_t>&)dstReg, (RegTensor<int8_t>&)srcReg, mask, modeValue);
    } else if constexpr (sizeof(ActualT) != 8) {
        vmov(dstReg, srcReg, mask, modeValue);
    } else {
        if constexpr (CheckRegTrait<U, RegTraitNumOne>()) {
            constexpr auto lowerDist =
                std::integral_constant<::HiloPart, static_cast<::HiloPart>(HighLowPart::LOWEST)>();
            MaskReg dstMask;
            MaskReg tmpMask;
            MaskReg dumpMask;
            ppack(tmpMask, mask, lowerDist);
            pintlv_b32(dstMask, dumpMask, tmpMask, tmpMask);
            vmov((RegTensor<uint32_t>&)dstReg, (RegTensor<uint32_t>&)srcReg, dstMask, modeValue);
        } else if constexpr (CheckRegTrait<U, RegTraitNumTwo>()) {
            vmov((RegTensor<uint32_t>&)dstReg.reg[0], (RegTensor<uint32_t>&)srcReg.reg[0], mask, modeValue);
            vmov((RegTensor<uint32_t>&)dstReg.reg[1], (RegTensor<uint32_t>&)srcReg.reg[1], mask, modeValue);
        }
    }
}

template <typename T = DefaultType, typename U>
__simd_callee__ inline void CopyImpl(U& dstReg, U& srcReg)
{
    using ActualT = typename U::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, bool, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t,
                  half, float, bfloat16_t>(), "current data type is not supported on current device!");
    if constexpr (IsSameType<ActualT, bool>::value) {
        vmov((RegTensor<int8_t>&)dstReg, (RegTensor<int8_t>&)srcReg);
    } else if constexpr (sizeof(ActualT) != 8) {
        vmov(dstReg, srcReg);
    } else {
        if constexpr (CheckRegTrait<U, RegTraitNumOne>()) {
            vmov((RegTensor<uint32_t>&)dstReg, (RegTensor<uint32_t>&)srcReg);
        } else if constexpr (CheckRegTrait<U, RegTraitNumTwo>()) {
            vmov((RegTensor<uint32_t>&)dstReg.reg[0], (RegTensor<uint32_t>&)srcReg.reg[0]);
            vmov((RegTensor<uint32_t>&)dstReg.reg[1], (RegTensor<uint32_t>&)srcReg.reg[1]);
        }
    }
}
} // namespace MicroAPI
} // namespace AscendC
#endif // ASCENDC_MODULE_MICRO_COPY_IMPL_H
