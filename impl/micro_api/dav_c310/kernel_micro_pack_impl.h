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
 * \file kernel_micro_pack_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_PACK_IMPL_H
#define ASCENDC_MODULE_MICRO_PACK_IMPL_H

namespace AscendC {
namespace MicroAPI {
template <typename T = DefaultType, typename U = DefaultType, HighLowPart part = HighLowPart::LOWEST, typename S,
          typename V>
__simd_callee__ inline void PackImpl(S& dstReg, V& srcReg)
{
    using ActualT = typename S::ActualT;
    using ActualU = typename V::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(std::is_same_v<U, DefaultType> || std::is_same_v<U, ActualU>, "U type is not correct!");
    static_assert((SupportType<Tuple<ActualT, ActualU>, Tuple<uint8_t, uint16_t>, Tuple<uint8_t, int16_t>,
                  Tuple<uint16_t, uint32_t>, Tuple<uint16_t, int32_t>, Tuple<uint32_t, uint64_t>, Tuple<uint32_t, 
                  int64_t>>()), "unsupport datatype");
    constexpr auto partValue = std::integral_constant<::HiloPart, static_cast<::HiloPart>(part)>();
    if constexpr (sizeof(ActualU) != 8) {
        vpack(dstReg, srcReg, partValue);
    } else {
        if constexpr (CheckRegTrait<V, RegTraitNumOne>()) {
            RegTensor<uint32_t> zeroReg;
            RegTensor<uint32_t> dumpReg;
            MaskReg mask0 = CreateMask<uint32_t, MaskPattern::ALL>();
            Duplicate(zeroReg, 0, mask0);
            if constexpr (part == HighLowPart::LOWEST) {
                DeInterleave((RegTensor<uint32_t>&)dstReg, dumpReg, (RegTensor<uint32_t>&)srcReg, zeroReg);
            } else {
                DeInterleave((RegTensor<uint32_t>&)dstReg, dumpReg, zeroReg, (RegTensor<uint32_t>&)srcReg);
            }
        } else if constexpr (CheckRegTrait<V, RegTraitNumTwo>()) {
            Copy((RegTensor<uint32_t>&)dstReg, (RegTensor<uint32_t>&)srcReg.reg[0]);
        }
    }
}

template <typename T = DefaultType, typename U = DefaultType, HighLowPart part = HighLowPart::LOWEST, typename S,
          typename V>
__simd_callee__ inline void UnPackImpl(S& dstReg, V& srcReg)
{
    using ActualT = typename S::ActualT;
    using ActualU = typename V::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(std::is_same_v<U, DefaultType> || std::is_same_v<U, ActualU>, "U type is not correct!");
    static_assert((SupportType<Tuple<ActualT, ActualU>, Tuple<uint32_t, uint16_t>, Tuple<int32_t, int16_t>,
                  Tuple<uint16_t, uint8_t>, Tuple<int16_t, int8_t>, Tuple<uint64_t, uint32_t>, Tuple<int64_t, 
                  int32_t>>()), "unsupport datatype");
    constexpr auto partValue = std::integral_constant<::HiloPart, static_cast<::HiloPart>(part)>();
    if constexpr (sizeof(ActualT) != 8) {
        vunpack(dstReg, srcReg, partValue);
    } else {
        RegTensor<uint32_t> padReg;
        MaskReg mask0 = CreateMask<ActualU, MaskPattern::ALL>();
        if constexpr (std::is_same_v<ActualU, int32_t>) {
            ShiftRights<int32_t, int16_t>((RegTensor<int32_t>&)padReg, srcReg, 31, mask0);
        } else {
            Duplicate(padReg, 0, mask0);
        }
        if constexpr (CheckRegTrait<S, RegTraitNumOne>()) {
            RegTensor<uint32_t> dumpReg;
            if constexpr (part == HighLowPart::LOWEST) {
                Interleave((RegTensor<uint32_t>&)dstReg, dumpReg, (RegTensor<uint32_t>&)srcReg, padReg);
            } else {
                Interleave(dumpReg, (RegTensor<uint32_t>&)dstReg, (RegTensor<uint32_t>&)srcReg, padReg);
            }
        } else if constexpr (CheckRegTrait<S, RegTraitNumTwo>()) {
            Copy((RegTensor<uint32_t>&)dstReg.reg[0], (RegTensor<uint32_t>&)srcReg);
            Copy((RegTensor<uint32_t>&)dstReg.reg[1], padReg);
        }
    }
}
} // namespace MicroAPI
} // namespace AscendC
#endif // ASCENDC_MODULE_MICRO_PACK_IMPL_H
