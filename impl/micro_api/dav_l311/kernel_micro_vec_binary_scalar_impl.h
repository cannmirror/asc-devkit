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
 * \file kernel_micro_vec_binary_scalar_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_VEC_BINARY_SCALAR_IMPL_H
#define ASCENDC_MODULE_MICRO_VEC_BINARY_SCALAR_IMPL_H

#include "kernel_micro_common_impl.h"
#include "../../../include/micro_api/kernel_micro_vec_binary_intf.h"

namespace AscendC {
namespace MicroAPI {
template <typename T = DefaultType, typename ScalarT, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__simd_callee__ inline void AddsImpl(RegT &dstReg, RegT &srcReg0, ScalarT scalarValue, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float>(),
        "current data type is not supported on current device!");
    static_assert(SupportType<ScalarT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float>(),
        "current scalarValue data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vadds(dstReg, srcReg0, scalarValue, mask, modeValue);
}

template <typename T = DefaultType, typename ScalarT, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__simd_callee__ inline void MulsImpl(RegT &dstReg, RegT &srcReg0, ScalarT scalarValue, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float>(),
        "current data type is not supported on current device!");
    static_assert(SupportType<ScalarT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float>(),
        "current scalarValue data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vmuls(dstReg, srcReg0, scalarValue, mask, modeValue);
}

template <typename T = DefaultType, typename ScalarT, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__simd_callee__ inline void MaxsImpl(RegT &dstReg, RegT &srcReg0, ScalarT scalarValue, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float>(),
        "current data type is not supported on current device!");
    static_assert(SupportType<ScalarT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float>(),
        "current scalarValue data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vmaxs(dstReg, srcReg0, scalarValue, mask, modeValue);
}

template <typename T = DefaultType, typename ScalarT, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__simd_callee__ inline void MinsImpl(RegT &dstReg, RegT &srcReg0, ScalarT scalarValue, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float>(),
        "current data type is not supported on current device!");
    static_assert(SupportType<ScalarT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float>(),
        "current scalarValue data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vmins(dstReg, srcReg0, scalarValue, mask, modeValue);
}

template <typename T = DefaultType, typename ScalarT, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__simd_callee__ inline void ShiftLeftsImpl(RegT &dstReg, RegT &srcReg0, ScalarT scalarValue, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t>(),
        "current data type is not supported on current device!");
    static_assert(SupportType<ScalarT, int16_t>(), "current scalarValue data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vshls(dstReg, srcReg0, scalarValue, mask, modeValue);
}

template <typename T = DefaultType, typename ScalarT, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__simd_callee__ inline void ShiftRightsImpl(RegT &dstReg, RegT &srcReg, ScalarT scalar, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t>(),
        "current data type is not supported on current device!");
    static_assert(SupportType<ScalarT, int16_t>(), "current scalar data type is not supported on current device!");
    static_assert(SupportEnum<mode, MaskMergeMode::ZEROING>(),
        "current ShiftRights api only supported Mode ZEROING on current device!");

    if constexpr (sizeof(ActualT) < 8) {
        constexpr auto modeValue = GetMaskMergeMode<mode>();
        vshrs(dstReg, srcReg, scalar, mask, modeValue);
    } else if constexpr (sizeof(ActualT) == 8) {
        if constexpr (CheckRegTrait<RegT, RegTraitNumTwo>()) {
            RegT dstTemp;
            ShiftRightsB64Impl(dstTemp, srcReg, scalar, mask);
            dstReg = dstTemp;
        } else if constexpr (CheckRegTrait<RegT, RegTraitNumOne>()) {
            MaskReg maskTrait2;
            MaskPack(maskTrait2, mask);
            RegTensor<ActualT, RegTraitNumTwo> traitTwoSrcReg0;
            RegTensor<ActualT, RegTraitNumTwo> traitTwoDstReg;
            B64TraitOneToTaitTwo(traitTwoSrcReg0, srcReg);
            ShiftRightsB64Impl(traitTwoDstReg, traitTwoSrcReg0, scalar, maskTrait2);
            B64TraitTwoToTaitOne(dstReg, traitTwoDstReg);
        }
    }
}

template <typename ScalarT, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__simd_callee__ inline void ShiftRightsB64Impl(RegT &dstReg, RegT &srcReg, ScalarT scalar, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    int16_t B32Width = 32;
    if constexpr (std::is_same_v<ActualT, uint64_t>) {
        RegTensor<uint32_t> tmpReg0;
        RegTensor<uint32_t> tmpReg1;
        vshrs(tmpReg0, (RegTensor<uint32_t> &)srcReg.reg[0], scalar, mask, modeValue);
        vshls(tmpReg1, (RegTensor<uint32_t> &)srcReg.reg[1], B32Width - scalar, mask, modeValue);
        Or((RegTensor<uint32_t> &)dstReg.reg[0], tmpReg0, tmpReg1, mask);
        vshrs(tmpReg0, (RegTensor<uint32_t> &)srcReg.reg[0], B32Width + scalar, mask, modeValue);
        vshrs(tmpReg1, (RegTensor<uint32_t> &)srcReg.reg[1], scalar, mask, modeValue);
        Or((RegTensor<uint32_t> &)dstReg.reg[1], tmpReg0, tmpReg1, mask);
    } else if constexpr (std::is_same_v<ActualT, int64_t>) {
        RegTensor<int32_t> tmpReg0;
        RegTensor<int32_t> tmpReg1;
        vshrs((RegTensor<uint32_t> &)tmpReg0, (RegTensor<uint32_t> &)srcReg.reg[0], scalar, mask, modeValue);
        vshls((RegTensor<int32_t> &)tmpReg1, (RegTensor<int32_t> &)srcReg.reg[1], B32Width - scalar, mask, modeValue);
        Or((RegTensor<int32_t> &)dstReg.reg[0], tmpReg0, tmpReg1, mask);
        vshrs((RegTensor<uint32_t> &)tmpReg0, (RegTensor<uint32_t> &)srcReg.reg[0], B32Width + scalar, mask, modeValue);
        vshrs(tmpReg1, (RegTensor<int32_t> &)srcReg.reg[1], scalar, mask, modeValue);
        Or((RegTensor<int32_t> &)dstReg.reg[1], tmpReg0, tmpReg1, mask);
    }
}

template <typename T = DefaultType, typename U, MaskMergeMode mode = MaskMergeMode::ZEROING, typename S>
__simd_callee__ inline void LeakyReluImpl(S& dstReg, S& srcReg, U scalarValue, MaskReg& mask)
{
    using ActualT = typename S::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, half, float>(), "current data type is not supported on current device!");
    static_assert(SupportType<U, half, float>(), "current scalarValue data type is not supported on current device!");
    static_assert(Std::is_convertible<U, ActualT>(), "scalarValue data type could be converted to RegTensor data type");
    static_assert(SupportEnum<mode, MaskMergeMode::ZEROING>(),
                  "current LeakyRelu api only supported Mode ZEROING on current device!");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vlrelu(dstReg, srcReg, scalarValue, mask, modeValue);
}
} // namespace MicroAPI
} // namespace AscendC
#endif // ASCENDC_MODULE_MICRO_VEC_BINARY_SCALAR_IMPL_H