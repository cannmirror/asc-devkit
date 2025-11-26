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
 * \file kernel_micro_vec_vconv_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_VEC_VCONV_IMPL_H
#define ASCENDC_MODULE_MICRO_VEC_VCONV_IMPL_H

#include "kernel_micro_common_impl.h"
namespace AscendC {
namespace MicroAPI {
template <typename T, typename U, RegLayout layoutMode, MaskMergeMode mode>
__aicore__ inline void CastImpl(RegTensor<T> &dstReg, RegTensor<U> &srcReg, MaskReg &mask)
{
    static_assert(SupportEnum<mode, MaskMergeMode::MERGING>(),
        "current Cast api only supported Mode MERGING on current device!");
    constexpr bool partCondition = SupportType<Tuple<T, U>, Tuple<uint16_t, uint8_t>, Tuple<int16_t, int8_t>,
        Tuple<uint32_t, uint16_t>, Tuple<uint32_t, int16_t>, Tuple<int32_t, int16_t>, Tuple<float, half>,
        Tuple<half, uint8_t>, Tuple<half, int8_t>, Tuple<float, int16_t>>();
    constexpr bool ppCondition = SupportType<Tuple<T, U>, Tuple<uint32_t, uint8_t>, Tuple<int32_t, int8_t>>();
    if constexpr (partCondition) {
        // vcvt_ii u82u16/s82s16/u162u32/s162u32/s162s32
        // vcvt_ff f162f32
        // vcvt_if u82f16/s82f16/s162f32
        static_assert(SupportEnum<layoutMode, RegLayout::ZERO, RegLayout::ONE>(),
            "current cast api RegLayout Mode is not supported on current device!");
        constexpr auto partModeValue = std::integral_constant<::Part, static_cast<::Part>(layoutMode)>();
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, half>) {
            vfcvt(dstReg, srcReg, partModeValue);
        } else {
            vcvt(dstReg, srcReg, partModeValue);
        }
    } else if constexpr (ppCondition) {
        // vcvt_ii u82u32/s82s32
        constexpr auto ppModeValue = std::integral_constant<::Part_T, static_cast<::Part_T>(layoutMode)>();
        vcvt(dstReg, srcReg, ppModeValue);
    } else {
        static_assert(!(partCondition && ppCondition), "current cast data type is not supported on current device!");
    }
}

template <typename T, typename U, SatMode satMode, RegLayout layoutMode, MaskMergeMode mode>
__aicore__ inline void CastImpl(RegTensor<T> &dstReg, RegTensor<U> &srcReg, MaskReg &mask)
{
    static_assert(SupportEnum<mode, MaskMergeMode::MERGING>(),
        "current Cast api only supported Mode MERGING on current device!");
    constexpr bool partCondition = SupportType<Tuple<T, U>, Tuple<uint8_t, uint16_t>, Tuple<uint8_t, int16_t>,
        Tuple<uint16_t, uint32_t>, Tuple<int16_t, uint32_t>, Tuple<uint16_t, int32_t>, Tuple<int16_t, int32_t>>();
    constexpr bool ppCondition = SupportType<Tuple<T, U>, Tuple<uint8_t, uint32_t>, Tuple<uint8_t, int32_t>>();
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    if constexpr (partCondition) {
        // vcvt_ii u162u8/s162u8/u322u16/u322s16/s322u16/s322s16
        static_assert(SupportEnum<layoutMode, RegLayout::ZERO, RegLayout::ONE>(),
            "current cast api RegLayout Mode is not supported on current device!");
        constexpr auto partModeValue = std::integral_constant<::Part, static_cast<::Part>(layoutMode)>();
        if constexpr (satMode == SatMode::SAT) {
            vscvt(dstReg, srcReg, partModeValue, modeValue);
        } else {
            vcvt(dstReg, srcReg, partModeValue, modeValue);
        }
    } else if constexpr (ppCondition) {
        // vcvt_ii u322u8/s322u8
        constexpr auto ppModeValue = std::integral_constant<::Part_T, static_cast<::Part_T>(layoutMode)>();
        if constexpr (satMode == SatMode::SAT) {
            vscvt(dstReg, srcReg, ppModeValue, modeValue);
        } else {
            vcvt(dstReg, srcReg, ppModeValue, modeValue);
        }
    } else {
        static_assert(!(partCondition && ppCondition), "current cast data type is not supported on current device!");
    }
}

// vfcvt/vsfcvt f322s16/f162s8/f162u8
// vfcvt f322f16
template <typename T, typename U, RoundMode roundMode, SatMode satMode, RegLayout layoutMode, MaskMergeMode mode>
__aicore__ inline void CastImpl(RegTensor<T> &dstReg, RegTensor<U> &srcReg, MaskReg &mask)
{
    static_assert(SupportEnum<mode, MaskMergeMode::MERGING>(),
        "current Cast api only supported Mode MERGING on current device!");
    static_assert(SupportType<Tuple<T, U>, Tuple<int16_t, float>, Tuple<uint8_t, half>, Tuple<int8_t, half>,
        Tuple<half, float>>(),
        "current cast data type is not supported on current device!");
    static_assert(SupportEnum<layoutMode, RegLayout::ZERO, RegLayout::ONE>(),
        "current cast api RegLayout Mode is not supported on current device!");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    constexpr auto partModeValue = std::integral_constant<::Part, static_cast<::Part>(layoutMode)>();
    constexpr auto roundModeValue = std::integral_constant<::ROUND, GetRound<roundMode>()>();
    if constexpr (satMode == SatMode::SAT) {
        vsfcvt(dstReg, srcReg, roundModeValue, partModeValue, modeValue);
    } else {
        vfcvt(dstReg, srcReg, roundModeValue, partModeValue, modeValue);
    }
}

// vsfcvt/vfcvt f322s32/f162s16
// f162s16 not support RoundMode in l210.
template <typename T, typename U, RoundMode roundMode, SatMode satMode, MaskMergeMode mode>
__aicore__ inline void CastImpl(RegTensor<T> &dstReg, RegTensor<U> &srcReg, MaskReg &mask)
{
    static_assert(SupportEnum<mode, MaskMergeMode::MERGING>(),
        "current Cast api only supported Mode MERGING on current device!");
    static_assert(SupportType<Tuple<T, U>, Tuple<int32_t, float>, Tuple<int16_t, half>>(),
        "current cast data type is not supported on current device!");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    constexpr auto roundModeValue = std::integral_constant<::ROUND, GetRound<roundMode>()>();
    if constexpr (std::is_same_v<T, int16_t> && std::is_same_v<U, half>) {
        static_assert(roundMode == RoundMode::CAST_NONE,
            "f162s16 not support this RoundMode set in l210! please use CAST_NONE!");
        if constexpr (satMode == SatMode::SAT) {
            vsfcvt(dstReg, srcReg, mask, modeValue);
        } else {
            vfcvt(dstReg, srcReg, mask, modeValue);
        }
    } else {
        if constexpr (satMode == SatMode::SAT) {
            vsfcvt(dstReg, srcReg, mask, roundModeValue, modeValue);
        } else {
            vfcvt(dstReg, srcReg, mask, roundModeValue, modeValue);
        }
    }
}

// vfcvt f162s32
template <typename T, typename U, RoundMode roundMode, RegLayout layoutMode, MaskMergeMode mode>
__aicore__ inline void CastImpl(RegTensor<T> &dstReg, RegTensor<U> &srcReg, MaskReg &mask)
{
    static_assert(SupportEnum<mode, MaskMergeMode::MERGING>(),
        "current Cast api only supported Mode MERGING on current device!");
    static_assert(SupportType<Tuple<T, U>, Tuple<int32_t, half>>(),
        "current cast data type is not supported on current device!");
    static_assert(SupportEnum<layoutMode, RegLayout::ZERO, RegLayout::ONE>(),
        "current cast api RegLayout Mode is not supported on current device!");
    constexpr auto partModeValue = std::integral_constant<::Part, static_cast<::Part>(layoutMode)>();
    constexpr auto roundModeValue = std::integral_constant<::ROUND, GetRound<roundMode>()>();
    vfcvt(dstReg, srcReg, roundModeValue, partModeValue);
}

// vfcvt s162f16/s322f32
// s162f16/s322f32 not support RoundMode in l210.
template <typename T, typename U, RoundMode roundMode, MaskMergeMode mode>
__aicore__ inline void CastImpl(RegTensor<T> &dstReg, RegTensor<U> &srcReg, MaskReg &mask)
{
    static_assert(SupportEnum<mode, MaskMergeMode::MERGING>(),
        "current Cast api only supported Mode MERGING on current device!");
    static_assert(SupportType<Tuple<T, U>, Tuple<half, int16_t>, Tuple<float, int32_t>>(),
        "current cast data type is not supported on current device!");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vfcvt(dstReg, srcReg, mask, modeValue);
}

// vtrc f162f16/f322f32
template <typename T, RoundMode roundMode = RoundMode::CAST_NONE, MaskMergeMode mode = MaskMergeMode::UNKNOWN>
__aicore__ inline void TruncateImpl(RegTensor<T>& dstReg, RegTensor<T>& srcReg, MaskReg &mask)
{
    static_assert(SupportEnum<mode, MaskMergeMode::MERGING>(),
        "current Cast api only supported Mode MERGING on current device!");
    static_assert(SupportType<T, half, float>(), "current trunc data type is not supported on current device!");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    constexpr auto roundModeValue = std::integral_constant<::ROUND, GetRound<roundMode>()>();
    vtrc(dstReg, srcReg, roundModeValue, mask, modeValue);
}

template <typename T, typename U, const CastTrait &trait, typename RegT, typename RegU>
__aicore__ inline void CastImpl(RegT &dstReg, RegU &srcReg, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    using ActualU = typename RegU::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(std::is_same_v<U, DefaultType> || std::is_same_v<U, ActualU>, "U type is not correct!");
    constexpr bool layoutMerge = SupportType<Tuple<ActualT, ActualU>, Tuple<uint16_t, uint8_t>, Tuple<int16_t, int8_t>,
        Tuple<uint32_t, uint16_t>, Tuple<uint32_t, int16_t>, Tuple<int32_t, int16_t>, Tuple<float, half>,
        Tuple<half, uint8_t>, Tuple<half, int8_t>, Tuple<float, int16_t>, Tuple<uint32_t, uint8_t>,
        Tuple<int32_t, int8_t>>();

    constexpr bool satLayMergeCast = SupportType<Tuple<ActualT, ActualU>, Tuple<uint8_t, uint16_t>,
        Tuple<uint8_t, int16_t>, Tuple<uint16_t, uint32_t>, Tuple<int16_t, uint32_t>, Tuple<uint16_t, int32_t>,
        Tuple<int16_t, int32_t>, Tuple<uint8_t, uint32_t>, Tuple<uint8_t, int32_t>>();
    constexpr bool rndSatLayoutMergeCast = SupportType<Tuple<ActualT, ActualU>, Tuple<int16_t, float>,
        Tuple<uint8_t, half>, Tuple<int8_t, half>, Tuple<int64_t, float>, Tuple<half, float>>();
    constexpr bool rndSatMergeCast =
        SupportType<Tuple<ActualT, ActualU>, Tuple<int32_t, float>, Tuple<int16_t, half>>();
    constexpr bool rndLayoutMergeCast = SupportType<Tuple<ActualT, ActualU>, Tuple<int32_t, half>>();
    constexpr bool rndMergeCast = SupportType<Tuple<ActualT, ActualU>, Tuple<half, int16_t>, Tuple<float, int32_t>>();
    constexpr bool allNotSupport = !(layoutMerge && satLayMergeCast && rndSatLayoutMergeCast && rndSatMergeCast &&
        rndLayoutMergeCast && rndMergeCast);
    if constexpr (layoutMerge) {
        CastImpl<ActualT, ActualU, trait.layoutMode, trait.mrgMode>(dstReg, srcReg, mask);
    } else if constexpr (satLayMergeCast) {
        CastImpl<ActualT, ActualU, trait.satMode, trait.layoutMode, trait.mrgMode>(dstReg, srcReg, mask);
    } else if constexpr (rndSatLayoutMergeCast) {
        CastImpl<ActualT, ActualU, trait.roundMode, trait.satMode, trait.layoutMode, trait.mrgMode>(dstReg, srcReg,
            mask);
    } else if constexpr (rndSatMergeCast) {
        CastImpl<ActualT, ActualU, trait.roundMode, trait.satMode, trait.mrgMode>(dstReg, srcReg, mask);
    } else if constexpr (rndLayoutMergeCast) {
        CastImpl<ActualT, ActualU, trait.roundMode, trait.layoutMode, trait.mrgMode>(dstReg, srcReg, mask);
    } else if constexpr (rndMergeCast) {
        CastImpl<ActualT, ActualU, trait.roundMode, trait.mrgMode>(dstReg, srcReg, mask);
    } else {
        static_assert(allNotSupport, "current cast data type is not supported on current device!");
    }
}
} // namespace MicroAPI
} // namespace AscendC
#endif // ASCENDC_MODULE_MICRO_VEC_VCONV_IMPL_H
