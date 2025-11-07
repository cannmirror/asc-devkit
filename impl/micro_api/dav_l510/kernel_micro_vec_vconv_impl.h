/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
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
    constexpr bool partCondition = SupportType<Tuple<T, U>, Tuple<uint16_t, uint8_t>, Tuple<int16_t, int8_t>,
        Tuple<uint32_t, uint16_t>, Tuple<uint32_t, int16_t>, Tuple<int32_t, int16_t>, Tuple<float, half>,
        Tuple<float, bfloat16_t>, Tuple<half, hifloat8_t>, Tuple<half, uint8_t>, Tuple<half, int8_t>,
        Tuple<float, int16_t>, Tuple<half, float8_e4m3_t>, Tuple<half, float8_e5m2_t>>();
    constexpr bool ppCondition = SupportType<Tuple<T, U>, Tuple<uint32_t, uint8_t>, Tuple<int32_t, int8_t>,
        Tuple<float, hifloat8_t>, Tuple<float, float8_e4m3_t>, Tuple<float, float8_e5m2_t>,
        Tuple<half, int4x2_t>, Tuple<bfloat16_t, int4x2_t>, Tuple<int16_t, int4x2_t>>();
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    if constexpr (partCondition) {
        // vcvt_ii u82u16/s82s16/u162u32/s162u32/s162s32
        // vcvt_ff f162f32
        // vcvt_if u82f16/s82f16/s162f32
        static_assert(SupportEnum<layoutMode, RegLayout::ZERO, RegLayout::ONE>(),
            "current cast api RegLayout Mode is not supported on current device!");
        constexpr auto partModeValue = std::integral_constant<::Part, static_cast<::Part>(layoutMode)>();
        vcvt(dstReg, srcReg, mask, partModeValue, modeValue);
    } else if constexpr (ppCondition) {
        constexpr auto ppModeValue = std::integral_constant<::Part_T, static_cast<::Part_T>(layoutMode)>();
        if constexpr (SupportType<Tuple<T, U>, Tuple<half, int4x2_t>>()) {
            // vcvt_if s42f16
            vcvt_s42f16(dstReg, srcReg, mask, ppModeValue, modeValue);
        } else if constexpr (SupportType<Tuple<T, U>, Tuple<bfloat16_t, int4x2_t>, Tuple<int16_t, int4x2_t>>()) {
            // vcvt_if s42s16
            RegTensor<half> halfTmp;
            constexpr auto roundModeValue = std::integral_constant<::ROUND, GetRound<RoundMode::CAST_RINT>()>();
            constexpr auto satModeValue =
                std::integral_constant<::RoundingSaturation, static_cast<::RoundingSaturation>(SatMode::SAT)>();
            vcvt_s42f16(halfTmp, srcReg, mask, ppModeValue, modeValue);
            if constexpr (SupportType<Tuple<T, U>, Tuple<bfloat16_t, int4x2_t>>()) {
                vcvt(dstReg, halfTmp, mask, roundModeValue, modeValue);
            } else {
                vcvt(dstReg, halfTmp, mask, roundModeValue, satModeValue, modeValue);
            }

        } else {
            // vcvt_ii u82u32/s82s32/
            // vcvt_ff hif82f32/e4m32f32/e5m22f32/fp4e2m12bf16/fp4e1m22bf16
            vcvt(dstReg, srcReg, mask, ppModeValue, modeValue);
        }
    } else {
        static_assert(!(partCondition && ppCondition), "current cast data type is not supported on current device!");
    }
}

template <typename T, typename U, SatMode satMode, RegLayout layoutMode, MaskMergeMode mode>
__aicore__ inline void CastImpl(RegTensor<T> &dstReg, RegTensor<U> &srcReg, MaskReg &mask)
{
    constexpr bool partCondition =
        SupportType<Tuple<T, U>, Tuple<uint8_t, uint16_t>, Tuple<uint8_t, int16_t>, Tuple<uint16_t, uint32_t>,
        Tuple<int16_t, uint32_t>, Tuple<uint16_t, int32_t>, Tuple<int16_t, int32_t>>();
    constexpr bool ppCondition = SupportType<Tuple<T, U>, Tuple<uint8_t, uint32_t>, Tuple<uint8_t, int32_t>,
        Tuple<int4x2_t, int16_t>>();
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    constexpr auto satModeValue =
        std::integral_constant<::RoundingSaturation, static_cast<::RoundingSaturation>(satMode)>();
    if constexpr (partCondition) {
        // vcvt_ii u162u8/s162u8/u322u16/u322s16/s322u16/s322s16
        static_assert(SupportEnum<layoutMode, RegLayout::ZERO, RegLayout::ONE>(),
            "current cast api RegLayout Mode is not supported on current device!");
        constexpr auto partModeValue = std::integral_constant<::Part, static_cast<::Part>(layoutMode)>();
        vcvt(dstReg, srcReg, mask, satModeValue, partModeValue, modeValue);
    } else if constexpr (ppCondition) {
        constexpr auto ppModeValue = std::integral_constant<::Part_T, static_cast<::Part_T>(layoutMode)>();
        if constexpr (SupportType<Tuple<T, U>, Tuple<int4x2_t, int16_t>>()) {
            // vcvt_ii s162s4x2
            RegTensor<half> halfTmp;
            constexpr auto roundModeValue = std::integral_constant<::ROUND, GetRound<RoundMode::CAST_RINT>()>();
            vcvt(halfTmp, srcReg, mask, roundModeValue, modeValue);
            vcvt_f162s4(dstReg, halfTmp, mask, roundModeValue, satModeValue, ppModeValue, modeValue);
        } else {
            // vcvt_ii u322u8/s322u8
            vcvt(dstReg, srcReg, mask, satModeValue, ppModeValue, modeValue);
        }
    } else {
        static_assert(!(partCondition && ppCondition), "current cast data type is not supported on current device!");
    }
}

template <typename T, typename U, RoundMode roundMode, SatMode satMode, RegLayout layoutMode, MaskMergeMode mode>
__aicore__ inline void CastImpl(RegTensor<T> &dstReg, RegTensor<U> &srcReg, MaskReg &mask)
{
    constexpr bool partCondition = SupportType<Tuple<T, U>, Tuple<int16_t, float>, Tuple<uint8_t, half>,
        Tuple<int8_t, half>, Tuple<int32_t, bfloat16_t>, Tuple<half, float>,
        Tuple<bfloat16_t, float>, Tuple<hifloat8_t, half>, Tuple<float8_e4m3_t, half>,
        Tuple<float8_e5m2_t, half>, Tuple<float8_e4m3_t, bfloat16_t>, Tuple<float8_e5m2_t, bfloat16_t>>();
    constexpr bool ppCondition = SupportType<Tuple<T, U>, Tuple<hifloat8_t, float>, Tuple<float8_e5m2_t, float>,
        Tuple<float8_e4m3_t, float>, Tuple<int4x2_t, half>>();
    static_assert(SupportEnum<layoutMode, RegLayout::ZERO, RegLayout::ONE>(),
        "current cast api RegLayout Mode is not supported on current device!");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    constexpr auto satModeValue =
        std::integral_constant<::RoundingSaturation, static_cast<::RoundingSaturation>(satMode)>();
    constexpr auto roundModeValue = std::integral_constant<::ROUND, GetRound<roundMode>()>();
    if constexpr (partCondition) {
        // vcvt_fi f322s16/f162u8/f162s8/bf162s32
        // vcvt_ff f322f16/f322bf16/f162hif8/f162e4m3/f162e5m2/bf162e4m3/bf162e5m2
        static_assert(SupportEnum<layoutMode, RegLayout::ZERO, RegLayout::ONE>(),
            "current cast api RegLayout Mode is not supported on current device!");
        constexpr auto partModeValue = std::integral_constant<::Part, static_cast<::Part>(layoutMode)>();
        vcvt(dstReg, srcReg, mask, roundModeValue, satModeValue, partModeValue, modeValue);
    } else if constexpr (ppCondition) {
        constexpr auto ppModeValue = std::integral_constant<::Part_T, static_cast<::Part_T>(layoutMode)>();
        // vcvt_fi f162s4x2
        if constexpr (SupportType<Tuple<T, U>, Tuple<int4x2_t, half>>()) {
            vcvt_f162s4(dstReg, srcReg, mask, roundModeValue, satModeValue, ppModeValue, modeValue);
        } else {
            // vcvt_ff f322hif8/f322e5m2/f322e4m3
            vcvt(dstReg, srcReg, mask, roundModeValue, satModeValue, ppModeValue, modeValue);
        }
    } else {
        static_assert(!(partCondition && ppCondition), "current cast data type is not supported on current device!");
    }
}

// vcvt_fi f322s32/f162s16
template <typename T, typename U, RoundMode roundMode, SatMode satMode, MaskMergeMode mode>
__aicore__ inline void CastImpl(RegTensor<T> &dstReg, RegTensor<U> &srcReg, MaskReg &mask)
{
    constexpr bool partCondition = SupportType<Tuple<T, U>, Tuple<int32_t, float>, Tuple<int16_t, half>>();
    constexpr bool ppCondition = SupportType<Tuple<T, U>, Tuple<half, bfloat16_t>>();
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    constexpr auto satModeValue =
        std::integral_constant<::RoundingSaturation, static_cast<::RoundingSaturation>(satMode)>();
    constexpr auto roundModeValue = std::integral_constant<::ROUND, GetRound<roundMode>()>();
    if constexpr (partCondition) {
        // vcvt_fi f322s32/f162s16
        vcvt(dstReg, srcReg, mask, roundModeValue, satModeValue, modeValue);
    } else if constexpr (ppCondition) {
        // bf162f16
        vcvt(dstReg, srcReg, mask, satModeValue, roundModeValue, modeValue);
    } else {
        static_assert(!(partCondition && ppCondition), "current cast data type is not supported on current device!");
    }
}

// vcvt_fi f162s32
template <typename T, typename U, RoundMode roundMode, RegLayout layoutMode, MaskMergeMode mode>
__aicore__ inline void CastImpl(RegTensor<T> &dstReg, RegTensor<U> &srcReg, MaskReg &mask)
{
    static_assert(SupportType<Tuple<T, U>, Tuple<int32_t, half>>(),
        "current cast data type is not supported on current device!");
    static_assert(SupportEnum<layoutMode, RegLayout::ZERO, RegLayout::ONE>(),
        "current cast api RegLayout Mode is not supported on current device!");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    constexpr auto partModeValue = std::integral_constant<::Part, static_cast<::Part>(layoutMode)>();
    constexpr auto roundModeValue = std::integral_constant<::ROUND, GetRound<roundMode>()>();
    vcvt(dstReg, srcReg, mask, roundModeValue, partModeValue, modeValue);
}

// vcvt_if s162f16/s322f32
// vcvt_ff f162bf16
template <typename T, typename U, RoundMode roundMode, MaskMergeMode mode>
__aicore__ inline void CastImpl(RegTensor<T> &dstReg, RegTensor<U> &srcReg, MaskReg &mask)
{
    static_assert(SupportType<Tuple<T, U>, Tuple<half, int16_t>, Tuple<float, int32_t>, Tuple<bfloat16_t, half>>(),
        "current cast data type is not supported on current device!");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    constexpr auto roundModeValue = std::integral_constant<::ROUND, GetRound<roundMode>()>();
    vcvt(dstReg, srcReg, mask, roundModeValue, modeValue);
}

// truncate f162f16/f322f32/bf162bf16
template <typename T, RoundMode roundMode, MaskMergeMode mode, typename RegT>
__aicore__ inline void TruncateImpl(RegT &dstReg, RegT &srcReg, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(SupportType<ActualT, half, float, bfloat16_t>(),
        "current trunc data type is not supported on current device!");
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
        Tuple<float, bfloat16_t>, Tuple<half, hifloat8_t>, Tuple<half, uint8_t>, Tuple<half, int8_t>,
        Tuple<float, int16_t>, Tuple<half, float8_e4m3_t>, Tuple<half, float8_e5m2_t>,
        Tuple<uint32_t, uint8_t>, Tuple<int32_t, int8_t>,
        Tuple<float, hifloat8_t>, Tuple<float, float8_e4m3_t>, Tuple<float, float8_e5m2_t>,
        Tuple<half, int4x2_t>, Tuple<bfloat16_t, int4x2_t>, Tuple<int16_t, int4x2_t>>();

    constexpr bool satLayMergeCast = SupportType<Tuple<ActualT, ActualU>, Tuple<uint8_t, uint16_t>,
        Tuple<uint8_t, int16_t>, Tuple<uint16_t, uint32_t>, Tuple<int16_t, uint32_t>, Tuple<uint16_t, int32_t>,
        Tuple<int16_t, int32_t>, Tuple<uint8_t, uint32_t>, Tuple<uint8_t, int32_t>, Tuple<int4x2_t, int16_t>>();
    constexpr bool rndSatLayoutMergeCast = SupportType<Tuple<ActualT, ActualU>, Tuple<int16_t, float>,
        Tuple<uint8_t, half>, Tuple<int8_t, half>, Tuple<int32_t, bfloat16_t>, Tuple<half, float>,
        Tuple<bfloat16_t, float>, Tuple<hifloat8_t, half>, Tuple<float8_e4m3_t, half>,
        Tuple<float8_e5m2_t, half>, Tuple<float8_e4m3_t, bfloat16_t>, Tuple<float8_e5m2_t, bfloat16_t>,
        Tuple<hifloat8_t, float>, Tuple<float8_e5m2_t, float>, Tuple<float8_e4m3_t, float>, Tuple<int4x2_t, half>>();
    constexpr bool rndSatMergeCast =
        SupportType<Tuple<ActualT, ActualU>, Tuple<int32_t, float>, Tuple<int16_t, half>, Tuple<half, bfloat16_t>>();
    constexpr bool rndLayoutMergeCast = SupportType<Tuple<ActualT, ActualU>, Tuple<int32_t, half>>();
    constexpr bool rndMergeCast = SupportType<Tuple<ActualT, ActualU>, Tuple<half, int16_t>, Tuple<float, int32_t>,
        Tuple<bfloat16_t, half>>();
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
