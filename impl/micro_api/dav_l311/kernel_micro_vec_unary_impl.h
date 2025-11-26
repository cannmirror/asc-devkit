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
 * \file kernel_micro_vec_unary_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_VEC_SINGLE_IMPL_H
#define ASCENDC_MODULE_MICRO_VEC_SINGLE_IMPL_H
#include "kernel_micro_common_impl.h"

namespace AscendC {
namespace MicroAPI {
template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void AbsImpl(RegT &dstReg, RegT &srcReg, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, int8_t, int16_t, int32_t, half, float>(),
        "current data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vabs(dstReg, srcReg, mask, modeValue);
}

template <typename T, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void ReluImpl(RegT &dstReg, RegT &srcReg, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, int32_t, half, float>(),
        "current data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vrelu(dstReg, srcReg, mask, modeValue);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void ExpImpl(RegT &dstReg, RegT &srcReg, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, half, float>(), "current data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vexp(dstReg, srcReg, mask, modeValue);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void SqrtImpl(RegT &dstReg, RegT &srcReg, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, half, float>(), "current data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vsqrt(dstReg, srcReg, mask, modeValue);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void RsqrtImpl(RegT &dstReg, RegT &srcReg, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, half, float>(), "current data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vrsqrt(dstReg, srcReg, mask, modeValue);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void RecImpl(RegT &dstReg, RegT &srcReg, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, half, float>(), "current data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vrec(dstReg, srcReg, mask, modeValue);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void LogImpl(RegT &dstReg, RegT &srcReg, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, half, float>(), "current data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vln(dstReg, srcReg, mask, modeValue);
}

template <MaskMergeMode mode = MaskMergeMode::ZEROING>
__aicore__ inline void LogXImpl(RegTensor<half> &dstReg, RegTensor<half> &srcReg,
    MaskReg &mask, const float lnXReciprocal)
{
    vector_f16 f16RegLow;
    vector_f16 f16RegHigh;
    vector_f32 f32RegLow;
    vector_f32 f32RegHigh;
    vector_bool MaskLow;
    vector_bool MaskHigh;
    vector_f16 tmpReg;
    constexpr auto patAll = std::integral_constant<::Pat, static_cast<::Pat>(MicroAPI::MaskPattern::ALL)>();
    vector_bool maskAll = pset_b32(patAll);
    constexpr auto patVal = std::integral_constant<::Pat, static_cast<::Pat>(MicroAPI::MaskPattern::H)>();
    vector_bool selMask = pset_b16(patVal);
    constexpr auto partModeEvenVal = std::integral_constant<::Part, static_cast<::Part>(RegLayout::ZERO)>();
    constexpr auto satModeValue =
        std::integral_constant<::RoundingSaturation, static_cast<::RoundingSaturation>(SatMode::NO_SAT)>();
    constexpr auto roundModeValue = std::integral_constant<::ROUND, GetRound<RoundMode::CAST_RINT>()>();
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    punpack(MaskLow, mask, LOWER);
    punpack(MaskHigh, mask, HIGHER);

    vintlv(f16RegLow, f16RegHigh, srcReg, srcReg);

    vcvt(f32RegLow, f16RegLow, maskAll, partModeEvenVal, modeValue);
    vcvt(f32RegHigh, f16RegHigh, maskAll, partModeEvenVal, modeValue);

    vln(f32RegLow, f32RegLow, maskAll, modeValue);
    vln(f32RegHigh, f32RegHigh, maskAll, modeValue);

    vmuls(f32RegLow, f32RegLow, lnXReciprocal, maskAll, modeValue);
    vmuls(f32RegHigh, f32RegHigh, lnXReciprocal, maskAll, modeValue);

    vcvt(f16RegLow, f32RegLow, MaskLow, roundModeValue, satModeValue, partModeEvenVal, modeValue);
    vcvt(f16RegHigh, f32RegHigh, MaskHigh, roundModeValue, satModeValue, partModeEvenVal, modeValue);

    vdintlv(f16RegLow, tmpReg, f16RegLow, tmpReg);
    vdintlv(f16RegHigh, tmpReg, tmpReg, f16RegHigh);
    vsel(dstReg, f16RegLow, f16RegHigh, selMask);
}


template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void Log2Impl(RegT &dstReg, RegT &srcReg, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, half, float>(), "current data type is not supported on current device!");

    constexpr float ln2Reciprocal = 1.4426950408889634; // 1.0/Ln2;
    if constexpr (SupportType<ActualT, half>()) {
        LogXImpl<mode>(dstReg, srcReg, mask, ln2Reciprocal);
    } else {
        constexpr auto modeValue = GetMaskMergeMode<mode>();
        vln(dstReg, srcReg, mask, modeValue);
        vmuls(dstReg, dstReg, ln2Reciprocal, mask, modeValue);
    }
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void Log10Impl(RegT &dstReg, RegT &srcReg, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, half, float>(), "current data type is not supported on current device!");

    constexpr float ln10Reciprocal = 0.43429448190325176; // 1.0/Ln10;
    if constexpr (SupportType<ActualT, half>()) {
        LogXImpl<mode>(dstReg, srcReg, mask, ln10Reciprocal);
    } else {
        constexpr auto modeValue = GetMaskMergeMode<mode>();
        vln(dstReg, srcReg, mask, modeValue);
        vmuls(dstReg, dstReg, ln10Reciprocal, mask, modeValue);
    }
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void NegImpl(RegT &dstReg, RegT &srcReg, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, int8_t, int16_t, int32_t, half, float>(),
        "current data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vneg(dstReg, srcReg, mask, modeValue);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void NotImpl(RegT &dstReg, RegT &srcReg, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float>(),
        "current data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vnot(dstReg, srcReg, mask, modeValue);
}

template <typename T = DefaultType, typename SrcT = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING,
    typename RegT, typename RegSrcT>
__aicore__ inline void CountBitImpl(RegT &dstReg, RegSrcT &srcReg, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    using ActualSrcT = typename RegSrcT::ActualT;
    static_assert(SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t>(),
        "current dst data type is not supported on current device!");
    static_assert(SupportType<ActualSrcT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t>(),
        "current src data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vbcnt(dstReg, srcReg, mask, modeValue);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void CountLeadingSignBitsImpl(RegT &dstReg, RegT &srcReg, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<T, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t>(),
        "current data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vcls(dstReg, srcReg, mask, modeValue);
}
} // namespace MicroAPI
} // namespace AscendC
#endif // ASCENDC_MODULE_MICRO_VEC_SINGLE_IMPL_H