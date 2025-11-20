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
 * \file kernel_micro_vec_binary_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_VEC_BINARY_IMPL_H
#define ASCENDC_MODULE_MICRO_VEC_BINARY_IMPL_H

#include "kernel_micro_common_impl.h"

namespace AscendC {
namespace MicroAPI {
template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void AddImpl(RegT &dstReg, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float>(),
        "current data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vadd(dstReg, srcReg0, srcReg1, mask, modeValue);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void SubImpl(RegT &dstReg, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float>(),
        "current data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vsub(dstReg, srcReg0, srcReg1, mask, modeValue);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void MulImpl(RegT &dstReg, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float>(),
        "current data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vmul(dstReg, srcReg0, srcReg1, mask, modeValue);
}

template <typename T = DefaultType, auto mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void DivImpl(RegT &dstReg, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(
        IsSameType<decltype(mode), MaskMergeMode>::value || IsSameType<decltype(mode), const DivSpecificMode *>::value,
        "mode type must be either MaskMergeMode or const DivSpecificMode* ");
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    constexpr DivSpecificMode sprMode = util::GetDivSpecificMode(mode);
    static_assert(!sprMode.precisionMode, "precision mode for MicroAPI Div is not supported on the current device!");
    static_assert(SupportType<ActualT, uint16_t, int16_t, uint32_t, int32_t, half, float>(),
        "current data type is not supported on current device!!");
    constexpr auto modeValue = GetMaskMergeMode<sprMode.mrgMode>();
    vdiv(dstReg, srcReg0, srcReg1, mask, modeValue);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void MaxImpl(RegT &dstReg, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float>(),
        "current data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vmax(dstReg, srcReg0, srcReg1, mask, modeValue);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void MinImpl(RegT &dstReg, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float>(),
        "current data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vmin(dstReg, srcReg0, srcReg1, mask, modeValue);
}

template <typename T = DefaultType, typename SHIFT_T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING,
    typename RegT, typename RegShiftT>
__aicore__ inline void ShiftLeftImpl(RegT &dstReg, RegT &srcReg0, RegShiftT &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    using ActualShiftT = typename RegShiftT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(std::is_same_v<SHIFT_T, DefaultType> || std::is_same_v<SHIFT_T, ActualShiftT>,
        "T type is not correct!");
    static_assert(SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t>(),
        "current data type is not supported on current device!");
    static_assert(SupportType<ActualShiftT, int8_t, int16_t, int32_t>(),
        "current src1 data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vshl(dstReg, srcReg0, srcReg1, mask, modeValue);
}

template <typename T = DefaultType, typename SHIFT_T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING,
    typename RegT, typename RegShiftT>
__aicore__ inline void ShiftRightImpl(RegT &dstReg, RegT &srcReg0, RegShiftT &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    using ActualShiftT = typename RegShiftT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(std::is_same_v<SHIFT_T, DefaultType> || std::is_same_v<SHIFT_T, ActualShiftT>,
        "T type is not correct!");
    static_assert(SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t>(),
        "current data type is not supported on current device!");
    static_assert(SupportType<ActualShiftT, int8_t, int16_t, int32_t>(),
        "current src1 data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vshr(dstReg, srcReg0, srcReg1, mask, modeValue);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void AndImpl(RegT &dstReg, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t>(),
        "current data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vand(dstReg, srcReg0, srcReg1, mask, modeValue);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void OrImpl(RegT &dstReg, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t>(),
        "current data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vor(dstReg, srcReg0, srcReg1, mask, modeValue);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void XorImpl(RegT &dstReg, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t>(),
        "current data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vxor(dstReg, srcReg0, srcReg1, mask, modeValue);
}

template <typename T = DefaultType, typename IndexT = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING,
    typename RegT, typename RegIndexT>
__aicore__ inline void RoundImpl(RegT &dstReg, RegT &srcReg0, RegIndexT &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    using ActualIndexT = typename RegIndexT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(std::is_same_v<IndexT, DefaultType> || std::is_same_v<IndexT, ActualIndexT>,
        "IndexT type is not correct!");
    static_assert(SupportType<ActualT, int16_t, int32_t>(), "current data type is not supported on current device!");
    static_assert(SupportType<ActualIndexT, uint16_t, uint32_t>(),
        "current src1 data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vrnd(dstReg, srcReg0, srcReg1, mask, modeValue);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void PreluImpl(RegT &dstReg, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, half, float>(), "current data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vprelu(dstReg, srcReg0, srcReg1, mask, modeValue);
}

template <typename T = DefaultType, typename RegT>
__aicore__ inline void ModImpl(RegT &dstReg, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    constexpr auto modeValue = std::integral_constant<::Mode, static_cast<::Mode>(Mode::ZEROING)>();
    vdiv(dstReg, srcReg0, srcReg1, mask, modeValue);
    vmul(dstReg, srcReg1, dstReg, mask, modeValue);
    vsub(dstReg, srcReg0, dstReg, mask, modeValue);
}

template <typename T = DefaultType, typename RegT>
__aicore__ inline void MullImpl(RegT &dstReg0, RegT &dstReg1, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Mull api is not supported on current device!"); });
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void MulAddDstImpl(RegT &dstReg, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(
        SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float, bfloat16_t>(),
        "current data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vmula(dstReg, srcReg0, srcReg1, mask, modeValue);
}

template <typename T = DefaultType, typename RegT>
__aicore__ inline void AddCarryOutImpl(MaskReg &carryp, RegT &dstReg, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "AddCarryOut api is not supported on current device!"); });
}

template <typename T = DefaultType, typename RegT>
__aicore__ inline void SubCarryOutImpl(MaskReg &carryp, RegT &dstReg, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SubCarryOut api is not supported on current device!"); });
}

template <typename T = DefaultType, typename RegT>
__aicore__ inline void AddCarryOutsImpl(MaskReg &carryp, RegT &dstReg, RegT &srcReg0, RegT &srcReg1, MaskReg &carrysrcp,
    MaskReg &mask)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "AddCarryOuts api is not supported on current device!"); });
}

template <typename T = DefaultType, typename RegT>
__aicore__ inline void SubCarryOutsImpl(MaskReg &carryp, RegT &dstReg, RegT &srcReg0, RegT &srcReg1, MaskReg &carrysrcp,
    MaskReg &mask)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "SubCarryOuts api is not supported on current device!"); });
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void SaturationAddImpl(RegT &dstReg, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, int16_t>(), "current data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vsadd(dstReg, srcReg0, srcReg1, mask, modeValue);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void SaturationSubImpl(RegT &dstReg, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, int16_t>(), "current data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vssub(dstReg, srcReg0, srcReg1, mask, modeValue);
}

template <typename T = DefaultType, typename RegT>
__aicore__ inline void SlideImpl(RegT &dstReg, RegT &srcReg0, RegT &srcReg1, int16_t slideAmount)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, int8_t, uint8_t, int16_t, uint16_t, half, int32_t, uint32_t, float>(),
        "current data type is not supported on current device!");
    vslide(dstReg, srcReg0, srcReg1, slideAmount);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void Add3Impl(RegT &dstReg, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t>(),
        "current data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vadd3(dstReg, srcReg0, srcReg1, mask, modeValue);
}

template <typename T = DefaultType, RoundControl rnd = RoundControl::NO_ROUND, typename RegT>
__aicore__ inline void MeanImpl(RegTensor<T> &dstReg, RegTensor<T> &srcReg0, RegTensor<T> &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t>(),
        "current data type is not supported on current device!");
    constexpr auto rndValue = std::integral_constant<::Rnd, static_cast<::Rnd>(rnd)>();
    constexpr auto modeValue = GetMaskMergeMode<MaskMergeMode::ZEROING>();
    vavg(dstReg, srcReg0, srcReg1, rndValue, mask, modeValue);
}

} // namespace MicroAPI
} // namespace AscendC
#endif // ASCENDC_MODULE_MICRO_VEC_BINARY_IMPL_H