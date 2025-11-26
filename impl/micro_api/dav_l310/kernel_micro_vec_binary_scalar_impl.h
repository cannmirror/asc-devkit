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

namespace AscendC {
namespace MicroAPI {
template <typename T = DefaultType, typename ScalarT, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void AddsImpl(RegT &dstReg, RegT &srcReg0, ScalarT scalarValue, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float>(),
        "current data type is not supported on current device");
    static_assert(SupportType<ScalarT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float>(),
        "current scalarValue data type is not supported on current device");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vadds(dstReg, srcReg0, scalarValue, mask, modeValue);
}

template <typename T = DefaultType, typename ScalarT, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void MulsImpl(RegT &dstReg, RegT &srcReg0, ScalarT scalarValue, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float>(),
        "current data type is not supported on current device");
    static_assert(SupportType<ScalarT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float>(),
        "current scalarValue data type is not supported on current device");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vmuls(dstReg, srcReg0, scalarValue, mask, modeValue);
}

template <typename T = DefaultType, typename ScalarT, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void MaxsImpl(RegT &dstReg, RegT &srcReg0, ScalarT scalarValue, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float>(),
        "current data type is not supported on current device");
    static_assert(SupportType<ScalarT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float>(),
        "current scalarValue data type is not supported on current device");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vmaxs(dstReg, srcReg0, scalarValue, mask, modeValue);
}

template <typename T = DefaultType, typename ScalarT, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void MinsImpl(RegT &dstReg, RegT &srcReg0, ScalarT scalarValue, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float>(),
        "current data type is not supported on current device");
    static_assert(SupportType<ScalarT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, half, float>(),
        "current scalarValue data type is not supported on current device");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vmins(dstReg, srcReg0, scalarValue, mask, modeValue);
}

template <typename T = DefaultType, typename ScalarT, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void ShiftLeftsImpl(RegT &dstReg, RegT &srcReg0, ScalarT scalarValue, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t>(),
        "current data type is not supported on current device");
    static_assert(SupportType<ScalarT, int16_t>(), "current scalarValue data type is not supported on current device");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vshls(dstReg, srcReg0, scalarValue, mask, modeValue);
}

template <typename T = DefaultType, typename ScalarT, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void ShiftRightsImpl(RegT &dstReg, RegT &srcReg0, ScalarT scalarValue, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t>(),
        "current data type is not supported on current device");
    static_assert(SupportType<ScalarT, int16_t>(), "current scalarValue data type is not supported on current device");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vshrs(dstReg, srcReg0, scalarValue, mask, modeValue);
}

template <typename T = DefaultType, typename ScalarT, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void RoundsImpl(RegT &dstReg, RegT &srcReg0, ScalarT scalarValue, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, int16_t, int32_t>(), "current data type is not supported on current device");
    static_assert(SupportType<ScalarT, uint16_t>(), "current scalarValue data type is not supported on current device");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vrnds(dstReg, srcReg0, scalarValue, mask, modeValue);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void LeakyReluImpl(RegT &dstReg, RegT &srcReg0, T scalarValue, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, half, float>(), "current data type is not supported on current device");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vlrelu(dstReg, srcReg0, scalarValue, mask, modeValue);
}
} // namespace MicroAPI
} // namespace AscendC
#endif // ASCENDC_MODULE_MICRO_VEC_BINARY_SCALAR_IMPL_H