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
 * \file kernel_micro_vec_cmpsel_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_VEC_FUSED_IMPL_H
#define ASCENDC_MODULE_MICRO_VEC_FUSED_IMPL_H

#include "kernel_micro_common_impl.h"

namespace AscendC {
namespace MicroAPI {

template <typename T, typename U, typename ScalarT, RegLayout layout, typename RegT, typename RegU>
__aicore__ inline void FusedMulsCastImpl(RegT &dstReg, RegU &srcReg, ScalarT scalar, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    using ActualU = typename RegU::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(std::is_same_v<U, DefaultType> || std::is_same_v<U, ActualU>, "U type is not correct!");
    static_assert(SupportType<Tuple<ActualT, ActualU, ScalarT>, Tuple<half, float, float>>(),
        "current data type is not supported on current device!");
    static_assert(SupportEnum<layout, RegLayout::ZERO, RegLayout::ONE>(),
        "current FusedMulsCast api only supported RegLayout ZERO, ONE on current device!");

    constexpr auto partModeValue = std::integral_constant<::Part, static_cast<::Part>(layout)>();
    vmulscvt(dstReg, srcReg, scalar, mask, partModeValue);
}

template <typename T, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void FusedMulDstAddImpl(RegT &dstReg, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, half, float, bfloat16_t>(), "current data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vmadd(dstReg, srcReg0, srcReg1, mask, modeValue);
}

template <typename T, MaskMergeMode mode, typename RegT>
__aicore__ inline void FusedAbsSubImpl(RegT &dstReg, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, half, float>(), "current data type is not supported on current device!");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vsub(dstReg, srcReg0, srcReg1, mask, modeValue);
    vabs(dstReg, dstReg, mask, modeValue);
}

template <typename T, typename U, RegLayout layout, MaskMergeMode mode, typename RegT, typename RegU>
__aicore__ inline void FusedExpSubImpl(RegT &dstReg, RegU &srcReg0, RegU &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    using ActualU = typename RegU::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(std::is_same_v<U, DefaultType> || std::is_same_v<U, ActualU>, "U type is not correct!");
    static_assert(SupportType<Tuple<ActualT, ActualU>, Tuple<float, float>, Tuple<half, half>>(),
        "current data type is not supported on current device!");
    static_assert(SupportEnum<layout, RegLayout::ZERO, RegLayout::ONE>(),
        "current FusedExpSub api only supported RegLayout ZERO, ONE on current device!");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    constexpr auto partModeValue = std::integral_constant<::Part, static_cast<::Part>(layout)>();
    vexpdif(dstReg, srcReg0, srcReg1, mask, partModeValue);
}
}
} // namespace AscendC
#endif // ASCENDC_MODULE_MICRO_VEC_FUSED_IMPL_H