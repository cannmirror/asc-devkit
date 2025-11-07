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
 * \file kernel_micro_vec_duplicate_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_VEC_DUPLICATE_IMPL_H
#define ASCENDC_MODULE_MICRO_VEC_DUPLICATE_IMPL_H

#include "kernel_micro_common_impl.h"

namespace AscendC {
namespace MicroAPI {
template <typename T = DefaultType, typename T1, typename RegT>
__aicore__ inline void DuplicateImpl(RegT &dstReg, T1 scalar)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert((SupportType<ActualT, int8_t, uint8_t, uint16_t, int16_t, uint32_t, int32_t, float, half>()),
        "unsupported datatype on current device!");
    vbr(dstReg, (ActualT)scalar);
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename T1, typename RegT>
__aicore__ inline void DuplicateImpl(RegT &dstReg, T1 scalar, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert((SupportType<ActualT, int8_t, uint8_t, uint16_t, int16_t, uint32_t, int32_t, float, half>()),
        "unsupported datatype on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vdup(dstReg, (ActualT)scalar, mask, modeValue);
}

template <typename T = DefaultType, HighLowPart pos = HighLowPart::LOWEST, MaskMergeMode mode = MaskMergeMode::ZEROING,
    typename RegT>
__aicore__ inline void DuplicateImpl(RegT &dstReg, RegT &srcReg, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert((SupportType<ActualT, int8_t, uint8_t, uint16_t, int16_t, uint32_t, int32_t, float, half>()),
        "unsupported datatype on current device!");

    constexpr auto posValue = std::integral_constant<::Pos, static_cast<::Pos>(pos)>();
    constexpr auto modeValue = GetMaskMergeMode<mode>();

    vdup(dstReg, srcReg, mask, posValue, modeValue);
}

template <typename T = DefaultType, typename RegT>
__aicore__ inline void InterleaveImpl(RegT &dstReg0, RegT &dstReg1, RegT &srcReg0, RegT &srcReg1)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "Interleave only support type b8/b16/b32 on current device!");
    vintlv(dstReg0, dstReg1, srcReg0, srcReg1);
}

template <typename T = DefaultType, typename RegT>
__aicore__ inline void DeInterleaveImpl(RegT &dstReg0, RegT &dstReg1, RegT &srcReg0, RegT &srcReg1)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "DeInterleave only support type b8/b16/b32 on current device!");
    vdintlv(dstReg0, dstReg1, srcReg0, srcReg1);
}
} // namespace MicroAPI
} // namespace AscendC
#endif // ASCENDC_MODULE_MICRO_VEC_DUPLICATE_IMPL_H