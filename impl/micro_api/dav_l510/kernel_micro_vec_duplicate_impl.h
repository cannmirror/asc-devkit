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
    static_assert((SupportType<ActualT, bool, int8_t, uint8_t, uint16_t, int16_t, bfloat16_t, uint32_t, int32_t, float,
        half>()),
        "current data type is not supported on current device!");

    if constexpr (IsSameType<ActualT, bool>::value) {
        vbr((RegTensor<int8_t> &)dstReg, (int8_t)scalar);
    } else {
        vbr(dstReg, (ActualT)scalar);
    }
}

template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename T1, typename RegT>
__aicore__ inline void DuplicateImpl(RegT &dstReg, T1 scalar, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert((SupportType<ActualT, bool, int8_t, uint8_t, uint16_t, int16_t, bfloat16_t, uint32_t, int32_t, float,
        half>()),
        "current data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    if constexpr (IsSameType<ActualT, bool>::value) {
        vdup((RegTensor<int8_t> &)dstReg, (int8_t)scalar, mask, modeValue);
    } else  {
        vdup(dstReg, (ActualT)scalar, mask, modeValue);
    }
}

template <typename T = DefaultType, HighLowPart pos = HighLowPart::LOWEST, MaskMergeMode mode = MaskMergeMode::ZEROING,
    typename RegT>
__aicore__ inline void DuplicateImpl(RegT &dstReg, RegT &srcReg, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert((SupportType<ActualT, bool, int8_t, uint8_t, uint16_t, int16_t, uint32_t, int32_t, float, half,
        bfloat16_t>()),
        "current data type is not supported on current device!");
    constexpr auto posValue = std::integral_constant<::Pos, static_cast<::Pos>(pos)>();
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    if constexpr (IsSameType<ActualT, bool>::value) {
        vdup((RegTensor<int8_t> &)dstReg, (RegTensor<int8_t> &)srcReg, mask, posValue, modeValue);
    } else {
        vdup(dstReg, srcReg, mask, posValue, modeValue);
    }
}

template <typename T = DefaultType, typename RegT>
__aicore__ inline void InterleaveImpl(RegT &dstReg0, RegT &dstReg1, RegT &srcReg0, RegT &srcReg1)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportBytes<ActualT, 1, 2, 4>(), "Interleave only support type bool/b8/b16/b32 on current device");
    if constexpr (sizeof(ActualT) == 1) {
        vintlv((RegTensor<int8_t> &)dstReg0, (RegTensor<int8_t> &)dstReg1, (RegTensor<int8_t> &)srcReg0, (RegTensor<int8_t> &)srcReg1);
    } else if constexpr (sizeof(ActualT) == 2) {
        vintlv((RegTensor<int16_t> &)dstReg0, (RegTensor<int16_t> &)dstReg1, (RegTensor<int16_t> &)srcReg0, (RegTensor<int16_t> &)srcReg1);
    } else {
        vintlv((RegTensor<int32_t> &)dstReg0, (RegTensor<int32_t> &)dstReg1, (RegTensor<int32_t> &)srcReg0, (RegTensor<int32_t> &)srcReg1);
    }
}

template <typename T = DefaultType, typename RegT>
__aicore__ inline void DeInterleaveImpl(RegT &dstReg0, RegT &dstReg1, RegT &srcReg0, RegT &srcReg1)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportBytes<ActualT, 1, 2, 4>(),
        "DeInterleave only support type bool/b8/b16/b32 on current device");
    if constexpr (sizeof(ActualT) == 1) {
        vdintlv((RegTensor<int8_t> &)dstReg0, (RegTensor<int8_t> &)dstReg1, (RegTensor<int8_t> &)srcReg0, (RegTensor<int8_t> &)srcReg1);
    } else if constexpr (sizeof(ActualT) == 2) {
        vdintlv((RegTensor<int16_t> &)dstReg0, (RegTensor<int16_t> &)dstReg1, (RegTensor<int16_t> &)srcReg0, (RegTensor<int16_t> &)srcReg1);
    } else {
        vdintlv((RegTensor<int32_t> &)dstReg0, (RegTensor<int32_t> &)dstReg1, (RegTensor<int32_t> &)srcReg0, (RegTensor<int32_t> &)srcReg1);
    }
}
} // namespace MicroAPI
} // namespace AscendC
#endif // ASCENDC_MODULE_MICRO_VEC_DUPLICATE_IMPL_H