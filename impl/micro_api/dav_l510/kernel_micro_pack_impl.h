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
 * \file kernel_micro_pack_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_PACK_IMPL_H
#define ASCENDC_MODULE_MICRO_PACK_IMPL_H

#include "kernel_micro_common_impl.h"
namespace AscendC {
namespace MicroAPI {
template <typename T = DefaultType, typename U = DefaultType, HighLowPart part = HighLowPart::LOWEST, typename RegT,
    typename RegU>
__aicore__ inline void PackImpl(RegT &dstReg, RegU &srcReg)
{
    using ActualT = typename RegT::ActualT;
    using ActualU = typename RegU::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(std::is_same_v<U, DefaultType> || std::is_same_v<U, ActualU>, "U type is not correct!");
    static_assert((SupportType<Tuple<ActualT, ActualU>, Tuple<uint8_t, uint16_t>, Tuple<uint8_t, int16_t>,
        Tuple<uint16_t, uint32_t>, Tuple<uint16_t, int32_t>>()),
        "unsupported datatype on current device");
    constexpr auto partValue = std::integral_constant<::HiloPart, static_cast<::HiloPart>(part)>();
    vpack(dstReg, srcReg, partValue);
}

template <typename T = DefaultType, typename U = DefaultType, HighLowPart part = HighLowPart::LOWEST, typename RegT,
    typename RegU>
__aicore__ inline void UnPackImpl(RegT &dstReg, RegU &srcReg)
{
    using ActualT = typename RegT::ActualT;
    using ActualU = typename RegU::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(std::is_same_v<U, DefaultType> || std::is_same_v<U, ActualU>, "U type is not correct!");
    static_assert((SupportType<Tuple<ActualT, ActualU>, Tuple<uint32_t, uint16_t>, Tuple<int32_t, int16_t>,
        Tuple<uint16_t, uint8_t>, Tuple<int16_t, int8_t>>()),
        "unsupported datatype on current device");
    constexpr auto partValue = std::integral_constant<::HiloPart, static_cast<::HiloPart>(part)>();
    vunpack(dstReg, srcReg, partValue);
}
} // namespace MicroAPI
} // namespace AscendC
#endif // ASCENDC_MODULE_MICRO_PACK_IMPL_H
