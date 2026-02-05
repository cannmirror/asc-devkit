/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file kernel_micro_histograms_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_HISTOGRAMS_IMPL_H
#define ASCENDC_MODULE_MICRO_HISTOGRAMS_IMPL_H

#include "../../../include/micro_api/kernel_micro_struct_intf.h"

namespace AscendC {
namespace MicroAPI {
template <typename T = DefaultType, typename U = DefaultType, HistogramsBinType mode, HistogramsType type,
          typename S, typename V>
__simd_callee__ inline void HistogramsImpl(V& dstReg, S& srcReg, MaskReg& mask)
{
    using ActualT = typename S::ActualT;
    using ActualU = typename V::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(std::is_same_v<U, DefaultType> || std::is_same_v<U, ActualU>, "U type is not correct!");
    static_assert((SupportType<ActualT, uint8_t>()), "current data type is not supported on current device!");
    static_assert((SupportType<ActualU, uint16_t>()), "current data type is not supported on current device!");
    auto constexpr HistogramsMode = std::integral_constant<::Bin, static_cast<::Bin>(mode)>();
    if constexpr (type == HistogramsType::FREQUENCY) {
        dhistv2(dstReg, srcReg, mask, HistogramsMode);
    } else if constexpr (type == HistogramsType::ACCUMULATE) {
        chistv2(dstReg, srcReg, mask, HistogramsMode);
    }
}
} // namespace MicroAPI
} // namespace AscendC
#endif // ASCENDC_MODULE_MICRO_HISTOGRAMS_IMPL_H