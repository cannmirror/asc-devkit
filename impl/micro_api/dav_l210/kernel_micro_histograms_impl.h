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
 * \file kernel_micro_histograms_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_HISTOGRAMS_IMPL_H
#define ASCENDC_MODULE_MICRO_HISTOGRAMS_IMPL_H

#include "kernel_micro_common_impl.h"
namespace AscendC {
namespace MicroAPI {
template <typename T = DefaultType, typename U = DefaultType, HistogramsBinType mode, HistogramsType type,
    typename RegT, typename RegU>
__aicore__ inline void HistogramsImpl(RegU &dstReg, RegT &srcReg, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    using ActualU = typename RegU::ActualT;
    static_assert((SupportType<ActualT, uint8_t>()), "current data type is not supported on current device!");
    static_assert((SupportType<ActualU, uint16_t>()), "current data type is not supported on current device!");
    if constexpr (type == HistogramsType::FREQUENCY) {
        if constexpr (mode == HistogramsBinType::BIN0) {
            dhist0(dstReg, srcReg, mask);
        } else if constexpr (mode == HistogramsBinType::BIN1) {
            dhist1(dstReg, srcReg, mask);
        } else if constexpr (mode == HistogramsBinType::BIN2) {
            dhist2(dstReg, srcReg, mask);
        } else if constexpr (mode == HistogramsBinType::BIN3) {
            dhist3(dstReg, srcReg, mask);
        }
    } else if constexpr (type == HistogramsType::ACCUMULATE) {
        if constexpr (mode == HistogramsBinType::BIN0) {
            chist0(dstReg, srcReg, mask);
        } else if constexpr (mode == HistogramsBinType::BIN1) {
            chist1(dstReg, srcReg, mask);
        } else if constexpr (mode == HistogramsBinType::BIN2) {
            chist2(dstReg, srcReg, mask);
        } else if constexpr (mode == HistogramsBinType::BIN3) {
            chist3(dstReg, srcReg, mask);
        }
    }
}
}  // namespace MicroAPI
}  // namespace AscendC

#endif // ASCENDC_MODULE_MICRO_HISTOGRAMS_IMPL_H