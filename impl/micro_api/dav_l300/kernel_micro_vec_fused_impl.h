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
 * \file kernel_micro_vec_cmpsel_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_VEC_FUSED_IMPL_H
#define ASCENDC_MODULE_MICRO_VEC_FUSED_IMPL_H

#include "kernel_micro_common_impl.h"

namespace AscendC {
namespace MicroAPI {

template <typename T, typename U, typename ScalarT, RegLayout layout, typename RegT, typename RegU>
__simd_callee__ inline void FusedMulsCastImpl(RegT &dstReg, RegU &srcReg, ScalarT scalarValue, MaskReg &mask)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "FusedMulsCast is not supported on current device!"); });
}

template <typename T, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__simd_callee__ inline void FusedMulDstAddImpl(RegT &dstReg, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, half, float>(), "current data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vmadd(dstReg, srcReg0, srcReg1, mask, modeValue);
}

template <typename T, MaskMergeMode mode, typename RegT>
__simd_callee__ inline void FusedAbsSubImpl(RegT &dstReg, RegT &srcReg0, RegT &srcReg1, MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, half, float>(), "current data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vabsdif(dstReg, srcReg0, srcReg1, mask, modeValue);
}

template <typename T, typename U, RegLayout layout, MaskMergeMode mode, typename RegT, typename RegU>
__simd_callee__ inline void FusedExpSubImpl(RegT &dstReg, RegU &srcReg0, RegU &srcReg1, MaskReg &mask)
{
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vsub(srcReg0, srcReg0, srcReg1, mask, modeValue);
    vexp(dstReg, srcReg0, mask, modeValue);
}
}
} // namespace AscendC
#endif // ASCENDC_MODULE_MICRO_VEC_FUSED_IMPL_H