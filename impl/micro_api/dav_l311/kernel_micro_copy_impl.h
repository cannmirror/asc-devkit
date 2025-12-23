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
 * \file kernel_micro_copy_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_COPY_IMPL_H
#define ASCENDC_MODULE_MICRO_COPY_IMPL_H

#include "kernel_micro_common_impl.h"
namespace AscendC {
namespace MicroAPI {
template <typename T = DefaultType, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__simd_callee__ inline void CopyImpl(RegT &dstReg, RegT &srcReg, MaskReg mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportEnum<mode, MaskMergeMode::MERGING>(),
        "current Move api only supported MaskMergeMode MERGING");
    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vmov(dstReg, srcReg, mask, modeValue);
}

template <typename T = DefaultType, typename RegT> __simd_callee__ inline void CopyImpl(RegT &dstReg, RegT &srcReg)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    vmov(dstReg, srcReg);
}
} // namespace MicroAPI
} // namespace AscendC
#endif