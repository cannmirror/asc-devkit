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
 * \file kernel_micro_vec_ternary_scalar_impl.h
 * \brief AscendC c310 support vaxpy level 0/2 api.
 */
#ifndef ASCENDC_MODULE_MICRO_VEC_TERNARY_SCALAR_IMPL_H
#define ASCENDC_MODULE_MICRO_VEC_TERNARY_SCALAR_IMPL_H
#include "kernel_micro_common_impl.h"

namespace AscendC {
namespace MicroAPI {
template <typename T, typename ScalarT, MaskMergeMode mode = MaskMergeMode::ZEROING, typename RegT>
__aicore__ inline void AxpyImpl(RegT &dstReg, RegT &srcReg, const ScalarT scalarValue, const MaskReg &mask)
{
    using ActualT = typename RegT::ActualT;
    static_assert(std::is_same_v<T, DefaultType> || std::is_same_v<T, ActualT>, "T type is not correct!");
    static_assert(SupportType<ActualT, half, float>(), "current Axpy data type is not supported on current device!");

    constexpr auto modeValue = GetMaskMergeMode<mode>();
    vaxpy(dstReg, srcReg, scalarValue, mask, modeValue);
}
} // namespace MicroAPI
} // namespace AscendC
#endif // ASCENDC_MODULE_MICRO_VEC_TERNARY_SCALAR_IMPL_H