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
 * \file sin_v200_impl.h
 * \brief
 */
#ifndef IMPL_MATH_SIN_SIN_V200_IMPL_H
#define IMPL_MATH_SIN_SIN_V200_IMPL_H
#include "kernel_basic_intf.h"
#include "kernel_tensor.h"

namespace AscendC {
__aicore__ inline void SinCast(
    const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor, RoundMode castType)
{
    const LocalTensor<int32_t>& castTensor = dstTensor.ReinterpretCast<int32_t>();
    Cast<int32_t, float, false>(castTensor, srcTensor, castType, MASK_PLACEHOLDER, 1,
        { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();

    Cast<float, int32_t, false>(dstTensor, castTensor, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
}
} // namespace AscendC

#endif // IMPL_MATH_SIN_SIN_V200_IMPL_H
