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
 * \file digamma_v220_impl.h
 * \brief
 */
#ifndef IMPL_MATH_DIGAMMA_DIGAMMA_V220_IMPL_H
#define IMPL_MATH_DIGAMMA_DIGAMMA_V220_IMPL_H
#include "kernel_tensor.h"
#include "kernel_tiling/kernel_tiling.h"

namespace AscendC {
__aicore__ inline void DigammaCast(const LocalTensor<float> &dst, const LocalTensor<float> &src, RoundMode castType)
{
    Cast<float, float, false>(dst, src, castType, MASK_PLACEHOLDER, 1,
                              {1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
    PipeBarrier<PIPE_V>();
}
} // namespace AscendC

#endif // IMPL_MATH_DIGAMMA_DIGAMMA_V220_IMPL_H
