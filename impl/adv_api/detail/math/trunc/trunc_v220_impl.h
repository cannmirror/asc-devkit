/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef IMPL_MATH_TRUNC_TRUNC_V220_IMPL_H
#define IMPL_MATH_TRUNC_TRUNC_V220_IMPL_H
#include "kernel_basic_intf.h"
#include "kernel_tensor.h"

namespace AscendC {
__aicore__ inline void TruncCastForTrunc(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
    const LocalTensor<uint8_t>& tmpTensor)
{
    Cast<float, float, false>(dstTensor, srcTensor, RoundMode::CAST_TRUNC, MASK_PLACEHOLDER, 1,
        { 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE });
    PipeBarrier<PIPE_V>();
}
} // namespace AscendC
#endif // IMPL_MATH_TRUNC_TRUNC_V220_IMPL_H