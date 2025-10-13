/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef IMPL_REDUCE_REDUCE_MEAN_REDUCE_MEAN_V220_IMPL_H
#define IMPL_REDUCE_REDUCE_MEAN_REDUCE_MEAN_V220_IMPL_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../reduce_common_util_v220_impl.h"
#include "../../common/check.h"
#include "../../api_check/kernel_api_check.h"

namespace AscendC {
namespace Internal {
template <class T, class pattern, bool isReuseSource = false>
__aicore__ inline void ReduceMeanImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t srcShape[], bool srcInnerPad)
{
    uint32_t last = srcShape[1];
    uint32_t first = srcShape[0];
    constexpr uint32_t elePerBlk = ONE_BLK_SIZE / sizeof(T);
    uint32_t padLast = AlignUp(last, elePerBlk);
    static_assert(SupportType<T, half, float>(), "failed to check the data type, current api supports data type is half/float!");
    static_assert(SupportType<pattern, Pattern::Reduce::AR, Pattern::Reduce::RA>(), 
        "failed to check the reduce pattern, it only supports AR/RA pattern!");
    CHECK_FUNC_HIGHLEVEL_API(ReduceMean, (T, pattern), (dstTensor, srcTensor,
        sharedTmpBuffer, srcShape, srcInnerPad, padLast));
    ReduceParams reduceParams = ReduceParams(first, last, padLast, 0, 0, elePerBlk);
    ReduceSumCommon<T, pattern, isReuseSource>(dstTensor, srcTensor, sharedTmpBuffer, srcShape, srcInnerPad, reduceParams);
    SetMaskCount();
    UnaryRepeatParams defaultUnaryParam;
    if constexpr (IsSameType<pattern, Pattern::Reduce::AR>::value) {
        float lastAxisValReciprocal = 1.0f/static_cast<int32_t>(last);
        SetVectorMask<T, MaskMode::COUNTER>(first);
        Muls<T, false>(dstTensor, dstTensor, lastAxisValReciprocal, MASK_PLACEHOLDER, 1, defaultUnaryParam);
        PipeBarrier<PIPE_V>();
    } else {
        float firstAxisValReciprocal = 1.0f/static_cast<int32_t>(first);
        SetVectorMask<T, MaskMode::COUNTER>(last);
        Muls<T, false>(dstTensor, dstTensor, firstAxisValReciprocal, MASK_PLACEHOLDER, 1, defaultUnaryParam);
        PipeBarrier<PIPE_V>();
    }
    SetMaskNorm();
    ResetMask();
}
} // namespace Internal
} // namespace AscendC
#endif // IMPL_REDUCE_REDUCE_MEAN_REDUCE_MEAN_V220_IMPL_H