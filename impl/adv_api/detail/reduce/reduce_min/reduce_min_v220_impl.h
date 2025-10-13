/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef IMPL_REDUCE_REDUCE_MIN_REDUCE_MIN_V220_IMPL_H_
#define IMPL_REDUCE_REDUCE_MIN_REDUCE_MIN_V220_IMPL_H_

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../reduce_common_util_v220_impl.h"
#include "../../common/check.h"
#include "../../api_check/kernel_api_check.h"

namespace AscendC {
namespace Internal {
template <class T, class pattern, bool isReuseSource = false>
__aicore__ inline void ReduceMinImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
                                      const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t srcShape[],
                                      bool srcInnerPad)
{
    uint32_t last = srcShape[1];
    uint32_t first = srcShape[0];
    constexpr uint32_t elePerBlk = ONE_BLK_SIZE / sizeof(T);
    uint32_t padLast = AlignUp(last, elePerBlk);
    static_assert(SupportType<T, half, float>(), "failed to check the data type, current api supports data type is half/float!");
    static_assert(SupportType<pattern, Pattern::Reduce::AR, Pattern::Reduce::RA>(), 
        "failed to check the reduce pattern, it only supports AR/RA pattern!");
    CHECK_FUNC_HIGHLEVEL_API(ReduceMin, (T, pattern), (dstTensor, srcTensor, sharedTmpBuffer, srcShape, srcInnerPad, padLast));

    LocalTensor<T> tmpTensor = sharedTmpBuffer.ReinterpretCast<T>();

    if constexpr (IsSameType<pattern, Pattern::Reduce::AR>::value) {
        BlockReduceByLastAxis<T, isReuseSource, ApiMode::API_MODE_MIN, Min<T, false>>(
            dstTensor, srcTensor, tmpTensor, first, last, padLast);
    } else {
        BinaryReduceByFirstAxis<T, isReuseSource, Min<T, false>>(
            dstTensor, srcTensor, tmpTensor, first, last, padLast);
    }
    SetMaskNorm();
    ResetMask();
}
} // namespace Internal
} // namespace AscendC
#endif // IMPL_REDUCE_REDUCE_MIN_REDUCE_MIN_V220_IMPL_H_