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
 * \file topk_common_impl.h
 * \brief
 */
#ifndef IMPL_SORT_TOPK_TOPK_COMMON_IMPL_H
#define IMPL_SORT_TOPK_TOPK_COMMON_IMPL_H
#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
#include "kernel_log.h"
#endif

#include "kernel_tensor.h"
#include "kernel_utils.h"
#include "kernel_pop_stack_buffer.h"
#include "kernel_tiling/kernel_tiling.h"
#include "topk_common_utils.h"
#include "include/adv_api/index/arithprogression.h"

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "topk_v220_impl.h"
#elif defined(__NPU_ARCH__) && __NPU_ARCH__ == 2002
#include "topk_v200_impl.h"
#endif

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201 || __NPU_ARCH__ == 2002)
namespace AscendC {
template <typename T, bool isInitIndex = false, bool isHasfinish = false, bool isReuseSrc = false>
__aicore__ inline void TopKNormal(const LocalTensor<T> &dstValueLocal, const LocalTensor<int32_t> &dstIndexLocal,
    const LocalTensor<T> &srcLocal, const LocalTensor<int32_t> &srcIndexLocal, const LocalTensor<bool> &finishLocal,
    const LocalTensor<uint8_t> &tmpLocal, const int32_t k, const TopkTiling &tilling, const TopKInfo &topKInfo,
    const bool isLargest = true)
{
    LocalTensor<T> tempBuffer = tmpLocal.template ReinterpretCast<T>();
    // if isInitIndex is false, The index of the input data needs to be generated here.
    if constexpr (!isInitIndex) {
        LocalTensor<int32_t> indexLocalTmp = tempBuffer[tilling.srcIndexOffset].template ReinterpretCast<int32_t>();
        ArithProgression(indexLocalTmp, static_cast<int32_t>(0), static_cast<int32_t>(1), topKInfo.inner);
        PipeBarrier<PIPE_V>();
    }

    SetMaskCount();
    TopKCompute<T, isInitIndex, isHasfinish>(dstValueLocal, dstIndexLocal, srcLocal, srcIndexLocal, finishLocal,
        tempBuffer, k, tilling, topKInfo, isLargest);

    if (!isLargest) {
        const UnaryRepeatParams unaryParams;
        SetVectorMask<T, MaskMode::COUNTER>(0, tilling.maskOffset);
        Muls<T, false>(dstValueLocal, dstValueLocal, T(-1), MASK_PLACEHOLDER, 1, unaryParams);
    }

    SetMaskNorm();
    ResetMask();
}

template <typename T, bool isInitIndex = false, bool isHasfinish = false, bool isReuseSrc = false>
__aicore__ inline void TopKNormal(const LocalTensor<T> &dstValueLocal, const LocalTensor<int32_t> &dstIndexLocal,
    const LocalTensor<T> &srcLocal, const LocalTensor<int32_t> &srcIndexLocal, const LocalTensor<bool> &finishLocal,
    const int32_t k, const TopkTiling &tilling, const TopKInfo &topKInfo, const bool isLargest = true)
{
    LocalTensor<uint8_t> stackTensor;
    PopStackBuffer<uint8_t, TPosition::LCM>(stackTensor);
    ASCENDC_ASSERT((stackTensor.GetSize() / sizeof(T) >= tilling.tmpLocalSize), {KERNEL_LOG(KERNEL_ERROR, "The stack "
        "buffer is insufficient, TopK api need %d, but only %ld exists.", tilling.tmpLocalSize, 
        stackTensor.GetSize() / sizeof(T));});
    stackTensor.SetSize(tilling.tmpLocalSize * sizeof(T));
    TopKNormal<T, isInitIndex, isHasfinish, isReuseSrc>(dstValueLocal, dstIndexLocal, srcLocal, srcIndexLocal, 
        finishLocal, stackTensor, k, tilling, topKInfo, isLargest);
}

template <typename T, bool isInitIndex = false, bool isHasfinish = false, bool isReuseSrc = false>
__aicore__ inline void TopKNSmall(const LocalTensor<T> &dstValueLocal, const LocalTensor<int32_t> &dstIndexLocal,
    const LocalTensor<T> &srcLocal, const LocalTensor<int32_t> &srcIndexLocal, const LocalTensor<bool> &finishLocal,
    const LocalTensor<uint8_t> &tmpLocal, const int32_t k, const TopkTiling &tilling, const TopKInfo &topKInfo,
    const bool isLargest = true)
{
    LocalTensor<T> tempBuffer = tmpLocal.template ReinterpretCast<T>();
    // if isInitIndex is false, The index of the input data needs to be generated here.
    if constexpr (!isInitIndex) {
        LocalTensor<int32_t> indexLocalTmp = tempBuffer[tilling.topkNSmallSrcIndexOffset].template
                                             ReinterpretCast<int32_t>();
        ArithProgression(indexLocalTmp, static_cast<int32_t>(0), static_cast<int32_t>(1), topKInfo.inner);
        PipeBarrier<PIPE_V>();
        if (topKInfo.outter > 1) {
            CopyData(indexLocalTmp, topKInfo);
        }
    }

    SetMaskCount();
    const UnaryRepeatParams unaryParams;
    // if isLargest if false, sort Ascending
    if (!isLargest) {
        SetVectorMask<T, MaskMode::COUNTER>(0, tilling.allDataSize);
        Muls<T, false>(tempBuffer[tilling.innerDataSize], srcLocal, T(-1), MASK_PLACEHOLDER, 1, unaryParams);
        PipeBarrier<PIPE_V>();
    }
    TopKNSmallCompute<T, isInitIndex, isHasfinish>(dstValueLocal, dstIndexLocal, srcLocal, srcIndexLocal, finishLocal,
        tempBuffer, k, tilling, topKInfo, isLargest);

    if (!isLargest) {
        PipeBarrier<PIPE_V>();
        SetMaskCount();
        SetVectorMask<T, MaskMode::COUNTER>(0, tilling.maskOffset);
        Muls<T, false>(dstValueLocal, dstValueLocal, T(-1), MASK_PLACEHOLDER, 1, unaryParams);
    }
    SetMaskNorm();
    ResetMask();
}

template <typename T, bool isInitIndex = false, bool isHasfinish = false, bool isReuseSrc = false>
__aicore__ inline void TopKNSmall(const LocalTensor<T> &dstValueLocal, const LocalTensor<int32_t> &dstIndexLocal,
    const LocalTensor<T> &srcLocal, const LocalTensor<int32_t> &srcIndexLocal, const LocalTensor<bool> &finishLocal,
    const int32_t k, const TopkTiling &tilling, const TopKInfo &topKInfo, const bool isLargest = true)
{
    LocalTensor<uint8_t> stackTensor;
    PopStackBuffer<uint8_t, TPosition::LCM>(stackTensor);
    ASCENDC_ASSERT((stackTensor.GetSize() / sizeof(T) >= tilling.tmpLocalSize), {KERNEL_LOG(KERNEL_ERROR, "The stack "
        "buffer is insufficient, TopK api need %d, but only %ld exists.", tilling.tmpLocalSize, 
        stackTensor.GetSize() / sizeof(T));});
    stackTensor.SetSize(tilling.tmpLocalSize * sizeof(T));

    TopKNSmall<T, isInitIndex, isHasfinish, isReuseSrc>(dstValueLocal, dstIndexLocal, srcLocal, srcIndexLocal,
        finishLocal, stackTensor, k, tilling, topKInfo, isLargest);
}

}  // namespace AscendC
#endif

#endif  // IMPL_SORT_TOPK_TOPK_COMMON_IMPL_H
