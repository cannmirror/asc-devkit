/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file init_global_memory_v220_impl.h
 * \brief
 */
#ifndef IMPL_UTILS_INIT_GLOBAL_MEMORY_INIT_GLOBAL_MEMORY_V220_IMPL_H
#define IMPL_UTILS_INIT_GLOBAL_MEMORY_INIT_GLOBAL_MEMORY_V220_IMPL_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "../../api_check/kernel_api_check.h"

namespace AscendC {
template <typename T>
__aicore__ inline void InitGlobalMemoryImpl(GlobalTensor<T> &gmWorkspaceAddr, const uint64_t size, const T value)
{
    if ASCEND_IS_AIC {
        return;
    }
    CHECK_FUNC_HIGHLEVEL_API(InitGlobalMemory, (T), (gmWorkspaceAddr, size, value));
    LocalTensor<T> popBuffer;
    constexpr uint32_t MAX_REPEAT_LEN = 256;
    bool ret = PopStackBuffer<T, TPosition::LCM>(popBuffer);
    ASCENDC_ASSERT(ret, { KERNEL_LOG(KERNEL_ERROR, "No space left to allocate in Unified Buffer"); });
    constexpr uint32_t maxBurstSize = (MAX_REPEAT_TIMES * MAX_REPEAT_LEN) / sizeof(T);
    const uint32_t popSize = popBuffer.GetSize() >= maxBurstSize ? maxBurstSize : popBuffer.GetSize();
    const uint32_t round = size / popSize;
    const uint32_t tail = size % popSize;
    const uint32_t roundSize = round != 0 ? popSize : 0;
    Duplicate<T>(popBuffer, value, popSize);
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    struct DataCopyExtParams repeatParams;
    repeatParams.blockCount = 1;
    uint32_t comOffset = 0;
    // compute the main block
    repeatParams.blockLen = static_cast<uint32_t>(roundSize * sizeof(T));
    for (uint32_t index = 0; index < round; ++index) {
        DataCopyPad(gmWorkspaceAddr[comOffset], popBuffer, repeatParams);
        comOffset += roundSize;
    }
    // compute the tail block
    repeatParams.blockLen = static_cast<uint32_t>(tail * sizeof(T));
    if (tail != 0) {
        comOffset = round * roundSize;
        DataCopyPad(gmWorkspaceAddr[comOffset], popBuffer, repeatParams);
    }
    PipeBarrier<PIPE_MTE3>();
}

template <typename T>
__aicore__ inline __in_pipe__(V)
    __out_pipe__(MTE3) void InitGlobalMemory(GlobalTensor<T> &gmWorkspaceAddr, const uint64_t size, const T value)
{
    InitGlobalMemoryImpl<T>(gmWorkspaceAddr, size, value);
}
} // namespace AscendC
#endif // IMPL_UTILS_INIT_GLOBAL_MEMORY_INIT_GLOBAL_MEMORY_V220_IMPL_H
