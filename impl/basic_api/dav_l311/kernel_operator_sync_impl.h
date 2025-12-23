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
 * \file kernel_operator_sync_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_SYNC_IMPL_H
#define ASCENDC_MODULE_OPERATOR_SYNC_IMPL_H

#include "utils/kernel_utils_constants.h"

namespace AscendC {

template<pipe_t AIV_PIPE = PIPE_MTE3, pipe_t AIC_PIPE = PIPE_FIX>
__aicore__ inline void SetNextTaskStartImpl()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported SetNextTaskStart!"); });
}

template<int8_t earlyStartConfig = -1>
__aicore__ inline void WaitPreTaskEndImpl()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported WaitPreTaskEnd!"); });
}

template<bool isAIVOnly = true>
__aicore__ inline void SoftSyncAllImpl(__gm__ int32_t* gmWorkspaceAddr, __ubuf__ int32_t* ubWorkspaceAddr,
    const int usedCores)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported software syncAll!"); });
}

template<bool isAIVOnly = true>
__aicore__ inline void SyncAllImpl()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "unsupported hardware syncAll!"); });
}

template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void NotifyEventImpl(uint16_t flagId)
{}

template <uint8_t modeId = 0, pipe_t pipe = PIPE_S>
__aicore__ inline void WaitEventImpl(uint16_t flagId)
{
    if (modeId == 2) {
        PipeBarrier<PIPE_ALL>();
    }
}
__aicore__ inline void SetSyncBaseAddrImpl(uint64_t config)
{
    ASCENDC_ASSERT((false), "SetSyncBaseAddr is not supported!");
}

} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_SYNC_IMPL_H
