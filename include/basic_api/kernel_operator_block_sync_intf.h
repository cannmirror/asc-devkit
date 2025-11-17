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
 * \file kernel_operator_block_sync_intf.h
 * \brief
 */

#ifndef ASCENDC_MODULE_OPERATOR_BLOCK_SYNC_INTF_H
#define ASCENDC_MODULE_OPERATOR_BLOCK_SYNC_INTF_H

#include "kernel_reg.h"
#include "kernel_tensor_base.h"

namespace AscendC {

template <HardEvent event>
__aicore__ inline void SetFlag(int32_t eventID)
{
    if ASCEND_IS_AIC {
        if constexpr (event == HardEvent::MTE2_V || event == HardEvent::V_MTE2 || event == HardEvent::MTE3_V
                      || event == HardEvent::V_MTE3 || event == HardEvent::V_V || event == HardEvent::S_V ||
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
                      event == HardEvent::V_S || event == HardEvent::MTE2_MTE3 || event == HardEvent::MTE3_MTE2
                      || event == HardEvent::MTE3_S || event == HardEvent::S_MTE3) {
#else
                      event == HardEvent::V_S) {
#endif
            return;
        }
    }
    if ASCEND_IS_AIV {
        if constexpr ((event == HardEvent::MTE2_MTE1) || (event == HardEvent::MTE1_MTE2) ||
                      (event == HardEvent::MTE1_M) || (event == HardEvent::M_MTE1) || (event == HardEvent::M_FIX) ||
                      (event == HardEvent::FIX_M)) {
            return;
        }
    }
    SetFlagImpl<event>(eventID);
}

template <HardEvent event>
__aicore__ inline void WaitFlag(int32_t eventID)
{
    if ASCEND_IS_AIC {
        if constexpr (event == HardEvent::MTE2_V || event == HardEvent::V_MTE2 || event == HardEvent::MTE3_V
                      || event == HardEvent::V_MTE3 || event == HardEvent::V_V || event == HardEvent::S_V ||
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
                      event == HardEvent::V_S || event == HardEvent::MTE2_MTE3 || event == HardEvent::MTE3_MTE2
                      || event == HardEvent::MTE3_S || event == HardEvent::S_MTE3) {
#else
                      event == HardEvent::V_S) {
#endif
            return;
        }
    }
    if ASCEND_IS_AIV {
        if constexpr ((event == HardEvent::MTE2_MTE1) || (event == HardEvent::MTE1_MTE2) ||
                      (event == HardEvent::MTE1_M) || (event == HardEvent::M_MTE1) || (event == HardEvent::M_FIX) ||
                      (event == HardEvent::FIX_M)) {
            return;
        }
    }
    WaitFlagImpl(event, eventID);
}

template <pipe_t pipe>
__aicore__ inline void PipeBarrier()
{
    PipeBarrierImpl<pipe>();
}

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 2201) || (__NPU_ARCH__ == 3002) ||       \
    (__NPU_ARCH__ == 3102) || (__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102) ||       \
    (__NPU_ARCH__ == 3003) || (__NPU_ARCH__ == 3103) ||                                 \
    (__NPU_ARCH__ == 3113))
template <MemDsbT arg0>
__aicore__ inline void DataSyncBarrier()
{
    DataSyncBarrierImpl<arg0>();
}
#endif

/*
 * @ingroup：IBSet, IBWait
 * @brief：Set the flag bit of a core
 * @param [in] gmWorkspace GlobalTensor to store core state
 * @param [in] ubWorkspce LocalTensor for current core
 * @param [in] blockIdx the idx number waiting for the core
 * @param [in] eventID Set and wait events
 */
template <bool isAIVOnly = true>
__aicore__ inline void IBSet(const GlobalTensor<int32_t>& gmWorkspace, const LocalTensor<int32_t>& ubWorkspace,
                                  int32_t blockIdx, int32_t eventID);

template <bool isAIVOnly = true>
__aicore__ inline void IBWait(const GlobalTensor<int32_t>& gmWorkspace, const LocalTensor<int32_t>& ubWorkspace,
                                   int32_t blockIdx, int32_t eventID);
/*
 * @ingroup：SyncALL
 * @brief：Set flag bits of all cores
 * @param [in] gmWorkspace GlobalTensor to store core state
 * @param [in] ubWorkspce LocalTensor for current core
 */
template <bool isAIVOnly = true>
__aicore__ inline void SyncAll(const GlobalTensor<int32_t>& gmWorkspace, const LocalTensor<int32_t>& ubWorkspace,
                                 const int32_t usedCores = 0);

#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3101) || (__NPU_ARCH__ == 5102))
template <bool isAIVOnly = true, const SyncAllConfig& config = DEFAULT_SYNC_ALL_CONFIG>
__aicore__ inline void SyncAll();
#else
template <bool isAIVOnly = true>
__aicore__ inline void SyncAll();
#endif

template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void CrossCoreSetFlag(uint16_t flagId);

template <uint8_t modeId = 0, pipe_t pipe = PIPE_S>
__aicore__ inline void CrossCoreWaitFlag(uint16_t flagId);

template<pipe_t src, pipe_t dst>
class TQueSync {
public:
    __aicore__ inline void SetFlag(TEventID id);
    __aicore__ inline void WaitFlag(TEventID id);
};

} // namespace AscendC

#endif // KERNEL_BLOCK_SYNC_INTF_H