/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hccl_control.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_HCCL_HCCL_CONTROL_H
#define AICORE_ADV_API_DETAIL_HCCL_HCCL_CONTROL_H

#include "hccl_msg.h"

namespace AscendC {

constexpr uint8_t SINGLE_COMM_NUM = 1;
constexpr uint8_t MULTI_COMM_NUM = 2;

__aicore__ inline void GetRestartFromContext(GM_ADDR context, uint8_t& restart)
{
    if (context == nullptr) {
        return;
    }
    __gm__ HcclCombineOpParam* hcclContext = (__gm__ HcclCombineOpParam*)context;
    uint64_t msgAddr = hcclContext->workSpace;
    __gm__ HcclMsgArea* hcclMsgArea = (__gm__ HcclMsgArea*)msgAddr;
    __gm__ ControlHcclMsg* controlMsgGM = &hcclMsgArea->controlMsg;
    FlushDataCache(controlMsgGM);
    restart += controlMsgGM->restart;
}

__aicore__ inline void ResetRestartFlag(GM_ADDR context)
{
    if (context == nullptr) {
        return;
    }
    __gm__ HcclCombineOpParam* hcclContext = (__gm__ HcclCombineOpParam*)context;
    uint64_t msgAddr = hcclContext->workSpace;
    __gm__ HcclMsgArea* hcclMsgArea = (__gm__ HcclMsgArea*)msgAddr;
    __gm__ ControlHcclMsg* controlMsgGM = &hcclMsgArea->controlMsg;
    controlMsgGM->restart = 0;
    controlMsgGM->restarting = 1;
    controlMsgGM->resetSeq = 1;
    FlushDataCache(controlMsgGM);
}

__aicore__ inline uint8_t GetRestart(uint8_t ctxNum)
{
    uint8_t restart = 0;
    // 最大支持双通信域，0 1 分别获取两个通信域
    if (ctxNum >= SINGLE_COMM_NUM) {
        GetRestartFromContext(AscendC::GetHcclContext<0>(), restart);
    }
    if (ctxNum >= MULTI_COMM_NUM) {
        GetRestartFromContext(AscendC::GetHcclContext<1>(), restart);
    }
    return restart;
}

__aicore__ inline void SetRestart(uint8_t ctxNum)
{
    if (GetBlockIdx() == 0) {
        // 最大支持双通信域，0 1 分别获取两个通信域
        if (ctxNum >= SINGLE_COMM_NUM) {
            ResetRestartFlag(AscendC::GetHcclContext<0>());
        }
        if (ctxNum >= MULTI_COMM_NUM) {
            ResetRestartFlag(AscendC::GetHcclContext<1>());
        }
    }
}

__aicore__ inline bool CheckIfRestart(__gm__ HcclMsgArea* hcclMsgArea_)
{
#if AICORE_EXCEPTION_RESTART == 1
    // 重执行开启场景，检测是否需重执行
    __gm__ ControlHcclMsg* controlMsgGM = &hcclMsgArea_->controlMsg;
    FlushDataCache(controlMsgGM);
    if (controlMsgGM->restart > 0) {
        return true;
    }
#endif
    return false;
}

#define HCCL_CHECK_RESTART(msgArea, expression)                                                                        \
    do {                                                                                                               \
        if (unlikely(CheckIfRestart(msgArea))) {                                                                       \
            expression;                                                                                                \
        }                                                                                                              \
    } while (0)

} // namespace AscendC

#endif // AICORE_ADV_API_DETAIL_HCCL_HCCL_CONTROL_H