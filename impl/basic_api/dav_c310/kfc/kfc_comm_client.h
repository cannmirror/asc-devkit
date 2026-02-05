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
 * \file kfc_comm_client.h
 * \brief
 */
#ifndef __KERNEL_KFC_COMM_CLIENT_H__
#define __KERNEL_KFC_COMM_CLIENT_H__

#include "kfc_comm.h"
#include "kernel_tpipe.h"

namespace AscendC {

class KfcCommClient;

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3101)
#ifdef SPLIT_CORE_CUBE
#elif defined(SPLIT_CORE_VEC)
__BLOCK_LOCAL__ __inline__ AscendC::KfcCommClient* g_kfcClient;
#else
__BLOCK_LOCAL__ __inline__ AscendC::KfcCommClient* g_kfcClient;
#endif
#endif
class KfcCommClient {
public:
    // 发送消息队列维护
    MSG_POS KfcMsg *msgSendHead;   // 消息头
    uint8_t msgSendPos;  // 用于循环队列的索引
    int8_t enableHardPoll;
    int8_t enableMixDualMaster;
public:
    __aicore__ inline KfcCommClient(GM_ADDR workspace, int subBlockID, int8_t enableHardPoll = 0,
        int8_t enableMixDualMaster = 0);
    __aicore__ inline ~KfcCommClient();
    __aicore__ inline GM_ADDR AllocUB(uint32_t size, int32_t &tailInfo)
    {
        return nullptr;
    }

    template <bool isAck>
    __aicore__ inline void PostMessage(MSG_POS KfcMsg *msg)
    {
        if (enableHardPoll == 1) {
            set_intra_block(PIPE_S, 15);
        }
    }

    template <bool isAck>
    __aicore__ inline void PostSameABFakeMsg(KFC_Enum funID, uint16_t instID)
    {
        if (enableHardPoll == 2) {
            auto msg = AllocMessage();
            msg->head = KfcMsgMakeFlag(funID, instID, 1);
        }
    }

    __aicore__ inline MSG_POS KfcMsg *AllocMessage()
    {
        auto msg = msgSendHead + msgSendPos;
        while (static_cast<bool>(KfcMsgGetState(msg->head))) {
        }
        msgSendPos = (msgSendPos + 1) & (MAX_MSG_COUNT_C310 - 1);
        return msg;
    }
};

__aicore__ inline KfcCommClient::KfcCommClient(GM_ADDR workspace, int subBlockID, int8_t enableHardPoll, int8_t enableMixDualMaster)
{
    if ASCEND_IS_AIV {
        ASCENDC_ASSERT((workspace != nullptr), { KERNEL_LOG(KERNEL_ERROR, "workspace can not be nullptr"); });
        ASCENDC_ASSERT((GetTPipePtr() != nullptr), { KERNEL_LOG(KERNEL_ERROR, "tpipe ptr can not be nullptr"); });
        ASCENDC_ASSERT((GetTPipePtr()->GetBaseAddr((int8_t)TPosition::VECIN) != nullptr),
                       { KERNEL_LOG(KERNEL_ERROR, "vecin base addr can not be nullptr"); });
        this->msgSendHead = (MSG_POS KfcMsg *)GetMsgHead(subBlockID); // 注意aic和aiv的地址是交换的
        this->msgSendPos = 0;
        this->enableHardPoll = enableHardPoll;
        this->enableMixDualMaster = enableMixDualMaster;
    }
}

__aicore__ inline KfcCommClient::~KfcCommClient()
{
    if ASCEND_IS_AIV {
        if (enableMixDualMaster == 1) {
            return;
        }
        MSG_POS KfcMsg *msg = AllocMessage();
        ASCENDC_ASSERT((msg != nullptr),
                       { KERNEL_LOG(KERNEL_ERROR, "ret of alloc message can not be nullptr when client quit"); });
        msg->head = KfcMsgMakeFlag(KFC_Enum::SERVICE_QUIT, 0);
        msg->userCustomData = GetTaskRationImpl(); // vector core nums: 1 & 2
        PostMessage<false>(msg);
    }
}

__aicore__ inline AscendC::KfcCommClient* GetKfcClient()
{
#ifndef SPLIT_CORE_CUBE
    return reinterpret_cast<AscendC::KfcCommClient*>(g_kfcClient);
#else
    return nullptr;
#endif
}

template <int NUM>
__aicore__ inline void SendSSbufData(uint64_t *ptr, MSG_POS uint64_t *ptrMsg)
{
#pragma unroll
    for (int i = 0; i < NUM; i++) {
        *(ptrMsg + i) = *(ptr + i);
    }
}
}  // namespace AscendC
#endif  // __KERNEL_KFC_COMM_CLIENT_H__
