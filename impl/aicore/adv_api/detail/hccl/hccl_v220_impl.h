/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hccl_v220_impl.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_HCCL_HCCL_V220_IMPL_H
#define AICORE_ADV_API_DETAIL_HCCL_HCCL_V220_IMPL_H

#include "hccl_common.h"
#include "hccl_control.h"

namespace AscendC {

template <HcclServerType serverType, const auto& config>
__aicore__ inline void HcclImpl<serverType, config>::SendMsgToServer(
    uint16_t queId, const CommonPrepareParam& para, int8_t srcGroupID, HcclHandle srcHandleID)
{
    if (!workingFlag_ && queueNum_ == 0U) {
        return;
    }
    __gm__ HcclMsg* hcclSendMsg;
    if (queueNum_ == 0U) {
        hcclSendMsg = hcclMsgArea_->sendMsgs + curMsgPosition_[0U];
    } else {
        __gm__ HcclMsgAreaForMultiQue* tmp = reinterpret_cast<__gm__ HcclMsgAreaForMultiQue*>(hcclMsgArea_);
        hcclSendMsg = tmp->sendMsgs[queId + GetBlockIdx() * queueNum_] + curMsgPosition_[queId];
    }
    do {
        HCCL_CHECK_RESTART(hcclMsgArea_, return );
        FlushDataCache(hcclSendMsg);
    } while ((debugMode_ != HCCL_ONLY_COMPUTE)
             && ((curVersion_ == 0 && hcclSendMsg->addMsg.v0Msg.valid == HCCL_MSG_VALID_MASK)
                 || (curVersion_ != 0 && hcclSendMsg->addMsg.v1Msg.valid == HCCL_MSG_VALID_MASK)));
    KERNEL_LOG(KERNEL_INFO, "Hccl send msg[%u] is available now.", curMsgPosition_[queId]);
    if (srcGroupID < 0) {
        uint64_t tiling = 0UL;
        if (para.commType < HcclCMDType::HCCL_CMD_ALL) {
            tiling = ccOpTilingDataTable_[static_cast<uint32_t>(para.commType)];
        }
        AssembleHcclMsg(para, curVersion_, curHandleId_, tiling, hcclSendMsg, &hcclMsgArea_->controlMsg);
    } else {
        AssembleHcclMsg(para, srcGroupID, srcHandleID, hcclSendMsg);
    }
    FlushDataCache(reinterpret_cast<__gm__ void*>(hcclSendMsg));
}

template <HcclServerType serverType, const auto& config>
__aicore__ inline void HcclImpl<serverType, config>::SendMsgToServer(const AlltoAllVParamExt& para)
{
    if (!workingFlag_) {
        return;
    }
    __gm__ HcclMsgExt* hcclSendMsg = &(hcclMsgArea_->paramExtMsgList[curMsgPosition_[0U]]);
    do {
        HCCL_CHECK_RESTART(hcclMsgArea_, return );
        FlushDataCache(hcclSendMsg);
    } while ((debugMode_ != HCCL_ONLY_COMPUTE) && (hcclSendMsg->valid == HCCL_MSG_VALID_MASK));
    KERNEL_LOG(KERNEL_INFO, "Hccl send extMsg[%u] is available now.", curMsgPosition_[0U]);
    para.AssembleHcclMsgExt(hcclContext_->rankNum, hcclSendMsg);
    GlobalTensor<int64_t> globalHcclMsgArea;
    for (uint32_t i = 0U; i < hcclContext_->rankNum; i += U64_CNT_PER_CACHELINE) {
        FlushDataCache(globalHcclMsgArea, (hcclSendMsg->sendCounts + i));
        FlushDataCache(globalHcclMsgArea, (hcclSendMsg->sendOffset + i));
        FlushDataCache(globalHcclMsgArea, (hcclSendMsg->recvCounts + i));
        FlushDataCache(globalHcclMsgArea, (hcclSendMsg->recvOffset + i));
    }
    FlushDataCache(globalHcclMsgArea, hcclSendMsg->reserved);
}

template <HcclServerType serverType, const auto& config>
__aicore__ inline uint16_t HcclImpl<serverType, config>::GetStepSizeByHandle(HcclHandle handle)
{
    const uint8_t commType = handleId2CmdType_[handle];
    if (commType != static_cast<uint8_t>(HcclCMDType::HCCL_CMD_ALLTOALLV)) {
        return 0U;
    }
    __gm__ uint8_t* tilingPtr = reinterpret_cast<__gm__ uint8_t*>(ccOpTilingDataTable_[commType]);
    if (tilingPtr == nullptr) {
        return 0U;
    }
    return *(tilingPtr + HCCL_STEP_SIZE_OFFSET);
}

template <HcclServerType serverType, const auto& config>
__aicore__ inline uint16_t HcclImpl<serverType, config>::GetStepCntsPerRepeatByHandle(HcclHandle handle)
{
    return (GetStepSizeByHandle(handle) == 0U ? 1U : GetRankDim());
}

template <HcclServerType serverType, const auto& config>
__aicore__ inline void HcclImpl<serverType, config>::SetCommitTurnCntToGm(uint8_t msgPos, uint64_t turnCnt)
{
    if (queueNum_ != 0U || !workingFlag_) {
        return;
    }

    __gm__ TurnCnt* commitGM = hcclMsgArea_->commitTurnCnt + msgPos;
    do {
        HCCL_CHECK_RESTART(hcclMsgArea_, return );
        FlushDataCache(commitGM);
    } while ((debugMode_ != HCCL_ONLY_COMPUTE) && (commitGM->cnt >= turnCnt));
    KERNEL_LOG(KERNEL_INFO, "Block idx[%d] write commit turn cnt[%lu].", DEFAULT_CFG.blockId, turnCnt);
    commitGM->cnt = turnCnt;
    commitGM->valid = COMMIT_VALID_MASK;
    FlushDataCache(commitGM);
}

template <HcclServerType serverType, const auto& config>
__aicore__ inline uint64_t HcclImpl<serverType, config>::WaitFinishCntFromGm(uint8_t msgPos, uint64_t expectedCnt)
{
    __gm__ TurnCnt* finishGM = hcclMsgArea_->finishedTurnCnt + msgPos;
    GlobalTensor<int64_t> globalHcclMsgArea;
    while (true) {
        HCCL_CHECK_RESTART(hcclMsgArea_, break);
        FlushDataCache(globalHcclMsgArea, finishGM);
        if ((debugMode_ == HCCL_ONLY_COMPUTE) || (finishGM->cnt >= expectedCnt)) {
            break;
        }
    }
    return finishGM->cnt;
}

template <HcclServerType serverType, const auto& config>
template <bool commit>
__aicore__ inline HcclHandle HcclImpl<serverType, config>::CommonPrepareImpl(const CommonPrepareParam& param)
{
    HCCL_CHECK_RESTART(hcclMsgArea_, return INVALID_HANDLE_ID);
    if (unlikely(param.repeat == 0U)) {
        return INVALID_HANDLE_ID;
    }
    ASCENDC_HCCL_API_ASSERT(
        CheckCommonPrepareParamValid(param), { return INVALID_HANDLE_ID; }, "Call Prepare[%d] failed, param invalid.",
        static_cast<int32_t>(param.commType));

    HcclHandle handleId = ++curHandleId_;
    ASCENDC_HCCL_API_ASSERT(
        handleId < HCCL_MAX_HANDLE_ID, { return INVALID_HANDLE_ID; },
        "Call Prepare[%d] failed, Prepare interface call num is[%d], expected no more than[%d].",
        static_cast<int32_t>(param.commType), handleId + 1, HCCL_MAX_HANDLE_ID);
    if (param.commType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        SendMsgToServer(param.paramExt);
    }
    const uint16_t queId = (queueNum_ == 0U ? 0U : static_cast<uint16_t>(param.dataType));
    SendMsgToServer(queId, param);
    handleIdMsgPosition_[handleId] = curMsgPosition_[queId];
    handleIdRepeat_[handleId] = param.repeat;
    handleId2CmdType_[handleId] = static_cast<uint8_t>(param.commType);
    if constexpr (commit) {
        handleIdCommitTurnCnt_[handleId] = param.repeat * GetStepCntsPerRepeatByHandle(handleId);
        SetCommitTurnCntToGm(curMsgPosition_[queId], handleIdCommitTurnCnt_[handleId]);
    }
    ++(curMsgPosition_[queId]);
    ASCENDC_HCCL_API_ASSERT(
        curMsgPosition_[queId] < HCCL_MSG_CNT, { return INVALID_HANDLE_ID; },
        "Message amount exceeds the maximum value when prepare.");
    return handleId;
}

template <HcclServerType serverType, const auto& config>
__aicore__ inline void HcclImpl<serverType, config>::Init(GM_ADDR context, __gm__ void* initTiling)
{
    ASCENDC_HCCL_API_ASSERT(
        context != nullptr, { return; }, "Init Hccl failed, context addr is nullptr.");
    hcclContext_ = (__gm__ HcclCombineOpParam*)context;
    // ensure hcclMsgArea 512B aligned
    uint64_t msgAddr = hcclContext_->workSpace;
    if (msgAddr & 0x1ff) {
        msgAddr = (msgAddr & (~((uint64_t)0x1ff))) + 0x200;
    }
    if (unlikely((msgAddr == 0UL) || (initTiling == nullptr && curVersion_ > 0)
                 || (initTiling != nullptr && curVersion_ == 0))) {
        KERNEL_LOG(KERNEL_ERROR, "Init Hccl failed, workspace addr is nullptr or invalid tiling.");
        curVersion_ = -1;
        return;
    }
    if (initTiling != nullptr) {
        curVersion_ = 1;
        auto initTilingPtr = reinterpret_cast<__gm__ char*>(initTiling);
        debugMode_ = *(reinterpret_cast<__gm__ uint8_t*>(initTilingPtr + HCCL_DEBUG_MODE_OFFSET));
        queueNum_ = *(reinterpret_cast<__gm__ uint16_t*>(initTilingPtr + HCCL_QUEUE_NUM_OFFSET));
    } else {
        curVersion_ = 0;
    }
    hcclMsgArea_ = (__gm__ HcclMsgArea*)msgAddr;
    for (uint32_t i = 0U; i < HCCL_MAX_HANDLE_ID; ++i) { handleIdMsgPosition_[i] = INVALID_MSG_POSITION; }
    InitWorkingFlag();
}

template <HcclServerType serverType, const auto& config>
__aicore__ inline int32_t HcclImpl<serverType, config>::Wait(HcclHandle handleId)
{
    HCCL_CHECK_RESTART(hcclMsgArea_, return HCCL_FAILED);
    ASCENDC_HCCL_API_ASSERT(
        curVersion_ >= 0, { return HCCL_FAILED; },
        "Call Wait failed, ensure Hccl::Init func has been called successfully!");
    if (queueNum_ != 0U) {
        return HCCL_SUCCESS;
    }
    if (unlikely(handleId <= INVALID_HANDLE_ID || handleId >= HCCL_MAX_HANDLE_ID)) {
        KERNEL_LOG(KERNEL_ERROR, "Failed to wait, handleId is[%d], expected to be in range of [0, %d).", handleId,
            HCCL_MAX_HANDLE_ID);
        return HCCL_FAILED;
    }
    uint16_t& waitCnt = handleIdWaitCallNum_[handleId];
    if (unlikely(waitCnt >= handleIdCommitTurnCnt_[handleId])) {
        KERNEL_LOG(KERNEL_ERROR,
            "Failed to wait, call num of Wait for handleId[%d] is[%u], expected to be no larger "
            "than Commit num[%u].",
            handleId, waitCnt + 1, handleIdCommitTurnCnt_[handleId]);
        return HCCL_FAILED;
    }
    int8_t curMsgPos = handleIdMsgPosition_[handleId];
    ASCENDC_HCCL_API_ASSERT(
        curMsgPos != INVALID_MSG_POSITION, { return HCCL_FAILED; },
        "Call Wait failed, handleId[%d] was not got by Prepare interface.", handleId);
    const uint16_t stepSize = GetStepSizeByHandle(handleId);
    waitCnt += (stepSize == 0U ? 1U : stepSize);
    (void)WaitFinishCntFromGm(curMsgPos, waitCnt);
    return HCCL_SUCCESS;
}

template <HcclServerType serverType, const auto& config>
__aicore__ inline void HcclImpl<serverType, config>::Commit(HcclHandle handleId)
{
    HCCL_CHECK_RESTART(hcclMsgArea_, return );
    ASCENDC_HCCL_API_ASSERT(
        curVersion_ >= 0, { return; }, "Call Commit failed, ensure Hccl::Init func has been called successfully!");
    if (unlikely(handleId <= INVALID_HANDLE_ID || handleId >= HCCL_MAX_HANDLE_ID)) {
        KERNEL_LOG(KERNEL_ERROR, "Call Commit failed, handleId is[%d], expected in range of [0, %d).", handleId,
            HCCL_MAX_HANDLE_ID);
        return;
    }
    uint16_t& commitCnt = handleIdCommitTurnCnt_[handleId];
    if (unlikely(commitCnt >= handleIdRepeat_[handleId] * GetStepCntsPerRepeatByHandle(handleId))) {
        KERNEL_LOG(KERNEL_ERROR,
            "Call Commit for handleId[%d] failed, call num is[%u], "
            "expected no larger than task num[%u].",
            handleId, commitCnt + 1, handleIdRepeat_[handleId] * GetStepCntsPerRepeatByHandle(handleId));
        return;
    }
    const uint16_t stepSize = GetStepSizeByHandle(handleId);
    commitCnt += (stepSize == 0U ? 1U : stepSize);
    SetCommitTurnCntToGm(handleIdMsgPosition_[handleId], commitCnt);
}

template <HcclServerType serverType, const auto& config>
template <ScopeType type>
__aicore__ inline void HcclImpl<serverType, config>::QueueBarrier(uint16_t queueID)
{
    SendMsgToServer(queueID, {HcclCMDType::HCCL_CMD_BARRIER});
    ++(curMsgPosition_[queueID]);
    ASCENDC_HCCL_API_ASSERT(
        curMsgPosition_[queueID] < HCCL_MSG_CNT, { return; }, "Message amount exceeds the maximum value when barrier.");
}

template <HcclServerType serverType, const auto& config>
template <bool sync>
__aicore__ inline int32_t HcclImpl<serverType, config>::Iterate(
    HcclHandle handleId, uint16_t* seqSlices, uint16_t seqSliceLen)
{
    ASCENDC_HCCL_API_ASSERT(
        curVersion_ == 1, { return HCCL_FAILED; }, "Initialization has not been done properly.");
    ASCENDC_HCCL_API_ASSERT(
        seqSlices != nullptr && seqSliceLen != 0U, { return HCCL_FAILED; }, "Invalid param for Iterate.");
    const uint16_t stepSize = GetStepSizeByHandle(handleId);
    const uint16_t stepsPerRepeat = GetStepCntsPerRepeatByHandle(handleId);
    ASCENDC_HCCL_API_ASSERT(
        stepSize > 0U && stepsPerRepeat > 1U, { return HCCL_FAILED; }, "Handle id %d is not for fine-grained.",
        handleId);
    uint16_t& curSlice = handleId2CurrSliceId_[handleId];
    KERNEL_LOG(KERNEL_INFO, "The step size for handle %d is %u, current slice and total slices are %u/%u.", handleId,
        stepSize, curSlice, stepsPerRepeat);

    // Only for All2AllV + pairwise
    if (curSlice >= stepsPerRepeat * handleIdRepeat_[handleId]) {
        KERNEL_LOG(KERNEL_INFO, "The step id %u for handle id %d reach the maximum.", handleId, curSlice);
        return 0;
    }
    const uint16_t slicesPerRepeat = stepsPerRepeat;
    const uint32_t rankId = GetRankId();
    const uint32_t rankDim = GetRankDim();
    ASCENDC_HCCL_API_ASSERT(
        rankDim != 0U, { return HCCL_FAILED; }, "Invalid rank-dim.");
    for (uint16_t i = 0U; i < seqSliceLen; ++i) {
        if constexpr (sync) {
            if ((curSlice + 1) % stepSize == 0) {
                (void)Wait(handleId);
            }
            seqSlices[i] = (rankId + rankDim - curSlice % slicesPerRepeat) % rankDim;
        } else {
            seqSlices[i] = (rankId + curSlice % slicesPerRepeat) % rankDim;
        }
        ++curSlice;
    }
    return seqSliceLen;
}

template <HcclServerType serverType, const auto& config>
template <bool sync>
__aicore__ inline void HcclImpl<serverType, config>::Finalize()
{
    ASCENDC_HCCL_API_ASSERT(
        curVersion_ >= 0, { return; }, "Call Finalize failed, ensure Hccl::Init func has been called successfully!");
    HCCL_CHECK_RESTART(hcclMsgArea_, return );

    if (!workingFlag_ && queueNum_ == 0U) {
        ++(curMsgPosition_[0U]);
        ASCENDC_HCCL_API_ASSERT(
            curMsgPosition_[0U] < HCCL_MSG_CNT, { return; }, "Message amount exceeds the maximum value when finalize.");
        return;
    }

    // 1. wait until last hccl task finished(the commitTurnCnt will be reset by aicpu-server before task finished),
    //    then commitTurnCnt can be used by next op.
    if constexpr (sync) {
        if (curHandleId_ > INVALID_HANDLE_ID) {
            KERNEL_LOG(KERNEL_INFO, "Wait hccl task finished for last HandleId[%d] when Finalize.", curHandleId_);
            while ((debugMode_ != HCCL_ONLY_COMPUTE) && (Query(curHandleId_) < handleIdRepeat_[curHandleId_])) {
                HCCL_CHECK_RESTART(hcclMsgArea_, return );
            }
        }
    }

    // 2. send Finalize msg
    SendFinalizeMsg<sync>();

    if constexpr (sync) {
        // 3. wait for server sqe task finished, and client can ResetFinishedTurnCnt
        // 4. reset finishedTurnCnt, then the finishedTurnCnt can be used by next op.
        __gm__ TurnCnt* finishGM = hcclMsgArea_->finishedTurnCnt + curMsgPosition_[0U];
        KERNEL_LOG(KERNEL_INFO, "Only block idx[%ld] wait until Finalize msg has been read.", GetBlockIdx());
        do {
            HCCL_CHECK_RESTART(hcclMsgArea_, return );
            FlushDataCache(finishGM);
        } while ((debugMode_ != HCCL_ONLY_COMPUTE) && (finishGM->cnt != FINALIZE_FINISH_CNT));
        KERNEL_LOG(KERNEL_INFO, "Only block idx[%ld] will ResetFinishedTurnCnt.", GetBlockIdx());
        ResetFinishedTurnCnt();
        ++(curMsgPosition_[0U]);
        ASCENDC_HCCL_API_ASSERT(
            curMsgPosition_[0U] < HCCL_MSG_CNT, { return; }, "Message amount exceeds the maximum value when finalize.");
    }
}

template <HcclServerType serverType, const auto& config>
__aicore__ inline void HcclImpl<serverType, config>::ResetFinishedTurnCnt()
{
    __gm__ TurnCnt* finishArea = hcclMsgArea_->finishedTurnCnt;
    GlobalTensor<int64_t> globalHcclMsgArea;
    for (uint32_t i = 0U; i <= curMsgPosition_[0U]; ++i) {
        __gm__ TurnCnt* finishGM = finishArea + i;
        finishGM->cnt = 0;
        FlushDataCache(globalHcclMsgArea, finishGM);
    }
}

template <HcclServerType serverType, const auto& config>
template <bool sync>
__aicore__ inline void HcclImpl<serverType, config>::SendFinalizeMsg()
{
    const uint16_t totalQueNum = (queueNum_ == 0U ? 1U : queueNum_);
    for (uint16_t idx = 0U; idx < totalQueNum; ++idx) {
        KERNEL_LOG(KERNEL_INFO, "Only block idx[%ld] write sendMsgList[%u] when Finalize.", GetBlockIdx(),
            curMsgPosition_[idx]);
        SendMsgToServer(idx, {HcclCMDType::HCCL_CMD_FINALIZE});
        if constexpr (!sync) {
            ++(curMsgPosition_[idx]);
            ASCENDC_HCCL_API_ASSERT(
                curMsgPosition_[idx] < HCCL_MSG_CNT, { return; },
                "Message amount exceeds the maximum value when finalize.");
        }
    }
}
} // namespace AscendC
#endif // __CCE_AICORE__ == 220