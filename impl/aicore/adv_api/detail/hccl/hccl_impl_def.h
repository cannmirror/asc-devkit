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
 * \file hccl_impl_def.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_HCCL_HCCL_IMPL_DEF_H
#define AICORE_ADV_API_DETAIL_HCCL_HCCL_IMPL_DEF_H
#include "hccl_msg.h"

namespace AscendC {
template <HcclServerType serverType, const auto& config>
class HcclImpl {
public:
    template <bool commit = false>
    __aicore__ inline HcclHandle AllReduce(
        GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, uint8_t repeat = 1);

    template <bool commit = false>
    __aicore__ inline HcclHandle AllGather(GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t sendCount, HcclDataType dataType,
        uint64_t strideCount, uint8_t repeat = 1);

    template <bool commit = false>
    __aicore__ inline HcclHandle ReduceScatter(GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t recvCount,
        HcclDataType dataType, HcclReduceOp op, uint64_t strideCount, uint8_t repeat = 1);

    template <bool commit = false>
    __aicore__ inline HcclHandle AlltoAll(GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t dataCount, HcclDataType dataType,
        uint64_t strideCount = 0, uint8_t repeat = 1);

    template <bool commit = false>
    __aicore__ inline HcclHandle AlltoAllV(GM_ADDR sendBuf, void* sendCounts, void* sdispls, HcclDataType sendType,
        GM_ADDR recvBuf, void* recvCounts, void* rdispls, HcclDataType recvType, uint8_t repeat = 1);

    template <bool commit = false>
    __aicore__ inline HcclHandle BatchWrite(GM_ADDR batchWriteInfo, uint32_t itemNum, uint16_t queueID);

    template <bool commit = false>
    __aicore__ inline HcclHandle AlltoAllvWrite(
        GM_ADDR usrIn, GM_ADDR sendOffsets, GM_ADDR sendSizes, uint64_t remoteWinOffset, uint64_t localDataSize);

public:
    __aicore__ inline void Init(GM_ADDR context, __gm__ void* initTiling = nullptr);

    __aicore__ inline int32_t SetCcTiling(__gm__ void* ccOpTilingData);

    __aicore__ inline void Commit(HcclHandle handleId);

    __aicore__ inline int32_t Wait(HcclHandle handleId);

    __aicore__ inline int32_t Query(HcclHandle handleId);

    __aicore__ inline void InterHcclGroupSync(int8_t srcGroupID, HcclHandle srcHandleID);

    template <ScopeType type = ScopeType::ALL>
    __aicore__ inline void QueueBarrier(uint16_t queueID);

    template <bool sync = true>
    __aicore__ inline int32_t Iterate(HcclHandle handleId, uint16_t* seqSlices, uint16_t seqSliceLen);

    template <bool sync = true>
    __aicore__ inline void Finalize();

public:
    __aicore__ inline GM_ADDR GetWindowsInAddr(uint32_t rankId);

    __aicore__ inline GM_ADDR GetWindowsOutAddr(uint32_t rankId);

    __aicore__ inline uint32_t GetRankId()
    {
        return hcclContext_->rankId;
    }

    __aicore__ inline uint32_t GetRankDim()
    {
        return hcclContext_->rankNum;
    }

    __aicore__ inline uint16_t GetQueueNum()
    {
        return queueNum_;
    }

private:
    // Generic implementation for corresponding interface of each Prepare primitive. Return identifier(handleId) of
    // corresponding comm task. HandleId >= 0 when successful, otherwise return -1.
    template <bool commit = false>
    __aicore__ inline HcclHandle CommonPrepareImpl(const CommonPrepareParam& param);

    __aicore__ inline bool CheckCommonPrepareParamValid(const CommonPrepareParam& param);

    // Clear the finishedTurnCnt before aicore exits to ensure the correctness of next launch.
    __aicore__ inline void ResetFinishedTurnCnt();

    template <bool sync>
    __aicore__ inline void SendFinalizeMsg();

    __aicore__ inline void SendMsgToServer(uint16_t queId, const CommonPrepareParam& para, int8_t srcGroupID = -1,
        HcclHandle srcHandleID = INVALID_HANDLE_ID);

    __aicore__ inline void SendMsgToServer(const AlltoAllVParamExt& para);

    __aicore__ inline uint16_t GetStepSizeByHandle(HcclHandle handle);

    __aicore__ inline uint16_t GetStepCntsPerRepeatByHandle(HcclHandle handle);

    __aicore__ inline void SetCommitTurnCntToGm(uint8_t msgPos, uint64_t turnCnt);

    __aicore__ inline uint64_t WaitFinishCntFromGm(uint8_t msgPos, uint64_t expectedCnt);

    __aicore__ inline void InitWorkingFlag();

private:
    uint64_t ccOpTilingDataTable_[static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALL)] = {0UL};
    __gm__ HcclCombineOpParam* hcclContext_;
    __gm__ HcclMsgArea* hcclMsgArea_;
    uint16_t queueNum_ = 0U;
    uint16_t handleId2CurrSliceId_[HCCL_MAX_HANDLE_ID] = {0U};
    uint16_t handleIdCommitTurnCnt_[HCCL_MAX_HANDLE_ID] = {0U};
    uint16_t handleIdWaitCallNum_[HCCL_MAX_HANDLE_ID] = {0U};
    uint8_t handleId2CmdType_[HCCL_MAX_HANDLE_ID] = {0U};
    int8_t handleIdMsgPosition_[HCCL_MAX_HANDLE_ID];
    uint8_t handleIdRepeat_[HCCL_MAX_HANDLE_ID] = {0U};
    uint8_t curMsgPosition_[MAX_QUE_NUM] = {0U};
    HcclHandle curHandleId_ = INVALID_HANDLE_ID;
    // Current msg position where Api write, starts from 0 and increases automatically, with a maximum of
    // HCCL_MSG_CNT-1. When HCCL_MSG_CNT is reached, take the remainder and recycling message area.
    // Prepare/BatchPrepare/Finalize/InterHcclGroupSync (supported in future versions) only use one message.
    int8_t curVersion_ = -1;
    uint8_t workingFlag_ = false;
    uint8_t debugMode_ = 0U;

#if defined(__DAV_C310__) || defined(__DAV_310R6__)
    CCUConfig ccuConfig_;
    CCUMsg ccuMsg_;
    ReduceDataTypeAbility ccuDataType_;
    bool needResetDataType_ = false;
    uint8_t commitCnt_;
    uint8_t alltoallvCnt_ = 0;
    bool isInited_ = false;
    __gm__ uint32_t* finishCntGM_;
    __aicore__ inline void FlushCache(GM_ADDR addrGM);
    __aicore__ inline void AssembleHcclFinalizeMsgForCCU(__gm__ HcclMsg* hcclSendMsg);
    __aicore__ inline void AssembleHcclSendMsgForCCU(
        const CommonPrepareParam& commonPrepareParam, __gm__ HcclMsg* hcclSendMsg);
    template <bool commit = false>
    __aicore__ inline void CCUPrepareInner(const CommonPrepareParam& commonPrepareParam, const HcclHandle handleId);
    __aicore__ inline void CCUPrepare(
        const CommonPrepareParam& commonPrepareParam, uint8_t reqId, uint64_t sendbuf, uint64_t recvBuf);
    __aicore__ inline uint64_t GetOpId(const CommonPrepareParam& commonPrepareParam);
    __aicore__ inline void TryReleaseMsgResource(HcclHandle handleId);
    __aicore__ inline void DataCopy2XnAddr(GM_ADDR dstGmAddr, GM_ADDR srcGmAddr, uint32_t size);
    __aicore__ inline void CcuSendMsg(uint8_t reqId);
    __aicore__ inline bool CheckWaitCKE(uint8_t resourceIndex, HcclHandle handleId);
    __aicore__ inline void TryReleaseMsg();
    __aicore__ inline void FlushCommitMsg(HcclHandle handleId = -1);
    __aicore__ inline GM_ADDR GetCommitCkeAddr(uint8_t msgId);
    __aicore__ inline GM_ADDR GetWaitCkeAddr(uint8_t msgId);
    __aicore__ inline bool IsFinish(uint8_t reqId);
    __aicore__ inline bool CheckNeedPush();
    __aicore__ inline uint64_t GetParallelParam(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum);
    __aicore__ inline void FlushDataCacheForCopy(GM_ADDR gmAddr, uint32_t size);
    __aicore__ inline void AssembleHcclMsgExtForCCU(const CommonPrepareParam& commonPrepareParam, uint32_t repeatIndex);

    CCUCommOp commReqBuf_[HCCL_MAX_HANDLE_ID];
    CircularFifo<uint8_t, HCCL_MAX_HANDLE_ID> committedReqFifo_;
    CircularFifo<uint8_t, CCU_MAX_MSG_NUM> ccuMsgFifo_;
    __gm__ CCUMsgExt* ccuMsgExt_;

public:
    __aicore__ inline bool SetReduceDataTypeAbility(
        HcclReduceOp op, HcclDataType dstDataType, HcclDataType srcDataType);
#endif
};
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_HCCL_HCCL_IMPL_DEF_H
