/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hccl_ccu_d100_def.h
 * \brief
 */
#ifndef IMPL_V310_HCCL_CCU_D100_DEF_H
#define IMPL_V310_HCCL_CCU_D100_DEF_H

namespace AscendC {

template<const auto &config>
class HcclImpl<HcclServerType::HCCL_SERVER_TYPE_CCU, config> {
public:
    template <bool commit = false>
    __aicore__ inline HcclHandle AllReduce(GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t count,
                                           HcclDataType dataType, HcclReduceOp op, uint8_t repeat = 1);

    template <bool commit = false>
    __aicore__ inline HcclHandle AllGather(GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t sendCount,
                                           HcclDataType dataType, uint64_t strideCount, uint8_t repeat = 1);

    template <bool commit = false>
    __aicore__ inline HcclHandle ReduceScatter(GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t recvCount,
                                               HcclDataType dataType, HcclReduceOp op, uint64_t strideCount,
                                               uint8_t repeat = 1);

    template <bool commit = false>
    __aicore__ inline HcclHandle AlltoAll(GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t dataCount,
                                          HcclDataType dataType, uint64_t strideCount = 0, uint8_t repeat = 1);

    template <bool commit = false>
    __aicore__ inline HcclHandle AlltoAllV(GM_ADDR sendBuf, void *sendCounts, void *sdispls, HcclDataType sendType,
                                           GM_ADDR recvBuf, void *recvCounts, void *rdispls, HcclDataType recvType,
                                           uint8_t repeat = 1);

    template <bool commit = false>
    __aicore__ inline HcclHandle AlltoAllvWrite(GM_ADDR usrIn, GM_ADDR sendOffsets, GM_ADDR sendSizes,
                                                uint64_t remoteWinOffset, uint64_t localDataSize);

    __aicore__ inline void Init(GM_ADDR context, __gm__ void *initTiling = nullptr);

    __aicore__ inline void InitV2(GM_ADDR context, const void *initTiling);

    __aicore__ inline int32_t SetCcTiling(__gm__ void *ccOpTilingData);

    __aicore__ inline int32_t SetCcTilingV2(uint64_t offset);

    __aicore__ inline void Commit(HcclHandle handleId);

    __aicore__ inline int32_t Wait(HcclHandle handleId);

    template <bool sync = true>
    __aicore__ inline void Finalize();

    __aicore__ inline uint32_t GetRankId() { return hcclContext_->rankId; }

    __aicore__ inline uint32_t GetRankDim() { return hcclContext_->rankNum; }

    __aicore__ inline bool
    SetReduceDataTypeAbility(HcclReduceOp op, HcclDataType dstDataType, HcclDataType srcDataType);

private:
    __aicore__ inline void InitWorkingFlag();

    template <bool commit = false>
    __aicore__ inline HcclHandle CommonPrepareImpl(const CommonPrepareParam &param);

    __aicore__ inline void FlushCache(GM_ADDR addrGM);

    __aicore__ inline void AssembleHcclFinalizeMsgForCCU(__gm__ HcclMsg *hcclSendMsg);

    __aicore__ inline void
    AssembleHcclSendMsgForCCU(const CommonPrepareParam &commonPrepareParam, __gm__ HcclMsg *hcclSendMsg);

    template<bool commit = false>
    __aicore__ inline void CCUPrepareInner(const CommonPrepareParam &commonPrepareParam, const HcclHandle handleId);

    __aicore__ inline void
    CCUPrepare(const CommonPrepareParam &commonPrepareParam, uint8_t reqId, uint64_t sendbuf, uint64_t recvBuf);

    __aicore__ inline uint64_t GetOpId(const CommonPrepareParam &commonPrepareParam);

    __aicore__ inline void DataCopy2XnAddr(GM_ADDR dstGmAddr, GM_ADDR srcGmAddr, uint32_t size);

    __aicore__ inline void CcuSendMsg(uint8_t reqId);

    __aicore__ inline GM_ADDR GetCommitCkeAddr(uint8_t msgId);

    __aicore__ inline GM_ADDR GetWaitCkeAddr(uint8_t msgId);

    __aicore__ inline bool IsFinish(uint8_t reqId);

    __aicore__ inline bool CheckNeedPush();

    __aicore__ inline uint64_t GetParallelParam(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum);

    __aicore__ inline void FlushDataCacheForCopy(GM_ADDR gmAddr, uint32_t size);

    __aicore__ inline void AssembleHcclMsgExtForCCU(const CommonPrepareParam &commonPrepareParam, uint32_t repeatIndex);

    __aicore__ inline void InitCommReq(uint8_t reqId);

    __aicore__ inline void InitHandleInfo(uint8_t handleId);

    __aicore__ inline void CommitMsg(HcclHandle handleId, uint8_t reqId);

    __aicore__ inline void InitInner(GM_ADDR context, HcclTilingVersion version);

private:
    __gm__ HcclCombineOpParam *hcclContext_;
    HcclHandle curHandleId_ = INVALID_HANDLE_ID;

    uint8_t workingFlag_ = false;
    bool needResetDataType_ = false;
    uint8_t commitCnt_;
    uint8_t alltoallvCnt_ = 0;
    bool isInited_ = false;
    __gm__ uint32_t *finishCntGM_;
    __gm__ CCUMsgExt *ccuMsgExt_;

    CCUConfig ccuConfig_;
    CCUMsg ccuMsg_;
    ReduceDataTypeAbility ccuDataType_;

    CCUMsgCommOp commReqBuf_[HCCL_MAX_HANDLE_ID];
    HandleCommOp handleInfo_[HCCL_MAX_HANDLE_ID];
    CircularFifo<uint8_t, HCCL_MAX_HANDLE_ID> committedReqFifo_;
    CircularFifo<uint8_t, CCU_MAX_MSG_NUM> ccuMsgFifo_;

    uint64_t ccOpTilingDataTable_[static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALL)] = {0UL};
    HcclTilingVersion curVersion_ = HcclTilingVersion::INVALID_TILING_VERSION;
    uint64_t tilingBaseAddr_;
};
} // namespace AscendC

#endif