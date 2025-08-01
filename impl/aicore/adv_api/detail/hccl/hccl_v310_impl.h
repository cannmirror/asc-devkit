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
 * \file hccl_v310_impl.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_HCCL_HCCL_V310_IMPL_H
#define AICORE_ADV_API_DETAIL_HCCL_HCCL_V310_IMPL_H

#include "hccl_common.h"

namespace AscendC {
template <HcclServerType serverType, const auto& config>
__aicore__ inline void HcclImpl<serverType, config>::FlushDataCacheForCopy(GM_ADDR gmAddr, uint32_t size)
{
    uint32_t offset = 0;
    for (uint32_t i = 0; i < size; i += MAX_DCCI_CNT) {
        FlushDataCache((GM_ADDR)(gmAddr + i));
        offset += MAX_DCCI_CNT;
    }
    FlushDataCache((GM_ADDR)(gmAddr + offset));
}

template <HcclServerType serverType, const auto& config>
__aicore__ inline void HcclImpl<serverType, config>::DataCopy2XnAddr(
    GM_ADDR dstGmAddr, GM_ADDR srcGmAddr, uint32_t size)
{
    for (int i = 0; i < CCU_USED_XN_NUM; i++) {
        *(reinterpret_cast<__gm__ uint64_t*>(dstGmAddr + CCU_XN_DATA_SIZE * i)) =
            *(reinterpret_cast<__gm__ uint64_t*>(srcGmAddr + CCU_XN_DATA_SIZE * i));
    }
    FlushDataCacheForCopy(ccuMsg_.XnAddr, size);
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
    ccuMsgExt_ = reinterpret_cast<__gm__ CCUMsgExt*>(msgAddr);

    ccuConfig_.XnAddr = hcclContext_->XnOffset;
    ccuConfig_.CKEAddr = hcclContext_->CKEOffset;
#ifndef __CCE_KT_TEST__
    // 检查是否是64字节对齐
    ASCENDC_ASSERT((reinterpret_cast<uintptr_t>(ccuConfig_.XnAddr) % ALIGN_64_BYTE == 0),
        { KERNEL_LOG(KERNEL_ERROR, "XnAddr is not 64-byte aligned!"); });
#endif
    ccuMsg_.XnData = reinterpret_cast<__gm__ uint8_t*>(hcclContext_->workSpace + CCU_MSG_EXT_MAX_OFFSET);
    finishCntGM_ = reinterpret_cast<__gm__ uint32_t*>(
        hcclContext_->workSpace + CCU_MSG_EXT_MAX_OFFSET + (CCU_MSG_XN_NUM * CCU_MAX_MSG_NUM));
    commitCnt_ = 0;
    if (ccuConfig_.XnAddr == nullptr || ccuConfig_.CKEAddr == nullptr || ccuMsg_.XnData == nullptr) {
        KERNEL_LOG(KERNEL_ERROR, "Init Hccl failed,"
                                 "ccuConfig_.XnAddr or ccuConfig_.CKEAddr or ccuMsg_.XnData is nullptr");
        return;
    }

    for (uint32_t i = 0U; i < HCCL_MAX_HANDLE_ID; ++i) {
        commReqBuf_[i].resourceId = -1;
        commReqBuf_[i].isFinish = 0;
        commReqBuf_[i].finishedCnt = 0;
        commReqBuf_[i].repeatCnt = 1;
        commReqBuf_[i].commitCnt = 0;
        commReqBuf_[i].waitCnt = 0;
    }

    InitWorkingFlag();
    curVersion_ = (initTiling != nullptr ? 1 : 0);
    curHandleId_ = 0;
    isInited_ = true;
}

template <HcclServerType serverType, const auto& config>
__aicore__ inline uint64_t HcclImpl<serverType, config>::GetOpId(const CommonPrepareParam& commonPrepareParam)
{
    if (commonPrepareParam.commType == HcclCMDType::HCCL_CMD_FINALIZE) {
        return 0xffffffffffffffff;
    }

    bool needReset = false;
    if (needResetDataType_ && commonPrepareParam.commType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
        needReset = true;
    }
    uint64_t algoType = 0U;
    uint64_t outDataType = needReset ? static_cast<uint64_t>(ccuDataType_.dstDataType) :
                                       static_cast<uint64_t>(commonPrepareParam.dataType);
    uint64_t reduceType = 0U;
    uint64_t dataType = needReset ? static_cast<uint64_t>(ccuDataType_.srcDataType) :
                                    static_cast<uint64_t>(commonPrepareParam.dataType);
    uint64_t commType = static_cast<uint64_t>(commonPrepareParam.commType);
    bool isReduceType = (commonPrepareParam.commType == HcclCMDType::HCCL_CMD_REDUCE)
                        || (commonPrepareParam.commType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER)
                        || (commonPrepareParam.commType == HcclCMDType::HCCL_CMD_ALLREDUCE);
    if (isReduceType) {
        reduceType = needReset ? static_cast<uint64_t>(ccuDataType_.op) : static_cast<uint64_t>(commonPrepareParam.op);
    }
    return (((algoType & 0x7f) << 32) | ((outDataType & 0x7f) << 24) | ((reduceType & 0x7f) << 16)
            | ((dataType & 0x7f) << 8) | (commType & 0x7f));
}

template <HcclServerType serverType, const auto& config>
__aicore__ inline uint64_t HcclImpl<serverType, config>::GetParallelParam(
    uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
{
    return ((repeatNum & 0x7f) << 55) | ((repeatLoopIndex & 0x7f) << 48) | ((totalLoopNum & 0x7f) << 41);
}

template <HcclServerType serverType, const auto& config>
__aicore__ inline void HcclImpl<serverType, config>::CCUPrepare(
    const CommonPrepareParam& commonPrepareParam, uint8_t reqId, uint64_t sendbuf, uint64_t recvBuf)
{
    auto& commOp = commReqBuf_[reqId];
    commOp.xnData[0] = GetOpId(commonPrepareParam);
    commOp.xnData[1] = sendbuf;
    commOp.xnData[2] = recvBuf;
    if (commonPrepareParam.commType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        commOp.xnData[3] = 0;
        commOp.xnData[4] = 0;
        // 按照卡分组，sendSize 、sendOffset、recvSize、recvOffset  以字节为单位 * DataSzie(DataType)
        commOp.xnData[5] = reinterpret_cast<uint64_t>(ccuMsgExt_) + CCU_MSG_EXT_RANK_OFFSET * alltoallvCnt_++;
        return;
    }

    uint64_t sliceCount = 0;
    uint64_t tmpCount = commonPrepareParam.count / hcclContext_->rankNum;
    uint64_t loopCount = CCU_LOOP_COUNT;

    if (commonPrepareParam.commType == HcclCMDType::HCCL_CMD_ALLGATHER
        || commonPrepareParam.commType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
        sliceCount = commonPrepareParam.count;
    } else if (commonPrepareParam.commType == HcclCMDType::HCCL_CMD_ALLREDUCE) {
        sliceCount = (hcclContext_->rankId == hcclContext_->rankNum - 1) ?
                         (commonPrepareParam.count - (hcclContext_->rankNum - 1) * tmpCount) :
                         tmpCount;
    }

    uint64_t sliceSize = sliceCount * DATA_TYPE_MAP[commonPrepareParam.dataType];

    if (commonPrepareParam.commType == HcclCMDType::HCCL_CMD_HALF_ALLTOALLV) {
        commOp.xnData[1] = reinterpret_cast<uint64_t>(commonPrepareParam.sendBuf);
        commOp.xnData[2] = commonPrepareParam.wParamExt.sendSizes;
        commOp.xnData[8] = commonPrepareParam.wParamExt.remoteWinOffset;
        sliceSize = commonPrepareParam.count;
        loopCount = CCU_LOOP_COUNT_ATAVW;
    }

    uint64_t loopSize = loopCount * CCU_MEMSLICE_SIZE;
    uint64_t m = sliceSize / loopSize;
    uint64_t n = (sliceSize - m * loopSize) / CCU_MEMSLICE_SIZE;
    uint64_t p = sliceSize - m * loopSize - n * CCU_MEMSLICE_SIZE;

    auto dataSize = DATA_TYPE_MAP[static_cast<uint64_t>(commonPrepareParam.dataType)];

    if (commonPrepareParam.commType == HcclCMDType::HCCL_CMD_ALLREDUCE) {
        commOp.xnData[3] = (commonPrepareParam.strideCount == 0) ?
                               tmpCount * dataSize * hcclContext_->rankId :
                               (commonPrepareParam.strideCount * dataSize * hcclContext_->rankId);
    } else if (commonPrepareParam.commType == HcclCMDType::HCCL_CMD_HALF_ALLTOALLV) {
        commOp.xnData[3] = commonPrepareParam.wParamExt.sendOffsets;
    } else {
        commOp.xnData[3] = (commonPrepareParam.strideCount == 0) ?
                               sliceSize * hcclContext_->rankId :
                               (commonPrepareParam.strideCount * dataSize * hcclContext_->rankId);
    }
    commOp.xnData[4] = loopSize * m;
    commOp.xnData[5] = m;

    if (n == 0 && p == 0) {
        // 数据量为loopSize的整数倍，跳过LoopGroup1
        commOp.xnData[6] = 0;
        commOp.xnData[7] = 0;
    } else if (n != 0 && p == 0) {
        // 数据量为256K * m + CCU_MEMSLICE_SIZE * n
        commOp.xnData[6] = GetParallelParam(n - 1, 0, 1);
        commOp.xnData[7] = CCU_MEMSLICE_SIZE;
    } else if (n == 0 && p != 0) {
        // 数据量为loopSize * m + p
        commOp.xnData[6] = GetParallelParam(0, 0, 1);
        commOp.xnData[7] = p;
    } else {
        // 数据量为loopSize * m + CCU_MEMSLICE_SIZE * n + p
        commOp.xnData[6] = GetParallelParam(n - 1, 1, 2);
        commOp.xnData[7] = p;
    }
}

template <HcclServerType serverType, const auto& config>
template <bool commit>
__aicore__ inline void HcclImpl<serverType, config>::CCUPrepareInner(
    const CommonPrepareParam& commonPrepareParam, const HcclHandle handleId)
{
    uint64_t sendbuf = (uint64_t)commonPrepareParam.sendBuf;
    uint64_t recvbuf = (uint64_t)commonPrepareParam.recvBuf;
    uint8_t reqId = handleId;
    for (uint32_t i = 0U; i < commReqBuf_[handleId].repeatCnt; ++i) {
        KERNEL_LOG(KERNEL_INFO, "do ccu prepare repeatIdx = %d, repeatCnt = %d, handleId = %d.", i,
            commReqBuf_[handleId].repeatCnt, handleId);
        // recv send buf + offset
        uint64_t offset = commonPrepareParam.count * i * DATA_TYPE_MAP[commonPrepareParam.dataType];
        if (workingFlag_ && commonPrepareParam.commType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
            KERNEL_LOG(KERNEL_INFO, "Only block idx[%ld] assemble msg ext when prepare alltoallv.", GetBlockIdx());
            AssembleHcclMsgExtForCCU(commonPrepareParam, i);
        }
        // 存msgOp_.xnData
        CCUPrepare(commonPrepareParam, reqId, sendbuf + offset, recvbuf + offset);
        if (commit) {
            KERNEL_LOG(KERNEL_INFO, "commit flag is true. repeatIdx = %d, repeatCnt = %d, handleId = %d.", i,
                commReqBuf_[handleId].repeatCnt, handleId);
            Commit(handleId);
        }
        reqId++;
    }
}

template <HcclServerType serverType, const auto& config>
__aicore__ inline void HcclImpl<serverType, config>::AssembleHcclMsgExtForCCU(
    const CommonPrepareParam& commonPrepareParam, uint32_t repeatIndex)
{
    auto dataSize = DATA_TYPE_MAP[static_cast<uint64_t>(commonPrepareParam.dataType)];

    __gm__ CCUMsgExt* ccuMsgExt = reinterpret_cast<__gm__ CCUMsgExt*>(
        reinterpret_cast<uint64_t>(ccuMsgExt_) + CCU_MSG_EXT_RANK_OFFSET * alltoallvCnt_);

    const uint32_t kRankNum = hcclContext_->rankNum;
    for (uint32_t i = 0U; i < kRankNum; ++i) {
        ccuMsgExt[i].sendSize = commonPrepareParam.paramExt.sendCounts[i] * dataSize;
        ccuMsgExt[i].recvSize = commonPrepareParam.paramExt.recvCounts[i] * dataSize;

        ccuMsgExt[i].sendOffset =
            commonPrepareParam.paramExt.sdispls[i] * dataSize + ccuMsgExt[i].sendSize * repeatIndex;
        ccuMsgExt[i].recvOffset =
            commonPrepareParam.paramExt.rdispls[i] * dataSize + ccuMsgExt[i].recvSize * repeatIndex;
    }

    uint32_t tmpCnt = (sizeof(CCUMsgExt) * kRankNum) / HCCL_DATACOPY_MAX_CNT;
    uint32_t copyCnt = (sizeof(CCUMsgExt) * kRankNum) % HCCL_DATACOPY_MAX_CNT ? tmpCnt + 1 : tmpCnt;
    GlobalTensor<int64_t> globalHcclMsgArea;

    uint64_t tmpSize = 0;
    for (uint32_t i = 0U; i < copyCnt; ++i) {
        FlushDataCache(globalHcclMsgArea, (ccuMsgExt + tmpSize));
        tmpSize += HCCL_DATACOPY_MAX_CNT;
    }
}

template <HcclServerType serverType, const auto& config>
template <bool commit>
__aicore__ inline HcclHandle HcclImpl<serverType, config>::CommonPrepareImpl(
    const CommonPrepareParam& commonPrepareParam)
{
    if (!CheckCommonPrepareParamValid(commonPrepareParam)) {
        return INVALID_HANDLE_ID;
    }

    HcclHandle handleId = curHandleId_;
    if (handleId >= HCCL_MAX_HANDLE_ID) {
        KERNEL_LOG(KERNEL_ERROR,
            "Call Prepare[%d] failed, Prepare interface call num is[%d], "
            "expected less than[%d].",
            static_cast<int32_t>(commonPrepareParam.commType), handleId + 1, HCCL_MAX_HANDLE_ID);
        return INVALID_HANDLE_ID;
    }

    commReqBuf_[handleId].repeatCnt = commonPrepareParam.repeat;
    CCUPrepareInner<commit>(commonPrepareParam, handleId);

    curHandleId_++;
    return handleId;
}

template <HcclServerType serverType, const auto& config>
__aicore__ inline void HcclImpl<serverType, config>::CcuSendMsg(uint8_t reqId)
{
    int8_t resourceId = commReqBuf_[reqId].resourceId;
    if (resourceId >= 8 || resourceId < 0) {
        KERNEL_LOG(KERNEL_ERROR, "CcuSendMsg resourceId %d is invalid.", resourceId);
        return;
    }
    ccuMsg_.XnAddr = ccuConfig_.XnAddr + resourceId * CCU_MSG_XN_NUM * CCU_MAX_MSG_NUM;
    ccuMsg_.commitCKEAddr = GetCommitCkeAddr(resourceId);
    ccuMsg_.waitCKEAddr = GetWaitCkeAddr(resourceId);
    auto ptr = ccuMsg_.XnData;
    for (int i = 0; i < CCU_USED_XN_NUM; i++) {
        *reinterpret_cast<__gm__ uint64_t*>(ptr) = commReqBuf_[reqId].xnData[i];
        ptr += 8; // 8 is sizeof(uint64)
    }
    DataCopy2XnAddr(ccuMsg_.XnAddr, ccuMsg_.XnData, (CCU_XN_DATA_SIZE * CCU_USED_XN_NUM));
    *(ccuMsg_.commitCKEAddr) |= 0x1;
    FlushDataCache(ccuMsg_.commitCKEAddr);
}

template <HcclServerType serverType, const auto& config>
__aicore__ inline bool HcclImpl<serverType, config>::CheckWaitCKE(uint8_t resourceId, HcclHandle handleId)
{
#ifdef __CCE_KT_TEST__
    return true;
#endif
    ccuMsg_.waitCKEAddr = GetWaitCkeAddr(resourceId);

    if (!workingFlag_ && handleId >= 0) {
        FlushDataCache(finishCntGM_);
        if (*finishCntGM_ >= commitCnt_) {
            return true;
        }
    }
    FlushDataCache(ccuMsg_.waitCKEAddr);
    if ((*ccuMsg_.waitCKEAddr & 0x1) != 0) {
        KERNEL_LOG(KERNEL_INFO, "the %dth msgbuf is avaliable.", resourceId);
        if (workingFlag_) {
            (*finishCntGM_)++;
            *ccuMsg_.waitCKEAddr &= ~0x1;
            FlushDataCache(finishCntGM_);
            FlushDataCache(ccuMsg_.waitCKEAddr);
        }
        return true;
    } else {
        KERNEL_LOG(KERNEL_INFO, "the %dth msgbuf is not avaliable.", resourceId);
        return false;
    }
}

template <HcclServerType serverType, const auto& config>
__aicore__ inline void HcclImpl<serverType, config>::TryReleaseMsgResource(HcclHandle handleId)
{
    while (CheckWaitCKE(ccuMsgFifo_.m_head, handleId)) {
        uint8_t reqId = 0;
        if (ccuMsgFifo_.pop(reqId)) {
            KERNEL_LOG(KERNEL_INFO, "request is finished, reqId = %d.", reqId);
            commReqBuf_[reqId].isFinish = 1;
        } else {
            break;
        }
    }
}

template <HcclServerType serverType, const auto& config>
__aicore__ inline bool HcclImpl<serverType, config>::CheckNeedPush()
{
    return (!ccuMsgFifo_.isFull()) && (!committedReqFifo_.isEmpty());
}

template <HcclServerType serverType, const auto& config>
__aicore__ inline void HcclImpl<serverType, config>::FlushCommitMsg(HcclHandle handleId)
{
    TryReleaseMsgResource(handleId);
    while (CheckNeedPush()) {
        uint8_t reqId = 0;
        (void)committedReqFifo_.pop(reqId);
        commReqBuf_[reqId].resourceId = ccuMsgFifo_.m_tail;
        (void)ccuMsgFifo_.push(reqId);
        KERNEL_LOG(KERNEL_INFO, "the %dth ccumsg start to send.", reqId);
        if (workingFlag_)
            CcuSendMsg(reqId);
    }
}

template <HcclServerType serverType, const auto& config>
__aicore__ inline GM_ADDR HcclImpl<serverType, config>::GetCommitCkeAddr(uint8_t msgId)
{
    return ccuConfig_.CKEAddr + msgId * CCU_CKE_SIZE;
}

template <HcclServerType serverType, const auto& config>
__aicore__ inline GM_ADDR HcclImpl<serverType, config>::GetWaitCkeAddr(uint8_t msgId)
{
    return ccuConfig_.CKEAddr + msgId * CCU_CKE_SIZE + CCU_CKE_SIZE * CCU_MAX_MSG_NUM;
}

template <HcclServerType serverType, const auto& config>
__aicore__ inline void HcclImpl<serverType, config>::Commit(HcclHandle handleId)
{
    if (!isInited_) {
        KERNEL_LOG(
            KERNEL_ERROR, "Call Commit failed, please ensure Hccl::Init func has been called successfully already!");
        return;
    }
    if ((handleId <= INVALID_HANDLE_ID) || (handleId >= HCCL_MAX_HANDLE_ID)) {
        KERNEL_LOG(KERNEL_ERROR, "Call Wait failed, handleId is[%d], expected in range of [0, %d).", handleId,
            HCCL_MAX_HANDLE_ID);
        return;
    }

    auto reqId = handleId + commReqBuf_[handleId].commitCnt;
    commReqBuf_[handleId].commitCnt++;
    commitCnt_++;
    committedReqFifo_.push(reqId);
    FlushCommitMsg(-1);
}

template <HcclServerType serverType, const auto& config>
__aicore__ inline bool HcclImpl<serverType, config>::IsFinish(uint8_t reqId)
{
    return commReqBuf_[reqId].isFinish == 1 ? true : false;
}

template <HcclServerType serverType, const auto& config>
__aicore__ inline int32_t HcclImpl<serverType, config>::Wait(HcclHandle handleId)
{
    if (!isInited_) {
        KERNEL_LOG(
            KERNEL_ERROR, "Call Wait failed, please ensure Hccl::Init func has been called successfully already!");
        return HCCL_FAILED;
    }

    if ((handleId <= INVALID_HANDLE_ID) || (handleId >= HCCL_MAX_HANDLE_ID)) {
        KERNEL_LOG(KERNEL_ERROR, "Call Wait failed, handleId is[%d], expected in range of [0, %d).", handleId,
            HCCL_MAX_HANDLE_ID);
        return HCCL_FAILED;
    }

    if (handleId >= curHandleId_) {
        KERNEL_LOG(KERNEL_ERROR, "Call Wait failed, handleId[%d] was not got by Prepare interface.", handleId);
        return HCCL_FAILED;
    }

    if (commReqBuf_[handleId].commitCnt <= 0) {
        KERNEL_LOG(KERNEL_ERROR, "commitCnt = %d is invalid", commReqBuf_[handleId].repeatCnt);
        return HCCL_FAILED;
    }

    if (commReqBuf_[handleId].repeatCnt <= 0) {
        KERNEL_LOG(KERNEL_ERROR, "repeat = %d is invalid", commReqBuf_[handleId].repeatCnt);
        return HCCL_FAILED;
    }

    auto reqId = handleId + commReqBuf_[handleId].waitCnt;
    commReqBuf_[handleId].waitCnt++;

    do {
        KERNEL_LOG(KERNEL_INFO, "Only block idx[%ld] do wait and resend msg, handleId = %d, finishedCnt = %d.",
            GetBlockIdx(), handleId, commReqBuf_[handleId].finishedCnt);
        FlushCommitMsg(handleId);
    } while (!IsFinish(reqId));

    commReqBuf_[handleId].finishedCnt++;
    return HCCL_SUCCESS;
}

template <HcclServerType serverType, const auto& config>
template <bool sync>
__aicore__ inline void HcclImpl<serverType, config>::Finalize()
{
    if (!isInited_) {
        KERNEL_LOG(
            KERNEL_ERROR, "Call Finalize failed, please ensure Hccl::Init func has been called successfully already!");
        return;
    }
    if (workingFlag_) {
        CommonPrepareParam commonPrepareParam = {HcclCMDType::HCCL_CMD_FINALIZE, 0, 0, 0,
            HcclDataType::HCCL_DATA_TYPE_RESERVED, HcclReduceOp::HCCL_REDUCE_RESERVED, 0, 0};
        CCUPrepare(commonPrepareParam, curHandleId_, 0, 0);
        Commit(curHandleId_);
    }
}

template <HcclServerType serverType, const auto& config>
__aicore__ inline bool HcclImpl<serverType, config>::SetReduceDataTypeAbility(
    HcclReduceOp op, HcclDataType dstDataType, HcclDataType srcDataType)
{
    // 检查DataType是否合法
    ASCENDC_HCCL_API_ASSERT(
        op != HcclReduceOp::HCCL_REDUCE_RESERVED, { return false; },
        "Set Reduce DataType Ability Failed, HcclReduceOp is invalid, reduceOpType = %u.", static_cast<uint32_t>(op));
    ASCENDC_HCCL_API_ASSERT(
        dstDataType != HcclDataType::HCCL_DATA_TYPE_RESERVED, { return false; },
        "Set Reduce DataType Ability Failed, Hccl OutputDataType is invalid, DataType = %u.",
        static_cast<uint32_t>(dstDataType));
    ASCENDC_HCCL_API_ASSERT(
        srcDataType != HcclDataType::HCCL_DATA_TYPE_RESERVED, { return false; },
        "Set Reduce DataType Ability Failed, Hccl InputDataType is invalid, DataType = %u.",
        static_cast<uint32_t>(srcDataType));

    // 设置datatype
    ccuDataType_.op = op;
    ccuDataType_.dstDataType = dstDataType;
    ccuDataType_.srcDataType = srcDataType;
    needResetDataType_ = true;
    return true;
}

} // namespace AscendC

#endif