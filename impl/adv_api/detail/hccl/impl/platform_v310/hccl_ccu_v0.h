/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hccl_ccu_d100.h
 * \brief
 */
#ifndef IMPL_V310_HCCL_CCU_D100_H
#define IMPL_V310_HCCL_CCU_D100_H

#include "../../common/hccl_utils.h"
#include "../../ccu/hccl_ccu_msg_prepare.h"

namespace AscendC {
template<const auto &config>
template <bool commit>
__aicore__ inline HcclHandle
HcclImpl<HcclServerType::HCCL_SERVER_TYPE_CCU, config>::AllReduce(GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t count,
                    HcclDataType dataType, HcclReduceOp op, uint8_t repeat)
{
    ASCENDC_HCCL_API_ASSERT(op >= HCCL_REDUCE_SUM && op < HCCL_REDUCE_RESERVED, { return INVALID_HANDLE_ID; },
                            "Call AllReduce failed, param HcclReduceOp is %d, invalid.", static_cast<int32_t>(op));

    return CommonPrepareImpl<commit>({ HcclCMDType::HCCL_CMD_ALLREDUCE, sendBuf, recvBuf, count, dataType,
                                       op, 0, repeat });
}

template<const auto &config>
template <bool commit>
__aicore__ inline HcclHandle
HcclImpl<HcclServerType::HCCL_SERVER_TYPE_CCU, config>::AllGather(GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t sendCount,
                HcclDataType dataType, uint64_t strideCount, uint8_t repeat)
{
    return CommonPrepareImpl<commit>({ HcclCMDType::HCCL_CMD_ALLGATHER, sendBuf, recvBuf, sendCount, dataType,
                                       HCCL_REDUCE_RESERVED, strideCount, repeat });
}

template<const auto &config>
template <bool commit>
__aicore__ inline HcclHandle
HcclImpl<HcclServerType::HCCL_SERVER_TYPE_CCU, config>::ReduceScatter(GM_ADDR sendBuf, GM_ADDR recvBuf,
                    uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, uint64_t strideCount, uint8_t repeat)
{
    ASCENDC_HCCL_API_ASSERT(op >= HCCL_REDUCE_SUM && op < HCCL_REDUCE_RESERVED, { return INVALID_HANDLE_ID; },
                            "Call ReduceScatter failed, param HcclReduceOp is %d, invalid.", static_cast<int32_t>(op));
    return CommonPrepareImpl<commit>({ HcclCMDType::HCCL_CMD_REDUCE_SCATTER, sendBuf, recvBuf, recvCount,
                                       dataType, op, strideCount, repeat });
}

template<const auto &config>
template <bool commit>
__aicore__ inline HcclHandle
HcclImpl<HcclServerType::HCCL_SERVER_TYPE_CCU, config>::AlltoAll(GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t dataCount,
                                    HcclDataType dataType, uint64_t strideCount, uint8_t repeat)
{
    return CommonPrepareImpl<commit>({ HcclCMDType::HCCL_CMD_ALLTOALL, sendBuf, recvBuf, dataCount, dataType,
                                       HCCL_REDUCE_RESERVED, strideCount, repeat });
}

template<const auto &config>
template <bool commit>
__aicore__ inline HcclHandle
HcclImpl<HcclServerType::HCCL_SERVER_TYPE_CCU, config>::AlltoAllV(GM_ADDR sendBuf, void *sendCounts, void *sdispls,
                                        HcclDataType sendType, GM_ADDR recvBuf, void *recvCounts, void *rdispls,
                                        HcclDataType recvType, uint8_t repeat)
{
    ASCENDC_HCCL_API_ASSERT(sendType == recvType, { return INVALID_HANDLE_ID; },
                            "Call AlltoAllV failed, param sendType[%d] is not equal to recvType[%d], invalid.",
                            static_cast<int32_t>(sendType), static_cast<int32_t>(recvType));
    return CommonPrepareImpl<commit>({ HcclCMDType::HCCL_CMD_ALLTOALLV, sendBuf, recvBuf, 0U, sendType,
                                       HCCL_REDUCE_RESERVED, 0U, repeat,
                                       {static_cast<uint64_t *>(sendCounts), static_cast<uint64_t *>(sdispls),
                                        static_cast<uint64_t *>(recvCounts), static_cast<uint64_t *>(rdispls)} });
}

template<const auto &config>
template <bool commit>
__aicore__ inline HcclHandle
HcclImpl<HcclServerType::HCCL_SERVER_TYPE_CCU, config>::AlltoAllvWrite(GM_ADDR usrIn, GM_ADDR sendOffsets,
    GM_ADDR sendSizes, uint64_t remoteWinOffset, uint64_t localDataSize)
{
    CommonPrepareParam commonPrepareParam = {HcclCMDType::HCCL_CMD_HALF_ALLTOALLV,
        usrIn,
        usrIn,
        localDataSize,
        HCCL_DATA_TYPE_INT8,
        HCCL_REDUCE_RESERVED,
        0,
        1,
        {},
        {reinterpret_cast<uint64_t>(sendOffsets), reinterpret_cast<uint64_t>(sendSizes), remoteWinOffset}
    };

    return CommonPrepareImpl<commit>(commonPrepareParam);
}

template<const auto &config>
__aicore__ inline void
HcclImpl<HcclServerType::HCCL_SERVER_TYPE_CCU, config>::FlushDataCacheForCopy(GM_ADDR gmAddr, uint32_t size)
{
    uint32_t offset = 0;
    for (uint32_t i = 0; i < size; i += MAX_DCCI_CNT) {
        FlushDataCache((GM_ADDR)(gmAddr + i));
        offset += MAX_DCCI_CNT;
    }
    FlushDataCache((GM_ADDR)(gmAddr + offset));
}
 
template<const auto &config>
__aicore__ inline void
HcclImpl<HcclServerType::HCCL_SERVER_TYPE_CCU, config>::DataCopy2XnAddr(GM_ADDR dstGmAddr, GM_ADDR srcGmAddr,
        uint32_t size)
{   
    for (int i = 0; i < CCU_USED_XN_NUM; i++) {
        *(reinterpret_cast<__gm__ uint64_t*>(dstGmAddr + CCU_XN_DATA_SIZE * i)) =
            *(reinterpret_cast<__gm__ uint64_t*>(srcGmAddr + CCU_XN_DATA_SIZE * i));
    }
    FlushDataCacheForCopy(ccuMsg_.xnAddr, size);
}

template<const auto &config>
__aicore__ inline void HcclImpl<HcclServerType::HCCL_SERVER_TYPE_CCU, config>::InitWorkingFlag()
{
    using T = decltype(config);
    static_assert(std::is_same<T, const HcclServerConfig &>::value);
    KERNEL_LOG(KERNEL_INFO, "Working core type %u id %u.", static_cast<uint8_t>(config.type), config.blockId);
    if constexpr (config.type == CoreType::ON_AIV) {
        workingFlag_ = (g_coreType == AscendC::AIV && GetBlockIdx() == config.blockId);
    } else if constexpr (config.type == CoreType::ON_AIC) {
        workingFlag_ = (g_coreType == AscendC::AIC && GetBlockIdx() == config.blockId);
    } else {
        workingFlag_ = (GetBlockIdx() == config.blockId);
    }
}

template<const auto &config>
__aicore__ inline void
HcclImpl<HcclServerType::HCCL_SERVER_TYPE_CCU, config>::InitInner(GM_ADDR context, HcclTilingVersion version)
{
    ASCENDC_HCCL_API_ASSERT(context != nullptr, { return; }, "Init Hccl failed, context addr is nullptr.");
    hcclContext_ = (__gm__ HcclCombineOpParam *)context;
    // ensure hcclMsgArea 512B aligned
    uint64_t msgAddr = hcclContext_->workSpace;
    if (msgAddr & 0x1ff) {
        msgAddr = (msgAddr & (~((uint64_t)0x1ff))) + 0x200;
    }
    ccuMsgExt_ = reinterpret_cast<__gm__ CCUMsgExt*>(msgAddr);

    ccuConfig_.xnAddr = hcclContext_->xnOffset;
    ccuConfig_.ckeAddr = hcclContext_->ckeOffset;
#ifndef __CCE_KT_TEST__
    ASCENDC_ASSERT((reinterpret_cast<uintptr_t>(ccuConfig_.xnAddr) % ALIGN_64_BYTE == 0),
        { KERNEL_LOG(KERNEL_ERROR, "xnAddr is not 64-byte aligned!"); });
#endif
    ccuMsg_.xnData = reinterpret_cast<__gm__ uint8_t*>(hcclContext_->workSpace + CCU_MSG_EXT_MAX_OFFSET);
    finishCntGM_ = reinterpret_cast<__gm__ uint32_t*>(hcclContext_->workSpace + CCU_MSG_EXT_MAX_OFFSET +
            (CCU_MSG_XN_NUM * CCU_MAX_MSG_NUM * CCU_XN_DATA_SIZE));
    commitCnt_ = 0;
    if (ccuConfig_.xnAddr == nullptr || ccuConfig_.ckeAddr == nullptr || ccuMsg_.xnData == nullptr) {
        KERNEL_LOG(KERNEL_ERROR, "Init Hccl failed,"
                   "ccuConfig_.xnAddr or ccuConfig_.ckeAddr or ccuMsg_.xnData is nullptr");
        return;
    }

    InitWorkingFlag();
    curVersion_ = version;
    curHandleId_ = 0;
    isInited_ = true;
}

template<const auto &config>
__aicore__ inline void
HcclImpl<HcclServerType::HCCL_SERVER_TYPE_CCU, config>::Init(GM_ADDR context, __gm__ void *initTiling)
{
    HcclTilingVersion version =
        (initTiling != nullptr ? HcclTilingVersion::NEW_TILING_VERSION : HcclTilingVersion::DEPRECATED_TILING_VERSION);
    InitInner(context, version);
}

template <const auto &config>
__aicore__ inline void
HcclImpl<HcclServerType::HCCL_SERVER_TYPE_CCU, config>::InitV2(GM_ADDR context, const void *initTiling)
{
    HcclTilingVersion version =
        (initTiling != nullptr ? HcclTilingVersion::NEW_TILING_VERSION : HcclTilingVersion::DEPRECATED_TILING_VERSION);
    InitInner(context, version);
}

template<const auto &config>
__aicore__ inline int32_t
HcclImpl<HcclServerType::HCCL_SERVER_TYPE_CCU, config>::SetCcTiling(__gm__ void *ccOpTilingData)
{
    ASCENDC_HCCL_API_ASSERT(curVersion_ == HcclTilingVersion::NEW_TILING_VERSION, { return HCCL_FAILED; },
                            "Call SetCcTiling failed, ensure Hccl::InitV1 func has been called successfully!");
    ASCENDC_HCCL_API_ASSERT(ccOpTilingData != nullptr, { return HCCL_FAILED; },
                            "Call SetCcTiling failed, ensure ccOpTilingData is not nullptr");
    const uint32_t opType = (static_cast<__gm__ Mc2CcTilingInner *>(ccOpTilingData))->opType;
    ASCENDC_HCCL_API_ASSERT(opType >= 0 && opType < static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALL),
                            { return HCCL_FAILED; }, "Call SetCcTiling failed, ensure cmdType is valid");
    KERNEL_LOG(KERNEL_INFO, "CmdType = %d, ccOpTilingData = %lu ", opType, reinterpret_cast<uint64_t>(ccOpTilingData));
    ccOpTilingDataTable_[opType] = reinterpret_cast<uint64_t>(ccOpTilingData);
    return HCCL_SUCCESS;
}

template <const auto &config>
__aicore__ inline int32_t HcclImpl<HcclServerType::HCCL_SERVER_TYPE_CCU, config>::SetCcTilingV2(uint64_t offset)
{
    ASCENDC_HCCL_API_ASSERT(curVersion_ == HcclTilingVersion::ONLINE_COMPILATION_TILING_VERSION,
                            { return HCCL_FAILED; },
                            "Call SetCcTiling failed, ensure Hccl::InitV2 func has been called successfully!");
    const uint32_t opType = (reinterpret_cast<Mc2CcTilingInner *>(tilingBaseAddr_ + offset))->opType;
    ASCENDC_HCCL_API_ASSERT(opType >= 0 && opType < static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALL),
                            { return HCCL_FAILED; }, "Call SetCcTiling failed, ensure cmdType is valid");
    ccOpTilingDataTable_[opType] = offset;
    return HCCL_SUCCESS;
}

template<const auto &config>
__aicore__ inline uint64_t
HcclImpl<HcclServerType::HCCL_SERVER_TYPE_CCU, config>::GetOpId(const CommonPrepareParam &commonPrepareParam)
{
    if (commonPrepareParam.commType.msgType == ControlMsgType::HCCL_CMD_FINALIZE) {
        return 0xffffffffffffffff;
    }

    uint64_t algoType = 0U;
    uint64_t commType = static_cast<uint64_t>(commonPrepareParam.commType.prepareType);
    uint64_t outDataType = static_cast<uint64_t>(commonPrepareParam.dataType);
    uint64_t dataType = static_cast<uint64_t>(commonPrepareParam.dataType);
    bool isReduceType = (commonPrepareParam.commType.prepareType == HcclCMDType::HCCL_CMD_REDUCE) ||
                        (commonPrepareParam.commType.prepareType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) ||
                        (commonPrepareParam.commType.prepareType == HcclCMDType::HCCL_CMD_ALLREDUCE);
    uint64_t reduceType = isReduceType ? static_cast<uint64_t>(commonPrepareParam.op) : 0U;
    uint64_t ccTiling = ccOpTilingDataTable_[static_cast<uint32_t>(commonPrepareParam.commType.prepareType)];
    if (needResetDataType_) {
        outDataType = static_cast<uint64_t>(ccuDataType_.dstDataType);
        dataType = static_cast<uint64_t>(ccuDataType_.srcDataType);
        reduceType = isReduceType ? static_cast<uint64_t>(ccuDataType_.op) : reduceType;
    } else if (ccTiling != 0) {
        __gm__ Mc2CcTilingInner *tilingPtr = reinterpret_cast<__gm__ Mc2CcTilingInner *>(ccTiling);
        outDataType = tilingPtr->dstDataType;
        dataType = tilingPtr->srcDataType;
        reduceType = isReduceType ? tilingPtr->reduceType : reduceType;
    }
    return (((algoType & 0x7f) << 32) | ((outDataType & 0x7f) << 24 ) | ((reduceType & 0x7f) << 16) |
            ((dataType & 0x7f) << 8) | (commType & 0x7f));
}

template<const auto &config>
__aicore__ inline uint64_t HcclImpl<HcclServerType::HCCL_SERVER_TYPE_CCU, config>::GetParallelParam(uint64_t repeatNum,
    uint64_t repeatLoopIndex, uint64_t totalLoopNum)
{
    return ((repeatNum & 0x7f) << 55) | ((repeatLoopIndex & 0x7f) << 48) | ((totalLoopNum & 0x7f) << 41);
}

template<const auto &config>
__aicore__ inline void
HcclImpl<HcclServerType::HCCL_SERVER_TYPE_CCU, config>::CCUPrepare(const CommonPrepareParam &commonPrepareParam,
    uint8_t reqId, uint64_t sendBuf, uint64_t recvBuf)
{
    auto& commOp = commReqBuf_[reqId];
    commOp.xnData[0] = GetOpId(commonPrepareParam);
    commOp.xnData[1] = sendBuf;
    commOp.xnData[2] = recvBuf;
    if (commonPrepareParam.commType.prepareType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        commOp.xnData[3] = 0;
        commOp.xnData[4] = 0;
        // Group by rank, sendSize, sendOffset, recvSize, recvOffset in bytes * DataSzie(DataType)
        commOp.xnData[5] = reinterpret_cast<uint64_t>(ccuMsgExt_) + CCU_MSG_EXT_RANK_OFFSET * alltoallvCnt_++;
        return;
    }

    uint64_t sliceCount = 0;
    uint64_t tmpCount = commonPrepareParam.count / hcclContext_->rankNum;
    uint64_t loopCount = CCU_LOOP_COUNT;

    if (commonPrepareParam.commType.prepareType == HcclCMDType::HCCL_CMD_ALLGATHER ||
        commonPrepareParam.commType.prepareType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
        sliceCount = commonPrepareParam.count;
    } else if (commonPrepareParam.commType.prepareType == HcclCMDType::HCCL_CMD_ALLREDUCE) {
        sliceCount = (hcclContext_->rankId == hcclContext_->rankNum - 1) ?
                         (commonPrepareParam.count - (hcclContext_->rankNum - 1) * tmpCount) : tmpCount;
    }

    uint64_t sliceSize = sliceCount * DATA_TYPE_MAP[commonPrepareParam.dataType];

    if (commonPrepareParam.commType.prepareType == HcclCMDType::HCCL_CMD_HALF_ALLTOALLV) {
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

    if (commonPrepareParam.commType.prepareType == HcclCMDType::HCCL_CMD_ALLREDUCE) {
        commOp.xnData[3] = (commonPrepareParam.strideCount == 0) ? tmpCount * dataSize * hcclContext_->rankId : 
            (commonPrepareParam.strideCount * dataSize * hcclContext_->rankId);
    } else if (commonPrepareParam.commType.prepareType == HcclCMDType::HCCL_CMD_HALF_ALLTOALLV) {
        commOp.xnData[3] = commonPrepareParam.wParamExt.sendOffsets;
    } else {
        commOp.xnData[3] = (commonPrepareParam.strideCount == 0) ? sliceSize * hcclContext_->rankId : 
            (commonPrepareParam.strideCount * dataSize * hcclContext_->rankId);
    }
    commOp.xnData[4] = loopSize * m;
    commOp.xnData[5] = m;

    if (n == 0 && p == 0) {
        // If the data volume is an integer multiple of loopSize, skip LoopGroup1.
        commOp.xnData[6] = 0;
        commOp.xnData[7] = 0;
    } else if (n != 0 && p == 0) {
        // The data volume is 256K * m + CCU_MEMSLICE_SIZE * n
        commOp.xnData[6] = GetParallelParam(n - 1, 0, 1);
        commOp.xnData[7] = CCU_MEMSLICE_SIZE;
    } else if (n == 0 && p != 0) {
        // The data volume is loopSize * m + p
        commOp.xnData[6] = GetParallelParam(0, 0, 1);
        commOp.xnData[7] = p;
    } else {
        // The data volume is loopSize * m + CCU_MEMSLICE_SIZE * n + p
        commOp.xnData[6] = GetParallelParam(n - 1, 1, 2);
        commOp.xnData[7] = p;
    }
}

template<const auto &config>
template<bool commit>
__aicore__ inline void
HcclImpl<HcclServerType::HCCL_SERVER_TYPE_CCU, config>::CCUPrepareInner(const CommonPrepareParam &commonPrepareParam,
        const HcclHandle handleId)
{
    uint64_t sendBuf = (uint64_t)commonPrepareParam.sendBuf;
    uint64_t recvBuf = (uint64_t)commonPrepareParam.recvBuf;
    uint8_t reqId = handleInfo_[handleId].reqId + handleInfo_[handleId].commitCnt;
    for (uint32_t i = 0U; i < handleInfo_[handleId].repeatCnt; ++i) {
        KERNEL_LOG(KERNEL_INFO, "do ccu prepare repeatIdx = %d, repeatCnt = %d, handleId = %d, reqId = %d.",
                   i, handleInfo_[handleId].repeatCnt, handleId, reqId);
        // recv send buf + offset
        uint64_t offset = commonPrepareParam.count * i * DATA_TYPE_MAP[commonPrepareParam.dataType];
        if (workingFlag_ && commonPrepareParam.commType.prepareType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
            KERNEL_LOG(KERNEL_INFO, "Only block idx[%ld] assemble msg ext when prepare alltoallv.", GetBlockIdx());
            AssembleHcclMsgExtForCCU(commonPrepareParam, i);
        }
        InitCommReq(reqId);
        CCUPrepare(commonPrepareParam, reqId, sendBuf + offset, recvBuf + offset); 
        if (commit) {
            KERNEL_LOG(KERNEL_INFO, "commit flag is true. repeatIdx = %d, repeatCnt = %d, handleId = %d.",
                       i, handleInfo_[handleId].repeatCnt, handleId);
            CommitMsg(handleId, reqId);
        }
        reqId++;
    }
}

template<const auto &config>
__aicore__ inline void
HcclImpl<HcclServerType::HCCL_SERVER_TYPE_CCU, config>::
AssembleHcclMsgExtForCCU(const CommonPrepareParam &commonPrepareParam, uint32_t repeatIndex)
{
    auto dataSize = DATA_TYPE_MAP[static_cast<uint64_t>(commonPrepareParam.dataType)];

    __gm__ CCUMsgExt *ccuMsgExt = reinterpret_cast<__gm__ CCUMsgExt*>(reinterpret_cast<uint64_t>(ccuMsgExt_) +
        CCU_MSG_EXT_RANK_OFFSET * alltoallvCnt_);

    const uint32_t kRankNum = hcclContext_->rankNum;
    for (uint32_t i = 0U; i < kRankNum; ++i) {
        ccuMsgExt[i].sendSize = commonPrepareParam.paramExt.sendCounts[i] * dataSize;
        ccuMsgExt[i].recvSize = commonPrepareParam.paramExt.recvCounts[i] * dataSize;
        
        ccuMsgExt[i].sendOffset = commonPrepareParam.paramExt.sdispls[i] * dataSize +
            ccuMsgExt[i].sendSize * repeatIndex;
        ccuMsgExt[i].recvOffset = commonPrepareParam.paramExt.rdispls[i] * dataSize +
            ccuMsgExt[i].recvSize * repeatIndex;
    }
 
    uint32_t tmpCnt = (sizeof(CCUMsgExt) * kRankNum) / MAX_DCCI_CNT;
    uint32_t copyCnt = (sizeof(CCUMsgExt) * kRankNum) % MAX_DCCI_CNT ? tmpCnt + 1 : tmpCnt;
    GlobalTensor<int64_t> globalHcclMsgArea;
 
    uint64_t tmpSize = 0;
    for (uint32_t i = 0U; i < copyCnt ; ++i) {
        FlushDataCache(globalHcclMsgArea, (ccuMsgExt + tmpSize));
        tmpSize += MAX_DCCI_CNT;
    }
}

template<const auto &config>
__aicore__ inline void HcclImpl<HcclServerType::HCCL_SERVER_TYPE_CCU, config>::InitCommReq(uint8_t reqId)
{
    commReqBuf_[reqId].resourceId = -1;
    commReqBuf_[reqId].isFinish = 0;
}

template<const auto &config>
__aicore__ inline void HcclImpl<HcclServerType::HCCL_SERVER_TYPE_CCU, config>::InitHandleInfo(uint8_t handleId)
{
    handleInfo_[handleId].reqId = 0;
    handleInfo_[handleId].repeatCnt = 1;
    handleInfo_[handleId].commitCnt = 0;
    handleInfo_[handleId].waitCnt = 0;
    handleInfo_[handleId].finishCnt = 0;
}

template<const auto &config>
template<bool commit>
__aicore__ inline HcclHandle
HcclImpl<HcclServerType::HCCL_SERVER_TYPE_CCU, config>::CommonPrepareImpl(const CommonPrepareParam &commonPrepareParam)
{
    ASCENDC_HCCL_API_ASSERT(commonPrepareParam.repeat > 0, { return INVALID_HANDLE_ID; },
                            "Call Prepare failed, ensure repeat larger than 0!");
    HcclHandle handleId = curHandleId_;
    InitHandleInfo(handleId);
    if (handleId >= HCCL_MAX_HANDLE_ID) {
        KERNEL_LOG(KERNEL_ERROR,
                   "Call Prepare[%d] failed, Prepare interface call num is[%d], "
                   "expected less than[%d].",
                   static_cast<int32_t>(commonPrepareParam.commType.prepareType), handleId + 1,
                   HCCL_MAX_HANDLE_ID);
        return INVALID_HANDLE_ID;
    }

    handleInfo_[handleId].repeatCnt = commonPrepareParam.repeat;
    handleInfo_[handleId].reqId = (handleId == 0) ? 0 :
        handleInfo_[handleId - 1].reqId + handleInfo_[handleId - 1].repeatCnt;
    CCUPrepareInner<commit>(commonPrepareParam, handleId);

    curHandleId_++;
    return handleId;
}

template<const auto &config>
__aicore__ inline void HcclImpl<HcclServerType::HCCL_SERVER_TYPE_CCU, config>::CcuSendMsg(uint8_t reqId)
{
    int8_t resourceId = reqId % CCU_MAX_MSG_NUM;
    if (resourceId >= 8 || resourceId < 0) {
        KERNEL_LOG(KERNEL_ERROR, "CcuSendMsg resourceId %d is invalid.", resourceId);
        return;
    }
    ccuMsg_.xnAddr = ccuConfig_.xnAddr + static_cast<uint64_t>(resourceId) * CCU_MSG_XN_NUM * CCU_MAX_MSG_NUM;
    ccuMsg_.commitCKEAddr = GetCommitCkeAddr(resourceId);
    ccuMsg_.waitCKEAddr = GetWaitCkeAddr(resourceId);
    auto ptr = ccuMsg_.xnData;
    for (int i = 0; i < CCU_USED_XN_NUM; i++) {
        *reinterpret_cast<__gm__ uint64_t*>(ptr) = commReqBuf_[reqId].xnData[i];
        ptr += 8; // 8 is sizeof(uint64)
    }
    DataCopy2XnAddr(ccuMsg_.xnAddr, ccuMsg_.xnData, (CCU_XN_DATA_SIZE * CCU_USED_XN_NUM));

    WriteGmByPassDCache(ccuMsg_.commitCKEAddr, CCU_MSG_CKE_SET_VALUE);
}

template<const auto &config>
__aicore__ inline bool HcclImpl<HcclServerType::HCCL_SERVER_TYPE_CCU, config>::CheckNeedPush()
{
    return (!ccuMsgFifo_.isFull()) && (!committedReqFifo_.isEmpty());
}

template<const auto &config>
__aicore__ inline GM_ADDR HcclImpl<HcclServerType::HCCL_SERVER_TYPE_CCU, config>::GetCommitCkeAddr(uint8_t msgId)
{
    return ccuConfig_.ckeAddr + static_cast<uint64_t>(msgId) * CCU_CKE_SIZE;
}

template<const auto &config>
__aicore__ inline GM_ADDR HcclImpl<HcclServerType::HCCL_SERVER_TYPE_CCU, config>::GetWaitCkeAddr(uint8_t msgId)
{
    uint64_t offset = static_cast<uint64_t>(msgId) * CCU_CKE_SIZE + static_cast<uint64_t>(CCU_CKE_SIZE) * CCU_MAX_MSG_NUM;
    return ccuConfig_.ckeAddr + offset;
}

template<const auto &config>
__aicore__ inline void
HcclImpl<HcclServerType::HCCL_SERVER_TYPE_CCU, config>::CommitMsg(HcclHandle handleId, uint8_t reqId)
{
    handleInfo_[handleId].commitCnt++;
    commitCnt_++;
    committedReqFifo_.push(reqId);
    while (CheckNeedPush()) {
        uint8_t pushReqId = 0;
        (void)committedReqFifo_.pop(pushReqId);
        commReqBuf_[pushReqId].resourceId = pushReqId % CCU_MAX_MSG_NUM;
        (void)ccuMsgFifo_.push(pushReqId);
        KERNEL_LOG(KERNEL_INFO, "the %dth ccumsg start to send.", pushReqId);
        if (workingFlag_)
            CcuSendMsg(pushReqId);
    }
}

template<const auto &config>
__aicore__ inline void HcclImpl<HcclServerType::HCCL_SERVER_TYPE_CCU, config>::Commit(HcclHandle handleId)
{
    ASCENDC_HCCL_API_ASSERT(isInited_, { return; },
    "Call Commit failed, please ensure Hccl::Init func has been called successfully already!");

    ASCENDC_HCCL_API_ASSERT(handleId > INVALID_HANDLE_ID && handleId < HCCL_MAX_HANDLE_ID, { return; },
        "Call Wait failed, handleId is[%d], expected in range of [0, %d).", handleId, HCCL_MAX_HANDLE_ID);

    auto reqId = handleInfo_[handleId].reqId + handleInfo_[handleId].commitCnt;

    CommitMsg(handleId, reqId);
}

template<const auto &config>
__aicore__ inline bool HcclImpl<HcclServerType::HCCL_SERVER_TYPE_CCU, config>::IsFinish(uint8_t reqId)
{
    return commReqBuf_[reqId].isFinish == 1 ? true : false;
}

template<const auto &config>
__aicore__ inline int32_t HcclImpl<HcclServerType::HCCL_SERVER_TYPE_CCU, config>::Wait(HcclHandle handleId)
{
    if (!isInited_) {
        KERNEL_LOG(KERNEL_ERROR,
                   "Call Wait failed, please ensure Hccl::Init func has been called successfully already!");
        return HCCL_FAILED;
    }

    if ((handleId <= INVALID_HANDLE_ID) || (handleId >= HCCL_MAX_HANDLE_ID)) {
        KERNEL_LOG(KERNEL_ERROR,
                   "Call Wait failed, handleId is[%d], expected in range of [0, %d).",
                   handleId, HCCL_MAX_HANDLE_ID);
        return HCCL_FAILED;
    }

    if (handleId >= curHandleId_) {
        KERNEL_LOG(KERNEL_ERROR, "Call Wait failed, handleId[%d] was not got by Prepare interface.", handleId);
        return HCCL_FAILED;
    }

    if (handleInfo_[handleId].commitCnt <= 0) {
        KERNEL_LOG(KERNEL_ERROR, "commitCnt = %d is invalid", handleInfo_[handleId].repeatCnt);
        return HCCL_FAILED;
    }

    if (handleInfo_[handleId].repeatCnt <= 0) {
        KERNEL_LOG(KERNEL_ERROR, "repeat = %d is invalid", handleInfo_[handleId].repeatCnt);
        return HCCL_FAILED;
    }
#ifdef __CCE_KT_TEST__
    return HCCL_SUCCESS;
#endif
    uint8_t reqId = handleInfo_[handleId].reqId + handleInfo_[handleId].waitCnt;
    GM_ADDR waitCKEAddr = GetWaitCkeAddr(reqId % CCU_MAX_MSG_NUM);
    handleInfo_[handleId].waitCnt++;
    do {
        if (!workingFlag_) {
            uint64_t fCnt = handleId == 0 ? handleInfo_[handleId].waitCnt : handleInfo_[handleId - 1].finishCnt + handleInfo_[handleId].waitCnt;
            FlushDataCache(finishCntGM_);
            if (*finishCntGM_ >= fCnt) {
                return true;
            }
        }
        int32_t waitCKEAddrValue = ReadGmByPassDCache(waitCKEAddr);
        if (waitCKEAddrValue != 0) {
            KERNEL_LOG(KERNEL_INFO, "the %dth msgbuf is avaliable.", reqId);
            if (workingFlag_) {
                (*finishCntGM_)++;
                FlushDataCache(finishCntGM_);

                WriteGmByPassDCache(waitCKEAddr, CCU_MSG_CKE_INIT_VALUE);
                uint8_t popReqId = 0;
                ccuMsgFifo_.pop(popReqId);
            }
            break;
        }
    } while (true);

    if (workingFlag_) {
        while (CheckNeedPush()) {
            uint8_t pushReqId = 0;
            (void)committedReqFifo_.pop(pushReqId);
            commReqBuf_[pushReqId].resourceId = pushReqId % CCU_MAX_MSG_NUM;
            (void)ccuMsgFifo_.push(pushReqId);
            CcuSendMsg(pushReqId);
        }
    }
    handleInfo_[handleId].finishCnt++;
    return HCCL_SUCCESS;
}


template<const auto &config>
template <bool sync>
__aicore__ inline void HcclImpl<HcclServerType::HCCL_SERVER_TYPE_CCU, config>::Finalize()
{
    if (!isInited_) {
        KERNEL_LOG(KERNEL_ERROR,
                   "Call Finalize failed, please ensure Hccl::Init func has been called successfully already!");
        return;
    }
    if (workingFlag_) {
        int8_t reqId = handleInfo_[curHandleId_ - 1].reqId + handleInfo_[curHandleId_ - 1].repeatCnt;
        uint8_t resourceId = reqId % CCU_MAX_MSG_NUM;
        ccuMsg_.commitCKEAddr = GetCommitCkeAddr(resourceId);
        ccuMsg_.xnAddr = hcclContext_->xnOffset + CCU_MSG_XN_NUM * CCU_MAX_MSG_NUM * resourceId;
        *reinterpret_cast<__gm__ uint64_t*>(ccuMsg_.xnAddr) = 0xffffffffffffffff;
        FlushDataCache(ccuMsg_.xnAddr);

        WriteGmByPassDCache(ccuMsg_.commitCKEAddr, CCU_MSG_CKE_SET_VALUE);
        *finishCntGM_ = 0;
        FlushDataCache(finishCntGM_);
    }
}

template<const auto &config>
__aicore__ inline bool
HcclImpl<HcclServerType::HCCL_SERVER_TYPE_CCU, config>::SetReduceDataTypeAbility(HcclReduceOp op,
    HcclDataType dstDataType, HcclDataType srcDataType)
{
    ASCENDC_HCCL_API_ASSERT(op != HcclReduceOp::HCCL_REDUCE_RESERVED, { return false; },
        "Set Reduce DataType Ability Failed, HcclReduceOp is invalid, reduceOpType = %u.",
        static_cast<uint32_t>(op));
    ASCENDC_HCCL_API_ASSERT(dstDataType != HcclDataType::HCCL_DATA_TYPE_RESERVED, { return false; },
        "Set Reduce DataType Ability Failed, Hccl OutputDataType is invalid, DataType = %u.",
        static_cast<uint32_t>(dstDataType));
    ASCENDC_HCCL_API_ASSERT(srcDataType != HcclDataType::HCCL_DATA_TYPE_RESERVED, { return false; },
        "Set Reduce DataType Ability Failed, Hccl InputDataType is invalid, DataType = %u.",
        static_cast<uint32_t>(srcDataType));
    ccuDataType_.op = op;
    ccuDataType_.dstDataType = dstDataType;
    ccuDataType_.srcDataType = srcDataType;
    needResetDataType_ = true;
    return true;
}
}
#endif