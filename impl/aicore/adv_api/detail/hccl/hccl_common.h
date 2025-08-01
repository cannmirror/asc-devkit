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
 * \file hccl_common.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_HCCL_HCCL_COMMON_H
#define AICORE_ADV_API_DETAIL_HCCL_HCCL_COMMON_H
#include "hccl_impl_def.h"

namespace AscendC {

#if defined(ASCENDC_CPU_DEBUG) && ASCENDC_CPU_DEBUG == 1
#define ASCENDC_HCCL_API_ASSERT(cond, ret, fmt, ...)                                                                   \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            KERNEL_LOG(KERNEL_ERROR, fmt, ##__VA_ARGS__);                                                              \
            ret;                                                                                                       \
        }                                                                                                              \
    } while (0)
#elif defined(ASCENDC_DEBUG)
#define ASCENDC_HCCL_API_ASSERT(cond, ret, fmt, ...)                                                                   \
    do {                                                                                                               \
        ASCENDC_DEBUG_ASSERT(cond, fmt, ##__VA_ARGS__);                                                                \
        if (!(cond)) {                                                                                                 \
            ret;                                                                                                       \
        }                                                                                                              \
    } while (0)
#else
#define ASCENDC_HCCL_API_ASSERT(cond, ret, fmt, ...)
#endif

__aicore__ inline void FlushDataCache(GlobalTensor<int64_t>& globalHcclMsgArea, __gm__ void* gmAddr)
{
    AscendC::Barrier();
    globalHcclMsgArea.SetGlobalBuffer((__gm__ int64_t*)gmAddr);
    __asm__("NOP");
    DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(globalHcclMsgArea);
    DataSyncBarrier<MemDsbT::ALL>();
}

__aicore__ inline void FlushDataCache(__gm__ void* gmAddr)
{
    GlobalTensor<int64_t> globalHcclMsgArea;
    FlushDataCache(globalHcclMsgArea, gmAddr);
}

__aicore__ inline void CopyHcclMsg(const uint8_t* src, __gm__ HcclMsg* dst)
{
    __gm__ DataBlock* tmpDst = reinterpret_cast<__gm__ DataBlock*>(dst);
    volatile uint32_t xorCheck = 0U;
    for (uint32_t i = 0; i < HCCL_MSG_DATA_CNT - 1U; ++i) {
        if (i == HCCL_VALID_POS) {
            xorCheck ^= HCCL_MSG_VALID_MASK;
        } else {
            xorCheck ^= tmpDst->data[i] = *(reinterpret_cast<const uint32_t*>(src));
        }
        src += sizeof(tmpDst->data[i]);
    }
    tmpDst->data[HCCL_MSG_DATA_CNT - 1U] = xorCheck;
    tmpDst->data[HCCL_VALID_POS] = HCCL_MSG_VALID_MASK;
}

__aicore__ inline void AssembleHcclMsg(const CommonPrepareParam& para, int8_t ver, HcclHandle handle, uint64_t tiling,
    __gm__ HcclMsg* dst, __gm__ ControlHcclMsg* controlMsgGM)
{
    HcclMsg tmp;
    static uint8_t primitiveId = 0U;
#if AICORE_EXCEPTION_RESTART == 1
    FlushDataCache(controlMsgGM);
    if (controlMsgGM->resetSeq > 0) {
        controlMsgGM->resetSeq = 0;
        primitiveId = 0U;
    }
#endif
    tmp.commType = para.commType;
    if (para.commType == HcclCMDType::HCCL_CMD_FINALIZE) {
        primitiveId = 0U;
        if (ver != 0) {
            tmp.addMsg.v1Msg.ccOpTilingData = 0UL;
        }
    } else {
        tmp.opType = para.op;
        tmp.sendBuffer = reinterpret_cast<uint64_t>(para.sendBuf);
        tmp.recvBuffer = reinterpret_cast<uint64_t>(para.recvBuf);
        tmp.dataCnt = para.count;
        tmp.strideCount = para.strideCount;
        if (ver == 0) {
            tmp.addMsg.v0Msg.hcclDataType = para.dataType;
            tmp.addMsg.v0Msg.repeatCnt = para.repeat;
            tmp.addMsg.v0Msg.selfHandleID = handle;
            tmp.addMsg.v0Msg.seqNum = primitiveId++;
            tmp.addMsg.v0Msg.version = ver;
        } else {
            tmp.addMsg.v1Msg.ccOpTilingData = tiling;
            tmp.addMsg.v1Msg.hcclDataType = para.dataType;
            tmp.addMsg.v1Msg.repeatCnt = para.repeat;
            tmp.addMsg.v1Msg.selfHandleID = handle;
            tmp.addMsg.v1Msg.seqNum = primitiveId++;
            tmp.addMsg.v1Msg.version = ver;
        }
    }
    if (ver == 0) {
        tmp.addMsg.v0Msg.valid = HCCL_MSG_VALID_MASK;
    } else {
        tmp.addMsg.v1Msg.valid = HCCL_MSG_VALID_MASK;
    }
    CopyHcclMsg(reinterpret_cast<const uint8_t*>(&tmp), dst);
}

__aicore__ inline void AssembleHcclMsg(
    const CommonPrepareParam& para, int8_t srcGroupID, HcclHandle srcHandleID, __gm__ HcclMsg* dst)
{
    HcclMsg tmp;
    tmp.commType = para.commType;
    tmp.addMsg.v0Msg.commDepGroupID = srcGroupID;
    tmp.addMsg.v0Msg.commDepHandleID = srcHandleID;
    tmp.addMsg.v0Msg.valid = HCCL_MSG_VALID_MASK;
    CopyHcclMsg(reinterpret_cast<const uint8_t*>(&tmp), dst);
}

template <HcclServerType serverType, const auto& config>
__aicore__ inline bool HcclImpl<serverType, config>::CheckCommonPrepareParamValid(const CommonPrepareParam& param)
{
    const HcclCMDType commType = param.commType;
    if (curVersion_ > 0) {
        ASCENDC_HCCL_API_ASSERT(
            ccOpTilingDataTable_[static_cast<uint32_t>(commType)] != 0UL, { return false; },
            "Failed to prepare for type %u, ensure SetCcTiling has been called.", static_cast<uint32_t>(commType));
    } else {
        ASCENDC_HCCL_API_ASSERT(
            curVersion_ >= 0, { return false; }, "Failed to prepare for type %u, ensure Init has been called",
            static_cast<uint32_t>(commType));
    }
    ASCENDC_HCCL_API_ASSERT(
        param.sendBuf != nullptr && param.recvBuf != nullptr, { return false; },
        "Call Prepare[%d] failed, the param sendBuf/recvBuf is nullptr, "
        "which is an invalid parameter.",
        static_cast<int32_t>(commType));
    ASCENDC_HCCL_API_ASSERT(
        commType == HcclCMDType::HCCL_CMD_BATCH_WRITE
            || (param.dataType >= HCCL_DATA_TYPE_INT8 && param.dataType < HCCL_DATA_TYPE_RESERVED),
        { return false; }, "Call Prepare[%d] failed, param HcclDataType is %d, invalid.",
        static_cast<int32_t>(commType), static_cast<int32_t>(param.dataType));
    if (commType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        ASCENDC_HCCL_API_ASSERT(
            param.paramExt.sendCounts != nullptr && param.paramExt.sdispls != nullptr
                && param.paramExt.recvCounts != nullptr && param.paramExt.rdispls != nullptr,
            { return false; },
            "Call AlltoAllV failed, "
            "param sendCounts/recvCounts/sdispls/rdispls is nullptr, invalid.");
    } else {
        ASCENDC_HCCL_API_ASSERT(
            param.count != 0, { return false; }, "Call Prepare[%d] failed, param sendCount/recvCount is 0, invalid.",
            static_cast<int32_t>(commType));
    }
    return true;
}

template <HcclServerType serverType, const auto& config>
template <bool commit>
__aicore__ inline HcclHandle HcclImpl<serverType, config>::AllReduce(
    GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t count, HcclDataType dataType, HcclReduceOp op, uint8_t repeat)
{
    ASCENDC_HCCL_API_ASSERT(
        op >= HCCL_REDUCE_SUM && op < HCCL_REDUCE_RESERVED, { return INVALID_HANDLE_ID; },
        "Call AllReduce failed, param HcclReduceOp is %d, invalid.", static_cast<int32_t>(op));

    return CommonPrepareImpl<commit>(
        {HcclCMDType::HCCL_CMD_ALLREDUCE, sendBuf, recvBuf, count, dataType, op, 0, repeat});
}

template <HcclServerType serverType, const auto& config>
template <bool commit>
__aicore__ inline HcclHandle HcclImpl<serverType, config>::AllGather(
    GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t sendCount, HcclDataType dataType, uint64_t strideCount, uint8_t repeat)
{
    return CommonPrepareImpl<commit>({HcclCMDType::HCCL_CMD_ALLGATHER, sendBuf, recvBuf, sendCount, dataType,
        HCCL_REDUCE_RESERVED, strideCount, repeat});
}

template <HcclServerType serverType, const auto& config>
template <bool commit>
__aicore__ inline HcclHandle HcclImpl<serverType, config>::ReduceScatter(GM_ADDR sendBuf, GM_ADDR recvBuf,
    uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, uint64_t strideCount, uint8_t repeat)
{
    ASCENDC_HCCL_API_ASSERT(
        op >= HCCL_REDUCE_SUM && op < HCCL_REDUCE_RESERVED, { return INVALID_HANDLE_ID; },
        "Call ReduceScatter failed, param HcclReduceOp is %d, invalid.", static_cast<int32_t>(op));
    return CommonPrepareImpl<commit>(
        {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, sendBuf, recvBuf, recvCount, dataType, op, strideCount, repeat});
}

template <HcclServerType serverType, const auto& config>
template <bool commit>
__aicore__ inline HcclHandle HcclImpl<serverType, config>::AlltoAll(
    GM_ADDR sendBuf, GM_ADDR recvBuf, uint64_t dataCount, HcclDataType dataType, uint64_t strideCount, uint8_t repeat)
{
    return CommonPrepareImpl<commit>({HcclCMDType::HCCL_CMD_ALLTOALL, sendBuf, recvBuf, dataCount, dataType,
        HCCL_REDUCE_RESERVED, strideCount, repeat});
}

template <HcclServerType serverType, const auto& config>
template <bool commit>
__aicore__ inline HcclHandle HcclImpl<serverType, config>::AlltoAllV(GM_ADDR sendBuf, void* sendCounts, void* sdispls,
    HcclDataType sendType, GM_ADDR recvBuf, void* recvCounts, void* rdispls, HcclDataType recvType, uint8_t repeat)
{
    ASCENDC_HCCL_API_ASSERT(
        sendType == recvType, { return INVALID_HANDLE_ID; },
        "Call AlltoAllV failed, param sendType[%d] is not equal to recvType[%d], invalid.",
        static_cast<int32_t>(sendType), static_cast<int32_t>(recvType));
    return CommonPrepareImpl<commit>(
        {HcclCMDType::HCCL_CMD_ALLTOALLV, sendBuf, recvBuf, 0U, sendType, HCCL_REDUCE_RESERVED, 0U, repeat,
            {static_cast<uint64_t*>(sendCounts), static_cast<uint64_t*>(sdispls), static_cast<uint64_t*>(recvCounts),
                static_cast<uint64_t*>(rdispls)}});
}

template <HcclServerType serverType, const auto& config>
template <bool commit>
__aicore__ inline HcclHandle HcclImpl<serverType, config>::BatchWrite(
    GM_ADDR batchWriteInfo, uint32_t itemNum, uint16_t queueID)
{
    return CommonPrepareImpl<true>({HcclCMDType::HCCL_CMD_BATCH_WRITE, batchWriteInfo, batchWriteInfo, itemNum,
        static_cast<HcclDataType>(queueID), static_cast<HcclReduceOp>(queueID + GetBlockIdx() * queueNum_)});
}

template <HcclServerType serverType, const auto& config>
template <bool commit>
__aicore__ inline HcclHandle HcclImpl<serverType, config>::AlltoAllvWrite(
    GM_ADDR usrIn, GM_ADDR sendOffsets, GM_ADDR sendSizes, uint64_t remoteWinOffset, uint64_t localDataSize)
{
    CommonPrepareParam commonPrepareParam = {HcclCMDType::HCCL_CMD_HALF_ALLTOALLV, usrIn, usrIn, localDataSize,
        HCCL_DATA_TYPE_INT8, HCCL_REDUCE_RESERVED, 0, 1, {},
        {reinterpret_cast<uint64_t>(sendOffsets), reinterpret_cast<uint64_t>(sendSizes), remoteWinOffset}};

    return CommonPrepareImpl<commit>(commonPrepareParam);
}

template <HcclServerType serverType, const auto& config>
__aicore__ inline int32_t HcclImpl<serverType, config>::SetCcTiling(__gm__ void* ccOpTilingData)
{
    ASCENDC_HCCL_API_ASSERT(
        curVersion_ == 1, { return HCCL_FAILED; },
        "Call SetCcTiling failed, ensure Hccl::InitV1 func has been called successfully!");
    ASCENDC_HCCL_API_ASSERT(
        ccOpTilingData != nullptr, { return HCCL_FAILED; },
        "Call SetCcTiling failed, ensure ccOpTilingData is not nullptr");
    auto ccTilingPtr = reinterpret_cast<__gm__ char*>(ccOpTilingData);
    auto cmdType = *(reinterpret_cast<__gm__ uint32_t*>(ccTilingPtr + HCCL_CMD_TYPE_OFFSET));
    ASCENDC_HCCL_API_ASSERT(
        cmdType >= 0 && cmdType < static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALL), { return HCCL_FAILED; },
        "Call SetCcTiling failed, ensure cmdType is valid");
    KERNEL_LOG(KERNEL_INFO, "CmdType = %d, ccOpTilingData = %lu ", cmdType, reinterpret_cast<uint64_t>(ccOpTilingData));
    ccOpTilingDataTable_[cmdType] = reinterpret_cast<uint64_t>(ccOpTilingData);
    return HCCL_SUCCESS;
}

template <HcclServerType serverType, const auto& config>
__aicore__ inline int32_t HcclImpl<serverType, config>::Query(HcclHandle handleId)
{
    ASCENDC_HCCL_API_ASSERT(
        curVersion_ >= 0, { return HCCL_FAILED; },
        "Call Query failed, ensure Hccl::Init func has been called successfully!");
    ASCENDC_HCCL_API_ASSERT((handleId > INVALID_HANDLE_ID) && (handleId < HCCL_MAX_HANDLE_ID), { return HCCL_FAILED; },
        "Call Query failed, handleId is[%d], expected in range of [0, %d).", handleId, HCCL_MAX_HANDLE_ID);
    if (queueNum_ != 0U) {
        return 0;
    }
    int8_t curMsgPos = handleIdMsgPosition_[handleId];
    ASCENDC_HCCL_API_ASSERT(
        curMsgPos != INVALID_MSG_POSITION, { return HCCL_FAILED; },
        "Call Query failed, handleId[%d] was not got by Prepare interface.", handleId);
    return WaitFinishCntFromGm(curMsgPos, 0UL);
}

template <HcclServerType serverType, const auto& config>
__aicore__ inline void HcclImpl<serverType, config>::InterHcclGroupSync(int8_t srcGroupID, HcclHandle srcHandleID)
{
    ASCENDC_HCCL_API_ASSERT(
        curVersion_ >= 0, { return; },
        "Call InterHcclGroupSync failed, ensure Hccl::Init func has been called successfully!");
    CommonPrepareParam param = {HcclCMDType::HCCL_CMD_INTER_GROUP_SYNC};
    SendMsgToServer(0U, param, srcGroupID, srcHandleID);
    ++(curMsgPosition_[0U]);
    ASCENDC_HCCL_API_ASSERT(
        curMsgPosition_[0U] < HCCL_MSG_CNT, { return; }, "Message amount exceeds the maximum value when sync group.");
}

template <HcclServerType serverType, const auto& config>
__aicore__ inline GM_ADDR HcclImpl<serverType, config>::GetWindowsInAddr(uint32_t rankId)
{
    ASCENDC_HCCL_API_ASSERT(
        rankId < GetRankDim(), { return nullptr; }, "GetWindowsInAddr failed, rankId[%u], expected less than[%u]",
        rankId, GetRankDim());
    if (hcclContext_->multiFlag == 0U) {
        return (GM_ADDR)hcclContext_->windowsIn[rankId];
    } else {
        if (rankId == hcclContext_->rankId) {
            return (GM_ADDR)(hcclContext_->data[rankId].localInput.addr);
        } else {
            return (GM_ADDR)(hcclContext_->data[rankId].remoteInput.addr);
        }
    }
}

template <HcclServerType serverType, const auto& config>
__aicore__ inline GM_ADDR HcclImpl<serverType, config>::GetWindowsOutAddr(uint32_t rankId)
{
    ASCENDC_HCCL_API_ASSERT(
        rankId < GetRankDim(), { return nullptr; }, "GetWindowsOutAddr failed, rankId[%u], expected less than[%u]",
        rankId, GetRankDim());
    if (hcclContext_->multiFlag == 0U) {
        return (GM_ADDR)hcclContext_->windowsOut[rankId];
    } else {
        if (rankId == hcclContext_->rankId) {
            return (GM_ADDR)(hcclContext_->data[rankId].localOutput.addr);
        } else {
            return (GM_ADDR)(hcclContext_->data[rankId].remoteOutput.addr);
        }
    }
}

template <HcclServerType serverType, const auto& config>
__aicore__ inline void HcclImpl<serverType, config>::InitWorkingFlag()
{
    using T = decltype(config);
    static_assert(std::is_same<T, const HcclServerConfig&>::value);
    KERNEL_LOG(KERNEL_INFO, "Working core type %u id %u.", static_cast<uint8_t>(config.type), config.blockId);
    if constexpr (config.type == CoreType::ON_AIV) {
        workingFlag_ = (g_coreType == AscendC::AIV && GetBlockIdx() == config.blockId);
    } else if constexpr (config.type == CoreType::ON_AIC) {
        workingFlag_ = (g_coreType == AscendC::AIC && GetBlockIdx() == config.blockId);
    } else {
        workingFlag_ = (GetBlockIdx() == config.blockId);
    }
}

} // namespace AscendC

#endif // AICORE_ADV_API_DETAIL_HCCL_HCCL_COMMON_H