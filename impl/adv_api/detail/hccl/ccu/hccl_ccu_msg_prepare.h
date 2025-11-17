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
 * \file hccl_ccu_msg_prepare.h
 * \brief
 */
#ifndef IMPL_HCCL_CCU_MSG_PREEPARE_H
#define IMPL_HCCL_CCU_MSG_PREEPARE_H

#include "../common/hccl_inner_def.h"
#include "../common/hccl_utils.h"

namespace AscendC {

constexpr uint64_t ALGO_TYPE_MASK = 0x7f;
constexpr uint64_t OUT_DATA_TYPE_MASK = 0x7f;
constexpr uint64_t REDUCE_TYPE_MASK = 0x7f;
constexpr uint64_t DATA_TYPE_MASK = 0x7f;
constexpr uint64_t COMM_TYPE_MASK = 0x7f;

constexpr uint64_t ALGO_TYPE_SHIFT = 32;
constexpr uint64_t OUT_DATA_TYPE_SHIFT = 24;
constexpr uint64_t REDUCE_TYPE_SHIFT = 16;
constexpr uint64_t DATA_TYPE_SHIFT = 8;
constexpr uint64_t COMM_TYPE_SHIFT = 0;

constexpr int CCU_PARAM_INDEX = 2;
constexpr int CCU_XNDATA_INDEX_SIX = 6;
constexpr int CCU_XNDATA_INDEX_SEVEN = 7;

constexpr uint64_t REPEAT_NUM_SHIFT = 55;
constexpr uint64_t REPEAT_LOOP_INDEX_SHIFT = 48;
constexpr uint64_t TOTAL_LOOP_NUM_SHIFT = 41;
constexpr uint64_t MASK = 0x7f;

__aicore__ inline uint64_t GetOpId(CCUParam &ccuParam)
{
    if (ccuParam.commParam.commType.msgType == ControlMsgType::HCCL_CMD_FINALIZE) {
        return 0xffffffffffffffff;
    }

    uint64_t algoType = 0U;
    uint64_t commType = static_cast<uint64_t>(ccuParam.commParam.commType.prepareType);
    uint64_t dataType = static_cast<uint64_t>(ccuParam.commParam.dataType);
    uint64_t outDataType = static_cast<uint64_t>(ccuParam.commParam.dstDataType);

    bool isReduceType = (ccuParam.commParam.commType.prepareType == HcclCMDType::HCCL_CMD_REDUCE) ||
                        (ccuParam.commParam.commType.prepareType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) ||
                        (ccuParam.commParam.commType.prepareType == HcclCMDType::HCCL_CMD_ALLREDUCE);

    uint64_t reduceType = isReduceType ? static_cast<uint64_t>(ccuParam.commParam.op) : 0U;

    KERNEL_LOG(KERNEL_INFO, "ApiClient GetOpId algoType:%d outDataType:%d, reduceType:%d, dataType:%d, commType:%d",
               algoType, outDataType, reduceType, dataType, commType);

    return ((algoType & ALGO_TYPE_MASK) << ALGO_TYPE_SHIFT) |
           ((outDataType & OUT_DATA_TYPE_MASK) << OUT_DATA_TYPE_SHIFT) |
           ((reduceType & REDUCE_TYPE_MASK) << REDUCE_TYPE_SHIFT) |
           ((dataType & DATA_TYPE_MASK) << DATA_TYPE_SHIFT) |
           (commType & COMM_TYPE_MASK);
}

__aicore__ inline void AssembleHcclMsgExtForCCU(CCUParam &ccuParam)
{
    __gm__ CCUMsgExt *ccuMsgExt = reinterpret_cast<__gm__ CCUMsgExt *>(
        reinterpret_cast<uint64_t>(ccuParam.ccuMsgExt) + CCU_MSG_EXT_RANK_OFFSET * ccuParam.alltoallvCnt);
    uint64_t dataSize = DATA_TYPE_MAP[static_cast<uint64_t>(ccuParam.commParam.dataType)];

    KERNEL_LOG(KERNEL_INFO, "ApiClient AssembleHcclMsgExtForCCU ccuParam.ccuMsgExt:0x%llx, ccuMsgExt:0x%llx",
               reinterpret_cast<uint64_t>(ccuParam.ccuMsgExt), reinterpret_cast<uint64_t>(ccuMsgExt));

    for (uint32_t i = 0U; i < ccuParam.rankNum; ++i) {
        ccuMsgExt[i].sendSize = ccuParam.commParam.paramExt.sendCounts[i] * dataSize;
        ccuMsgExt[i].recvSize = ccuParam.commParam.paramExt.recvCounts[i] * dataSize;
        ccuMsgExt[i].sendOffset =
            ccuParam.commParam.paramExt.sdispls[i] * dataSize + ccuMsgExt[i].sendSize * ccuParam.repeatIndex;
        ccuMsgExt[i].recvOffset =
            ccuParam.commParam.paramExt.rdispls[i] * dataSize + ccuMsgExt[i].recvSize * ccuParam.repeatIndex;
        KERNEL_LOG(KERNEL_INFO, "ApiClient ccuMsgExt rankIndex:%u, sendSize:%d, recvSize:%d, sendOffset:%d, recvOffset:%d",
                   i, ccuMsgExt[i].sendSize, ccuMsgExt[i].recvSize, ccuMsgExt[i].sendOffset, ccuMsgExt[i].recvOffset);
    }
 
    uint32_t tmpCnt = (sizeof(CCUMsgExt) * ccuParam.rankNum) / MAX_DCCI_CNT;
    uint32_t copyCnt = (sizeof(CCUMsgExt) * ccuParam.rankNum) % MAX_DCCI_CNT ? tmpCnt + 1 : tmpCnt;
    KERNEL_LOG(KERNEL_INFO, "ApiClient AssembleHcclMsgExtForCCU tmpCnt:%d, copyCnt:%d", tmpCnt, copyCnt);

    GlobalTensor<int64_t> globalHcclMsgArea;
    uint64_t tmpSize = 0;
    for (uint32_t i = 0U; i < copyCnt ; ++i) {
        FlushDataCache(globalHcclMsgArea, (GM_ADDR)(reinterpret_cast<uint64_t>(ccuMsgExt) + tmpSize));
        tmpSize += MAX_DCCI_CNT;
    }
}

__aicore__ inline uint64_t GetSliceCount(CCUParam &ccuParam, uint64_t tmpCount)
{
    uint64_t sliceCount = 0;
    if (ccuParam.commParam.commType.prepareType == HcclCMDType::HCCL_CMD_ALLGATHER ||
            ccuParam.commParam.commType.prepareType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER) {
        sliceCount = ccuParam.commParam.count;
    } else if (ccuParam.commParam.commType.prepareType == HcclCMDType::HCCL_CMD_ALLREDUCE) {
        sliceCount = (ccuParam.rankId == ccuParam.rankNum - 1) ?
                     (ccuParam.commParam.count - (ccuParam.rankNum - 1) * tmpCount) : tmpCount;
    }
    KERNEL_LOG(KERNEL_INFO, "ApiClient GetSliceCount sliceCount:%d", sliceCount);
    return sliceCount;
}

__aicore__ inline uint64_t GetParallelParameters(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
{
    return ((repeatNum & MASK) << REPEAT_NUM_SHIFT) | ((repeatLoopIndex & MASK) << REPEAT_LOOP_INDEX_SHIFT) |
        ((totalLoopNum & MASK) << TOTAL_LOOP_NUM_SHIFT);
}

__aicore__ inline void SetupCCUDataPartA(CCUParam &ccuParam, uint64_t *xnData)
{
    xnData[0] = GetOpId(ccuParam); // ccu xn0
    uint64_t offset = ccuParam.commParam.count * ccuParam.repeatIndex * DATA_TYPE_MAP[ccuParam.commParam.dataType];
    xnData[1] = (uint64_t)ccuParam.commParam.sendBuf + offset; // ccu xn1
    xnData[2] = (uint64_t)ccuParam.commParam.recvBuf + offset; // ccu xn2
}

__aicore__ inline void SetupCCUDataPartB(CCUParam &ccuParam, uint64_t *xnData, uint64_t m, uint64_t n, uint64_t p)
{
    if (n == 0 && p == 0) {
        // 数据量为loopSize的整数倍，跳过LoopGroup1
        xnData[CCU_XNDATA_INDEX_SIX] = 0; // ccu xn6
        xnData[CCU_XNDATA_INDEX_SEVEN] = 0; // ccu xn7
    } else if (n != 0 && p == 0) {
        // 数据量为256K * m + CCU_MEMSLICE_SIZE * n
        xnData[CCU_XNDATA_INDEX_SIX] = GetParallelParameters(n - 1, 0, 1); // ccu xn6
        xnData[CCU_XNDATA_INDEX_SEVEN] = CCU_MEMSLICE_SIZE; // ccu xn7
    } else if (n == 0 && p != 0) {
        // 数据量为loopSize * m + p
        xnData[CCU_XNDATA_INDEX_SIX] = GetParallelParameters(0, 0, 1); // ccu xn6
        xnData[CCU_XNDATA_INDEX_SEVEN] = p; // ccu xn7
    } else {
        // 数据量为loopSize * m + CCU_MEMSLICE_SIZE * n + p
        xnData[CCU_XNDATA_INDEX_SIX] = GetParallelParameters(n - 1, 1, CCU_PARAM_INDEX); // ccu xn6
        xnData[CCU_XNDATA_INDEX_SEVEN] = p; // ccu xn7
    }
}

__aicore__ inline void CCUPrepare(CCUParam &ccuParam, uint64_t *xnData)
{
    SetupCCUDataPartA(ccuParam, xnData);
    if (ccuParam.commParam.commType.prepareType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        xnData[3] = 0; // 3 is index of xnData
        xnData[4] = 0; // 4 is index of xnData
        // 按照卡分组，sendSize 、sendOffset、recvSize、recvOffset  以字节为单位 * DataSzie(DataType)
        // ccu xn5
        AssembleHcclMsgExtForCCU(ccuParam);
        xnData[5] = reinterpret_cast<uint64_t>(ccuParam.ccuMsgExt) + CCU_MSG_EXT_RANK_OFFSET * ccuParam.alltoallvCnt++;
        return;
    } else if (ccuParam.commParam.commType.prepareType == HcclCMDType::HCCL_CMD_ALLTOALL) {
        uint64_t sliceSizeAlltoall = ccuParam.commParam.count * DATA_TYPE_MAP[ccuParam.commParam.dataType];
        xnData[3] = sliceSizeAlltoall;
        xnData[4] = sliceSizeAlltoall;
        xnData[5] = (uint64_t)ccuParam.commParam.sendBuf;
        xnData[6] = sliceSizeAlltoall * ccuParam.rankId;
        return;
    }

    uint64_t tmpCount = ccuParam.commParam.count / ccuParam.rankNum;
    uint64_t loopCount = CCU_LOOP_COUNT;
    uint64_t sliceCount = GetSliceCount(ccuParam, tmpCount);
    uint64_t sliceSize = sliceCount * DATA_TYPE_MAP[ccuParam.commParam.dataType];

    if (ccuParam.commParam.commType.prepareType == HcclCMDType::HCCL_CMD_HALF_ALLTOALLV) {
        xnData[1] = reinterpret_cast<uint64_t>(ccuParam.commParam.sendBuf); // 1 is index of xnData
        xnData[2] = ccuParam.commParam.wParamExt.sendSizes; // 2 is index of xnData
        xnData[8] = ccuParam.commParam.wParamExt.remoteWinOffset; // 8 is index of xnData
        sliceSize = ccuParam.commParam.count;
        loopCount = CCU_LOOP_COUNT_ATAVW;
    }

    uint64_t loopSize = loopCount * CCU_MEMSLICE_SIZE;
    uint64_t m = sliceSize / loopSize;
    uint64_t n = (sliceSize - m * loopSize) / CCU_MEMSLICE_SIZE;
    uint64_t p = sliceSize - m * loopSize - n * CCU_MEMSLICE_SIZE;
    KERNEL_LOG(KERNEL_INFO, "ApiClient CCUPrepare loopSize:%d, m:%d, n:%d, p:%d", loopSize, m, n, p);

    auto dataSize = DATA_TYPE_MAP[static_cast<uint64_t>(ccuParam.commParam.dataType)];
    if (ccuParam.commParam.commType.prepareType == HcclCMDType::HCCL_CMD_ALLREDUCE) {
        xnData[3] = (ccuParam.commParam.strideCount == 0) ? tmpCount * dataSize * ccuParam.rankId : 
                    (ccuParam.commParam.strideCount * dataSize * ccuParam.rankId); // 3 is index of xnData
    } else if (ccuParam.commParam.commType.prepareType == HcclCMDType::HCCL_CMD_HALF_ALLTOALLV) {
        xnData[3] = ccuParam.commParam.wParamExt.sendOffsets;// 3 is index of xnData
    } else {
        xnData[3] = (ccuParam.commParam.strideCount == 0) ? sliceSize * ccuParam.rankId : 
                    (ccuParam.commParam.strideCount * dataSize * ccuParam.rankId);// 3 is index of xnData
    }
    xnData[4] = loopSize * m; // 4 is index of xnData
    xnData[5] = m; // 5 is index of xnData
    SetupCCUDataPartB(ccuParam, xnData, m, n, p);
}
}

#endif