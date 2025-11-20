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
 * \file hccl_inner_def.h
 * \brief
 */
#ifndef IMPL_HCCL_HCCL_INNER_DEF_H
#define IMPL_HCCL_HCCL_INNER_DEF_H

#include "include/adv_api/hccl/internal/hccl_msg.h"
#include "include/adv_api/hccl/internal/hccl_tiling_msg.h"

using namespace HcclApi;
namespace AscendC {
constexpr int32_t HCCL_FAILED = -1;
constexpr int32_t HCCL_SUCCESS = 0;
constexpr uint8_t HCCL_ONLY_COMPUTE = 1U;
constexpr uint8_t HCCL_ASCEND910B = 1U;
constexpr uint32_t MAX_DCCI_CNT = 64U;
constexpr uint64_t DATA_TYPE_MAP[] = {1, 2, 4, 2, 4, 8, 8, 1, 8, 4, 8, 2, 0, 0, 1, 1, 1, 1, 0};

// Used to calc xor checksum for HcclMsg
constexpr uint32_t HCCL_MSG_DATA_CNT = 16U;
struct DataBlock {
    uint32_t data[HCCL_MSG_DATA_CNT];
};

struct AlltoAllvWriteParamExt {
    uint64_t sendOffsets;
    uint64_t sendSizes;
    uint64_t remoteWinOffset;
};

struct CommonPrepareParam {
    HcclCommType commType;
    GM_ADDR sendBuf;
    GM_ADDR recvBuf;
    uint64_t count;
    HcclDataType dataType;
    HcclDataType dstDataType;
    HcclReduceOp op;
    uint64_t strideCount;
    uint8_t repeat = 1U;
    AlltoAllVParamExt paramExt; // only used by AlltoAllV
    AlltoAllvWriteParamExt wParamExt; // only used by AlltoAllvWrite
};

struct MemDetails {
    uint64_t size;
    uint64_t addr;
    uint32_t key;
};

struct IbVerbsData {
    MemDetails remoteInput;
    MemDetails remoteOutput;
    MemDetails localInput;
    MemDetails localOutput;
    uint8_t res[24];
};

constexpr uint32_t HCCL_MAX_RANK_NUM = 32U;
struct HcclCombineOpParam {
    uint64_t workSpace;                         // Address for communication between client and server,
                                                // hccl requests and clears
    uint64_t workSpaceSize;                     // Space for communication between client and server
    uint32_t rankId;                            // id of this rank
    uint32_t rankNum;                           // num of ranks in this comm group
    uint64_t winSize;                           // size of each windows memory
    uint64_t windowsIn[HCCL_MAX_RANK_NUM];      // windows address for input, windowsIn[rankId] corresponds
                                                // to the local card address,
                                                // and others are cross-card mapping addresses.
    uint64_t windowsOut[HCCL_MAX_RANK_NUM];     // windows address for output, windowsOut[rankId] corresponds
                                                // to the local card address,
                                                // and others are cross-card mapping addresses.
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3101
    GM_ADDR xnOffset;
    GM_ADDR ckeOffset;
    uint8_t res[8312];
#else
    uint8_t res[8328];
#endif
    uint8_t multiFlag;
    __gm__ IbVerbsData *data;
};

namespace HcclContextDef {
struct HcclRankRelationResV2 {
    uint32_t remoteUsrRankId = 0;
    uint32_t remoteWorldRank = 0;
    uint64_t windowsIn = 0;
    uint64_t windowsOut = 0;
    uint64_t windowsExp = 0;
};

struct RemoteResPtr {
    HcclRankRelationResV2 *nextHostPtr;
    HcclRankRelationResV2 *nextDevicePtr;
};

struct HcclOpResParam {
    uint64_t workSpace;
    uint64_t workSpaceSize;
    uint32_t rankId;
    uint32_t rankNum;
    uint64_t winSize;
    uint64_t localWindowsIn;
    uint64_t localWindowsOut;
    char hcomId[128];
    uint64_t winExpSize;
    uint64_t localWindowsExp;
    uint32_t rWinStart;
};
}
constexpr uint16_t CCU_CKE_SIZE = 8;
constexpr uint64_t CCU_XN_DATA_SIZE = 8; // Number of bytes per xn
constexpr uint16_t CCU_USED_XN_NUM = 9;  // Currently only the first 9 xn are used
constexpr uint16_t CCU_MAX_MSG_NUM = 8;  // The message queue length sent to CCU is 8
constexpr uint16_t CCU_MSG_XN_NUM = 64;  // Maximum xn number, each CCU message body occupies 8 registers
                                         // the message body length is 64*8B=512B
constexpr uint64_t CCU_LOOP_COUNT = 64;  // CCU cycle number, MC2 is not aware of it
constexpr uint64_t CCU_LOOP_COUNT_ATAVW = 8; // CCU cycle number, only for AlltoAllvWrite
constexpr uint64_t ALIGN_64_BYTE = 64;
constexpr uint64_t CCU_MEMSLICE_SIZE = 4096; // CCU MS size, MC2 is not aware of it
constexpr uint8_t CCU_MSG_CKE_INIT_VALUE = 0;
constexpr uint8_t CCU_MSG_CKE_SET_VALUE = 1;

struct CCUConfig { // HCCL instance of AIC, corresponding message content of communication domain
    GM_ADDR xnAddr; // The address of the Xn register mapped by the CCU requires the address to be continuous and
                    // 8B aligned; the total message body length is 864B, and the valid range is [xnAddr, xnAddr+864B]
    GM_ADDR ckeAddr; // The address of the CKE register mapped by the CCU requires the address to be continuous and
                     // 8B aligned; the total message body length is 816b, and the valid range is [CKEAddr, ckeAddr+8*16b]
    // ckeAddr: [15:0] is valid, other bits are invalid and cannot be written! 
    // commitCKEAddr AIC writes 1 CCU reads clear; waitCKEAddr CCU writes 1, AIC reads
};

struct CCUMsg {
    GM_ADDR xnData; // Msg is converted to CCU register value
    GM_ADDR xnAddr;
    GM_ADDR commitCKEAddr; // The commit address corresponding to each msg
    GM_ADDR waitCKEAddr;   // The wait address corresponding to each Msg
};

struct ReduceDataTypeAbility {
    HcclReduceOp op;
    HcclDataType dstDataType;
    HcclDataType srcDataType;
};

template <typename T, int Size>
class CircularFifo {
public:
    __aicore__ CircularFifo() : mHead(0), mTail(0), mSize(0)
    {}

    __aicore__ inline bool push(const T &value)
    {
        if (mSize == Size) {
            return false;
        }

        m_buffer[mTail] = value;
        mTail = (mTail + 1) % Size;
        ++mSize;

        return true;
    }

    __aicore__ inline bool pop(T &value)
    {
        if (mSize == 0) {
            return false;
        }

        value = m_buffer[mHead];
        mHead = (mHead + 1) % Size;
        --mSize;

        return true;
    }

    __aicore__ inline bool isFull() const
    {
        return mSize == Size;
    }

    __aicore__ inline bool isEmpty() const
    {
        return mSize == 0;
    }

    __aicore__ inline T Head() const
    {
        return m_buffer[mHead];
    }

    __aicore__ inline T Tail() const
    {
        return m_buffer[mTail];
    }

public:
    int mHead;
    int mTail;

private:
    T m_buffer[Size];
    int mSize;
};

struct CCUMsgExt { // AllToAllv HcclMsgExt trans for ccu
    uint64_t sendSize;
    uint64_t sendOffset;
    uint64_t recvSize;
    uint64_t recvOffset;
};

struct CCUMsgCommOp {
    int8_t resourceId;
    int8_t isFinish;
    uint8_t reserved[6];
    uint64_t xnData[CCU_USED_XN_NUM];
};

struct HandleCommOp {
    uint8_t reqId;
    uint8_t repeatCnt;
    uint8_t commitCnt;
    uint8_t waitCnt;
    uint8_t finishCnt;
    uint8_t reserved[3];
};

struct CCUParam {
    uint32_t rankNum;
    uint32_t rankId;
    CommonPrepareParam commParam;
    uint32_t repeatIndex;
    uint8_t alltoallvCnt = 0;
    __gm__ CCUMsgExt *ccuMsgExt;
};

constexpr uint64_t CCU_MSG_EXT_RANK_OFFSET = sizeof(CCUMsgExt) * HCCL_MAX_RANK_NUM_V2;
constexpr uint64_t CCU_MSG_EXT_MAX_OFFSET = CCU_MSG_EXT_RANK_OFFSET * HCCL_MSG_CNT;
} // namespace AscendC
#endif // IMPL_HCCL_HCCL_INNER_DEF_H