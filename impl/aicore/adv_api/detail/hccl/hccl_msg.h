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
 * \file hccl_msg.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_HCCL_HCCL_MSG_H
#define AICORE_ADV_API_DETAIL_HCCL_HCCL_MSG_H

namespace AscendC {
constexpr int32_t HCCL_FAILED = -1;
constexpr int32_t HCCL_SUCCESS = 0;
constexpr int32_t HCCL_MAX_HANDLE_ID = 63;
constexpr int8_t INVALID_HANDLE_ID = -1;
constexpr int8_t INVALID_MSG_POSITION = -1;

constexpr uint32_t HCCL_MAX_RANK_NUM = 32U;
constexpr uint32_t HCCL_MAX_RANK_NUM_V2 = 256;
constexpr uint32_t HCCL_MSG_CNT = 64;
constexpr uint32_t HCCL_MSG_VALID_MASK = 0x5CDF123A;

constexpr uint32_t HCCL_CCTILING_SIZE = 280;
constexpr uint32_t HCCL_CMD_TYPE_OFFSET = HCCL_CCTILING_SIZE - 8;
constexpr uint32_t HCCL_ALG_NAME_OFFSET = HCCL_CMD_TYPE_OFFSET - 128U;
constexpr uint32_t HCCL_STEP_SIZE_OFFSET = 2U;
constexpr uint32_t HCCL_DEBUG_MODE_OFFSET = 40U;
constexpr uint32_t HCCL_QUEUE_NUM_OFFSET = 42U;

constexpr uint32_t U64_CNT_PER_CACHELINE = 8U;
constexpr uint8_t HCCL_MSG_EXT_RESERVED_CNT = 6U;
constexpr uint32_t HCCL_VALID_POS = 12U;
constexpr uint32_t HCCL_MSG_DATA_CNT = 16U;
constexpr uint32_t HCCL_DATA_TYPE_MAP = 18U;
constexpr uint8_t HCCL_ONLY_COMPUTE = 1U;

constexpr uint32_t MAX_DCCI_CNT = 64;

constexpr uint32_t HCCL_DATACOPY_MAX_CNT = 64; // 64 msgs used max
constexpr uint64_t DATA_TYPE_MAP[HCCL_DATA_TYPE_MAP] = {1, 2, 4, 2, 4, 8, 8, 1, 8, 4, 8, 2, 0, 0, 1, 1, 1, 0};

// Used to calc xor checksum for HcclMsg
struct DataBlock {
    uint32_t data[HCCL_MSG_DATA_CNT];
};

// 32 bytes aligned if using ubuf and dma to send/recv
// 64 bytes aligned if using scalar to write/read
struct V0MsgAdditionInfo {
    HcclDataType hcclDataType;
    uint32_t p2pSrcDestRankId;  // RankId of the peer end of send/recv, destRank for send, srcRank for recv
    uint32_t valid;             // msg valid when setting as HCCL_MSG_VALID_MASK
    uint8_t repeatCnt;          // The number of comm task launched by this msg is repeatCnt. The default is 1.
    uint8_t everyTurnRsp;       // Wait for the current turn to finish and a response before the next turn is executed
    uint8_t everyTurnWait;      // Each turn needs to wait for the work message before execution
    int8_t commDepGroupID;      // The comm group id that needs to wait for the execution of this msg. -1 default,
                                // indicating no need to wait.
    HcclHandle commDepHandleID; // The comm task of handleId needed to wait for the execution of this msg. -1 default,
                                // indicating no need to wait.
    HcclHandle selfHandleID;    // handleId of this comm msg, -1 for control msg.
    uint8_t seqNum;
    uint8_t version;
    uint32_t xorCheck; // xor checksum
};

struct V1MsgAdditionInfo {
    uint64_t ccOpTilingData;
    uint32_t valid; // msg valid when setting as HCCL_MSG_VALID_MASK
    HcclDataType hcclDataType;
    uint8_t repeatCnt;       // The number of comm task launched by this msg is repeatCnt. The default is 1.
    HcclHandle selfHandleID; // handleId of this comm msg, -1 for control msg.
    uint8_t seqNum;
    uint8_t version;
    uint32_t xorCheck; // xor checksum
};

struct HcclMsg {
    HcclCMDType commType; // comm primitive type，AllReduce/AllGather.../Finalize/InterHcclGroupSync
    HcclReduceOp opType;  // reduce op type，sum/prod/max/min
    uint64_t sendBuffer;  // src buffer addr
    uint64_t recvBuffer;  // dst buffer addr
    uint64_t dataCnt;     // number of data participating in comm task
    uint64_t strideCount; // Communication and computing fusion scenario will involve tiling,
                          // which may lead to data discontinuity.
                          // Thus, use strideCount filed to describe the offset of each data-block
                          // in discontinuous memory.
    union {
        V0MsgAdditionInfo v0Msg;
        V1MsgAdditionInfo v1Msg;
    } addMsg;
};

// HcclMsgExt is only used by AlltoAllV, and is separate from HcclMsg to improve read/write performance of HcclMsg.
// Current HcclMsgExt support 256 ranks max.
// Current size of HcclMsgExt is 8256B, while stack frame size is 32KB limited. Thus, do not define HcclMsgExt object.
struct HcclMsgExt {
    // sendCounts[i] represents the data count sent to rank i by this rank.
    uint64_t sendCounts[HCCL_MAX_RANK_NUM_V2];
    // sendOffset[i] represents the offset count of the data sent to rank i by this rank relative to sendBuf.
    uint64_t sendOffset[HCCL_MAX_RANK_NUM_V2];
    // recvCounts[i] represents the data count received from rank i to this rank.
    uint64_t recvCounts[HCCL_MAX_RANK_NUM_V2];
    // recvOffset[i] represents the offset count of the data received from rank i to this rank relative to recvBuf.
    uint64_t recvOffset[HCCL_MAX_RANK_NUM_V2];
    uint64_t reserved[HCCL_MSG_EXT_RESERVED_CNT]; // cacheline aligned for valid and xorCheck
    uint64_t valid;                               // set by api, reset by server
    uint64_t xorCheck;                            // set by api, checked by server to ensure msg integrity
};

struct AlltoAllVParamExt {
    uint64_t* sendCounts;
    uint64_t* sdispls;
    uint64_t* recvCounts;
    uint64_t* rdispls;
    __aicore__ inline void AssembleHcclMsgExt(uint32_t rankDim, __gm__ HcclMsgExt* dst) const
    {
        uint64_t xorCheck = 0U;
        for (uint32_t i = 0U; i < rankDim; ++i) {
            xorCheck ^= dst->sendCounts[i] = sendCounts[i];
            xorCheck ^= dst->sendOffset[i] = sdispls[i];
            xorCheck ^= dst->recvCounts[i] = recvCounts[i];
            xorCheck ^= dst->recvOffset[i] = rdispls[i];
        }
        dst->xorCheck = (xorCheck ^ HCCL_MSG_VALID_MASK);
        dst->valid = HCCL_MSG_VALID_MASK;
    }
};

struct AlltoAllvWriteParamExt {
    uint64_t sendOffsets;
    uint64_t sendSizes;
    uint64_t remoteWinOffset;
};

constexpr uint64_t COMMIT_VALID_MASK = 987654321U;              // commit msg valid mask
constexpr uint64_t FINALIZE_FINISH_CNT = 1234567899999999999UL; // server write finish msg when all hccl task finished

// cacheline size aligned by 64 bytes
struct TurnCnt {
    uint64_t valid; // COMMIT_VALID_MASK, writen by client when Commit, checked by server
    uint64_t cnt;   // commit cnt, writen by client, reset by server
    uint64_t reserved[6];
};

struct ControlHcclMsg {
    uint8_t restart;
    uint8_t restarting;
    uint8_t restartCnt;
    uint8_t resetSeq;
    uint8_t reserved[60];
};

constexpr uint32_t BYTE_PER_KB = 1024U;
constexpr uint32_t BYTE_PER_MB = BYTE_PER_KB * BYTE_PER_KB;
// Current HcclMsgArea use count mode. Two msg bodies are used, one for read and one for write, to avoid aicore and
// aicpu reading or writing sendcnt/recvcnt at the same time.
// If using msg queue mode, then the state change can be in one msg, because it will not be written simultaneously.
// HcclMsgArea is the 16MB space reserved by workspace in struct HcclCombinOpParam and belongs to each comm group.
struct HcclMsgArea {
    HcclMsg sendMsgs[HCCL_MSG_CNT];
    HcclMsg recvMsgs[HCCL_MSG_CNT];
    uint8_t reserved0[8 * BYTE_PER_KB]; // for abi compatibility

    // commitTurnCnt and sendMsgList correspond one-to-one to inform the server times the task needs to be executed.
    // Ascend 910B and Ascend 310P support repeat>1 scenarios, so the element values are 1~repeat by Commit.
    // ccu does not support repeat>1 scenarios, so the element value can only be written to 1 by Commit.
    // The use of uint64_t is compatible with the requirement that ccu monitor 64bit.
    TurnCnt commitTurnCnt[HCCL_MSG_CNT];   // writen by client, indicating task num needed to be executed.
    TurnCnt finishedTurnCnt[HCCL_MSG_CNT]; // writen by server, indicating task num has been executed.
    uint8_t reserved1[BYTE_PER_MB];
    HcclMsgExt paramExtMsgList[HCCL_MSG_CNT];
    ControlHcclMsg controlMsg;
};

constexpr uint32_t MAX_QUE_NUM = 48U;
constexpr uint32_t PADDING_SIZE = 0xF7000;
struct HcclMsgAreaForMultiQue {
    HcclMsg sendMsgs[MAX_QUE_NUM][HCCL_MSG_CNT];
    TurnCnt commitTurnCnt[MAX_QUE_NUM][HCCL_MSG_CNT];
    TurnCnt finishedTurnCnt[MAX_QUE_NUM][HCCL_MSG_CNT];
    uint8_t pad[PADDING_SIZE];
    ControlHcclMsg controlMsg;
};

struct CommonPrepareParam {
    HcclCMDType commType;
    GM_ADDR sendBuf;
    GM_ADDR recvBuf;
    uint64_t count;
    HcclDataType dataType;
    HcclReduceOp op;
    uint64_t strideCount;
    uint8_t repeat = 1U;
    AlltoAllVParamExt paramExt;       // only used by AlltoAllV
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

struct HcclCombineOpParam {
    uint64_t workSpace;                     // Address for communication between client and server,
                                            // hccl requests and clears
    uint64_t workSpaceSize;                 // Space for communication between client and server
    uint32_t rankId;                        // id of this rank
    uint32_t rankNum;                       // num of ranks in this comm group
    uint64_t winSize;                       // size of each windows memory
    uint64_t windowsIn[HCCL_MAX_RANK_NUM];  // windows address for input, windowsIn[rankId] corresponds
                                            // to the local card address,
                                            // and others are cross-card mapping addresses.
    uint64_t windowsOut[HCCL_MAX_RANK_NUM]; // windows address for output, windowsOut[rankId] corresponds
                                            // to the local card address,
                                            // and others are cross-card mapping addresses.
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
    GM_ADDR XnOffset;
    GM_ADDR CKEOffset;
    uint8_t res[8312];
#else
    uint8_t res[8328];
#endif
    uint8_t multiFlag;
    __gm__ IbVerbsData* data;
};

constexpr uint16_t CCU_CKE_SIZE = 8;
constexpr uint64_t CCU_XN_DATA_SIZE = 8; // 每个xn的字节数
constexpr uint16_t CCU_USED_XN_NUM = 9;  // 当前只用前9个Xn
constexpr uint16_t CCU_MAX_MSG_NUM = 8;  //发给CCU 的消息队列长度为8
constexpr uint16_t CCU_MSG_XN_NUM = 64; // 最大xn数量，每个CCU 的消息体占用8个寄存器，即消息体长度为 64*8B=512B
constexpr uint64_t CCU_LOOP_COUNT = 64;      // CCU 循环数，MC2 不感知
constexpr uint64_t CCU_LOOP_COUNT_ATAVW = 8; // CCU 循环数，only for AlltoAllvWrite
constexpr uint64_t ALIGN_64_BYTE = 64;
constexpr uint64_t CCU_MEMSLICE_SIZE = 4096; // CCU MS 大小，MC2 不感知
struct CCUConfig {                           // AIC 的 HCCL 实例，对应的通信域，的消息内容
    // 映射1个地址
    GM_ADDR XnAddr; // CCU映射出的Xn寄存器的地址, 要求地址连续： 8B 对齐; 总消息体长度为： 864B， 有效范围： 【XnAddr，
                    // XnAddr+864B】
    GM_ADDR CKEAddr; // CCU映射出的CKE寄存器的地址, 要求地址连续： 8B 对齐; 总消息体长度为： 816b， 有效范围：
                     // 【CKEAddr， CKEAddr+8*16b】
    // CKEAddr: [15:0] 有效，其他bit 无效，不能写！ commitCKEAddr AIC写1 CCU 读清； waitCKEAddr CCU写1，AIC读取
};

struct CCUMsg {
    GM_ADDR XnData;        // Msg转换为CCU寄存器值
    GM_ADDR XnAddr;        //
    GM_ADDR commitCKEAddr; //每个msg 对应的 commit 地址
    GM_ADDR waitCKEAddr;   //每个Msg 对应的 wait 地址
};

struct ReduceDataTypeAbility {
    HcclReduceOp op;
    HcclDataType dstDataType;
    HcclDataType srcDataType;
};

template <typename T, int Size>
class CircularFifo {
public:
    __aicore__ CircularFifo() : m_head(0), m_tail(0), m_size(0) {}

    __aicore__ inline bool push(const T& value)
    {
        if (m_size == Size) {
            return false;
        }

        m_buffer[m_tail] = value;
        m_tail = (m_tail + 1) % Size;
        ++m_size;

        return true;
    }

    __aicore__ inline bool pop(T& value)
    {
        if (m_size == 0) {
            return false;
        }

        value = m_buffer[m_head];
        m_head = (m_head + 1) % Size;
        --m_size;

        return true;
    }

    __aicore__ inline bool isFull() const
    {
        return m_size == Size;
    }

    __aicore__ inline bool isEmpty() const
    {
        return m_size == 0;
    }

    __aicore__ inline T Head() const
    {
        return m_buffer[m_head];
    }

    __aicore__ inline T Tail() const
    {
        return m_buffer[m_tail];
    }

public:
    int m_head;
    int m_tail;

private:
    T m_buffer[Size];
    int m_size;
};

struct CCUCommOp {
    int8_t resourceId; // 是否分配资源
    int8_t isFinish;   // 通信是否已经完成
    uint8_t finishedCnt;
    uint8_t repeatCnt;
    uint8_t commitCnt;
    uint8_t waitCnt;
    uint64_t xnData[CCU_USED_XN_NUM]; // 只需要存转换后的数据
};

// AllToAllv HcclMsgExt trans for ccu
struct CCUMsgExt {
    uint64_t sendSize;
    uint64_t sendOffset;
    uint64_t recvSize;
    uint64_t recvOffset;
};

constexpr uint64_t CCU_MSG_EXT_RANK_OFFSET = sizeof(CCUMsgExt) * HCCL_MAX_RANK_NUM_V2;
constexpr uint64_t CCU_MSG_EXT_MAX_OFFSET = CCU_MSG_EXT_RANK_OFFSET * HCCL_MSG_CNT;

} // namespace AscendC

#endif // AICORE_ADV_API_DETAIL_HCCL_HCCL_MSG_H