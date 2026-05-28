/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <gtest/gtest.h>
#include <vector>
#define private public
#include "kernel_operator.h"

using namespace std;
using namespace AscendC;
using namespace HcclApi;

namespace {
class HcclSuiteAICPU : public testing::Test {
protected:
    virtual void SetUp(){
        AscendC::SetGCoreType(1);
    }
    virtual void TearDown(){
        AscendC::SetGCoreType(0);
    }
};

constexpr uint32_t RANK_NUM = 4U;
constexpr size_t WORKSPACE_SIZE = sizeof(HcclMsgArea);
constexpr uint32_t REPEAT_TIME_3 = 3U;

OpResCtx GetOpResCtx(const vector<uint8_t> &workSpace) {
    OpResCtx opResCtx{
        1,
        reinterpret_cast<uintptr_t>(workSpace.data()),
        WORKSPACE_SIZE,
        0,
        RANK_NUM,
        {{100, 0x1000}, {200, 0x2000}},
    };
    return opResCtx;
}

static HcclMsgArea *GetHcclMsgArea(uint8_t *workspaceGM) {
    uint64_t msgAddr = reinterpret_cast<uintptr_t>(workspaceGM);
    if (msgAddr & 0x1ff) {
        msgAddr = (msgAddr & (~((uint64_t)0x1ff))) + 0x200;
    }
    return reinterpret_cast<HcclMsgArea *>(msgAddr);
}

class Mc2InitTilingAicpuTest {
    uint32_t version = 0U;
    uint32_t mc2HcommCnt = 0U;
    uint32_t offset[MAX_CC_TILING_NUM] = {0U};
    uint8_t debugMode = 0U;
    uint8_t preparePosition = 0U;
    uint16_t queueNum = 0U;
    uint16_t commBlockNum = 0U;
    uint8_t devType = 0U;
    char reserved[17] = {0U};
};

struct Mc2CcTilingAicpuTest {
    uint8_t skipLocalRankCopy;
    uint8_t skipBufferWindowCopy;
    uint8_t stepSize = 1U;
    uint8_t version;
    char reserved[12];
    char groupName[128];
    char algConfig[128];
    uint32_t opType;
    uint32_t reduceType;
};

class Mc2TilingAicpuTest {
    Mc2InitTilingAicpuTest init;
    Mc2CcTilingAicpuTest cc;
};

TEST_F(HcclSuiteAICPU, AllGather_repeat1_prepare1_commit1_wait1_success)
{
    std::vector<uint8_t> workSpace(WORKSPACE_SIZE + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    OpResCtx opResCtx = GetOpResCtx(workSpace);
    Mc2TilingAicpuTest tilingData;
    tilingData.cc.opType = (uint32_t)HcclCMDType::HCCL_CMD_ALLGATHER;

    Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU> hccl;
    hccl.InitV2(reinterpret_cast<GM_ADDR>(&opResCtx), static_cast<const void*>(&tilingData));
    EXPECT_EQ(hccl.GetRankDim(), 4);
    auto ret = hccl.SetCcTilingV2(sizeof(Mc2InitTilingAicpuTest));
    EXPECT_EQ(ret, 0);
    HcclHandle handleId = hccl.AllGather(reinterpret_cast<__gm__ uint8_t*>(0x1234),
                                         reinterpret_cast<__gm__ uint8_t*>(0x4321), 100,
                                         HcclDataType::HCCL_DATA_TYPE_INT8, 0, 1);
    EXPECT_EQ(handleId, 0);
    hccl.Commit(handleId);
    EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
}

TEST_F(HcclSuiteAICPU, AllGather_repeat3_prepare1_commit0_wait3_success)
{
    std::vector<uint8_t> workSpace(WORKSPACE_SIZE + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    OpResCtx opResCtx = GetOpResCtx(workSpace);
    Mc2TilingAicpuTest tilingData;
    tilingData.cc.opType = (uint32_t)HcclCMDType::HCCL_CMD_ALLGATHER;

    Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU> hccl;
    hccl.InitV2(reinterpret_cast<GM_ADDR>(&opResCtx), static_cast<const void*>(&tilingData));
    auto ret = hccl.SetCcTilingV2(sizeof(Mc2InitTilingAicpuTest));
    EXPECT_EQ(ret, 0);
    HcclHandle handleId = hccl.AllGather<true>(reinterpret_cast<__gm__ uint8_t*>(0x1234),
                                         reinterpret_cast<__gm__ uint8_t*>(0x4321), 100,
                                         HcclDataType::HCCL_DATA_TYPE_INT8, 0, 3);
    EXPECT_EQ(handleId, 0);
    for (uint8_t i = 0; i < REPEAT_TIME_3; i++) {
        EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
    }
}

TEST_F(HcclSuiteAICPU, AllReduce_repeat1_prepare1_commit1_wait1_success)
{
    std::vector<uint8_t> workSpace(WORKSPACE_SIZE + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    OpResCtx opResCtx = GetOpResCtx(workSpace);
    Mc2TilingAicpuTest tilingData;
    tilingData.cc.opType = (uint32_t)HcclCMDType::HCCL_CMD_ALLREDUCE;

    Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU> hccl;
    hccl.InitV2(reinterpret_cast<GM_ADDR>(&opResCtx), static_cast<const void*>(&tilingData));
    auto ret = hccl.SetCcTilingV2(sizeof(Mc2InitTilingAicpuTest));
    EXPECT_EQ(ret, 0);
    HcclHandle handleId = hccl.AllReduce(reinterpret_cast<__gm__ uint8_t*>(0x1234),
                                         reinterpret_cast<__gm__ uint8_t*>(0x4321), 100,
                                         HcclDataType::HCCL_DATA_TYPE_INT8,
                                         HcclReduceOp::HCCL_REDUCE_SUM, 1);
    hccl.Commit(handleId);
    EXPECT_EQ(handleId, 0);
    EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
}

TEST_F(HcclSuiteAICPU, AllReduce_repeat3_prepare1_commit0_wait3_success)
{
    std::vector<uint8_t> workSpace(WORKSPACE_SIZE + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    OpResCtx opResCtx = GetOpResCtx(workSpace);
    Mc2TilingAicpuTest tilingData;
    tilingData.cc.opType = (uint32_t)HcclCMDType::HCCL_CMD_ALLREDUCE;

    Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU> hccl;
    hccl.InitV2(reinterpret_cast<GM_ADDR>(&opResCtx), static_cast<const void*>(&tilingData));
    auto ret = hccl.SetCcTilingV2(sizeof(Mc2InitTilingAicpuTest));
    EXPECT_EQ(ret, 0);
    HcclHandle handleId = hccl.AllReduce<true>(reinterpret_cast<__gm__ uint8_t*>(0x1234),
                                         reinterpret_cast<__gm__ uint8_t*>(0x4321), 100,
                                         HcclDataType::HCCL_DATA_TYPE_INT8,
                                         HcclReduceOp::HCCL_REDUCE_SUM, 3);
    EXPECT_EQ(handleId, 0);
    for (uint8_t i = 0; i < REPEAT_TIME_3; i++) {
        EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
    }
}

TEST_F(HcclSuiteAICPU, ReduceScatter_repeat1_prepare1_commit1_wait1_success)
{
    std::vector<uint8_t> workSpace(WORKSPACE_SIZE + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    OpResCtx opResCtx = GetOpResCtx(workSpace);
    Mc2TilingAicpuTest tilingData;
    tilingData.cc.opType = (uint32_t)HcclCMDType::HCCL_CMD_REDUCE_SCATTER;

    Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU> hccl;
    hccl.InitV2(reinterpret_cast<GM_ADDR>(&opResCtx), static_cast<const void*>(&tilingData));
    auto ret = hccl.SetCcTilingV2(sizeof(Mc2InitTilingAicpuTest));
    EXPECT_EQ(ret, 0);
    HcclHandle handleId = hccl.ReduceScatter(reinterpret_cast<__gm__ uint8_t*>(0x1234),
                                         reinterpret_cast<__gm__ uint8_t*>(0x4321), 100,
                                         HcclDataType::HCCL_DATA_TYPE_INT8, 
                                         HcclReduceOp::HCCL_REDUCE_SUM, 0, 1);
    hccl.Commit(handleId);
    EXPECT_EQ(handleId, 0);
    EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
}

TEST_F(HcclSuiteAICPU, ReduceScatter_repeat3_prepare1_commit0_wait3_success)
{
    std::vector<uint8_t> workSpace(WORKSPACE_SIZE + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    OpResCtx opResCtx = GetOpResCtx(workSpace);
    Mc2TilingAicpuTest tilingData;
    tilingData.cc.opType = (uint32_t)HcclCMDType::HCCL_CMD_REDUCE_SCATTER;

    Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU> hccl;
    hccl.InitV2(reinterpret_cast<GM_ADDR>(&opResCtx), static_cast<const void*>(&tilingData));
    auto ret = hccl.SetCcTilingV2(sizeof(Mc2InitTilingAicpuTest));
    EXPECT_EQ(ret, 0);
    HcclHandle handleId = hccl.ReduceScatter<true>(reinterpret_cast<__gm__ uint8_t*>(0x1234),
                                         reinterpret_cast<__gm__ uint8_t*>(0x4321), 100,
                                         HcclDataType::HCCL_DATA_TYPE_INT8,
                                         HcclReduceOp::HCCL_REDUCE_SUM, 0, 3);
    EXPECT_EQ(handleId, 0);
    for (uint8_t i = 0; i < REPEAT_TIME_3; i++) {
        EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
    }
}

TEST_F(HcclSuiteAICPU, AlltoAll_repeat1_prepare1_commit1_wait1_success)
{
    std::vector<uint8_t> workSpace(WORKSPACE_SIZE + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    OpResCtx opResCtx = GetOpResCtx(workSpace);
    Mc2TilingAicpuTest tilingData;
    tilingData.cc.opType = (uint32_t)HcclCMDType::HCCL_CMD_ALLTOALL;

    Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU> hccl;
    hccl.InitV2(reinterpret_cast<GM_ADDR>(&opResCtx), static_cast<const void*>(&tilingData));
    auto ret = hccl.SetCcTilingV2(sizeof(Mc2InitTilingAicpuTest));
    EXPECT_EQ(ret, 0);
    HcclHandle handleId = hccl.AlltoAll(reinterpret_cast<__gm__ uint8_t*>(0x1234),
                                         reinterpret_cast<__gm__ uint8_t*>(0x4321), 100,
                                         HcclDataType::HCCL_DATA_TYPE_INT8, 0, 1);
    hccl.Commit(handleId);
    EXPECT_EQ(handleId, 0);
    EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
}

TEST_F(HcclSuiteAICPU, AlltoAll_repeat3_prepare1_commit0_wait3_success)
{
    std::vector<uint8_t> workSpace(WORKSPACE_SIZE + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    OpResCtx opResCtx = GetOpResCtx(workSpace);
    Mc2TilingAicpuTest tilingData;
    tilingData.cc.opType = (uint32_t)HcclCMDType::HCCL_CMD_ALLTOALL;

    Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU> hccl;
    hccl.InitV2(reinterpret_cast<GM_ADDR>(&opResCtx), static_cast<const void*>(&tilingData));
    auto ret = hccl.SetCcTilingV2(sizeof(Mc2InitTilingAicpuTest));
    EXPECT_EQ(ret, 0);
    HcclHandle handleId = hccl.AlltoAll<true>(reinterpret_cast<__gm__ uint8_t*>(0x1234),
                                         reinterpret_cast<__gm__ uint8_t*>(0x4321), 100,
                                         HcclDataType::HCCL_DATA_TYPE_INT8, 0, 3);
    EXPECT_EQ(handleId, 0);
    for (uint8_t i = 0; i < REPEAT_TIME_3; i++) {
        EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
    }
}

TEST_F(HcclSuiteAICPU, AlltoAllV_repeat1_prepare1_commit1_wait1_success)
{
    std::vector<uint8_t> workSpace(WORKSPACE_SIZE + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    OpResCtx opResCtx = GetOpResCtx(workSpace);
    Mc2TilingAicpuTest tilingData;
    tilingData.cc.opType = (uint32_t)HcclCMDType::HCCL_CMD_ALLTOALLV;

    uint64_t sendCounts[RANK_NUM] ={0};
    uint64_t sDisplacements[RANK_NUM] ={0};
    uint64_t recvCounts[RANK_NUM] ={0};
    uint64_t rDisplacements[RANK_NUM] ={0};

    Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU> hccl;
    hccl.InitV2(reinterpret_cast<GM_ADDR>(&opResCtx), static_cast<const void*>(&tilingData));
    auto ret = hccl.SetCcTilingV2(sizeof(Mc2InitTilingAicpuTest));
    EXPECT_EQ(ret, 0);
    if (hccl.GetRankId() == 0) {
        sendCounts[0] = 3; sendCounts[1] = 3; sendCounts[2] = 3; sendCounts[3] = 3;
        sDisplacements[1] = 3; sDisplacements[2] = 6; sDisplacements[3] = 9;
        recvCounts[0] = 3; recvCounts[1] = 2; recvCounts[2] = 1; recvCounts[3] = 3;
        rDisplacements[1] = 3; rDisplacements[2] = 5; rDisplacements[3] = 6;
    }
    HcclHandle handleId = hccl.AlltoAllV(reinterpret_cast<__gm__ uint8_t*>(0x1234),
                                        sendCounts, sDisplacements, HcclDataType::HCCL_DATA_TYPE_INT8,
                                        reinterpret_cast<__gm__ uint8_t*>(0x4321),
                                        recvCounts, rDisplacements, HcclDataType::HCCL_DATA_TYPE_INT8);
    hccl.Commit(handleId);
    EXPECT_EQ(handleId, 0);
    EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
}

TEST_F(HcclSuiteAICPU, AlltoAllV_repeat3_prepare1_commit0_wait3_success)
{
    std::vector<uint8_t> workSpace(WORKSPACE_SIZE + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    OpResCtx opResCtx = GetOpResCtx(workSpace);
    Mc2TilingAicpuTest tilingData;
    tilingData.cc.opType = (uint32_t)HcclCMDType::HCCL_CMD_ALLTOALLV;

    uint32_t sendCounts[RANK_NUM] ={0};
    uint32_t sDisplacements[RANK_NUM] ={0};
    uint32_t recvCounts[RANK_NUM] ={0};
    uint32_t rDisplacements[RANK_NUM] ={0};

    Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU> hccl;
    hccl.InitV2(reinterpret_cast<GM_ADDR>(&opResCtx), static_cast<const void*>(&tilingData));
    auto ret = hccl.SetCcTilingV2(sizeof(Mc2InitTilingAicpuTest));
    EXPECT_EQ(ret, 0);
    if (hccl.GetRankId() == 0) {
        sendCounts[0] = 3; sendCounts[1] = 3; sendCounts[2] = 3; sendCounts[3] = 3;
        sDisplacements[1] = 3; sDisplacements[2] = 6; sDisplacements[3] = 9;
        recvCounts[0] = 3; recvCounts[1] = 2; recvCounts[2] = 1; recvCounts[3] = 3;
        rDisplacements[1] = 3; rDisplacements[2] = 5; rDisplacements[3] = 6;
    }
    HcclHandle handleId = hccl.AlltoAllV<true>(reinterpret_cast<__gm__ uint8_t*>(0x1234),
                                        sendCounts, sDisplacements, HcclDataType::HCCL_DATA_TYPE_INT8,
                                        reinterpret_cast<__gm__ uint8_t*>(0x4321),
                                        recvCounts, rDisplacements, HcclDataType::HCCL_DATA_TYPE_INT8, 3);
    EXPECT_EQ(handleId, 0);
    for (uint8_t i = 0; i < REPEAT_TIME_3; i++) {
        EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
    }
}

TEST_F(HcclSuiteAICPU, PrimitiveIdStateInControlMsg_success)
{
    std::vector<uint8_t> workSpace(WORKSPACE_SIZE + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    ControlHcclMsg* controlMsg = &hcclMsgArea->controlMsg;

    controlMsg->reserved[HCCL_CONTROL_RESERVED_PRIMITIVE_ID_IDX] = 9U;
    controlMsg->reserved[HCCL_CONTROL_RESERVED_PRIMITIVE_RESET_IDX] = 1U;
    ResetPrimitiveIdStateInControlMsg(controlMsg);
    EXPECT_EQ(controlMsg->reserved[HCCL_CONTROL_RESERVED_PRIMITIVE_ID_IDX], 0U);
    EXPECT_EQ(controlMsg->reserved[HCCL_CONTROL_RESERVED_PRIMITIVE_RESET_IDX], 0U);

    EXPECT_EQ(FetchAndIncPrimitiveIdInControlMsg(controlMsg), 0U);
    EXPECT_EQ(FetchAndIncPrimitiveIdInControlMsg(controlMsg), 1U);
    EXPECT_EQ(controlMsg->reserved[HCCL_CONTROL_RESERVED_PRIMITIVE_ID_IDX], 2U);

    controlMsg->reserved[HCCL_CONTROL_RESERVED_PRIMITIVE_ID_IDX] = 7U;
    controlMsg->reserved[HCCL_CONTROL_RESERVED_PRIMITIVE_RESET_IDX] = 0U;
    ResetPrimitiveIdOnceInControlMsg(controlMsg);
    EXPECT_EQ(controlMsg->reserved[HCCL_CONTROL_RESERVED_PRIMITIVE_ID_IDX], 0U);
    EXPECT_EQ(controlMsg->reserved[HCCL_CONTROL_RESERVED_PRIMITIVE_RESET_IDX], 1U);

    controlMsg->reserved[HCCL_CONTROL_RESERVED_PRIMITIVE_ID_IDX] = 5U;
    ResetPrimitiveIdOnceInControlMsg(controlMsg);
    EXPECT_EQ(controlMsg->reserved[HCCL_CONTROL_RESERVED_PRIMITIVE_ID_IDX], 5U);
    EXPECT_EQ(controlMsg->reserved[HCCL_CONTROL_RESERVED_PRIMITIVE_RESET_IDX], 1U);
}

TEST_F(HcclSuiteAICPU, AssembleHcclMsgV2_deprecatedTiling_success)
{
    std::vector<uint8_t> workSpace(WORKSPACE_SIZE + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    ControlHcclMsg* controlMsg = &hcclMsgArea->controlMsg;
    HcclMsg* sendMsg = &hcclMsgArea->commMsg.singleMsg.sendMsgs[0];
    CommonPrepareParam param{};
    param.commType.prepareType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    param.sendBuf = reinterpret_cast<GM_ADDR>(0x1234);
    param.recvBuf = reinterpret_cast<GM_ADDR>(0x4321);
    param.count = 16U;
    param.dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    param.op = HcclReduceOp::HCCL_REDUCE_SUM;
    param.strideCount = 8U;
    param.repeat = 3U;
    controlMsg->resetSeq = 1U;
    controlMsg->reserved[HCCL_CONTROL_RESERVED_PRIMITIVE_ID_IDX] = 11U;
    controlMsg->reserved[HCCL_CONTROL_RESERVED_PRIMITIVE_RESET_IDX] = 0U;

    AssembleHcclMsgV2(param, HcclTilingVersion::DEPRECATED_TILING_VERSION, 2, 0UL, sendMsg, controlMsg);

    EXPECT_EQ(controlMsg->resetSeq, 0U);
    EXPECT_EQ(controlMsg->reserved[HCCL_CONTROL_RESERVED_PRIMITIVE_ID_IDX], 1U);
    EXPECT_EQ(controlMsg->reserved[HCCL_CONTROL_RESERVED_PRIMITIVE_RESET_IDX], 1U);
    EXPECT_EQ(sendMsg->commType.prepareType, HcclCMDType::HCCL_CMD_ALLREDUCE);
    EXPECT_EQ(sendMsg->opType, HcclReduceOp::HCCL_REDUCE_SUM);
    EXPECT_EQ(sendMsg->sendBuffer, 0x1234UL);
    EXPECT_EQ(sendMsg->recvBuffer, 0x4321UL);
    EXPECT_EQ(sendMsg->dataCnt, 16U);
    EXPECT_EQ(sendMsg->strideCount, 8U);
    EXPECT_EQ(sendMsg->addMsg.v0Msg.hcclDataType, HcclDataType::HCCL_DATA_TYPE_INT8);
    EXPECT_EQ(sendMsg->addMsg.v0Msg.repeatCnt, 3U);
    EXPECT_EQ(sendMsg->addMsg.v0Msg.selfHandleID, 2);
    EXPECT_EQ(sendMsg->addMsg.v0Msg.seqNum, 0U);
    EXPECT_EQ(sendMsg->addMsg.v0Msg.version, HcclTilingVersion::DEPRECATED_TILING_VERSION);
    EXPECT_EQ(sendMsg->addMsg.v0Msg.valid, HCCL_MSG_VALID_MASK);
}

TEST_F(HcclSuiteAICPU, AssembleHcclMsgV2_contextDecoupleAndFinalize_success)
{
    std::vector<uint8_t> workSpace(WORKSPACE_SIZE + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    ControlHcclMsg* controlMsg = &hcclMsgArea->controlMsg;
    HcclMsg* sendMsg = &hcclMsgArea->commMsg.singleMsg.sendMsgs[0];
    HcclMsg* finalizeMsg = &hcclMsgArea->commMsg.singleMsg.sendMsgs[1];
    CommonPrepareParam param{};
    param.commType.prepareType = HcclCMDType::HCCL_CMD_ALLGATHER;
    param.sendBuf = reinterpret_cast<GM_ADDR>(0x2345);
    param.recvBuf = reinterpret_cast<GM_ADDR>(0x5432);
    param.count = 32U;
    param.dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    param.strideCount = 4U;
    param.repeat = 2U;
    controlMsg->reserved[HCCL_CONTROL_RESERVED_PRIMITIVE_ID_IDX] = 6U;
    controlMsg->reserved[HCCL_CONTROL_RESERVED_PRIMITIVE_RESET_IDX] = 1U;

    AssembleHcclMsgV2(param, HcclTilingVersion::CONTEXT_DECOUPLE_VERSION, 3, 0x1000UL, sendMsg, controlMsg);

    EXPECT_EQ(controlMsg->reserved[HCCL_CONTROL_RESERVED_PRIMITIVE_ID_IDX], 7U);
    EXPECT_EQ(sendMsg->commType.prepareType, HcclCMDType::HCCL_CMD_ALLGATHER);
    EXPECT_EQ(sendMsg->addMsg.v1Msg.ccOpTilingData, 0x1000UL);
    EXPECT_EQ(sendMsg->addMsg.v1Msg.hcclDataType, HcclDataType::HCCL_DATA_TYPE_INT8);
    EXPECT_EQ(sendMsg->addMsg.v1Msg.repeatCnt, 2U);
    EXPECT_EQ(sendMsg->addMsg.v1Msg.selfHandleID, 3);
    EXPECT_EQ(sendMsg->addMsg.v1Msg.seqNum, 6U);
    EXPECT_EQ(sendMsg->addMsg.v1Msg.version, HcclTilingVersion::CONTEXT_DECOUPLE_VERSION);
    EXPECT_EQ(sendMsg->addMsg.v1Msg.valid, HCCL_MSG_VALID_MASK);

    CommonPrepareParam finalizeParam{};
    finalizeParam.commType.msgType = ControlMsgType::HCCL_CMD_FINALIZE;
    AssembleHcclMsgV2(finalizeParam, HcclTilingVersion::CONTEXT_DECOUPLE_VERSION, 0, 0UL, finalizeMsg, controlMsg);

    EXPECT_EQ(controlMsg->reserved[HCCL_CONTROL_RESERVED_PRIMITIVE_ID_IDX], 0U);
    EXPECT_EQ(controlMsg->reserved[HCCL_CONTROL_RESERVED_PRIMITIVE_RESET_IDX], 0U);
    EXPECT_EQ(finalizeMsg->commType.msgType, ControlMsgType::HCCL_CMD_FINALIZE);
    EXPECT_EQ(finalizeMsg->addMsg.v0Msg.valid, HCCL_MSG_VALID_MASK);
}
}
