/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <vector>
#define private public
#define protect public
#include "kernel_operator.h"

using namespace std;
using namespace AscendC;
namespace {
constexpr uint32_t kRankNum = 8U;
constexpr size_t workSpaceSize = sizeof(HcclMsgArea);
HcclCombineOpParam GetHcclCombineOpParam(const vector<uint8_t>& workSpace)
{
    HcclCombineOpParam hcclCombineOpParam{
        reinterpret_cast<uintptr_t>(workSpace.data()), workSpaceSize, 0, kRankNum, 0, {0}, {0}};
    return hcclCombineOpParam;
}

HcclMsgArea* GetHcclMsgArea(uint8_t* workspaceGM)
{
    uint64_t msgAddr = reinterpret_cast<uintptr_t>(workspaceGM);
    if (msgAddr & 0x1ff) {
        msgAddr = (msgAddr & (~((uint64_t)0x1ff))) + 0x200;
    }
    return reinterpret_cast<HcclMsgArea*>(msgAddr);
}

uint32_t GenXorForHcclMsg(void* msg)
{
    DataBlock* block = reinterpret_cast<DataBlock*>(msg);
    constexpr uint32_t kBlockCntForXor = 15U;
    uint32_t xorVal = 0U;
    for (uint32_t i = 0; i < kBlockCntForXor; ++i) { xorVal ^= block->data[i]; }
    return xorVal;
}

uint64_t GenXorForHcclMsgExt(const HcclMsgExt* msgExt, const uint32_t rankNum)
{
    if (msgExt == nullptr) {
        return 0;
    }
    uint64_t xorVal = 0U;
    for (uint32_t i = 0U; i < rankNum; ++i) {
        xorVal ^= msgExt->sendCounts[i];
        xorVal ^= msgExt->sendOffset[i];
        xorVal ^= msgExt->recvCounts[i];
        xorVal ^= msgExt->recvOffset[i];
    }
    xorVal ^= HCCL_MSG_VALID_MASK;
    return xorVal;
}

void AlltoAllVThreadFunc(int blockIdx, HcclCombineOpParam&& hcclCombineOpParam, bool afterWorkBlockIdx = false)
{
    // blockIdx切换+备份
    auto block_idx_backup = block_idx;
    block_idx = blockIdx;
    KERNEL_LOG(
        KERNEL_INFO, "aicore blockIdx=%ld(recoverBlockIdx=%ld) start working...", GetBlockIdx(), block_idx_backup);

    // 线程要执行的代码
    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    std::vector<uint64_t> sendCounts(kRankNum, 10);
    std::vector<uint64_t> recvCounts(kRankNum, 11);
    std::vector<uint64_t> sendOffsets(kRankNum, 12);
    std::vector<uint64_t> recvOffsets(kRankNum, 13);
    HcclHandle handleId = hccl.AlltoAllV(reinterpret_cast<__gm__ uint8_t*>(0x11), sendCounts.data(), sendOffsets.data(),
        HcclDataType::HCCL_DATA_TYPE_INT8, reinterpret_cast<__gm__ uint8_t*>(0x11), recvCounts.data(),
        recvOffsets.data(), HcclDataType::HCCL_DATA_TYPE_INT8);
    EXPECT_EQ(handleId, 0);
    auto hcclMsgArea = GetHcclMsgArea(reinterpret_cast<uint8_t*>(hcclCombineOpParam.workSpace));
    ASSERT_NE(hcclMsgArea, nullptr);
    if (afterWorkBlockIdx) {
        EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.valid, HCCL_MSG_VALID_MASK);
        EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.repeatCnt, 1);
        EXPECT_EQ(hcclMsgArea->sendMsgs[0].dataCnt, 0);
        EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.hcclDataType, HcclReduceOp::HCCL_REDUCE_SUM);
        EXPECT_EQ(hcclMsgArea->sendMsgs[0].strideCount, 0);
        EXPECT_EQ(hcclMsgArea->sendMsgs[0].commType, HcclCMDType::HCCL_CMD_ALLTOALLV);
        EXPECT_EQ(hcclMsgArea->sendMsgs[0].opType, HcclReduceOp::HCCL_REDUCE_RESERVED);
        ASSERT_EQ(hcclMsgArea->paramExtMsgList[0].valid, HCCL_MSG_VALID_MASK);
        ASSERT_EQ(
            hcclMsgArea->paramExtMsgList[0].xorCheck, GenXorForHcclMsgExt(&hcclMsgArea->paramExtMsgList[0], kRankNum));
        for (int32_t i = 0; i < kRankNum; ++i) {
            EXPECT_EQ(hcclMsgArea->paramExtMsgList[0].sendCounts[i], 10);
            EXPECT_EQ(hcclMsgArea->paramExtMsgList[0].recvCounts[i], 11);
            EXPECT_EQ(hcclMsgArea->paramExtMsgList[0].sendOffset[i], 12);
            EXPECT_EQ(hcclMsgArea->paramExtMsgList[0].recvOffset[i], 13);
        }
    } else {
        EXPECT_NE(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.valid, HCCL_MSG_VALID_MASK);
    }

    hccl.Commit(handleId);
    if (afterWorkBlockIdx) {
        EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].cnt, 1);
    } else {
        EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].cnt, 0);
    }

    hcclMsgArea->finishedTurnCnt[0].cnt = 1;
    ASSERT_EQ(hccl.Query(handleId), 1);
    EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);

    // blockIdx恢复
    block_idx = block_idx_backup;
    KERNEL_LOG(KERNEL_INFO, "aicore blockIdx=%ld finished working, and recover to blockIdx=%ld.", GetBlockIdx(),
        block_idx_backup);
}

void FinalizeThreadFunc(int blockIdx, Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU>& hccl)
{
    // blockIdx切换+备份
    auto block_idx_backup = block_idx;
    block_idx = blockIdx;
    KERNEL_LOG(
        KERNEL_INFO, "aicore blockIdx=%ld(recoverBlockIdx=%ld) start Finalize...", GetBlockIdx(), block_idx_backup);

    hccl.Finalize();

    // blockIdx恢复
    block_idx = block_idx_backup;
    KERNEL_LOG(KERNEL_INFO, "aicore blockIdx=%ld finished Finalize, and recover to blockIdx=%ld.", GetBlockIdx(),
        block_idx_backup);
}

void ReadFinalizeMsgThreadFunc(const uint8_t msgPos, HcclMsgArea* hcclMsgArea)
{
    while (hcclMsgArea->sendMsgs[msgPos].addMsg.v0Msg.valid != HCCL_MSG_VALID_MASK) {}
    EXPECT_EQ(hcclMsgArea->sendMsgs[msgPos].addMsg.v0Msg.xorCheck, GenXorForHcclMsg(&hcclMsgArea->sendMsgs[msgPos]));
    hcclMsgArea->sendMsgs[msgPos].addMsg.v0Msg.valid = ~HCCL_MSG_VALID_MASK;
    hcclMsgArea->finishedTurnCnt[msgPos].cnt = FINALIZE_FINISH_CNT;
    KERNEL_LOG(KERNEL_INFO, "Aicpu has read Finalize msg[%u].", msgPos);
}
} // namespace
class HcclCommonTestSuite : public testing::Test {
protected:
    virtual void SetUp()
    {
        blockIdxBak_ = block_idx;
    }
    virtual void TearDown()
    {
        block_idx = blockIdxBak_;
    }

private:
    int64_t blockIdxBak_;
};

class HcclAbnormalTestSuite : public testing::Test {
protected:
    virtual void SetUp()
    {
        blockIdxBak_ = block_idx;
    }
    virtual void TearDown()
    {
        block_idx = blockIdxBak_;
    }

private:
    int64_t blockIdxBak_;
};

class HcclCombineTestSuite : public testing::Test {
protected:
    virtual void SetUp()
    {
        blockIdxBak_ = block_idx;
    }
    virtual void TearDown()
    {
        block_idx = blockIdxBak_;
    }

private:
    int64_t blockIdxBak_;
};

// 测试内容：Prepare1次(AllReduce接口, repeat=1)+Commit1次，校验消息区内容
TEST_F(HcclCommonTestSuite, AllReduce_Repeat1)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    HcclHandle handleId = hccl.AllReduce(reinterpret_cast<__gm__ uint8_t*>(0x11),
        reinterpret_cast<__gm__ uint8_t*>(0x11), 100, HcclDataType::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM);
    EXPECT_EQ(handleId, 0);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.valid, HCCL_MSG_VALID_MASK);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.repeatCnt, 1);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].commType, HcclCMDType::HCCL_CMD_ALLREDUCE);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].opType, HcclReduceOp::HCCL_REDUCE_SUM);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.selfHandleID, handleId);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.xorCheck, GenXorForHcclMsg(&hcclMsgArea->sendMsgs[0]));
    EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].cnt, 0);
    hccl.Commit(handleId);
    EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].valid, COMMIT_VALID_MASK);
    EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].cnt, 1);
    ASSERT_EQ(hccl.Query(handleId), 0);
    hcclMsgArea->finishedTurnCnt[0].cnt++;
    ASSERT_EQ(hccl.Query(handleId), 1);
    EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);

    // hccl ctx接口验证
    EXPECT_EQ(hccl.GetRankDim(), kRankNum);
    EXPECT_EQ(hccl.GetRankId(), 0);
    EXPECT_EQ(hccl.GetWindowsInAddr(0), nullptr);
    EXPECT_EQ(hccl.GetWindowsOutAddr(0), nullptr);
}

// 测试内容：Prepare1次(AllGather接口, repeat=1)+Commit1次，校验消息区内容
TEST_F(HcclCommonTestSuite, AllGather_Repeat1)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    HcclHandle handleId = hccl.AllGather(reinterpret_cast<__gm__ uint8_t*>(0x11),
        reinterpret_cast<__gm__ uint8_t*>(0x11), 100, HcclDataType::HCCL_DATA_TYPE_INT8, 0);
    EXPECT_EQ(handleId, 0);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.valid, HCCL_MSG_VALID_MASK);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.repeatCnt, 1);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].commType, HcclCMDType::HCCL_CMD_ALLGATHER);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].opType, HcclReduceOp::HCCL_REDUCE_RESERVED);
    EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].cnt, 0);
    hccl.Commit(handleId);
    EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].cnt, 1);
    ASSERT_EQ(hccl.Query(handleId), 0);
    hcclMsgArea->finishedTurnCnt[0].cnt++;
    ASSERT_EQ(hccl.Query(handleId), 1);
    EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
}

// 测试内容：Prepare1次(ReduceScatter接口, repeat=1)+Commit1次，校验消息区内容
TEST_F(HcclCommonTestSuite, ReduceScatter_Repeat1)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    HcclHandle handleId =
        hccl.ReduceScatter(reinterpret_cast<__gm__ uint8_t*>(0x11), reinterpret_cast<__gm__ uint8_t*>(0x11), 100,
            HcclDataType::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, 100 * 8);
    EXPECT_EQ(handleId, 0);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.valid, HCCL_MSG_VALID_MASK);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.repeatCnt, 1);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].commType, HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].opType, HcclReduceOp::HCCL_REDUCE_SUM);
    EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].cnt, 0);
    hccl.Commit(handleId);
    EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].cnt, 1);
    ASSERT_EQ(hccl.Query(handleId), 0);
    hcclMsgArea->finishedTurnCnt[0].cnt++;
    ASSERT_EQ(hccl.Query(handleId), 1);
    EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
}

// 测试内容：Prepare1次(AlltoAll接口, repeat=1)+Commit1次，校验消息区内容
TEST_F(HcclCommonTestSuite, AlltoAll_Repeat1)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    HcclHandle handleId = hccl.AlltoAll(reinterpret_cast<__gm__ uint8_t*>(0x11),
        reinterpret_cast<__gm__ uint8_t*>(0x11), 100, HcclDataType::HCCL_DATA_TYPE_INT8, 100 * 8);
    EXPECT_EQ(handleId, 0);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.valid, HCCL_MSG_VALID_MASK);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.repeatCnt, 1);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].dataCnt, 100);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.hcclDataType, HcclDataType::HCCL_DATA_TYPE_INT8);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].strideCount, 100 * 8);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].commType, HcclCMDType::HCCL_CMD_ALLTOALL);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].opType, HcclReduceOp::HCCL_REDUCE_RESERVED);
    EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].cnt, 0);
    hccl.Commit(handleId);
    EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].cnt, 1);
    ASSERT_EQ(hccl.Query(handleId), 0);
    hcclMsgArea->finishedTurnCnt[0].cnt++;
    ASSERT_EQ(hccl.Query(handleId), 1);
    EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
    std::thread t1(FinalizeThreadFunc, 0, std::ref(hccl));
    std::thread t2(ReadFinalizeMsgThreadFunc, 1, hcclMsgArea);
    t1.join();
    t2.join();
    ASSERT_EQ(hccl.Query(handleId), 0);
}

// 测试内容：Prepare1次(AlltoAll接口, repeat=2)，校验消息区内容
TEST_F(HcclCommonTestSuite, AlltoAll_Repeat2_CommitWhenPrepare)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    HcclHandle handleId = hccl.AlltoAll<true>(reinterpret_cast<__gm__ uint8_t*>(0x11),
        reinterpret_cast<__gm__ uint8_t*>(0x11), 100, HcclDataType::HCCL_DATA_TYPE_INT8, 100 * 8, 2);
    EXPECT_EQ(handleId, 0);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.valid, HCCL_MSG_VALID_MASK);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.repeatCnt, 2);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].dataCnt, 100);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.hcclDataType, HcclDataType::HCCL_DATA_TYPE_INT8);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].strideCount, 100 * 8);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].commType, HcclCMDType::HCCL_CMD_ALLTOALL);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].opType, HcclReduceOp::HCCL_REDUCE_RESERVED);
    EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].cnt, 2);
    EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].valid, COMMIT_VALID_MASK);
    for (int i = 0; i < 2; ++i) {
        ASSERT_EQ(hccl.Query(handleId), i);
        hcclMsgArea->finishedTurnCnt[0].cnt++;
        ASSERT_EQ(hccl.Query(handleId), i + 1);
        EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
    }
    std::thread t1(FinalizeThreadFunc, 0, std::ref(hccl));
    std::thread t2(ReadFinalizeMsgThreadFunc, 1, hcclMsgArea);
    t1.join();
    t2.join();
    ASSERT_EQ(hccl.Query(handleId), 0);
}

// 测试内容：Prepare10次(AlltoAll接口, repeat=1)+Commit10次，校验消息区内容
TEST_F(HcclCommonTestSuite, AlltoAll_Prepare10Repeat1)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    for (int i = 0; i < 10; ++i) {
        HcclHandle handleId = hccl.AlltoAll(reinterpret_cast<__gm__ uint8_t*>(0x11),
            reinterpret_cast<__gm__ uint8_t*>(0x11), 100, HcclDataType::HCCL_DATA_TYPE_INT8, 100 * 8);
        EXPECT_EQ(handleId, i);
        EXPECT_EQ(hcclMsgArea->sendMsgs[i].addMsg.v0Msg.valid, HCCL_MSG_VALID_MASK);
        EXPECT_EQ(hcclMsgArea->sendMsgs[i].addMsg.v0Msg.repeatCnt, 1);
        EXPECT_EQ(hcclMsgArea->sendMsgs[i].dataCnt, 100);
        EXPECT_EQ(hcclMsgArea->sendMsgs[i].addMsg.v0Msg.hcclDataType, HcclDataType::HCCL_DATA_TYPE_INT8);
        EXPECT_EQ(hcclMsgArea->sendMsgs[i].strideCount, 100 * 8);
        EXPECT_EQ(hcclMsgArea->sendMsgs[i].commType, HcclCMDType::HCCL_CMD_ALLTOALL);
        EXPECT_EQ(hcclMsgArea->sendMsgs[i].opType, HcclReduceOp::HCCL_REDUCE_RESERVED);
        EXPECT_EQ(hcclMsgArea->commitTurnCnt[i].cnt, 0);
        hccl.Commit(handleId);
        EXPECT_EQ(hcclMsgArea->commitTurnCnt[i].cnt, 1);
        ASSERT_EQ(hccl.Query(handleId), 0);
        hcclMsgArea->finishedTurnCnt[i].cnt++;
        ASSERT_EQ(hccl.Query(handleId), 1);
        EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
    }
    std::thread t1(FinalizeThreadFunc, 0, std::ref(hccl));
    std::thread t2(ReadFinalizeMsgThreadFunc, 10, hcclMsgArea);
    t1.join();
    t2.join();
}

// 测试内容：Prepare1次(AlltoAllV接口, repeat=1)+Commit1次，校验消息区内容
TEST_F(HcclCommonTestSuite, AlltoAllV_Repeat1)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    int64_t blockIdx = 0;
    std::thread t1(AlltoAllVThreadFunc, blockIdx, hcclCombineOpParam, true);
    t1.join();
}

// 测试内容：Prepare1次(AlltoAllV接口, repeat=2)，校验消息区内容
TEST_F(HcclCommonTestSuite, AlltoAllV_Repeat2_CommitWhenPrepare)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    std::vector<uint64_t> sendCounts(kRankNum, 10);
    std::vector<uint64_t> recvCounts(kRankNum, 11);
    std::vector<uint64_t> sendOffsets(kRankNum, 12);
    std::vector<uint64_t> recvOffsets(kRankNum, 13);
    HcclHandle handleId = hccl.AlltoAllV<true>(reinterpret_cast<__gm__ uint8_t*>(0x11), sendCounts.data(),
        sendOffsets.data(), HcclDataType::HCCL_DATA_TYPE_INT8, reinterpret_cast<__gm__ uint8_t*>(0x11),
        recvCounts.data(), recvOffsets.data(), HcclDataType::HCCL_DATA_TYPE_INT8, 2);
    EXPECT_EQ(handleId, 0);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.valid, HCCL_MSG_VALID_MASK);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.repeatCnt, 2);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].dataCnt, 0);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.hcclDataType, HcclReduceOp::HCCL_REDUCE_SUM);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].strideCount, 0);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].commType, HcclCMDType::HCCL_CMD_ALLTOALLV);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].opType, HcclReduceOp::HCCL_REDUCE_RESERVED);
    ASSERT_EQ(hcclMsgArea->paramExtMsgList[0].valid, HCCL_MSG_VALID_MASK);
    ASSERT_EQ(
        hcclMsgArea->paramExtMsgList[0].xorCheck, GenXorForHcclMsgExt(&hcclMsgArea->paramExtMsgList[0], kRankNum));
    for (int32_t i = 0; i < kRankNum; ++i) {
        EXPECT_EQ(hcclMsgArea->paramExtMsgList[0].sendCounts[i], 10);
        EXPECT_EQ(hcclMsgArea->paramExtMsgList[0].recvCounts[i], 11);
        EXPECT_EQ(hcclMsgArea->paramExtMsgList[0].sendOffset[i], 12);
        EXPECT_EQ(hcclMsgArea->paramExtMsgList[0].recvOffset[i], 13);
    }
    EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].cnt, 2);
    for (int i = 0; i < 2; ++i) {
        ASSERT_EQ(hccl.Query(handleId), i);
        hcclMsgArea->finishedTurnCnt[0].cnt++;
        ASSERT_EQ(hccl.Query(handleId), i + 1);
        EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
    }
    std::thread t1(FinalizeThreadFunc, 0, std::ref(hccl));
    std::thread t2(ReadFinalizeMsgThreadFunc, 1, hcclMsgArea);
    t1.join();
    t2.join();
    ASSERT_EQ(hccl.Query(handleId), 0);
}

// 测试内容：Prepare10次(AlltoAllV接口, repeat=1)+Commit10次，校验消息区内容
TEST_F(HcclCommonTestSuite, AlltoAllV_Prepare10Repeat1)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    std::vector<uint64_t> sendCounts(kRankNum, 10);
    std::vector<uint64_t> recvCounts(kRankNum, 11);
    std::vector<uint64_t> sendOffsets(kRankNum, 12);
    std::vector<uint64_t> recvOffsets(kRankNum, 13);
    for (int i = 0; i < 10; ++i) {
        HcclHandle handleId = hccl.AlltoAllV(reinterpret_cast<__gm__ uint8_t*>(0x11), sendCounts.data(),
            sendOffsets.data(), HcclDataType::HCCL_DATA_TYPE_INT8, reinterpret_cast<__gm__ uint8_t*>(0x11),
            recvCounts.data(), recvOffsets.data(), HcclDataType::HCCL_DATA_TYPE_INT8);
        EXPECT_EQ(handleId, i);
        EXPECT_EQ(hcclMsgArea->sendMsgs[i].addMsg.v0Msg.valid, HCCL_MSG_VALID_MASK);
        EXPECT_EQ(hcclMsgArea->sendMsgs[i].addMsg.v0Msg.repeatCnt, 1);
        EXPECT_EQ(hcclMsgArea->sendMsgs[i].dataCnt, 0);
        EXPECT_EQ(hcclMsgArea->sendMsgs[i].commType, HcclCMDType::HCCL_CMD_ALLTOALLV);

        ASSERT_EQ(hcclMsgArea->paramExtMsgList[0].valid, HCCL_MSG_VALID_MASK);
        ASSERT_EQ(
            hcclMsgArea->paramExtMsgList[0].xorCheck, GenXorForHcclMsgExt(&hcclMsgArea->paramExtMsgList[0], kRankNum));
        for (int32_t j = 0; j < kRankNum; ++j) {
            EXPECT_EQ(hcclMsgArea->paramExtMsgList[i].sendCounts[j], 10);
            EXPECT_EQ(hcclMsgArea->paramExtMsgList[i].recvCounts[j], 11);
            EXPECT_EQ(hcclMsgArea->paramExtMsgList[i].sendOffset[j], 12);
            EXPECT_EQ(hcclMsgArea->paramExtMsgList[i].recvOffset[j], 13);
        }

        EXPECT_EQ(hcclMsgArea->commitTurnCnt[i].cnt, 0);
        hccl.Commit(handleId);
        EXPECT_EQ(hcclMsgArea->commitTurnCnt[i].cnt, 1);
        ASSERT_EQ(hccl.Query(handleId), 0);
        hcclMsgArea->finishedTurnCnt[i].cnt++;
        ASSERT_EQ(hccl.Query(handleId), 1);
        EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
    }
    std::thread t1(FinalizeThreadFunc, 0, std::ref(hccl));
    std::thread t2(ReadFinalizeMsgThreadFunc, 10, hcclMsgArea);
    t1.join();
    t2.join();
}

// 测试内容: 测试blockIdx≠0的核, AlltoAllV接口实际不会发送消息
TEST_F(HcclCommonTestSuite, AlltoAllV_BlockIdxNot0_MsgIsNotWritten)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    int64_t blockIdx = 2;
    std::thread t1(AlltoAllVThreadFunc, blockIdx, hcclCombineOpParam, false);
    t1.join();
}

TEST_F(HcclCommonTestSuite, InterHcclGroupSync)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    hccl.InterHcclGroupSync(0, 1);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.valid, HCCL_MSG_VALID_MASK);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.commDepGroupID, 0);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.commDepHandleID, 1);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.xorCheck, GenXorForHcclMsg(&hcclMsgArea->sendMsgs[0]));
    std::thread t1(FinalizeThreadFunc, 0, std::ref(hccl));
    std::thread t2(ReadFinalizeMsgThreadFunc, 1, hcclMsgArea);
    t1.join();
    t2.join();
}

// 测试内容：Prepare2次(AllReduce接口, repeat=1)+Commit2次，校验消息区内容
TEST_F(HcclCommonTestSuite, AllReduce_Repeat1Prepare2Commit2)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    for (size_t i = 0U; i < 2; ++i) {
        HcclHandle handleId =
            hccl.AllReduce(reinterpret_cast<__gm__ uint8_t*>(0x11), reinterpret_cast<__gm__ uint8_t*>(0x11), 100,
                HcclDataType::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM);
        EXPECT_EQ(handleId, i);
        EXPECT_EQ(hcclMsgArea->sendMsgs[i].addMsg.v0Msg.valid, HCCL_MSG_VALID_MASK);
        EXPECT_EQ(hcclMsgArea->sendMsgs[i].addMsg.v0Msg.repeatCnt, 1);
        EXPECT_EQ(hcclMsgArea->sendMsgs[i].commType, HcclCMDType::HCCL_CMD_ALLREDUCE);
        EXPECT_EQ(hcclMsgArea->sendMsgs[i].opType, HcclReduceOp::HCCL_REDUCE_SUM);
        EXPECT_EQ(hcclMsgArea->commitTurnCnt[i].cnt, 0);
        hccl.Commit(handleId);
        EXPECT_EQ(hcclMsgArea->commitTurnCnt[i].cnt, 1);
        ASSERT_EQ(hccl.Query(handleId), 0);
        hcclMsgArea->finishedTurnCnt[i].cnt++;
        ASSERT_EQ(hccl.Query(handleId), 1);
        EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
    }
}

// 测试内容：Prepare1次(AllReduce接口, repeat=2)+Commit2次，校验消息区内容
TEST_F(HcclCommonTestSuite, AllReduce_Repeat2Prepare1Commit2)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    HcclHandle handleId =
        hccl.AllReduce(reinterpret_cast<__gm__ uint8_t*>(0x11), reinterpret_cast<__gm__ uint8_t*>(0x11), 100,
            HcclDataType::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, 2);
    EXPECT_EQ(handleId, 0);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.valid, HCCL_MSG_VALID_MASK);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.repeatCnt, 2);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].commType, HcclCMDType::HCCL_CMD_ALLREDUCE);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].opType, HcclReduceOp::HCCL_REDUCE_SUM);
    for (size_t i = 0U; i < 2; ++i) {
        EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].cnt, i);
        hccl.Commit(handleId);
        EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].cnt, i + 1);
        ASSERT_EQ(hccl.Query(handleId), i);
        hcclMsgArea->finishedTurnCnt[0].cnt++;
        ASSERT_EQ(hccl.Query(handleId), i + 1);
        EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
    }
}

// 测试内容：先计算后通信场景，用户可以不调用Wait接口，退出前只调用Finalize接口
TEST_F(HcclCommonTestSuite, AllReduceCallFinalize_ResetFinishedCount)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    HcclHandle handleId =
        hccl.AllReduce(reinterpret_cast<__gm__ uint8_t*>(0x11), reinterpret_cast<__gm__ uint8_t*>(0x11), 100,
            HcclDataType::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, 2);
    EXPECT_EQ(handleId, 0);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.valid, HCCL_MSG_VALID_MASK);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.repeatCnt, 2);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].commType, HcclCMDType::HCCL_CMD_ALLREDUCE);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].opType, HcclReduceOp::HCCL_REDUCE_SUM);
    for (size_t i = 0U; i < 2; ++i) {
        EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].cnt, i);
        hccl.Commit(handleId);
        EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].cnt, i + 1);
        ASSERT_EQ(hccl.Query(handleId), i);
        hcclMsgArea->finishedTurnCnt[0].cnt++;
        ASSERT_EQ(hccl.Query(handleId), i + 1);
    }
    std::thread t1(FinalizeThreadFunc, 0, std::ref(hccl));
    std::thread t2(ReadFinalizeMsgThreadFunc, 1, hcclMsgArea);
    t1.join();
    t2.join();
    EXPECT_EQ(hcclMsgArea->finishedTurnCnt[0].cnt, 0);
}

// 测试内容：BatchPrepare1次(AllReduce接口, repeat=2)，校验消息区内容
TEST_F(HcclCommonTestSuite, AllReduce_Repeat2Prepare1WithCommit)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    HcclHandle handleId =
        hccl.AllReduce<true>(reinterpret_cast<__gm__ uint8_t*>(0x11), reinterpret_cast<__gm__ uint8_t*>(0x11), 100,
            HcclDataType::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, 2);
    EXPECT_EQ(handleId, 0);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.valid, HCCL_MSG_VALID_MASK);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.repeatCnt, 2);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].commType, HcclCMDType::HCCL_CMD_ALLREDUCE);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].opType, HcclReduceOp::HCCL_REDUCE_SUM);
    EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].cnt, 2);
    ASSERT_EQ(hccl.Query(handleId), 0);
    hcclMsgArea->finishedTurnCnt[0].cnt += 2;
    ASSERT_EQ(hccl.Query(handleId), 2);
    for (size_t i = 0U; i < 2; ++i) { EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS); }
}

// 测试内容：BatchPrepare和Prepare接口混用 + Finalize，检查消息区内容
TEST_F(HcclCombineTestSuite, 3Prepare_3TasksForeach)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));

    // AllReduce
    HcclHandle handleId =
        hccl.AllReduce(reinterpret_cast<__gm__ uint8_t*>(0x11), reinterpret_cast<__gm__ uint8_t*>(0x11), 100,
            HcclDataType::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, 3);
    EXPECT_EQ(handleId, 0);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.valid, HCCL_MSG_VALID_MASK);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.repeatCnt, 3);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].commType, HcclCMDType::HCCL_CMD_ALLREDUCE);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].opType, HcclReduceOp::HCCL_REDUCE_SUM);

    for (size_t i = 0U; i < 3; ++i) {
        EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].cnt, i);
        hccl.Commit(handleId);
        EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].cnt, i + 1);
        ASSERT_EQ(hccl.Query(handleId), i);
        hcclMsgArea->finishedTurnCnt[0].cnt++;
        ASSERT_EQ(hccl.Query(handleId), i + 1);
        EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
    }

    // AllGather
    handleId = hccl.AllGather<true>(reinterpret_cast<__gm__ uint8_t*>(0x11), reinterpret_cast<__gm__ uint8_t*>(0x11),
        100, HcclDataType::HCCL_DATA_TYPE_INT8, 0, 3);
    EXPECT_EQ(handleId, 1);
    EXPECT_EQ(hcclMsgArea->sendMsgs[1].addMsg.v0Msg.valid, HCCL_MSG_VALID_MASK);
    EXPECT_EQ(hcclMsgArea->sendMsgs[1].addMsg.v0Msg.repeatCnt, 3);
    EXPECT_EQ(hcclMsgArea->sendMsgs[1].commType, HcclCMDType::HCCL_CMD_ALLGATHER);
    EXPECT_EQ(hcclMsgArea->sendMsgs[1].opType, HcclReduceOp::HCCL_REDUCE_RESERVED);
    EXPECT_EQ(hcclMsgArea->commitTurnCnt[1].cnt, 3);
    for (size_t i = 0U; i < 3; ++i) {
        ASSERT_EQ(hccl.Query(handleId), i);
        hcclMsgArea->finishedTurnCnt[1].cnt++;
        ASSERT_EQ(hccl.Query(handleId), i + 1);
        EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
    }

    // ReduceScatter
    for (int i = 0; i < 3; ++i) {
        handleId = hccl.ReduceScatter(reinterpret_cast<__gm__ uint8_t*>(0x11), reinterpret_cast<__gm__ uint8_t*>(0x11),
            100, HcclDataType::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, 100 * 8);
        EXPECT_EQ(handleId, 2 + i);
        EXPECT_EQ(hcclMsgArea->sendMsgs[2 + i].addMsg.v0Msg.valid, HCCL_MSG_VALID_MASK);
        EXPECT_EQ(hcclMsgArea->sendMsgs[2 + i].addMsg.v0Msg.repeatCnt, 1);
        EXPECT_EQ(hcclMsgArea->sendMsgs[2 + i].commType, HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
        EXPECT_EQ(hcclMsgArea->sendMsgs[2 + i].opType, HcclReduceOp::HCCL_REDUCE_SUM);
        EXPECT_EQ(hcclMsgArea->commitTurnCnt[2 + i].cnt, 0);
        hccl.Commit(handleId);
        EXPECT_EQ(hcclMsgArea->commitTurnCnt[2 + i].cnt, 1);
        ASSERT_EQ(hccl.Query(handleId), 0);
        hcclMsgArea->finishedTurnCnt[2 + i].cnt++;
        ASSERT_EQ(hccl.Query(handleId), 1);
        EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
    }

    // Finalize
    std::thread t1(FinalizeThreadFunc, 0, std::ref(hccl));
    std::thread t2(ReadFinalizeMsgThreadFunc, 5, hcclMsgArea);
    t1.join();
    t2.join();
    EXPECT_EQ(hcclMsgArea->sendMsgs[5].addMsg.v0Msg.valid, ~HCCL_MSG_VALID_MASK);
    EXPECT_EQ(hcclMsgArea->sendMsgs[5].commType, HcclCMDType::HCCL_CMD_FINALIZE);
}

// 测试内容：Commit顺序和Prepare的顺序不一致，但是Wait顺序和Prepare的顺序一致，用例正常工作
// step1 Prepare(AllReduce)
// step2 Prepare(AlltoAll)同步Commit
// step3 Prepare(AllGather)同步Commit
// step4 AllReduce Commit+Wait
// step5 AlltoAll Wait
// step6 AllGather Wait
// step7 Finalize
TEST_F(HcclCombineTestSuite, MultiPrepareInvoked_CommitNotAccordingToPrepareOrder)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));

    // AllReduce Prepare
    HcclHandle handleId_0 =
        hccl.AllReduce(reinterpret_cast<__gm__ uint8_t*>(0x11), reinterpret_cast<__gm__ uint8_t*>(0x11), 100,
            HcclDataType::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, 3);
    EXPECT_EQ(handleId_0, 0);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.valid, HCCL_MSG_VALID_MASK);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.repeatCnt, 3);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].commType, HcclCMDType::HCCL_CMD_ALLREDUCE);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].opType, HcclReduceOp::HCCL_REDUCE_SUM);

    // AlltoAll Prepare+Commit
    HcclHandle handleId_1 = hccl.AlltoAll<true>(reinterpret_cast<__gm__ uint8_t*>(0x11),
        reinterpret_cast<__gm__ uint8_t*>(0x11), 100, HcclDataType::HCCL_DATA_TYPE_INT8, 100 * 8, 2);
    EXPECT_EQ(handleId_1, 1);
    EXPECT_EQ(hcclMsgArea->sendMsgs[1].addMsg.v0Msg.valid, HCCL_MSG_VALID_MASK);
    EXPECT_EQ(hcclMsgArea->sendMsgs[1].addMsg.v0Msg.repeatCnt, 2);
    EXPECT_EQ(hcclMsgArea->sendMsgs[1].commType, HcclCMDType::HCCL_CMD_ALLTOALL);
    EXPECT_EQ(hcclMsgArea->sendMsgs[1].opType, HcclReduceOp::HCCL_REDUCE_RESERVED);
    EXPECT_EQ(hcclMsgArea->commitTurnCnt[1].cnt, 2);

    // AllGather Prepare+Commit
    HcclHandle handleId_2 = hccl.AllGather<true>(reinterpret_cast<__gm__ uint8_t*>(0x11),
        reinterpret_cast<__gm__ uint8_t*>(0x11), 100, HcclDataType::HCCL_DATA_TYPE_INT8, 0, 3);
    EXPECT_EQ(handleId_2, 2);
    EXPECT_EQ(hcclMsgArea->sendMsgs[2].addMsg.v0Msg.valid, HCCL_MSG_VALID_MASK);
    EXPECT_EQ(hcclMsgArea->sendMsgs[2].addMsg.v0Msg.repeatCnt, 3);
    EXPECT_EQ(hcclMsgArea->sendMsgs[2].commType, HcclCMDType::HCCL_CMD_ALLGATHER);
    EXPECT_EQ(hcclMsgArea->sendMsgs[2].opType, HcclReduceOp::HCCL_REDUCE_RESERVED);
    EXPECT_EQ(hcclMsgArea->commitTurnCnt[2].cnt, 3);

    // AllReduce Commit+Wait
    for (size_t i = 0U; i < 3; ++i) {
        EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].cnt, i);
        hccl.Commit(handleId_0);
        EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].cnt, i + 1);
        ASSERT_EQ(hccl.Query(handleId_0), i);
        hcclMsgArea->finishedTurnCnt[0].cnt++;
        ASSERT_EQ(hccl.Query(handleId_0), i + 1);
        EXPECT_EQ(hccl.Wait(handleId_0), HCCL_SUCCESS);
    }

    // AlltoAll Wait
    for (size_t i = 0U; i < 2; ++i) {
        ASSERT_EQ(hccl.Query(handleId_1), i);
        hcclMsgArea->finishedTurnCnt[1].cnt++;
        ASSERT_EQ(hccl.Query(handleId_1), i + 1);
        EXPECT_EQ(hccl.Wait(handleId_1), HCCL_SUCCESS);
    }

    // AllGather Wait
    for (size_t i = 0U; i < 3; ++i) {
        ASSERT_EQ(hccl.Query(handleId_2), i);
        hcclMsgArea->finishedTurnCnt[2].cnt++;
        ASSERT_EQ(hccl.Query(handleId_2), i + 1);
        EXPECT_EQ(hccl.Wait(handleId_2), HCCL_SUCCESS);
    }

    // Finalize
    std::thread t1(FinalizeThreadFunc, 0, std::ref(hccl));
    std::thread t2(ReadFinalizeMsgThreadFunc, 3, hcclMsgArea);
    t1.join();
    t2.join();
    EXPECT_EQ(hcclMsgArea->sendMsgs[3].addMsg.v0Msg.valid, ~HCCL_MSG_VALID_MASK);
    EXPECT_EQ(hcclMsgArea->sendMsgs[3].commType, HcclCMDType::HCCL_CMD_FINALIZE);
}

// 临界测试: 消息区hcclSendMsg超出MAX_MSG_COUNT后，循环使用
TEST_F(HcclCommonTestSuite, HcclCriticalTest_MsgUsageOverMAX_MSG_COUNT)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    for (int i = 0; i < MAX_MSG_COUNT; ++i) { hccl.InterHcclGroupSync(0, 0); }
    hcclMsgArea->sendMsgs[0].addMsg.v0Msg.valid = ~hcclMsgArea->sendMsgs[0].addMsg.v0Msg.valid;
    HcclHandle handleId =
        hccl.AllReduce(reinterpret_cast<__gm__ uint8_t*>(0x11), reinterpret_cast<__gm__ uint8_t*>(0x11), 100,
            HcclDataType::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, 3);
    EXPECT_EQ(handleId, INVALID_HANDLE_ID);
}

// 异常测试: Commit发生在Prepare之前，CommitTurnCnt值不会被写
TEST_F(HcclAbnormalTestSuite, CommitTurnCntNotUpdate_CommitBeforePrepare)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    hccl.Commit(0);
    EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].cnt, 0);
}

// 异常测试: Wait和Query发生在Prepare之前，拦截退出
TEST_F(HcclAbnormalTestSuite, ReturnInvalid_WaitAndQueryBeforePrepare)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    EXPECT_EQ(hccl.Wait(0), HCCL_FAILED);
    EXPECT_EQ(hccl.Query(0), HCCL_FAILED);
}

// 异常测试: Wait发生在Commit之前，拦截退出
TEST_F(HcclAbnormalTestSuite, ReturnInvalid_WaitBeforeCommit)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    HcclHandle handleId =
        hccl.AllReduce(reinterpret_cast<__gm__ uint8_t*>(0x11), reinterpret_cast<__gm__ uint8_t*>(0x11), 100,
            HcclDataType::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, 3);
    ASSERT_EQ(handleId, 0);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.valid, HCCL_MSG_VALID_MASK);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.repeatCnt, 3);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].commType, HcclCMDType::HCCL_CMD_ALLREDUCE);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].opType, HcclReduceOp::HCCL_REDUCE_SUM);
    EXPECT_EQ(hccl.Wait(handleId), HCCL_FAILED);
}

// 异常测试: hccl未初始化调用各接口，且hccl析构时不发生core dump
TEST_F(HcclAbnormalTestSuite, HcclNotInit)
{
    char ch[2048] = {'\0'}; // 防止栈内存复用
    Hccl hccl;
    auto hanleId1 = hccl.AllReduce(reinterpret_cast<__gm__ uint8_t*>(0x11), reinterpret_cast<__gm__ uint8_t*>(0x11),
        100, HcclDataType::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, 3);
    EXPECT_EQ(hanleId1, INVALID_HANDLE_ID);
    auto hanleId2 = hccl.AllGather(reinterpret_cast<__gm__ uint8_t*>(0x11), reinterpret_cast<__gm__ uint8_t*>(0x11),
        100, HcclDataType::HCCL_DATA_TYPE_INT8, 0, 3);
    EXPECT_EQ(hanleId2, INVALID_HANDLE_ID);
    auto hanleId3 = hccl.ReduceScatter(reinterpret_cast<__gm__ uint8_t*>(0x11), reinterpret_cast<__gm__ uint8_t*>(0x11),
        100, HcclDataType::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, 100 * 3, 3);
    EXPECT_EQ(hanleId3, INVALID_HANDLE_ID);
    hccl.Finalize();

    hccl.Commit(hanleId1);
    auto ret = hccl.Wait(hanleId1);
    EXPECT_EQ(ret, HCCL_FAILED);
    ret = hccl.Query(hanleId1);
    EXPECT_EQ(ret, HCCL_FAILED);
}

// 异常测试: hccl客户端下发的Prepare消息过多
TEST_F(HcclAbnormalTestSuite, PrepareCntLargerThanHCCL_MAX_HANDLE_ID)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    HcclHandle handleId;
    for (int i = 0; i < HCCL_MAX_HANDLE_ID; ++i) {
        handleId = hccl.AllReduce(reinterpret_cast<__gm__ uint8_t*>(0x11), reinterpret_cast<__gm__ uint8_t*>(0x11), 100,
            HcclDataType::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, 3);
        EXPECT_EQ(handleId, i);
        EXPECT_EQ(hcclMsgArea->sendMsgs[i].addMsg.v0Msg.valid, HCCL_MSG_VALID_MASK);
        EXPECT_EQ(hcclMsgArea->sendMsgs[i].addMsg.v0Msg.repeatCnt, 3);
        EXPECT_EQ(hcclMsgArea->sendMsgs[i].commType, HcclCMDType::HCCL_CMD_ALLREDUCE);
        EXPECT_EQ(hcclMsgArea->sendMsgs[i].opType, HcclReduceOp::HCCL_REDUCE_SUM);
    }
    handleId = hccl.AllReduce(reinterpret_cast<__gm__ uint8_t*>(0x11), reinterpret_cast<__gm__ uint8_t*>(0x11), 100,
        HcclDataType::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, 3);

    EXPECT_EQ(handleId, INVALID_HANDLE_ID);
    EXPECT_NE(hcclMsgArea->sendMsgs[63].addMsg.v0Msg.valid, HCCL_MSG_VALID_MASK);
}

// 异常测试: Repeat=0的通信消息，客户端拦截不发送
TEST_F(HcclAbnormalTestSuite, PrepareFailed_RepeatIs0)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    HcclHandle handleId =
        hccl.AllReduce(reinterpret_cast<__gm__ uint8_t*>(0x11), reinterpret_cast<__gm__ uint8_t*>(0x11), 100,
            HcclDataType::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, 0);
    EXPECT_EQ(handleId, INVALID_HANDLE_ID);
    EXPECT_NE(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.valid, HCCL_MSG_VALID_MASK);
}

// 异常测试: dtype不合法的通信消息，客户端拦截不发送
TEST_F(HcclAbnormalTestSuite, PrepareFailed_InvalidDtype)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    HcclHandle handleId =
        hccl.AllReduce(reinterpret_cast<__gm__ uint8_t*>(0x11), reinterpret_cast<__gm__ uint8_t*>(0x11), 100,
            HcclDataType::HCCL_DATA_TYPE_RESERVED, HcclReduceOp::HCCL_REDUCE_SUM, 1);
    EXPECT_EQ(handleId, INVALID_HANDLE_ID);
    EXPECT_NE(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.valid, HCCL_MSG_VALID_MASK);

    handleId = hccl.AllReduce(reinterpret_cast<__gm__ uint8_t*>(0x11), reinterpret_cast<__gm__ uint8_t*>(0x11), 100,
        static_cast<HcclDataType>(-1), HcclReduceOp::HCCL_REDUCE_SUM, 1);
    EXPECT_EQ(handleId, INVALID_HANDLE_ID);
    EXPECT_NE(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.valid, HCCL_MSG_VALID_MASK);
}

// sendBuf/recvBuf nullptr, HCCL_REDUCE_RESERVED for AllReduce.
TEST_F(HcclAbnormalTestSuite, PrepareFailed_CheckCommonPrepareParamValid)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    HcclHandle handleId =
        hccl.AllReduce(nullptr, nullptr, 100, HcclDataType::HCCL_DATA_TYPE_RESERVED, HcclReduceOp::HCCL_REDUCE_SUM, 1);
    EXPECT_EQ(handleId, INVALID_HANDLE_ID);
    EXPECT_NE(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.valid, HCCL_MSG_VALID_MASK);

    handleId = hccl.AllReduce(reinterpret_cast<__gm__ uint8_t*>(0x11), reinterpret_cast<__gm__ uint8_t*>(0x11), 100,
        static_cast<HcclDataType>(-1), HcclReduceOp::HCCL_REDUCE_SUM, 1);
    EXPECT_EQ(handleId, INVALID_HANDLE_ID);
    EXPECT_NE(hcclMsgArea->sendMsgs[0].addMsg.v0Msg.valid, HCCL_MSG_VALID_MASK);
}

// 异常测试: Wait和Query和Commit接口handleId越界
TEST_F(HcclAbnormalTestSuite, WaitAndQueryFailed_HanldIdOutOfRange)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));

    auto ret = hccl.Wait(-1);
    EXPECT_EQ(ret, HCCL_FAILED);
    ret = hccl.Query(-1);
    EXPECT_EQ(ret, HCCL_FAILED);
    ret = hccl.Wait(HCCL_MAX_HANDLE_ID);
    EXPECT_EQ(ret, HCCL_FAILED);
    ret = hccl.Query(HCCL_MAX_HANDLE_ID);
    EXPECT_EQ(ret, HCCL_FAILED);
}

// 异常测试: Wait和Query接口，handleId非Prepare生成，拦截退出
TEST_F(HcclAbnormalTestSuite, WaitAndQueryFailed_HandleIdNotFromPrepare)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    EXPECT_EQ(hccl.Query(1), HCCL_FAILED);
    EXPECT_EQ(hccl.Wait(1), HCCL_FAILED);
}

// 测试内容：AlltoAll Prepare33次，最后一次失败
TEST_F(HcclAbnormalTestSuite, AlltoAll_FailedWhenPrepare33)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    for (int i = 0; i < 63; ++i) {
        HcclHandle handleId = hccl.AlltoAll(reinterpret_cast<__gm__ uint8_t*>(0x11),
            reinterpret_cast<__gm__ uint8_t*>(0x11), 100, HcclDataType::HCCL_DATA_TYPE_INT8, 100 * 8);
        EXPECT_EQ(handleId, i);
    }
    HcclHandle handleId = hccl.AlltoAll(reinterpret_cast<__gm__ uint8_t*>(0x11),
        reinterpret_cast<__gm__ uint8_t*>(0x11), 100, HcclDataType::HCCL_DATA_TYPE_INT8, 100 * 8);
    EXPECT_EQ(handleId, INVALID_HANDLE_ID);
}

// 测试内容：AlltoAllV Prepare33次，最后一次失败
TEST_F(HcclAbnormalTestSuite, AlltoAllV_FailedWhenPrepare33)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    std::vector<uint64_t> sendCounts(kRankNum, 10);
    std::vector<uint64_t> recvCounts(kRankNum, 11);
    std::vector<uint64_t> sendOffsets(kRankNum, 12);
    std::vector<uint64_t> recvOffsets(kRankNum, 13);
    for (int i = 0; i < 63; ++i) {
        HcclHandle handleId = hccl.AlltoAllV<true>(reinterpret_cast<__gm__ uint8_t*>(0x11), sendCounts.data(),
            sendOffsets.data(), HcclDataType::HCCL_DATA_TYPE_INT8, reinterpret_cast<__gm__ uint8_t*>(0x11),
            recvCounts.data(), recvOffsets.data(), HcclDataType::HCCL_DATA_TYPE_INT8, 1);
        EXPECT_EQ(handleId, i);
    }
    HcclHandle handleId = hccl.AlltoAllV<true>(reinterpret_cast<__gm__ uint8_t*>(0x11), sendCounts.data(),
        sendOffsets.data(), HcclDataType::HCCL_DATA_TYPE_INT8, reinterpret_cast<__gm__ uint8_t*>(0x11),
        recvCounts.data(), recvOffsets.data(), HcclDataType::HCCL_DATA_TYPE_INT8, 1);
    EXPECT_EQ(handleId, INVALID_HANDLE_ID);
}

// 测试内容：AlltoAll 异常入参测试
TEST_F(HcclAbnormalTestSuite, AlltoAll_InputParamInvalid_ReturnInvalidHandleId)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    GM_ADDR sendBuf = nullptr;
    HcclHandle handleId = hccl.AlltoAll(
        sendBuf, reinterpret_cast<__gm__ uint8_t*>(0x11), 100, HcclDataType::HCCL_DATA_TYPE_INT8, 100 * 8);
    EXPECT_EQ(handleId, INVALID_HANDLE_ID);

    uint64_t dataCount = 0;
    handleId = hccl.AlltoAll(reinterpret_cast<__gm__ uint8_t*>(0x11), reinterpret_cast<__gm__ uint8_t*>(0x11),
        dataCount, HcclDataType::HCCL_DATA_TYPE_INT8, 100 * 8);
    EXPECT_EQ(handleId, INVALID_HANDLE_ID);

    uint8_t repeat = 0;
    handleId = hccl.AlltoAll(reinterpret_cast<__gm__ uint8_t*>(0x11), reinterpret_cast<__gm__ uint8_t*>(0x11), 100,
        HcclDataType::HCCL_DATA_TYPE_INT8, 100 * 8, repeat);
    EXPECT_EQ(handleId, INVALID_HANDLE_ID);
}

// 测试内容：AlltoAllV 异常入参测试
TEST_F(HcclAbnormalTestSuite, AlltoAllV_InputParamInvalid_ReturnInvalidHandleId)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    std::vector<uint64_t> sendCounts(kRankNum, 10);
    std::vector<uint64_t> recvCounts(kRankNum, 11);
    std::vector<uint64_t> sendOffsets(kRankNum, 12);
    std::vector<uint64_t> recvOffsets(kRankNum, 13);
    GM_ADDR sendBuf = nullptr;
    HcclHandle handleId = hccl.AlltoAllV(sendBuf, sendCounts.data(), sendOffsets.data(),
        HcclDataType::HCCL_DATA_TYPE_INT8, reinterpret_cast<__gm__ uint8_t*>(0x11), recvCounts.data(),
        recvOffsets.data(), HcclDataType::HCCL_DATA_TYPE_INT8, 1);
    EXPECT_EQ(handleId, INVALID_HANDLE_ID);

    uint8_t repeat = 0;
    handleId = hccl.AlltoAllV(reinterpret_cast<__gm__ uint8_t*>(0x11), sendCounts.data(), sendOffsets.data(),
        HcclDataType::HCCL_DATA_TYPE_INT8, reinterpret_cast<__gm__ uint8_t*>(0x11), recvCounts.data(),
        recvOffsets.data(), HcclDataType::HCCL_DATA_TYPE_INT8, repeat);
    EXPECT_EQ(handleId, INVALID_HANDLE_ID);

    // 全0输入需要支持
    std::vector<uint64_t> InvalidSendCounts(kRankNum, 0);
    handleId = hccl.AlltoAllV(reinterpret_cast<__gm__ uint8_t*>(0x11), InvalidSendCounts.data(), sendOffsets.data(),
        HcclDataType::HCCL_DATA_TYPE_INT8, reinterpret_cast<__gm__ uint8_t*>(0x11), recvCounts.data(),
        recvOffsets.data(), HcclDataType::HCCL_DATA_TYPE_INT8, 1);
    EXPECT_NE(handleId, INVALID_HANDLE_ID);

    handleId = hccl.AlltoAllV(reinterpret_cast<__gm__ uint8_t*>(0x11), sendCounts.data(), sendOffsets.data(),
        HcclDataType::HCCL_DATA_TYPE_INT8, reinterpret_cast<__gm__ uint8_t*>(0x11), nullptr, recvOffsets.data(),
        HcclDataType::HCCL_DATA_TYPE_INT8, 1);
    EXPECT_EQ(handleId, INVALID_HANDLE_ID);
}

// 组合测试: 使用3个线程模拟不同blockIdx的核，只有0核能写，所有核都能读消息区
TEST_F(HcclCombineTestSuite, AlltoAllV_3CoresWork_OnlyCore0Write)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    int64_t blockIdx = 2;
    std::thread t1(AlltoAllVThreadFunc, blockIdx, hcclCombineOpParam, false);
    t1.join();

    blockIdx = 0;
    std::thread t2(AlltoAllVThreadFunc, blockIdx, hcclCombineOpParam, true);
    t2.join();

    blockIdx = 1;
    std::thread t3(AlltoAllVThreadFunc, blockIdx, hcclCombineOpParam, true);
    t3.join();
}

class Mc2InitTilingTest {
    uint8_t reserved[48] = {0U};
};

struct Mc2CcTilingTest {
    uint8_t skipLocalRankCopy;
    uint8_t skipBufferWindowCopy;
    uint8_t stepSize;
    uint8_t version;
    char reserved[12];
    char groupName[128];
    char algConfig[128];
    uint32_t opType;
    uint32_t reduceType;
};

// 新tiling测试
// 正常调用
TEST_F(HcclCommonTestSuite, TilingV1_AllReduce_Repeat1)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);
    Mc2InitTilingTest mc2InitTiling;
    Mc2CcTilingTest mc2CcTiling;
    mc2CcTiling.opType = (uint32_t)HcclCMDType::HCCL_CMD_ALLREDUCE;
    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam), static_cast<__gm__ void*>(&mc2InitTiling));
    auto ret = hccl.SetCcTiling(static_cast<__gm__ void*>(&mc2CcTiling));
    HcclHandle handleId =
        hccl.AllReduce(reinterpret_cast<__gm__ uint8_t*>(0x11), reinterpret_cast<__gm__ uint8_t*>(0x11), 100,
            HcclDataType::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, 1);
    EXPECT_EQ(handleId, 0);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v1Msg.version, 1);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v1Msg.ccOpTilingData, reinterpret_cast<uint64_t>(&mc2CcTiling));
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v1Msg.xorCheck, GenXorForHcclMsg(&hcclMsgArea->sendMsgs[0]));
    EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].cnt, 0);
    hccl.Commit(handleId);
    EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].valid, COMMIT_VALID_MASK);
    EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].cnt, 1);
    ASSERT_EQ(hccl.Query(handleId), 0);
    hcclMsgArea->finishedTurnCnt[0].cnt++;
    ASSERT_EQ(hccl.Query(handleId), 1);
    EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);

    // hccl ctx接口验证
    EXPECT_EQ(hccl.GetRankDim(), kRankNum);
    EXPECT_EQ(hccl.GetRankId(), 0);
    EXPECT_EQ(hccl.GetWindowsInAddr(0), nullptr);
    EXPECT_EQ(hccl.GetWindowsOutAddr(0), nullptr);
}

// 异常调用
// 调用initv1和v0
TEST_F(HcclCommonTestSuite, TilingV1_AllReduce_Init_InitV1)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);
    Mc2InitTilingTest mc2InitTiling;
    Mc2CcTilingTest mc2CcTiling;
    mc2CcTiling.opType = (uint32_t)HcclCMDType::HCCL_CMD_ALLREDUCE;

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam), static_cast<__gm__ void*>(&mc2InitTiling));
    auto ret = hccl.SetCcTiling(static_cast<__gm__ void*>(&mc2CcTiling));
    EXPECT_EQ(ret, HCCL_FAILED);
    HcclHandle handleId =
        hccl.AllReduce(reinterpret_cast<__gm__ uint8_t*>(0x11), reinterpret_cast<__gm__ uint8_t*>(0x11), 100,
            HcclDataType::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, 1);
    EXPECT_EQ(handleId, INVALID_HANDLE_ID);
}

// 不调用Init直接调用SetCctiling
TEST_F(HcclCommonTestSuite, TilingV1_AllReduce_Prepare_noInitV1)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);
    Mc2CcTilingTest mc2CcTiling;
    mc2CcTiling.opType = (uint32_t)HcclCMDType::HCCL_CMD_ALLREDUCE;

    Hccl hccl;
    auto ret = hccl.SetCcTiling(static_cast<__gm__ void*>(&mc2CcTiling));
    EXPECT_EQ(ret, HCCL_FAILED);
}

// 调用InitV0 调用set
TEST_F(HcclCommonTestSuite, TilingV1_AllReduce_Init_PrepareV1)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);
    Mc2InitTilingTest mc2InitTiling;
    Mc2CcTilingTest mc2CcTiling;
    mc2CcTiling.opType = (uint32_t)HcclCMDType::HCCL_CMD_ALLREDUCE;

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    auto ret = hccl.SetCcTiling(static_cast<__gm__ void*>(&mc2CcTiling));
    EXPECT_EQ(ret, HCCL_FAILED);
}

// 设置异常opType
TEST_F(HcclCommonTestSuite, TilingV1_AllReduce_OpType_100)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);
    Mc2InitTilingTest mc2InitTiling;
    Mc2CcTilingTest mc2CcTiling;
    mc2CcTiling.opType = (uint32_t)HcclCMDType::HCCL_CMD_FINALIZE;

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam), static_cast<__gm__ void*>(&mc2InitTiling));
    auto ret = hccl.SetCcTiling(static_cast<__gm__ void*>(&mc2CcTiling));
    EXPECT_EQ(ret, HCCL_FAILED);
}

TEST_F(HcclCommonTestSuite, BatchWrite_InvalidPrepare)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    IbVerbsData ibData;
    HcclCombineOpParam hcclCombineOpParam{
        reinterpret_cast<uintptr_t>(workSpace.data()), workSpaceSize, 0, kRankNum, 0, {0}, {0}, {0}, true, &ibData};
    Mc2InitTilingTest mc2InitTiling;
    Mc2CcTilingTest mc2CcTiling;
    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam), static_cast<__gm__ void*>(&mc2InitTiling));

    mc2CcTiling.opType = (uint32_t)HcclCMDType::HCCL_CMD_BATCH_WRITE;
    auto ret = hccl.SetCcTiling(static_cast<__gm__ void*>(&mc2CcTiling));
    HcclHandle handleId = hccl.BatchWrite(reinterpret_cast<__gm__ uint8_t*>(0), 1);
    EXPECT_EQ(handleId, INVALID_HANDLE_ID);
}

TEST_F(HcclCommonTestSuite, BatchWrite_Prepare)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    IbVerbsData ibData[2] = {{{0, 0x11, 0}, {0, 0x22, 0}, {0, 0x33, 0}, {0, 0x44, 0}},
        {{0, 0x55, 0}, {0, 0x66, 0}, {0, 0x77, 0}, {0, 0x88, 0}}};
    HcclCombineOpParam hcclCombineOpParam{
        reinterpret_cast<uintptr_t>(workSpace.data()), workSpaceSize, 0, kRankNum, 0, {0}, {0}, {0}, true, &ibData[0]};
    Mc2InitTilingTest mc2InitTiling;
    Mc2CcTilingTest mc2CcTiling;
    mc2CcTiling.opType = (uint32_t)HcclCMDType::HCCL_CMD_BATCH_WRITE;
    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam), static_cast<__gm__ void*>(&mc2InitTiling));
    auto ret = hccl.SetCcTiling(static_cast<__gm__ void*>(&mc2CcTiling));
    HcclHandle handleId = hccl.BatchWrite(reinterpret_cast<__gm__ uint8_t*>(0x11), 1);
    EXPECT_EQ(handleId, 0);
    EXPECT_EQ(hccl.GetRankDim(), kRankNum);
    EXPECT_EQ(hccl.GetRankId(), 0);
    EXPECT_EQ((uint64_t)hccl.GetWindowsInAddr(0), 0x33);
    EXPECT_EQ((uint64_t)hccl.GetWindowsOutAddr(0), 0x44);
    EXPECT_EQ((uint64_t)hccl.GetWindowsInAddr(1), 0x55);
    EXPECT_EQ((uint64_t)hccl.GetWindowsOutAddr(1), 0x66);
}

TEST_F(HcclCommonTestSuite, All2AllV_FineGrainedSend)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);
    Mc2InitTilingTest mc2InitTiling;
    Mc2CcTilingTest mc2CcTiling;
    mc2CcTiling.stepSize = 1U;
    mc2CcTiling.opType = (uint32_t)HcclCMDType::HCCL_CMD_ALLTOALLV;
    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam), static_cast<__gm__ void*>(&mc2InitTiling));
    auto ret = hccl.SetCcTiling(static_cast<__gm__ void*>(&mc2CcTiling));
    std::vector<uint64_t> tmpCounts(kRankNum, 10);
    HcclHandle handleId = hccl.AlltoAllV(reinterpret_cast<__gm__ uint8_t*>(0x11), tmpCounts.data(), tmpCounts.data(),
        HcclDataType::HCCL_DATA_TYPE_INT8, reinterpret_cast<__gm__ uint8_t*>(0x11), tmpCounts.data(), tmpCounts.data(),
        HcclDataType::HCCL_DATA_TYPE_INT8);
    EXPECT_EQ(handleId, 0);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v1Msg.version, 1);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v1Msg.ccOpTilingData, reinterpret_cast<uint64_t>(&mc2CcTiling));
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v1Msg.xorCheck, GenXorForHcclMsg(&hcclMsgArea->sendMsgs[0]));
    uint16_t sliceList[kRankNum] = {0, 1, 2, 3, 4, 5, 6, 7};
    for (int i = 0; i < kRankNum; ++i) {
        EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].cnt, i);
        uint16_t sliceId;
        ret = hccl.Iterate<false>(handleId, &sliceId, 1);
        EXPECT_EQ(ret, 1);
        EXPECT_EQ(sliceId, sliceList[i]);
        hccl.Commit(handleId);
        EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].valid, COMMIT_VALID_MASK);
        EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].cnt, i + 1);
        ASSERT_EQ(hccl.Query(handleId), i);
        hcclMsgArea->finishedTurnCnt[0].cnt++;
        ASSERT_EQ(hccl.Query(handleId), i + 1);
        EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
    }
}

TEST_F(HcclCommonTestSuite, All2AllV_FineGrainedRecv)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);
    Mc2InitTilingTest mc2InitTiling;
    Mc2CcTilingTest mc2CcTiling;
    mc2CcTiling.stepSize = 1U;
    mc2CcTiling.opType = (uint32_t)HcclCMDType::HCCL_CMD_ALLTOALLV;
    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam), static_cast<__gm__ void*>(&mc2InitTiling));
    auto ret = hccl.SetCcTiling(static_cast<__gm__ void*>(&mc2CcTiling));
    std::vector<uint64_t> tmpCounts(kRankNum, 10);
    HcclHandle handleId = hccl.AlltoAllV(reinterpret_cast<__gm__ uint8_t*>(0x11), tmpCounts.data(), tmpCounts.data(),
        HcclDataType::HCCL_DATA_TYPE_INT8, reinterpret_cast<__gm__ uint8_t*>(0x11), tmpCounts.data(), tmpCounts.data(),
        HcclDataType::HCCL_DATA_TYPE_INT8);
    EXPECT_EQ(handleId, 0);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v1Msg.version, 1);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v1Msg.ccOpTilingData, reinterpret_cast<uint64_t>(&mc2CcTiling));
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v1Msg.xorCheck, GenXorForHcclMsg(&hcclMsgArea->sendMsgs[0]));
    uint16_t sliceList[kRankNum] = {0, 7, 6, 5, 4, 3, 2, 1};
    uint16_t sliceId;
    for (int i = 0; i < kRankNum; ++i) {
        EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].cnt, i);
        hccl.Commit(handleId);
        EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].valid, COMMIT_VALID_MASK);
        EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].cnt, i + 1);
        hcclMsgArea->finishedTurnCnt[0].cnt++;
        ret = hccl.Iterate<true>(handleId, &sliceId, 1);
        EXPECT_EQ(ret, 1);
        EXPECT_EQ(sliceId, sliceList[i]);
        ASSERT_EQ(hccl.Query(handleId), i + 1);
    }
    EXPECT_EQ(hccl.Iterate<true>(handleId, &sliceId, 1), 0);
}

TEST_F(HcclCommonTestSuite, All2AllV_FineGrainedRecvWithStepSize)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);
    Mc2InitTilingTest mc2InitTiling;
    Mc2CcTilingTest mc2CcTiling;
    mc2CcTiling.stepSize = 4U;
    mc2CcTiling.opType = (uint32_t)HcclCMDType::HCCL_CMD_ALLTOALLV;
    uint8_t repeat = 8U;
    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam), static_cast<__gm__ void*>(&mc2InitTiling));
    auto ret = hccl.SetCcTiling(static_cast<__gm__ void*>(&mc2CcTiling));
    std::vector<uint64_t> tmpCounts(kRankNum, 10);
    HcclHandle handleId = hccl.AlltoAllV<true>(reinterpret_cast<__gm__ uint8_t*>(0x11), tmpCounts.data(),
        tmpCounts.data(), HcclDataType::HCCL_DATA_TYPE_INT8, reinterpret_cast<__gm__ uint8_t*>(0x11), tmpCounts.data(),
        tmpCounts.data(), HcclDataType::HCCL_DATA_TYPE_INT8, repeat);
    EXPECT_EQ(handleId, 0);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v1Msg.version, 1);
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v1Msg.ccOpTilingData, reinterpret_cast<uint64_t>(&mc2CcTiling));
    EXPECT_EQ(hcclMsgArea->sendMsgs[0].addMsg.v1Msg.xorCheck, GenXorForHcclMsg(&hcclMsgArea->sendMsgs[0]));
    EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].valid, COMMIT_VALID_MASK);
    EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].cnt, kRankNum * repeat);
    uint16_t sliceList[kRankNum] = {0, 7, 6, 5, 4, 3, 2, 1};
    uint16_t sliceId[4];
    const uint8_t sliceCnt = sizeof(sliceId) / sizeof(sliceId[0]);
    for (uint8_t i = 0U; i < repeat; ++i) {
        for (uint32_t j = 0; j < kRankNum / mc2CcTiling.stepSize; ++j) {
            hcclMsgArea->finishedTurnCnt[0].cnt += mc2CcTiling.stepSize;
            ret = hccl.Iterate<true>(handleId, sliceId, sliceCnt);
            EXPECT_EQ(ret, sliceCnt);
            for (uint8_t k = 0; k < sizeof(sliceId) / sizeof(sliceId[0]); ++k) {
                EXPECT_EQ(sliceId[k], sliceList[k % sliceCnt + j * sliceCnt]);
            }
            ASSERT_EQ(hccl.Query(handleId), mc2CcTiling.stepSize * (1 + j + i * kRankNum / mc2CcTiling.stepSize));
        }
    }
    EXPECT_EQ(hccl.Iterate<true>(handleId, sliceId, sliceCnt), 0);
}

constexpr HcclServerConfig HCCL_CFG = {CoreType::ON_AIV, 10};
TEST_F(HcclCommonTestSuite, TestHcclConfig)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);
    Mc2InitTilingTest mc2InitTiling;
    Mc2CcTilingTest mc2CcTiling;
    mc2CcTiling.opType = (uint32_t)HcclCMDType::HCCL_CMD_ALLREDUCE;
    Hccl<HcclServerType::HCCL_SERVER_TYPE_AICPU, HCCL_CFG> hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam), static_cast<__gm__ void*>(&mc2InitTiling));
    auto ret = hccl.SetCcTiling(static_cast<__gm__ void*>(&mc2CcTiling));
    HcclHandle handleId =
        hccl.AllReduce(reinterpret_cast<__gm__ uint8_t*>(0x11), reinterpret_cast<__gm__ uint8_t*>(0x11), 100,
            HcclDataType::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, 1);
    EXPECT_EQ(handleId, 0);
}
