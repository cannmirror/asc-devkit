/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 */
#include <gtest/gtest.h>
#include <vector>
#define private public
#include "kernel_operator.h"

using namespace std;
using namespace AscendC;
using namespace HcclApi;

namespace {
class HcclSuiteAIC : public testing::Test {
protected:
    virtual void SetUp(){
        AscendC::SetGCoreType(1);
    }
    virtual void TearDown(){
        AscendC::SetGCoreType(0);
    }
};

constexpr uint32_t kRankNum = 8U;
constexpr size_t workSpaceSize = sizeof(HcclMsgArea);

HcclCombineOpParam GetHcclCombineOpParam(const vector<uint8_t> &workSpace) {

    uint64_t buffer[8];
    GM_ADDR ckeOffset = reinterpret_cast<GM_ADDR>(buffer);

    uint64_t buffer1[16 * 8 * 8];
    GM_ADDR xnOffset = reinterpret_cast<GM_ADDR>(buffer1);

    HcclCombineOpParam hcclCombineOpParam{
            reinterpret_cast<uintptr_t>(workSpace.data()),
            workSpaceSize,
            0,
            kRankNum,
            0,
            {0},
            {0},
            xnOffset,
            ckeOffset};
    return hcclCombineOpParam;
}

HcclMsgArea *GetHcclMsgArea(uint8_t *workspaceGM) {
    uint64_t msgAddr = reinterpret_cast<uintptr_t>(workspaceGM);
    if (msgAddr & 0x1ff) {
        msgAddr = (msgAddr & (~((uint64_t)0x1ff))) + 0x200;
    }
    return reinterpret_cast<HcclMsgArea *>(msgAddr);
}

// repeat_prepare_commit Repeat = 1 Call the Prepare interface Expected handleId = 0
TEST_F(HcclSuiteAIC, AllGather_Repeat1)
{
    const HcclServerType serverType = HcclServerType::HCCL_SERVER_TYPE_AICPU;
    std::vector<uint8_t> workSpace(workSpaceSize + 1024 * 100 * 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));

    HcclHandle handleId = hccl.AllGather(reinterpret_cast<__gm__ uint8_t*>(0x1234),
                                         reinterpret_cast<__gm__ uint8_t*>(0x4321), 100,
                                         HcclDataType::HCCL_DATA_TYPE_INT8, 0, 1);
    hccl.Commit(handleId);
    EXPECT_EQ(handleId, 0);
    EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
}

// repeat_prepare_commit_2_2 repeat = 1 calls the Prepare interface for loop call expected handleId = 0, 1
TEST_F(HcclSuiteAIC, AllGather_repeat_prepare_commit_2)
{
    const HcclServerType serverType = HcclServerType::HCCL_SERVER_TYPE_AICPU;
    std::vector<uint8_t> workSpace(workSpaceSize + 1024 * 14 * 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    for (int i = 0 ; i < 2; i++) {
         HcclHandle handleId = hccl.AllGather(reinterpret_cast<__gm__ uint8_t*>(0x11),
                                         reinterpret_cast<__gm__ uint8_t*>(0x11), 100,
                                         HcclDataType::HCCL_DATA_TYPE_INT8, 0, 1);
        hccl.Commit(handleId);
        EXPECT_EQ(handleId, i);
        EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
    }
}

// repeat_prepare_commit_2_1_1 calls the Prepare interface, repeat = 2, expected handleId = 0
TEST_F(HcclSuiteAIC, AllGather_repeat_prepare_commit_2_1_1)
{
    const HcclServerType serverType = HcclServerType::HCCL_SERVER_TYPE_AICPU;
    std::vector<uint8_t> workSpace(workSpaceSize + 1024 * 14 * 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    *(hcclCombineOpParam.ckeOffset) = 0x1;
    *(hcclCombineOpParam.ckeOffset + 1 * 8) = 0x1;
    HcclHandle handleId = hccl.AllGather(reinterpret_cast<__gm__ uint8_t*>(0x11),
                                         reinterpret_cast<__gm__ uint8_t*>(0x11), 100,
                                         HcclDataType::HCCL_DATA_TYPE_INT8, 0, 2);
    EXPECT_EQ(handleId, 0);
    hccl.Commit(handleId);
     *(hcclCombineOpParam.ckeOffset + 8 * 8) = 0x1;
    *(hcclCombineOpParam.ckeOffset + 1 * 8 + 8 * 8) = 0x1;
    EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
}

// prepare_commit_2_1_1 calls the Prepare interface, repeat = 2, expected handleId = 0
TEST_F(HcclSuiteAIC, AllGather_repeat_prepare_commit_16_1_1)
{
    const HcclServerType serverType = HcclServerType::HCCL_SERVER_TYPE_AICPU;
    std::vector<uint8_t> workSpace(workSpaceSize + 1024 * 14);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    for (int i = 0 ; i < 16; i++) {
        HcclHandle handleId = hccl.AllGather(reinterpret_cast<__gm__ uint8_t*>(0x11),
                                         reinterpret_cast<__gm__ uint8_t*>(0x11), 100,
                                         HcclDataType::HCCL_DATA_TYPE_INT8, 0, 1);
        *(hcclCombineOpParam.ckeOffset + i *8 + 8 * 8) = 0x1;
        hccl.Commit(handleId);
        EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
    }
}

// Abnormal test: Commit occurs before Prepare, and the CommitTurnCnt value will not be written
TEST_F(HcclSuiteAIC, AllGather_CommitBeforePrepare)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024 * 14);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    hccl.Commit(0);
    EXPECT_EQ(hcclMsgArea->commMsg.singleMsg.commitTurnCnt[0].cnt, 0);
}

// Abnormal test: Wait occurs before Prepare, intercept exit
TEST_F(HcclSuiteAIC, AllGather_WaitBeforePrepare)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024 * 14);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    EXPECT_EQ(hccl.Wait(0), HCCL_FAILED);
}

// Abnormal test: Wait occurs before Commit, intercepting exit
TEST_F(HcclSuiteAIC, AllGather_WaitBeforeCommit)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024 * 14);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    HcclHandle handleId = hccl.AllReduce(reinterpret_cast<__gm__ uint8_t*>(0x11),
                                         reinterpret_cast<__gm__ uint8_t*>(0x11), 100,
                                         HcclDataType::HCCL_DATA_TYPE_INT8,
                                         HcclReduceOp::HCCL_REDUCE_SUM, 3);
    ASSERT_EQ(handleId, 0);
    EXPECT_EQ(hccl.Wait(handleId), HCCL_FAILED);
}

// Added interface SetReduceDataTypeAbility test, exception test ReudceOpType is greater than 3
TEST_F(HcclSuiteAIC, AllGather_reduceOpType_reserved)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024 * 14);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    bool ret;
    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl;
    ret = hccl.SetReduceDataTypeAbility(HcclReduceOp::HCCL_REDUCE_RESERVED, AscendC::HcclDataType::HCCL_DATA_TYPE_BFP16, AscendC::HcclDataType::HCCL_DATA_TYPE_BFP16);
    EXPECT_EQ(ret, false);
}

// Abnormal test: ReudceOpType is greater than 16
TEST_F(HcclSuiteAIC, AllGather_DataType_reserved)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024 * 14);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    bool ret;
    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl;
    ret = hccl.SetReduceDataTypeAbility(HcclReduceOp::HCCL_REDUCE_SUM, AscendC::HcclDataType::HCCL_DATA_TYPE_FP32, AscendC::HcclDataType::HCCL_DATA_TYPE_RESERVED);
    EXPECT_EQ(ret, false);
}

// Abnormal test: ReudceOpType is greater than 16
TEST_F(HcclSuiteAIC, AllGather_DataType_reserved_test)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024 * 14);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    bool ret;
    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl;
    ret = hccl.SetReduceDataTypeAbility(HcclReduceOp::HCCL_REDUCE_SUM, AscendC::HcclDataType::HCCL_DATA_TYPE_RESERVED, AscendC::HcclDataType::HCCL_DATA_TYPE_FP32);
    EXPECT_EQ(ret, false);
}

TEST_F(HcclSuiteAIC, AllGather_Repeat1_SetReduceDataTypeAbility)
{
    const HcclServerType serverType = HcclServerType::HCCL_SERVER_TYPE_AICPU;
    std::vector<uint8_t> workSpace(workSpaceSize + 1024 * 14 * 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl;
    EXPECT_EQ(hccl.SetReduceDataTypeAbility(HcclReduceOp::HCCL_REDUCE_SUM,
        AscendC::HcclDataType::HCCL_DATA_TYPE_FP32, AscendC::HcclDataType::HCCL_DATA_TYPE_FP32), true);
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    HcclHandle handleId = hccl.AllGather(reinterpret_cast<__gm__ uint8_t*>(0x11),
                                         reinterpret_cast<__gm__ uint8_t*>(0x11), 100,
                                         HcclDataType::HCCL_DATA_TYPE_INT8, 1);
    *(hcclCombineOpParam.ckeOffset + 8 * 8) = 0x1;
    hccl.Commit(handleId);
    EXPECT_EQ(handleId, 0);
    EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
}

// alltoallv repeat_prepare_commit Repeat = 1 Call the Prepare interface Expected handleId = 0
TEST_F(HcclSuiteAIC, AllToAllv_prepare)
{
    const HcclServerType serverType = HcclServerType::HCCL_SERVER_TYPE_AICPU;
    std::vector<uint8_t> workSpace(workSpaceSize + 1024 * 100 * 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl<HcclServerType::HCCL_SERVER_TYPE_CCU> hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));

    uint64_t sendCounts[4] = {1};
    uint64_t sdispls[4] = {1};
    uint64_t recvCounts[4] = {1};
    uint64_t rdispls[4] = {1};

    HcclHandle handleId = hccl.AlltoAllV<true>(reinterpret_cast<__gm__ uint8_t *>(0x1234), sendCounts, sdispls, HcclDataType::HCCL_DATA_TYPE_INT8, 
                                               reinterpret_cast<__gm__ uint8_t *>(0x4321), recvCounts, rdispls, HcclDataType::HCCL_DATA_TYPE_INT8, 1);

    EXPECT_EQ(handleId, 0);
    EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
}

}