/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
class HcclSuiteAIC : public testing::Test {
protected:
    virtual void SetUp()
    {
        AscendC::SetGCoreType(1);
    }
    virtual void TearDown()
    {
        AscendC::SetGCoreType(0);
    }
};

constexpr uint32_t kRankNum = 8U;
constexpr size_t workSpaceSize = sizeof(HcclMsgArea);

HcclCombineOpParam GetHcclCombineOpParam(const vector<uint8_t>& workSpace)
{
    uint64_t buffer[8];
    GM_ADDR CKEOffset = reinterpret_cast<GM_ADDR>(buffer);

    uint64_t buffer1[16 * 8 * 8];
    GM_ADDR XnOffset = reinterpret_cast<GM_ADDR>(buffer1);

    HcclCombineOpParam hcclCombineOpParam{
        reinterpret_cast<uintptr_t>(workSpace.data()), workSpaceSize, 0, kRankNum, 0, {0}, {0}, XnOffset, CKEOffset};
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

// repeat_prepare_commit Repeat = 1调用Prepare接口 预期handleId = 0
TEST_F(HcclSuiteAIC, AllGather_Repeat1)
{
    const HcclServerType serverType = HcclServerType::HCCL_SERVER_TYPE_AICPU;
    std::vector<uint8_t> workSpace(workSpaceSize + 1024 * 100 * 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));

    HcclHandle handleId = hccl.AllGather(reinterpret_cast<__gm__ uint8_t*>(0x1234),
        reinterpret_cast<__gm__ uint8_t*>(0x4321), 100, HcclDataType::HCCL_DATA_TYPE_INT8, 0, 1);
    printf("test handleID = %d.\n", handleId);
    hccl.Commit(handleId);
    printf("test handleID = %d.\n", handleId);
    EXPECT_EQ(handleId, 0);
    EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
}

// repeat_prepare_commit_2_2 repeat = 1调用Prepare接口 for循环调用 预期handleId = 0, 1
TEST_F(HcclSuiteAIC, AllGather_repeat_prepare_commit_2)
{
    const HcclServerType serverType = HcclServerType::HCCL_SERVER_TYPE_AICPU;
    std::vector<uint8_t> workSpace(workSpaceSize + 1024 * 14 * 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    for (int i = 0; i < 2; i++) {
        HcclHandle handleId = hccl.AllGather(reinterpret_cast<__gm__ uint8_t*>(0x11),
            reinterpret_cast<__gm__ uint8_t*>(0x11), 100, HcclDataType::HCCL_DATA_TYPE_INT8, 0, 1);
        hccl.Commit(handleId);
        EXPECT_EQ(handleId, i);
        EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
    }
}

// repeat_prepare_commit_2_1_1调用Prepare接口，repeat = 2, 预期handleId = 0
TEST_F(HcclSuiteAIC, AllGather_repeat_prepare_commit_2_1_1)
{
    const HcclServerType serverType = HcclServerType::HCCL_SERVER_TYPE_AICPU;
    std::vector<uint8_t> workSpace(workSpaceSize + 1024 * 14 * 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    *(hcclCombineOpParam.CKEOffset) = 0x1;
    *(hcclCombineOpParam.CKEOffset + 1 * 8) = 0x1;
    HcclHandle handleId = hccl.AllGather<true>(reinterpret_cast<__gm__ uint8_t*>(0x11),
        reinterpret_cast<__gm__ uint8_t*>(0x11), 100, HcclDataType::HCCL_DATA_TYPE_INT8, 0, 2);
    EXPECT_EQ(handleId, 0);
    hccl.Commit(handleId);
    *(hcclCombineOpParam.CKEOffset + 8 * 8) = 0x1;
    *(hcclCombineOpParam.CKEOffset + 1 * 8 + 8 * 8) = 0x1;
    EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
}

// prepare_commit_2_1_1调用Prepare接口，repeat = 2, 预期handleId = 0
TEST_F(HcclSuiteAIC, AllGather_repeat_prepare_commit_16_1_1)
{
    const HcclServerType serverType = HcclServerType::HCCL_SERVER_TYPE_AICPU;
    std::vector<uint8_t> workSpace(workSpaceSize + 1024 * 14);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    for (int i = 0; i < 16; i++) {
        HcclHandle handleId = hccl.AllGather(reinterpret_cast<__gm__ uint8_t*>(0x11),
            reinterpret_cast<__gm__ uint8_t*>(0x11), 100, HcclDataType::HCCL_DATA_TYPE_INT8, 0, 1);
        *(hcclCombineOpParam.CKEOffset + i * 8 + 8 * 8) = 0x1;
        hccl.Commit(handleId);
        EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
    }
}

// 异常测试: Commit发生在Prepare之前，CommitTurnCnt值不会被写
TEST_F(HcclSuiteAIC, AllGather_CommitBeforePrepare)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024 * 14);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    hccl.Commit(0);
    EXPECT_EQ(hcclMsgArea->commitTurnCnt[0].cnt, 0);
}

// 异常测试: Wait发生在Prepare之前，拦截退出
TEST_F(HcclSuiteAIC, AllGather_WaitBeforePrepare)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024 * 14);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    EXPECT_EQ(hccl.Wait(0), HCCL_FAILED);
}

// 异常测试: Wait发生在Commit之前，拦截退出
TEST_F(HcclSuiteAIC, AllGather_WaitBeforeCommit)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024 * 14);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    HcclHandle handleId =
        hccl.AllReduce(reinterpret_cast<__gm__ uint8_t*>(0x11), reinterpret_cast<__gm__ uint8_t*>(0x11), 100,
            HcclDataType::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, 3);
    ASSERT_EQ(handleId, 0);
    EXPECT_EQ(hccl.Wait(handleId), HCCL_FAILED);
}

// 新增接口SetReduceDataTypeAbility测试
// 异常测试 ReudceOpType者大于3:
TEST_F(HcclSuiteAIC, AllGather_reduceOpType_reserved)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024 * 14);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    bool ret;
    Hccl hccl;
    ret = hccl.SetReduceDataTypeAbility(HcclReduceOp::HCCL_REDUCE_RESERVED, AscendC::HcclDataType::HCCL_DATA_TYPE_BFP16,
        AscendC::HcclDataType::HCCL_DATA_TYPE_BFP16);
    EXPECT_EQ(ret, false);
}

// 异常测试 ReudceOpType大于16:
TEST_F(HcclSuiteAIC, AllGather_DataType_reserved)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024 * 14);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    bool ret;
    Hccl hccl;
    ret = hccl.SetReduceDataTypeAbility(HcclReduceOp::HCCL_REDUCE_SUM, AscendC::HcclDataType::HCCL_DATA_TYPE_FP32,
        AscendC::HcclDataType::HCCL_DATA_TYPE_RESERVED);
    EXPECT_EQ(ret, false);
}

// 异常测试 ReudceOpType大于16:
TEST_F(HcclSuiteAIC, AllGather_DataType_reserved_test)
{
    std::vector<uint8_t> workSpace(workSpaceSize + 1024 * 14);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    bool ret;
    Hccl hccl;
    ret = hccl.SetReduceDataTypeAbility(HcclReduceOp::HCCL_REDUCE_SUM, AscendC::HcclDataType::HCCL_DATA_TYPE_RESERVED,
        AscendC::HcclDataType::HCCL_DATA_TYPE_FP32);
    EXPECT_EQ(ret, false);
}

TEST_F(HcclSuiteAIC, AllGather_Repeat1_SetReduceDataTypeAbility)
{
    const HcclServerType serverType = HcclServerType::HCCL_SERVER_TYPE_AICPU;
    std::vector<uint8_t> workSpace(workSpaceSize + 1024 * 14 * 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    EXPECT_EQ(hccl.SetReduceDataTypeAbility(HcclReduceOp::HCCL_REDUCE_SUM, AscendC::HcclDataType::HCCL_DATA_TYPE_FP32,
                  AscendC::HcclDataType::HCCL_DATA_TYPE_FP32),
        true);
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));
    HcclHandle handleId = hccl.AllGather(reinterpret_cast<__gm__ uint8_t*>(0x11),
        reinterpret_cast<__gm__ uint8_t*>(0x11), 100, HcclDataType::HCCL_DATA_TYPE_INT8, 1);
    *(hcclCombineOpParam.CKEOffset + 8 * 8) = 0x1;
    hccl.Commit(handleId);
    EXPECT_EQ(handleId, 0);
    EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
}

// alltoallv repeat_prepare_commit Repeat = 1调用Prepare接口 预期handleId = 0
TEST_F(HcclSuiteAIC, AllToAllv_prepare)
{
    const HcclServerType serverType = HcclServerType::HCCL_SERVER_TYPE_AICPU;
    std::vector<uint8_t> workSpace(workSpaceSize + 1024 * 100 * 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));

    uint64_t sendCounts[4] = {1};
    uint64_t sdispls[4] = {1};
    uint64_t recvCounts[4] = {1};
    uint64_t rdispls[4] = {1};

    HcclHandle handleId = hccl.AlltoAllV<true>(reinterpret_cast<__gm__ uint8_t*>(0x1234), sendCounts, sdispls,
        HcclDataType::HCCL_DATA_TYPE_INT8, reinterpret_cast<__gm__ uint8_t*>(0x4321), recvCounts, rdispls,
        HcclDataType::HCCL_DATA_TYPE_INT8, 1);

    EXPECT_EQ(handleId, 0);
    EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
}

// AlltoAllvWrite repeat_prepare_commit Repeat = 1调用Prepare接口 预期handleId = 0
TEST_F(HcclSuiteAIC, AlltoAllvWrite_prepare)
{
    const HcclServerType serverType = HcclServerType::HCCL_SERVER_TYPE_AICPU;
    std::vector<uint8_t> workSpace(workSpaceSize + 1024 * 100 * 1024);
    HcclMsgArea* hcclMsgArea = GetHcclMsgArea(workSpace.data());
    HcclCombineOpParam hcclCombineOpParam = GetHcclCombineOpParam(workSpace);

    Hccl hccl;
    hccl.Init(reinterpret_cast<GM_ADDR>(&hcclCombineOpParam));

    HcclHandle handleId = hccl.AlltoAllvWrite<false>(reinterpret_cast<__gm__ uint8_t*>(0x1234),
        reinterpret_cast<__gm__ uint8_t*>(0x1324), reinterpret_cast<__gm__ uint8_t*>(0x4321), 1, 1);

    hccl.Commit(handleId);
    EXPECT_EQ(handleId, 0);
    EXPECT_EQ(hccl.Wait(handleId), HCCL_SUCCESS);
}

} // namespace
