/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <gtest/gtest.h>
#include "graph/tensor.h"
#include <dlfcn.h>
#define private public
#define protected public
#include "tiling_api.h"
#include "platform_stub.h"
#include "include/adv_api/hccl/internal/hccl_tiling_msg.h"
#include "include/adv_api/hccl/hccl_tiling.h"
#include "include/adv_api/hccl/hccl_common.h"
#include "tiling/platform/platform_ascendc.h"
using namespace ge;
using namespace std;
using namespace optiling;
using namespace AscendC;
using namespace HcclApi;

namespace {
using HcclResult = uint32_t;
using HcclComm = void*;
HcclResult HcomGetCommHandleByGroup(const char *group, HcclComm *comm)
{
    return 0U;
}

HcclResult HcclAllocComResourceByTiling(HcclComm comm, void *stream, void *tiling, void **context)
{
    return 0U;
}

HcclResult HcclGetRankId(HcclComm comm, uint32_t *rank)
{
    return 0U;
}

HcclResult HcclGetRankSize(HcclComm comm, uint32_t *rankSize)
{
    return 0U;
}

HcclResult CommGetKFCWorkSpace(HcclComm comm, void **addr, uint64_t *size)
{
    return 0U;
}

map<string, void *> HcclFuncMap = {
        {"HcomGetCommHandleByGroup",     (void *) HcomGetCommHandleByGroup},
        {"HcclAllocComResourceByTiling", (void *) HcclAllocComResourceByTiling},
        {"HcclGetRankId",                (void *) HcclGetRankId},
        {"HcclGetRankSize",              (void *) HcclGetRankSize},
        {"CommGetKFCWorkSpace",          (void *) CommGetKFCWorkSpace},
};

void *DlsymStub(void *handle, const char *symbol)
{
    if (symbol == nullptr) {
        return nullptr;
    }
    string symStr = symbol;
    auto it = HcclFuncMap.find(symStr);
    if (it != HcclFuncMap.cend()) {
        return it->second;
    }
    return nullptr;
}
}

class TestHcclTiling : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {}
    void TearDown() {}
};

TEST_F(TestHcclTiling, Mc2CcTilingConfig_normal)
{
    ::Mc2InitTiling initTilingInner;
    ::Mc2CcTiling ccTilingInner;
    string groupName = "test";
    uint32_t opType = 1;
    string algConfig = "fullmesh";
    uint32_t reduceType = 1;
    Mc2CcTilingConfig ccTilingConfig(groupName, opType, algConfig, reduceType);
    EXPECT_EQ(ccTilingConfig.SetDebugMode(1U), EXIT_SUCCESS);
    EXPECT_EQ(ccTilingConfig.SetQueueNum(40U), EXIT_SUCCESS);
    EXPECT_EQ(ccTilingConfig.SetCommBlockNum(48U), EXIT_SUCCESS);
    uint32_t ret = ccTilingConfig.GetTiling(initTilingInner);
    EXPECT_EQ(ret, EXIT_SUCCESS);

    opType = 0;
    EXPECT_NE(ccTilingConfig.SetOpType(opType), EXIT_SUCCESS);
    groupName = "test1";
    EXPECT_EQ(ccTilingConfig.SetGroupName(groupName), EXIT_SUCCESS);
    algConfig = "doublering";
    EXPECT_EQ(ccTilingConfig.SetAlgConfig(algConfig), EXIT_SUCCESS);
    reduceType = 0;
    EXPECT_EQ(ccTilingConfig.SetReduceType(reduceType), EXIT_SUCCESS);
    uint8_t stepSize = 1;
    EXPECT_EQ(ccTilingConfig.SetStepSize(stepSize), EXIT_SUCCESS);
    uint8_t skipLocalRankCopy = 1;
    EXPECT_EQ(ccTilingConfig.SetSkipLocalRankCopy(skipLocalRankCopy), EXIT_SUCCESS);
    uint8_t skipBufferWindowCopy = 1;
    EXPECT_EQ(ccTilingConfig.SetSkipBufferWindowCopy(skipBufferWindowCopy), EXIT_SUCCESS);
    uint8_t commEngine = 1;
    EXPECT_EQ(ccTilingConfig.SetCommEngine(commEngine), EXIT_SUCCESS);
    EXPECT_NE(ccTilingConfig.GetTiling(ccTilingInner), EXIT_SUCCESS);
}

TEST_F(TestHcclTiling, Mc2CcTilingConfig_failed1)
{
    // member variable boundary value validation use case
    ::Mc2CcTiling ccTilingInner;
    string groupName = "test";
    uint32_t opType = 1;
    string algConfig = "fullmesh";
    uint32_t reduceType = 1;
    Mc2CcTilingConfig ccTilingConfig(groupName, opType, algConfig, reduceType);
    EXPECT_NE(ccTilingConfig.SetOpType(static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALL)), EXIT_SUCCESS);
    string value129 = "012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678";
    string value128 = "01234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567";
    EXPECT_NE(ccTilingConfig.SetGroupName(value129), EXIT_SUCCESS);
    EXPECT_EQ(ccTilingConfig.SetGroupName(value128), EXIT_SUCCESS);
    EXPECT_NE(ccTilingConfig.SetAlgConfig(value129), EXIT_SUCCESS);
    EXPECT_EQ(ccTilingConfig.SetAlgConfig(value128), EXIT_SUCCESS);
    EXPECT_NE(ccTilingConfig.SetReduceType(HCCL_REDUCE_RESERVED), EXIT_SUCCESS);
    EXPECT_NE(ccTilingConfig.SetSkipLocalRankCopy(2), EXIT_SUCCESS);
    EXPECT_NE(ccTilingConfig.SetSkipBufferWindowCopy(3), EXIT_SUCCESS);
    EXPECT_NE(ccTilingConfig.GetTiling(ccTilingInner), EXIT_SUCCESS);
}

TEST_F(TestHcclTiling, Mc2CcTilingConfig_failed2)
{
    // when opType is of the reduce type, reduceType must comply with the range validation cases
    ::Mc2InitTiling initTilingInner;
    string groupName = "test";
    uint32_t opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLREDUCE);
    string algConfig = "fullmesh";
    uint32_t reduceType = HCCL_REDUCE_RESERVED;
    Mc2CcTilingConfig ccTilingConfig(groupName, opType, algConfig, reduceType);
    uint32_t ret = ccTilingConfig.GetTiling(initTilingInner);
    EXPECT_NE(ret, EXIT_SUCCESS);

    // when opType is not of the reduce type, there are not range requirements for reduceType
    EXPECT_EQ(ccTilingConfig.SetOpType(static_cast<uint32_t>(HcclCMDType::HCCL_CMD_SEND)), EXIT_SUCCESS);
    ret = ccTilingConfig.GetTiling(initTilingInner);
    EXPECT_EQ(ret, EXIT_SUCCESS);
}

TEST_F(TestHcclTiling, Mc2CcTilingConfig_failed3)
{
    // do not invoke the initilazation validation test case
    ::Mc2InitTiling initTilingInner;
    ::Mc2CcTiling ccTilingInner;
    string groupName = "test";
    uint32_t opType = 1;
    string algConfig = "fullmesh";
    uint32_t reduceType = 1;
    Mc2CcTilingConfig ccTilingConfig(groupName, opType, algConfig, reduceType);
    EXPECT_NE(ccTilingConfig.GetTiling(ccTilingInner), EXIT_SUCCESS);
}

TEST_F(TestHcclTiling, Mc2CcTilingConfig_SetReduceType_ReduceOp)
{
    const char *groupName = "testGroup";
    uint32_t opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
    std::string algConfig = "ReduceScatter=level0:doublering";
    uint32_t reduceType = static_cast<uint32_t>(HcclReduceOp::HCCL_REDUCE_RESERVED);
    uint8_t srcDataType = static_cast<uint32_t>(HcclDataType::HCCL_DATA_TYPE_FP16);
    uint8_t dstDataType = static_cast<uint32_t>(HcclDataType::HCCL_DATA_TYPE_FP16);

    Mc2CcTilingConfig mc2CcTilingConfig(groupName, opType, algConfig, reduceType, srcDataType, dstDataType);
    EXPECT_EQ(mc2CcTilingConfig.SetReduceType(HcclReduceOp::HCCL_REDUCE_SUM, srcDataType, dstDataType), EXIT_SUCCESS);

    // invalid dstDataType
    dstDataType = -1;
    EXPECT_EQ(mc2CcTilingConfig.SetReduceType(HcclReduceOp::HCCL_REDUCE_SUM, srcDataType, dstDataType), EXIT_FAILURE);

    // invalid srcDataType
    dstDataType = static_cast<uint32_t>(HcclDataType::HCCL_DATA_TYPE_FP16);
    srcDataType = -1;
    EXPECT_EQ(mc2CcTilingConfig.SetReduceType(HcclReduceOp::HCCL_REDUCE_SUM, srcDataType, dstDataType), EXIT_FAILURE);
}

TEST_F(TestHcclTiling, Mc2CcTilingConfig_SetReduceType_NotReduceOp)
{
    const char *groupName = "testGroup";
    uint32_t opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLGATHER);
    std::string algConfig = "AllGather=level0:doublering";
    uint32_t reduceType = static_cast<uint32_t>(HcclReduceOp::HCCL_REDUCE_RESERVED);
    uint8_t srcDataType = static_cast<uint32_t>(HcclDataType::HCCL_DATA_TYPE_FP16);
    uint8_t dstDataType = static_cast<uint32_t>(HcclDataType::HCCL_DATA_TYPE_FP16);

    Mc2CcTilingConfig mc2CcTilingConfig(groupName, opType, algConfig, reduceType, srcDataType, dstDataType);
    EXPECT_EQ(mc2CcTilingConfig.SetReduceType(HcclReduceOp::HCCL_REDUCE_SUM, srcDataType, dstDataType), EXIT_SUCCESS);
}