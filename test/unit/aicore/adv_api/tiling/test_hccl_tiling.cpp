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
#include "graph/tensor.h"
#include <dlfcn.h>
#define private public
#define protected public
#include "tiling_api.h"
#include "platform_stub.h"
#include "detail/hccl/hccl_tiling_msg.h"
#include "hccl/hccl_tiling.h"
#include "hccl/hccl_common.h"
#include "tiling/platform/platform_ascendc.h"
using namespace ge;
using namespace std;
using namespace optiling;
using namespace AscendC;

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
    EXPECT_EQ(ccTilingConfig.SetDebugMode(1U), TILING_SUCCESS);
    EXPECT_EQ(ccTilingConfig.SetQueueNum(40U), TILING_SUCCESS);
    EXPECT_EQ(ccTilingConfig.SetCommBlockNum(48U), TILING_SUCCESS);
    uint32_t ret = ccTilingConfig.GetTiling(initTilingInner);
    EXPECT_EQ(ret, TILING_SUCCESS);

    opType = 0;
    EXPECT_EQ(ccTilingConfig.SetOpType(opType), TILING_FAILED);
    groupName = "test1";
    EXPECT_EQ(ccTilingConfig.SetGroupName(groupName), TILING_SUCCESS);
    algConfig = "doublering";
    EXPECT_EQ(ccTilingConfig.SetAlgConfig(algConfig), TILING_SUCCESS);
    reduceType = 0;
    EXPECT_EQ(ccTilingConfig.SetReduceType(reduceType), TILING_SUCCESS);
    uint8_t stepSize = 1;
    EXPECT_EQ(ccTilingConfig.SetStepSize(stepSize), TILING_SUCCESS);
    uint8_t skipLocalRankCopy = 1;
    EXPECT_EQ(ccTilingConfig.SetSkipLocalRankCopy(skipLocalRankCopy), TILING_SUCCESS);
    uint8_t skipBufferWindowCopy = 1;
    EXPECT_EQ(ccTilingConfig.SetSkipBufferWindowCopy(skipBufferWindowCopy), TILING_SUCCESS);
    EXPECT_EQ(ccTilingConfig.GetTiling(ccTilingInner), TILING_SUCCESS);
}

TEST_F(TestHcclTiling, Mc2CcTilingConfig_failed1)
{
    // 成员变量边界值校验用例
    ::Mc2CcTiling ccTilingInner;
    string groupName = "test";
    uint32_t opType = 1;
    string algConfig = "fullmesh";
    uint32_t reduceType = 1;
    Mc2CcTilingConfig ccTilingConfig(groupName, opType, algConfig, reduceType);
    EXPECT_EQ(ccTilingConfig.SetOpType(static_cast<uint32_t>(HcclCMDType::HCCL_CMD_MAX)), TILING_FAILED);
    string value129 =
        "012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678";
    string value128 =
        "01234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567";
    EXPECT_EQ(ccTilingConfig.SetGroupName(value129), TILING_FAILED);
    EXPECT_EQ(ccTilingConfig.SetGroupName(value128), TILING_SUCCESS);
    EXPECT_EQ(ccTilingConfig.SetAlgConfig(value129), TILING_FAILED);
    EXPECT_EQ(ccTilingConfig.SetAlgConfig(value128), TILING_SUCCESS);
    EXPECT_EQ(ccTilingConfig.SetReduceType(HCCL_REDUCE_RESERVED), TILING_FAILED);
    EXPECT_EQ(ccTilingConfig.SetSkipLocalRankCopy(2), TILING_FAILED);
    EXPECT_EQ(ccTilingConfig.SetSkipBufferWindowCopy(3), TILING_FAILED);
    EXPECT_EQ(ccTilingConfig.GetTiling(ccTilingInner), TILING_FAILED);
}

TEST_F(TestHcclTiling, Mc2CcTilingConfig_failed2)
{
    // opType是reduce类型是，reduceType要符合范围的校验用例
    ::Mc2InitTiling initTilingInner;
    string groupName = "test";
    uint32_t opType = static_cast<uint32_t>(HcclCMDType::HCCL_CMD_ALLREDUCE);
    string algConfig = "fullmesh";
    uint32_t reduceType = HCCL_REDUCE_RESERVED;
    Mc2CcTilingConfig ccTilingConfig(groupName, opType, algConfig, reduceType);
    uint32_t ret = ccTilingConfig.GetTiling(initTilingInner);
    EXPECT_EQ(ret, TILING_FAILED);

    // opType是非reduce类型是，reduceType没有范围要求
    EXPECT_EQ(ccTilingConfig.SetOpType(static_cast<uint32_t>(HcclCMDType::HCCL_CMD_SEND)), TILING_SUCCESS);
    ret = ccTilingConfig.GetTiling(initTilingInner);
    EXPECT_EQ(ret, TILING_SUCCESS);
}

TEST_F(TestHcclTiling, Mc2CcTilingConfig_failed3)
{
    // 不调用初始化的校验用例
    ::Mc2InitTiling initTilingInner;
    ::Mc2CcTiling ccTilingInner;
    string groupName = "test";
    uint32_t opType = 1;
    string algConfig = "fullmesh";
    uint32_t reduceType = 1;
    Mc2CcTilingConfig ccTilingConfig(groupName, opType, algConfig, reduceType);
    EXPECT_EQ(ccTilingConfig.GetTiling(ccTilingInner), TILING_FAILED);
}
