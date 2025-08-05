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
#include "common/atvc_opdef.h"
#include "elewise/common/elewise_common.h"
#include "elewise/elewise_host.h"
#include "reduce/common/reduce_common.h"
#include "reduce/reduce_host.h"
#include "broadcast/common/broadcast_common.h"
#include "broadcast/broadcast_host.h"

class TestAtvcTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {}
    static void TearDownTestCase()
    {}
    virtual void SetUp()
    {}
    void TearDown()
    {}
};

TEST_F(TestAtvcTiling, TestAtvcTilingEleWiseAddCase)
{
    using ADD_OPTRAITS = ATVC::OpTraits<ATVC::OpInputs<float, float>, ATVC::OpOutputs<float>>;
    int32_t eleNum = 8 * 1024;
    ATVC::EleWiseParam param;

    ATVC::Host::CalcEleWiseTiling<ADD_OPTRAITS>(eleNum, param);

    EXPECT_EQ(param.nBufferNum, 2);
}

TEST_F(TestAtvcTiling, TestAtvcTilingReduceSumCase)
{
    using ReduceOpTraits = ATVC::OpTraits<ATVC::OpInputs<float>, ATVC::OpOutputs<float>>;
    std::vector<int64_t> dim{0};
    std::vector<int64_t> shape{8, 1024};
    ATVC::ReduceParam param;
    ATVC::ReducePolicy policy;

    ATVC::Host::CalcReduceTiling<ReduceOpTraits>(shape, dim, &policy, &param);

    EXPECT_EQ(param.nBufferNum, 2);
}

TEST_F(TestAtvcTiling, TestAtvcTilingBroadcastToCase)
{
    using BroadcastOpTraits = ATVC::OpTraits<ATVC::OpInputs<float>, ATVC::OpOutputs<float>>;
    std::vector<int64_t> shapeIn{1, 1024};
    std::vector<int64_t> shapeOut{8, 1024};
    ATVC::BroadcastParam param;
    ATVC::BroadcastPolicy policy;

    ATVC::Host::CalcBroadcastTiling<BroadcastOpTraits>(shapeIn, shapeOut, &policy, &param);

    EXPECT_EQ(param.nBufferNum, 2);
}