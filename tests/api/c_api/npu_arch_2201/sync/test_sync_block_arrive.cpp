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
#include <mockcpp/mockcpp.hpp>
#include "c_api/stub/cce_stub.h"
#include "c_api/asc_simd.h"
#include "c_api/utils_intf.h"

class TestSyncBlkArrive : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

namespace {
void asc_sync_block_arrive_stub(pipe_t pipe, uint64_t config)
{
    EXPECT_EQ(273, config);
}
}

TEST_F(TestSyncBlkArrive, sync_block_arrive_Succ)
{
    uint8_t mode = 1;
    int64_t flagID = 1;
    uint64_t config = 273;
    pipe_t pipe = PIPE_S;
    MOCKER_CPP(ffts_cross_core_sync, void(pipe_t, uint64_t))
            .times(1)
            .will(invoke(asc_sync_block_arrive_stub));

    asc_sync_block_arrive(pipe, mode, flagID);
    GlobalMockObject::verify();
}