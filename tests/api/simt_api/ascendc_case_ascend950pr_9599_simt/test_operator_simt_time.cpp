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

#undef __CHECK_FEATURE_AT_PRECOMPILE
#define __simt_callee__ 
#define __gm__ 
#define __aicore__
uint64_t __cce_simt_get_CLOCK64();
#include "utils/debug/asc_time.h"

class ClockTestsuite : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

TEST_F(ClockTestsuite, ClockTest)
{
    uint64_t t1 = __asc_simt_vf::clock();
    uint64_t t2 = __asc_simt_vf::clock();
    EXPECT_LE(t1, t2);
}
