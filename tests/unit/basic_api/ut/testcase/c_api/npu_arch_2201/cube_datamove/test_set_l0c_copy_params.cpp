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
#include <mockcpp/mockcpp.hpp>
#include "c_api/stub/cce_stub.h"
#include "c_api/asc_simd.h"

class TestCubeComputeSetL0cCopyParams : public testing::Test { 
protected:
    void SetUp() {g_c_api_core_type = C_API_AIC_TYPE;}
    void TearDown() {g_c_api_core_type = C_API_AIC_TYPE;}
};

namespace {
void set_set_l0c_copy_params_Stub(uint16_t nd_num, uint16_t src_nd_stride, uint16_t dst_nd_stride)
{
    EXPECT_EQ(1, nd_num);
    EXPECT_EQ(4, src_nd_stride);
    EXPECT_EQ(8, dst_nd_stride);
}
}

TEST_F(TestCubeComputeSetL0cCopyParams, set_l0c_copy_params_Succ)
{
    MOCKER(asc_set_l0c_copy_params, void(uint16_t, uint16_t, uint16_t))
        .times(1)
        .will(invoke(set_set_l0c_copy_params_Stub));
    uint64_t nd_num = 1;
    uint64_t src_nd_stride = 4;
    uint64_t dst_nd_stride = 8;
    
    asc_set_l0c_copy_params(nd_num, src_nd_stride, dst_nd_stride);
    GlobalMockObject::verify();
}
