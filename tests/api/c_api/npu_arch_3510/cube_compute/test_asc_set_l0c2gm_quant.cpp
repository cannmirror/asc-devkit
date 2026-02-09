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

class TestCubeComputeSetL0c2gmQuant : public testing::Test { 
protected:
    void SetUp() {
        g_c_api_core_type = C_API_AIC_TYPE;
    }
    void TearDown() {
        g_c_api_core_type = C_API_AIV_TYPE;
    }
};

namespace {
void set_fpc_quant_Stub(uint64_t config)
{
    uint64_t conf = 31488;
    EXPECT_EQ(conf, config);
}
}

TEST_F(TestCubeComputeSetL0c2gmQuant, set_l0c2gm_quant_Succ)
{
    MOCKER(set_fpc, void(uint64_t))
        .times(1)
        .will(invoke(set_fpc_quant_Stub));
    uint64_t config = 123;
    
    asc_set_l0c2gm_quant(config);
    GlobalMockObject::verify();
}
