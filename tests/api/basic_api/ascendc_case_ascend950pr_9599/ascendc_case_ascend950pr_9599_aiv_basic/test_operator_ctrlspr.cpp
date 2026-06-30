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
#include "kernel_operator.h"
#include "mockcpp/mockcpp.hpp"

using namespace AscendC;

template <SaturationMode mode, int8_t ctrlBit, int64_t enabledValue, int64_t disabledValue>
void CheckSaturationFlag()
{
    SetSaturationFlag<mode>(true);
    EXPECT_EQ(GetSaturationFlag<mode>(), true);
    EXPECT_EQ((GetCtrlSpr<ctrlBit, ctrlBit>()), enabledValue);

    SetSaturationFlag<mode>(false);
    EXPECT_EQ(GetSaturationFlag<mode>(), false);
    EXPECT_EQ((GetCtrlSpr<ctrlBit, ctrlBit>()), disabledValue);

    SetSaturationFlag<mode>(true);
    EXPECT_EQ(GetSaturationFlag<mode>(), true);
    EXPECT_EQ((GetCtrlSpr<ctrlBit, ctrlBit>()), enabledValue);
}

void CtrlSprKernel()
{
    CheckSaturationFlag<SaturationMode::FLOAT, 48, 0, 1>();
    CheckSaturationFlag<SaturationMode::FLOAT8, 50, 0, 1>();
    CheckSaturationFlag<SaturationMode::INT, 53, 1, 0>();
    CheckSaturationFlag<SaturationMode::CAST, 59, 0, 1>();

    SetSaturationStrategy<OverrideStrategy::USE_API>();
    EXPECT_EQ(GetSaturationStrategy(), OverrideStrategy::USE_API);
    EXPECT_EQ((GetCtrlSpr<60, 60>()), 0);

    SetSaturationStrategy<OverrideStrategy::USE_GLOBAL>();
    EXPECT_EQ(GetSaturationStrategy(), OverrideStrategy::USE_GLOBAL);
    EXPECT_EQ((GetCtrlSpr<60, 60>()), 1);

    SetSaturationStrategy<OverrideStrategy::USE_API>();
    EXPECT_EQ(GetSaturationStrategy(), OverrideStrategy::USE_API);
    EXPECT_EQ((GetCtrlSpr<60, 60>()), 0);
}

struct CtrlSprTestParams {
    void (*calFunc)();
};

class CtrlSprTestsuite : public testing::Test, public testing::WithParamInterface<CtrlSprTestParams> {
protected:
    void SetUp() { AscendC::SetGCoreType(2); }
    void TearDown()
    {
        AscendC::SetGCoreType(0);
        GlobalMockObject::verify();
    }
};

INSTANTIATE_TEST_CASE_P(TEST_CTRL_SPR, CtrlSprTestsuite, ::testing::Values(CtrlSprTestParams{CtrlSprKernel}));

TEST_P(CtrlSprTestsuite, CtrlSprTestCase)
{
    auto param = GetParam();
    param.calFunc();
}
