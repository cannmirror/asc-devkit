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
#include "kernel_operator.h"
#include "detail/api_check/kernel_api_check.h"
#include "../reduce_case_common.h"

class ReduceAnyAPICheck : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp()
    {
        AscendC::KernelRaise::GetInstance().SetRaiseMode(false);
    }
    void TearDown()
    {
        AscendC::KernelRaise::GetInstance().SetRaiseMode(true);
    }
};

TEST_F(ReduceAnyAPICheck, ReduceAnyAPICheckReduceAnySrcShapeLastAxis)
{
    CheckReduceSrcShapeLastAxis<uint8_t, HighLevelApiCheck::CheckFuncReduceAny<uint8_t, AscendC::Pattern::Reduce::AR>>(
        "ReduceAny");
}

TEST_F(ReduceAnyAPICheck, ReduceAnyAPICheckReduceAnySrcShapeArFirstAxis)
{
    CheckReduceSrcShapeArFirstAxis<float, HighLevelApiCheck::CheckFuncReduceAny<float, AscendC::Pattern::Reduce::AR>>(
        "ReduceAny");
}

TEST_F(ReduceAnyAPICheck, ReduceAnyAPICheckReduceAnySrcShapeRaLastAxis)
{
    CheckReduceSrcShapeRaLastAxis<float, HighLevelApiCheck::CheckFuncReduceAny<float, AscendC::Pattern::Reduce::AR>>(
        "ReduceAny");
}

TEST_F(ReduceAnyAPICheck, ReduceAnyAPICheckReduceAnySrcShapeSrcSize)
{
    CheckReduceSrcShapeSrcSize<float, HighLevelApiCheck::CheckFuncReduceAny<float, AscendC::Pattern::Reduce::RA>>(
        "ReduceAny");
}

TEST_F(ReduceAnyAPICheck, ReduceAnyAPICheckReduceAnySrcInnerPad)
{
    CheckReduceSrcInnerPad<float, HighLevelApiCheck::CheckFuncReduceAny<float, AscendC::Pattern::Reduce::RA>>(
        "ReduceAny");
}

TEST_F(ReduceAnyAPICheck, ReduceAnyAPICheckReduceAnySrcPos)
{
    CheckReduceSrcPos<float, HighLevelApiCheck::CheckFuncReduceAny<float, AscendC::Pattern::Reduce::RA>>("ReduceAny");
}

TEST_F(ReduceAnyAPICheck, ReduceAnyAPICheckReduceAnyDstPos)
{
    CheckReduceDstPos<float, HighLevelApiCheck::CheckFuncReduceAny<float, AscendC::Pattern::Reduce::RA>>("ReduceAny");
}

TEST_F(ReduceAnyAPICheck, ReduceAnyAPICheckTmpPos)
{
    CheckReduceTmpPos<float, HighLevelApiCheck::CheckFuncReduceAny<float, AscendC::Pattern::Reduce::RA>>("ReduceAny");
}

TEST_F(ReduceAnyAPICheck, ReduceAnyAPICheckOverlap)
{
    CheckReduceOverlap<float, HighLevelApiCheck::CheckFuncReduceAny<float, AscendC::Pattern::Reduce::RA>>("ReduceAny");
}
