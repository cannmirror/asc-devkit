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

class ReduceProdAPICheck : public testing::Test {
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

TEST_F(ReduceProdAPICheck, ReduceProdAPICheckReduceProdSrcShapeLastAxis)
{
    CheckReduceSrcShapeLastAxis<float, HighLevelApiCheck::CheckFuncReduceProd<float, AscendC::Pattern::Reduce::AR>>(
        "ReduceProd");
}

TEST_F(ReduceProdAPICheck, ReduceProdAPICheckReduceProdSrcShapeArFirstAxis)
{
    CheckReduceSrcShapeArFirstAxis<float, HighLevelApiCheck::CheckFuncReduceProd<float, AscendC::Pattern::Reduce::AR>>(
        "ReduceProd");
}

TEST_F(ReduceProdAPICheck, ReduceProdAPICheckReduceProdSrcShapeRaLastAxis)
{
    CheckReduceSrcShapeRaLastAxis<float, HighLevelApiCheck::CheckFuncReduceProd<float, AscendC::Pattern::Reduce::AR>>(
        "ReduceProd");
}

TEST_F(ReduceProdAPICheck, ReduceProdAPICheckReduceProdSrcShapeSrcSize)
{
    CheckReduceSrcShapeSrcSize<float, HighLevelApiCheck::CheckFuncReduceProd<float, AscendC::Pattern::Reduce::RA>>(
        "ReduceProd");
}

TEST_F(ReduceProdAPICheck, ReduceProdAPICheckReduceProdSrcInnerPad)
{
    CheckReduceSrcInnerPad<float, HighLevelApiCheck::CheckFuncReduceProd<float, AscendC::Pattern::Reduce::RA>>(
        "ReduceProd");
}

TEST_F(ReduceProdAPICheck, ReduceProdAPICheckReduceProdSrcPos)
{
    CheckReduceSrcPos<float, HighLevelApiCheck::CheckFuncReduceProd<float, AscendC::Pattern::Reduce::RA>>("ReduceProd");
}

TEST_F(ReduceProdAPICheck, ReduceProdAPICheckReduceProdDstPos)
{
    CheckReduceDstPos<float, HighLevelApiCheck::CheckFuncReduceProd<float, AscendC::Pattern::Reduce::RA>>("ReduceProd");
}

TEST_F(ReduceProdAPICheck, ReduceProdAPICheckTmpPos)
{
    CheckReduceTmpPos<float, HighLevelApiCheck::CheckFuncReduceProd<float, AscendC::Pattern::Reduce::RA>>("ReduceProd");
}

TEST_F(ReduceProdAPICheck, ReduceProdAPICheckOverlap)
{
    CheckReduceOverlap<float, HighLevelApiCheck::CheckFuncReduceProd<float, AscendC::Pattern::Reduce::RA>>(
        "ReduceProd");
}
