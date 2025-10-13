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
#include "impl/adv_api/detail/api_check/kernel_api_check.h"
#include "../reduce_case_common.h"


class ReduceMaxAPICheck : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {
        AscendC::SetGCoreType(2);
        AscendC::KernelRaise::GetInstance().SetRaiseMode(false);
    }
    void TearDown() {
        AscendC::SetGCoreType(0);
        AscendC::KernelRaise::GetInstance().SetRaiseMode(true);
    }
};


TEST_F(ReduceMaxAPICheck, ReduceMaxAPICheckReduceMaxSrcShapeLastAxis)
{
    CheckReduceSrcShapeLastAxis<float, 
        HighLevelApiCheck::CheckFuncReduceMax<float, AscendC::Pattern::Reduce::AR>>("ReduceMax");

}

TEST_F(ReduceMaxAPICheck, ReduceMaxAPICheckReduceMaxSrcShapeArFirstAxis)
{
    CheckReduceSrcShapeArFirstAxis<float, 
        HighLevelApiCheck::CheckFuncReduceMax<float, AscendC::Pattern::Reduce::AR>>("ReduceMax");

}

TEST_F(ReduceMaxAPICheck, ReduceMaxAPICheckReduceMaxSrcShapeRaLastAxis)
{
    CheckReduceSrcShapeRaLastAxis<float, 
        HighLevelApiCheck::CheckFuncReduceMax<float, AscendC::Pattern::Reduce::AR>>("ReduceMax");

}

TEST_F(ReduceMaxAPICheck, ReduceMaxAPICheckReduceMaxSrcShapeSrcSize)
{
    CheckReduceSrcShapeSrcSize<float, 
        HighLevelApiCheck::CheckFuncReduceMax<float, AscendC::Pattern::Reduce::RA>>("ReduceMax");

}

TEST_F(ReduceMaxAPICheck, ReduceMaxAPICheckReduceMaxSrcInnerPad)
{
    CheckReduceSrcInnerPad<float, 
        HighLevelApiCheck::CheckFuncReduceMax<float, AscendC::Pattern::Reduce::RA>>("ReduceMax");

}

TEST_F(ReduceMaxAPICheck, ReduceMaxAPICheckReduceMaxSrcPos)
{
    CheckReduceSrcPos<float, 
        HighLevelApiCheck::CheckFuncReduceMax<float, AscendC::Pattern::Reduce::RA>>("ReduceMax");

}

TEST_F(ReduceMaxAPICheck, ReduceMaxAPICheckReduceMaxDstPos)
{
    CheckReduceDstPos<float, 
        HighLevelApiCheck::CheckFuncReduceMax<float, AscendC::Pattern::Reduce::RA>>("ReduceMax");

}


TEST_F(ReduceMaxAPICheck, ReduceMaxAPICheckTmpPos)
{
    CheckReduceTmpPos<float, 
        HighLevelApiCheck::CheckFuncReduceMax<float, AscendC::Pattern::Reduce::RA>>("ReduceMax");

}

TEST_F(ReduceMaxAPICheck, ReduceMaxAPICheckOverlap)
{
    CheckReduceOverlap<float, 
        HighLevelApiCheck::CheckFuncReduceMax<float, AscendC::Pattern::Reduce::RA>>("ReduceMax");

}
