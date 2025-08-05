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
#include "common/const_def.h"
#include "common/kernel_utils.h"
#include "elewise/common/elewise_common.h"
#include "elewise/elewise_op_template.h"

using ADD_OPTRAITS = ATVC::OpTraits<ATVC::OpInputs<float, float>, ATVC::OpOutputs<float>>;

template <typename Traits>
struct AddComputeFunc {
    template <typename T>
    __aicore__ inline void operator()(AscendC::LocalTensor<T> a, AscendC::LocalTensor<T> b, AscendC::LocalTensor<T> c)
    {
        AscendC::Add(c, a, b, c.GetSize());
    }
};

template <class Traits>
__global__ __aicore__ void AddCustom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR param)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    auto op = ATVC::Kernel::EleWiseOpTemplate<AddComputeFunc<Traits>>();
    op.Run(a, b, c, param);
}

class AtvcAddTestsuite : public testing::Test {
protected:
    void SetUp()
    {
        AscendC::SetGCoreType(2);
    }

    void TearDown()
    {
        AscendC::SetGCoreType(0);
    }
};

TEST_F(AtvcAddTestsuite, AtvcAddTestCase)
{
    uint8_t eleNum = 4;
    uint8_t aGm[eleNum] = {0};
    uint8_t bGm[eleNum] = {0};
    uint8_t cGm[eleNum] = {0};

    ATVC::EleWiseParam *eParam = new ATVC::EleWiseParam{};
    eParam->nBufferNum = 2;
    eParam->tilingData.blockNum = 1;
    eParam->tilingData.numPerBlock = 0;
    eParam->tilingData.tailBlockCnt = 0;
    eParam->tilingData.tiledCnt = 32;
    eParam->tilingData.tailElemCnt = 4;
    eParam->totalCnt = 4;

    AddCustom<ADD_OPTRAITS>(aGm, bGm, cGm, reinterpret_cast<uint8_t *>(eParam));
    for (int i = 0; i < eleNum; i++) {
        EXPECT_EQ(cGm[0], 0);
    }
}