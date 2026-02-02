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
#include "tensor_api/stub/cce_stub.h"
#include "impl/experimental/tensor_api/tensor_api_impl.h"


class Tensor_Api_Coord : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {}
    void TearDown() {}
};

template <size_t v>
using Int = AscendC::Std::integral_constant<size_t, v>; 

TEST_F(Tensor_Api_Coord, MakeCoordOperation)
{
    constexpr int M = 11;
    constexpr int N = 12;
    constexpr int blockM = 13;
    constexpr int blockN = 14;

    auto coord = AscendC::MakeCoord(Int<20>{}, Int<30>{});
    auto shape = AscendC::MakeShape(AscendC::MakeShape(Int<blockM>{}, Int<M/blockM>{}), AscendC::MakeShape(Int<blockN>{}, Int<N/blockN>{}));
    auto stride = AscendC::MakeStride(AscendC::MakeStride(Int<blockN>{}, Int<blockM*blockN>{}),AscendC::MakeStride(Int<1>{}, Int<M*blockN>{}));

    auto layout = AscendC::MakeLayout(shape, stride);
    auto index = layout(coord);
    EXPECT_EQ(index, 590);

    index = AscendC::Crd2Idx(coord, layout);
    EXPECT_EQ(index, 590);
}

TEST_F(Tensor_Api_Coord, Crd2IdxOperation)
{
    auto blockCoordM    = Int<11>{};
    auto blockCoordN    = Int<12>{};
    auto baseShapeM     = Int<13>{};
    auto baseShapeN     = Int<14>{};
    auto basestrideM    = Int<15>{};
    auto basestrideN    = Int<16>{};

    auto coord = AscendC::MakeCoord(blockCoordM, blockCoordN);
    auto shape = AscendC::MakeShape(AscendC::MakeShape(baseShapeM, baseShapeM), AscendC::MakeShape(baseShapeN, baseShapeN));
    auto stride = AscendC::MakeStride(AscendC::MakeStride(basestrideM, basestrideM),AscendC::MakeStride(basestrideN, basestrideN));
    
    auto index = AscendC::Crd2Idx(coord, shape, stride);
    EXPECT_EQ(index, 357);
}

TEST_F(Tensor_Api_Coord, Crd2IdxIntZeroOperation)
{
    auto blockCoordM    = Int<11>{};
    auto blockCoordN    = Int<12>{};
    auto baseShapeM     = Int<13>{};
    auto baseShapeN     = Int<14>{};
    auto basestrideM    = Int<15>{};
    auto basestrideN    = Int<16>{};

    auto coord = AscendC::MakeCoord(Int<0>{}, Int<0>{});
    auto shape = AscendC::MakeShape(AscendC::MakeShape(baseShapeM, baseShapeM), AscendC::MakeShape(baseShapeN, baseShapeN));
    auto stride = AscendC::MakeStride(AscendC::MakeStride(basestrideM, basestrideM),AscendC::MakeStride(basestrideN, basestrideN));
    
    auto index = AscendC::Crd2Idx(coord, shape, stride);
    EXPECT_EQ(index, 0);
}

TEST_F(Tensor_Api_Coord, Crd2IdxCoordSingleZeroOperation)
{
    auto blockCoordM    = Int<11>{};
    auto blockCoordN    = Int<12>{};
    auto baseShapeM     = Int<13>{};
    auto baseShapeN     = Int<14>{};
    auto basestrideM    = Int<15>{};
    auto basestrideN    = Int<16>{};

    auto coord = AscendC::MakeCoord(Int<0>{}, blockCoordN);
    auto shape = AscendC::MakeShape(AscendC::MakeShape(baseShapeM, baseShapeM), AscendC::MakeShape(baseShapeN, baseShapeN));
    auto stride = AscendC::MakeStride(AscendC::MakeStride(basestrideM, basestrideM),AscendC::MakeStride(basestrideN, basestrideN));
    
    auto index = AscendC::Crd2Idx(coord, shape, stride);
    EXPECT_EQ(index, 192);
}
