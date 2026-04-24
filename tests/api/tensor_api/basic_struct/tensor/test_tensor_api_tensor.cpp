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
#include "include/tensor_api/tensor.h"

class Tensor_Api_Tensor_Struct : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {}
    void TearDown() {}
};

TEST_F(Tensor_Api_Tensor_Struct, TestLocalTensorStruct)
{
    using namespace AscendC::Te;

    __gm__ float data[6] = {0, 1, 2, 3, 4, 5};
    auto tensor = MakeTensor(MakeMemPtr<Location::GM>(data), MakeFrameLayout<NDLayoutPtn, LayoutTraitDefault<float>>(2, 3));

    EXPECT_EQ(tensor.Tensor().Data(), tensor.Data());
    EXPECT_EQ(tensor.Engine().Begin(), tensor.Data());
    EXPECT_EQ(tensor.Size(), 6);
    EXPECT_EQ(tensor.Capacity(), 6);
    EXPECT_EQ(AscendC::Std::get<0>(tensor.Shape()), 2);
    EXPECT_EQ(AscendC::Std::get<1>(tensor.Shape()), 3);
    EXPECT_EQ(AscendC::Std::get<0>(tensor.Stride()), 3);
    EXPECT_EQ(AscendC::Std::get<1>(tensor.Stride()), 1);
    EXPECT_EQ(tensor[MakeCoord(1, 2)], 5);
}

TEST_F(Tensor_Api_Tensor_Struct, TestLocalTensorCoord)
{
    using namespace AscendC::Te;

    __gm__ float data[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    auto layout = MakeFrameLayout<NDLayoutPtn, LayoutTraitDefault<float>>(3, 4);
    auto tensor = MakeTensor(MakeMemPtr<Location::GM>(data), layout);
    auto subTensor = tensor(MakeCoord(1, 1));

    EXPECT_EQ(subTensor.Data(), tensor.Data() + layout(MakeCoord(1, 1)));
    EXPECT_EQ(AscendC::Std::get<0>(subTensor.Shape()), 2);
    EXPECT_EQ(AscendC::Std::get<1>(subTensor.Shape()), 3);
    EXPECT_EQ(subTensor[MakeCoord(1, 2)], 11);
}
