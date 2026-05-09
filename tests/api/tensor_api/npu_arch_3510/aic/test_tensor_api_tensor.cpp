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

class Tensor_Api_Tensor_3510 : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
    virtual void SetUp() {}
    void TearDown() {}
};

TEST_F(Tensor_Api_Tensor_3510, SetL2CacheHint)
{
    using namespace AscendC::Te;

    constexpr uint32_t TILE_LENGTH = 8;
    __gm__ float data[TILE_LENGTH] = {0, 1, 2, 3, 4, 5, 6, 7};
    auto ptr = MakeMemPtr<Location::GM>(data);
    auto tensor = MakeTensor(ptr, MakeShape(AscendC::Std::Int<2>{}, AscendC::Std::Int<2>{}, AscendC::Std::Int<2>{}),
                             MakeStride(AscendC::Std::Int<4>{}, AscendC::Std::Int<2>{}, AscendC::Std::Int<1>{}));

    tensor.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    EXPECT_EQ(tensor.Engine().GetCacheMode(), static_cast<uint8_t>(CacheMode::CACHE_MODE_DISABLE));
    
    tensor.SetL2CacheHint(CacheMode::CACHE_MODE_NORMAL);
    EXPECT_EQ(tensor.Engine().GetCacheMode(), static_cast<uint8_t>(CacheMode::CACHE_MODE_NORMAL));

    tensor.SetL2CacheHint(CacheMode::CACHE_MODE_LAST);
    EXPECT_EQ(tensor.Engine().GetCacheMode(), static_cast<uint8_t>(CacheMode::CACHE_MODE_LAST));

    tensor.SetL2CacheHint(CacheMode::CACHE_MODE_PERSISTENT);
    EXPECT_EQ(tensor.Engine().GetCacheMode(), static_cast<uint8_t>(CacheMode::CACHE_MODE_PERSISTENT));
}
