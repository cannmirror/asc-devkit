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
#include "tensor_api/tensor.h"

class Tensor_Api_Vector_Cast_3510 : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}

    void SetUp() override {}

    void TearDown() override {}
};

template<typename DstDataType, typename SrcDataType, typename Func>
__aicore__ inline void TestTransformBinary(__gm__ DstDataType* z, __gm__ SrcDataType* x, __ubuf__ DstDataType zUB[2048], __ubuf__ SrcDataType xUB[2048])
{
    constexpr uint32_t TILE_LENGTH = 2048;
    constexpr uint32_t BLK_NUM = 1;

    using namespace AscendC::Te;
    asc_init();

    constexpr uint8_t cacheMode = 0;
    constexpr uint32_t burstLength = 0;
    constexpr uint64_t srcStride = 0;
    constexpr uint32_t dstStride = 0;

    auto gmPtrX = MakeMemPtr<Location::GM>(x);
    auto gmPtrZ = MakeMemPtr<Location::GM>(z);

    auto xGm = MakeTensor(gmPtrX, MakeFrameLayout<NDLayoutPtn>(_1{}, AscendC::Std::Int<TILE_LENGTH>{}));
    auto zGm = MakeTensor(gmPtrZ, MakeFrameLayout<NDLayoutPtn>(_1{}, AscendC::Std::Int<TILE_LENGTH>{}));

    auto xLocal = MakeTensor(MakeMemPtr(xUB), MakeFrameLayout<NDLayoutPtn>(_1{}, AscendC::Std::Int<TILE_LENGTH>{}));
    auto zLocal = MakeTensor(MakeMemPtr(zUB), MakeFrameLayout<NDLayoutPtn>(_1{}, AscendC::Std::Int<TILE_LENGTH>{}));

    asc_copy_gm2ub_align(xLocal.Data().Get(), xGm.Data().Get(), BLK_NUM, TILE_LENGTH * sizeof(SrcDataType), 0, 0, true, cacheMode, srcStride, dstStride);

    asc_sync_notify(PIPE_MTE2, PIPE_V, EVENT_ID0);
    asc_sync_wait(PIPE_MTE2, PIPE_V, EVENT_ID0);

    Transform<Func>(zLocal, xLocal);

    asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID0);
    asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID0);

    asc_copy_ub2gm_align(zGm.Data().Get(), zLocal.Data().Get(), BLK_NUM, TILE_LENGTH * sizeof(DstDataType), cacheMode, srcStride, dstStride);
}

#define VECTOR_CAST_3510(Function, DstDataType, SrcDataType) \
TEST_F(Tensor_Api_Vector_Cast_3510, VECTOR_##Function##_##DstDataType##_##SrcDataType) \
{   \
    constexpr uint32_t TILE_LENGTH = 2048;  \
    \
    __gm__ SrcDataType x[TILE_LENGTH] = {0};  \
    __gm__ DstDataType z[TILE_LENGTH] = {0};  \
    \
    __ubuf__ SrcDataType xUB[TILE_LENGTH] = {0};  \
    __ubuf__ DstDataType zUB[TILE_LENGTH] = {0};  \
    \
    TestTransformBinary<DstDataType, SrcDataType, AscendC::Te::Inst::Function>(z, x, zUB, xUB);   \
    EXPECT_EQ(z[0], zUB[0]); \
}

VECTOR_CAST_3510(Ceil, half, half)
VECTOR_CAST_3510(Ceil, float, float)
VECTOR_CAST_3510(Ceil, bfloat16_t, bfloat16_t)

VECTOR_CAST_3510(U82U16, uint16_t, uint8_t)