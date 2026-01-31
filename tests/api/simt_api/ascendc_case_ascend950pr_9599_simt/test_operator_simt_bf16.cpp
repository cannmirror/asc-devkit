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
#include <type_traits>
#include <random>
#include "simt_compiler_stub.h"
#include "kernel_operator.h"
#include "simt_api/asc_bf16.h"

using namespace std;
using namespace AscendC;
using namespace AscendC::Simt;

// ================================ Test bfloat16_t start ===============================
struct BF16Params {
    int32_t mode;
};

class BF16Testsuite : public testing::Test, public testing::WithParamInterface<BF16Params> {
protected:
    void SetUp() {}
    void TearDown() {}
};

TEST_F(BF16Testsuite, MathApiTest)
{
    float x = static_cast<float>(rand()) / RAND_MAX;
    bfloat16_t x_h = bfloat16_t(x);

    bfloat16_t result = hcos(x_h);
    bfloat16_t expect = bfloat16_t(cosf((float)x_h));
    EXPECT_EQ(expect, result);
    
    result = hsin(x_h);
    expect = bfloat16_t(sinf((float)x_h));
    EXPECT_EQ(expect, result);

    result = htanh(x_h);
    expect = bfloat16_t(tanhf((float)x_h));
    EXPECT_EQ(expect, result);

    result = hexp(x_h);
    expect = bfloat16_t(expf((float)x_h));
    EXPECT_EQ(expect, result);

    result = hexp2(x_h);
    expect = bfloat16_t(exp2f((float)x_h));
    EXPECT_EQ(expect, result);
    
    result = hexp10(x_h);
    expect = bfloat16_t(powf(10.0, (float)x_h));
    EXPECT_EQ(expect, result);

    result = hrcp(x_h);
    if (x_h != bfloat16_t(0)) {
        expect = bfloat16_t(1.0f / (float)x_h);
        EXPECT_EQ(expect, result);
    }
}
// ================================ Test bfloat16_t end ===============================

// ================================ Test bfloat16x2_t start ===============================
struct BF162Params {
    int32_t mode;
};

class BF162Testsuite : public testing::Test, public testing::WithParamInterface<BF162Params> {
protected:
    void SetUp() {}
    void TearDown() {}
};

TEST_F(BF162Testsuite, MathApiTest_bfloat16x2t)
{
    float x = static_cast<float>(rand()) / RAND_MAX;
    float y = static_cast<float>(rand()) / RAND_MAX;
    bfloat16_t x_h = bfloat16_t(x);
    bfloat16_t y_h = bfloat16_t(y);
    bfloat16x2_t xy_h2 = {x_h, y_h};

    bfloat16x2_t result = h2cos(xy_h2);
    bfloat16_t expect1 = bfloat16_t(cosf((float)x_h));
    bfloat16_t expect2 = bfloat16_t(cosf((float)y_h));
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);
    
    result = h2sin(xy_h2);
    expect1 = bfloat16_t(sinf((float)x_h));
    expect2 = bfloat16_t(sinf((float)y_h));
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);

    result = h2tanh(xy_h2);
    expect1 = bfloat16_t(tanhf((float)x_h));
    expect2 = bfloat16_t(tanhf((float)y_h));
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);

    result = h2exp(xy_h2);
    expect1 = bfloat16_t(expf((float)x_h));
    expect2 = bfloat16_t(expf((float)y_h));
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);

    result = h2exp2(xy_h2);
    expect1 = bfloat16_t(exp2f((float)x_h));
    expect2 = bfloat16_t(exp2f((float)y_h));
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);
    
    result = h2exp10(xy_h2);
    expect1 = bfloat16_t(powf(10.0f, (float)x_h));
    expect2 = bfloat16_t(powf(10.0f, (float)y_h));
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);

    result = h2log(xy_h2);
    if (x_h > bfloat16_t(0) && y_h > bfloat16_t(0)) {
        expect1 = bfloat16_t(log((float)x_h));
        expect2 = bfloat16_t(log((float)y_h));
        EXPECT_EQ(expect1, result.x);
        EXPECT_EQ(expect2, result.y);
    }

    result = h2log2(xy_h2);
    if (x_h > bfloat16_t(0) && y_h > bfloat16_t(0)) {
        expect1 = bfloat16_t(log2((float)x_h));
        expect2 = bfloat16_t(log2((float)y_h));
        EXPECT_EQ(expect1, result.x);
        EXPECT_EQ(expect2, result.y);
    }

    result = h2log10(xy_h2);
    if (x_h > bfloat16_t(0) && y_h > bfloat16_t(0)) {
        expect1 = bfloat16_t(log10((float)x_h));
        expect2 = bfloat16_t(log10((float)y_h));
        EXPECT_EQ(expect1, result.x);
        EXPECT_EQ(expect2, result.y);
    }

    result = h2rcp(xy_h2);
    if (x_h != bfloat16_t(0) && y_h != bfloat16_t(0)) {
        expect1 = bfloat16_t(1.0f / (float)x_h);
        expect2 = bfloat16_t(1.0f / (float)y_h);
        EXPECT_EQ(expect1, result.x);
        EXPECT_EQ(expect2, result.y);
    }
    
    result = h2sqrt(xy_h2);
    if (x_h > bfloat16_t(0) && y_h > bfloat16_t(0)) {
        expect1 = bfloat16_t(sqrt((float)x_h));
        expect2 = bfloat16_t(sqrt((float)y_h));
        EXPECT_EQ(expect1, result.x);
        EXPECT_EQ(expect2, result.y);
    }

    result = h2rsqrt(xy_h2);
    if (x_h > bfloat16_t(0) && y_h > bfloat16_t(0)) {
        expect1 = bfloat16_t(1.0f / sqrt((float)x_h));
        expect2 = bfloat16_t(1.0f / sqrt((float)y_h));
        EXPECT_EQ(expect1, result.x);
        EXPECT_EQ(expect2, result.y);
    }

    result = h2ceil(xy_h2);
    expect1 = bfloat16_t(ceil((float)x_h));
    expect2 = bfloat16_t(ceil((float)y_h));
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);

    result = h2floor(xy_h2);
    expect1 = bfloat16_t(floor((float)x_h));
    expect2 = bfloat16_t(floor((float)y_h));
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);

    result = h2rint(xy_h2);
    expect1 = bfloat16_t(rint((float)x_h));
    expect2 = bfloat16_t(rint((float)y_h));
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);

    result = h2trunc(xy_h2);
    expect1 = bfloat16_t(trunc((float)x_h));
    expect2 = bfloat16_t(trunc((float)y_h));
    EXPECT_EQ(expect1, result.x);
    EXPECT_EQ(expect2, result.y);
}
// ================================ Test bfloat16_t end ===============================