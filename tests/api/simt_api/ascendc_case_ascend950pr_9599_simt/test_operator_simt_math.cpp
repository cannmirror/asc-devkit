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
#include <type_traits>
#include <cmath>
#include "kernel_operator.h"

using namespace std;
using namespace AscendC;
using namespace AscendC::Simt;

class MathComputeTestsuite : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}
};

TEST_F(MathComputeTestsuite, HalfComputeTestCaseAbs)
{
    half x = 123.0;
    half y = -123.0;
    EXPECT_EQ(static_cast<half>(123.0), Simt::Abs(x));
    EXPECT_EQ(static_cast<half>(123.0), Simt::Abs(y));
}

TEST_F(MathComputeTestsuite, HalfComputeTestCaseFma)
{
    half x = 1.0;
    half y = 2.0;
    half z = 3.0;
    EXPECT_EQ(static_cast<half>(5.0), Simt::Fma(x, y, z));
}

TEST_F(MathComputeTestsuite, FloatComputeTestCaseCover)
{
    int32_t x = 1;
    int32_t y = 2;
    int32_t z = Simt::Max(x, y);
    EXPECT_EQ(z, y);
    z = Simt::Min(x, y);
    EXPECT_EQ(z, x);
    float f_1 = 1.0f;
    float f_2 = 2.0f;
    float f_3 = Simt::Fdim(f_1, f_2);
    f_3 = Simt::RemQuo(f_1, f_2, &z);
    f_3 = Simt::Mod(f_1, f_2);
    f_3 = Simt::Remainder(f_1, f_2);
    f_3 = Simt::CopySign(f_1, f_2);
    f_3 = Simt::NearbyInt(f_1);
    f_3 = Simt::NextAfter(f_1, f_2);
    f_3 = Simt::ScaLbn(f_1, z);
    f_3 = Simt::Tan(f_1);
    f_3 = Simt::Tanh(f_1);
    f_3 = Simt::Atan(f_1);
    f_3 = Simt::Atanh(f_1);
    f_3 = Simt::Exp(f_1);
    f_3 = Simt::Log(f_1);
    f_3 = Simt::Cos(f_1);
    f_3 = Simt::Cosh(f_1);
    f_3 = Simt::Sin(f_1);
    f_3 = Simt::Sinh(f_1);
}

TEST_F(MathComputeTestsuite, HalfComputeTestCompare)
{
    half x = 1.0;
    half y = 2.0;
    half z = 3.0;
    bool result = Simt::IsNan(x);
    result = Simt::IsInf(x);
    EXPECT_EQ(result, false);
}
