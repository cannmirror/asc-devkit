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

class ConditionalTest : public testing::Test {
protected:
    virtual void SetUp() {}
    void TearDown() {}

    static void SetUpTestCase()
    {
        std::cout << "ConditionalTest SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "ConditionalTest TearDownTestCase" << std::endl;
    }
};

// 测试 AscendC::Std::conditional 在布尔条件为 true 时选择第一个类型（int 和 double）
TEST_F(ConditionalTest, ConditionalTrueIntDouble) {
    constexpr bool condition = true;
    using ResultType = typename AscendC::Std::conditional<condition, int, double>::type;
    EXPECT_TRUE((AscendC::Std::is_same<ResultType, int>::value));
}

// 测试 AscendC::Std::conditional 在布尔条件为 false 时选择第二个类型（int 和 double）
TEST_F(ConditionalTest, ConditionalFalseIntDouble) {
    constexpr bool condition = false;
    using ResultType = typename AscendC::Std::conditional<condition, int, double>::type;
    EXPECT_TRUE((AscendC::Std::is_same<ResultType, double>::value));
}

// 测试 AscendC::Std::conditional 在布尔条件为 true 时选择第一个类型（short 和 long）
TEST_F(ConditionalTest, ConditionalTrueShortLong) {
    constexpr bool condition = true;
    using ResultType = typename AscendC::Std::conditional<condition, short, long>::type;
    EXPECT_TRUE((AscendC::Std::is_same<ResultType, short>::value));
}

// 测试 AscendC::Std::conditional 在布尔条件为 false 时选择第二个类型（short 和 long）
TEST_F(ConditionalTest, ConditionalFalseShortLong) {
    constexpr bool condition = false;
    using ResultType = typename AscendC::Std::conditional<condition, short, long>::type;
    EXPECT_TRUE((AscendC::Std::is_same<ResultType, long>::value));
}

// 测试 AscendC::Std::conditional 在布尔条件为 true 时选择第一个类型（char 和 float）
TEST_F(ConditionalTest, ConditionalTrueCharFloat) {
    constexpr bool condition = true;
    using ResultType = typename AscendC::Std::conditional<condition, char, float>::type;
    EXPECT_TRUE((AscendC::Std::is_same<ResultType, char>::value));
}

// 测试 AscendC::Std::conditional 在布尔条件为 false 时选择第二个类型（char 和 float）
TEST_F(ConditionalTest, ConditionalFalseCharFloat) {
    constexpr bool condition = false;
    using ResultType = typename AscendC::Std::conditional<condition, char, float>::type;
    EXPECT_TRUE((AscendC::Std::is_same<ResultType, float>::value));
}

// 测试使用常量表达式函数作为条件，结果为 true
constexpr bool constantTrueFunction() { return true; }
TEST_F(ConditionalTest, ConditionalTrueFromFunction) {
    using ResultType = typename AscendC::Std::conditional<constantTrueFunction(), int, long long>::type;
    EXPECT_TRUE((AscendC::Std::is_same<ResultType, int>::value));
}

// 测试使用常量表达式函数作为条件，结果为 false
constexpr bool constantFalseFunction() { return false; }
TEST_F(ConditionalTest, ConditionalFalseFromFunction) {
    using ResultType = typename AscendC::Std::conditional<constantFalseFunction(), int, long long>::type;
    EXPECT_TRUE((AscendC::Std::is_same<ResultType, long long>::value));
}

// 测试使用枚举类型作为条件，结果为 true
enum class ConditionEnum { TrueValue = true, FalseValue = false };
TEST_F(ConditionalTest, ConditionalTrueFromEnum) {
    using ResultType = typename AscendC::Std::conditional<static_cast<bool>(ConditionEnum::TrueValue), short, double>::type;
    EXPECT_TRUE((AscendC::Std::is_same<ResultType, short>::value));
}

// 测试使用枚举类型作为条件，结果为 false
TEST_F(ConditionalTest, ConditionalFalseFromEnum) {
    using ResultType = typename AscendC::Std::conditional<static_cast<bool>(ConditionEnum::FalseValue), short, double>::type;
    EXPECT_TRUE((AscendC::Std::is_same<ResultType, double>::value));
}

