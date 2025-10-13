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
#include "kernel_operator.h"

class IntegralConstantTest : public testing::Test {
protected:
    virtual void SetUp() {}
    void TearDown() {}

    static void SetUpTestCase()
    {
        std::cout << "IntegralConstantTest SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "IntegralConstantTest TearDownTestCase" << std::endl;
    }
};

// 测试 integral_constant 的基本特性
TEST(IntegralConstantTest, BasicProperties) {
    using IntTrue = AscendC::Std::integral_constant<int, 1>;
    using IntFalse = AscendC::Std::integral_constant<int, 0>;
    using CharA = AscendC::Std::integral_constant<char, 'A'>;

    // 测试 value 静态常量
    EXPECT_EQ(IntTrue::value, 1);
    EXPECT_EQ(IntFalse::value, 0);
    EXPECT_EQ(CharA::value, 'A');

    // 测试()操作符重载
    EXPECT_EQ(IntTrue()(), 1);
    EXPECT_EQ(IntFalse()(), 0);
    EXPECT_EQ(CharA()(), 'A');

    // 测试类型定义
    EXPECT_TRUE((AscendC::Std::is_same_v<typename IntTrue::value_type, int>));
    EXPECT_TRUE((AscendC::Std::is_same_v<typename IntTrue::type, IntTrue>));
    EXPECT_TRUE((AscendC::Std::is_same_v<typename CharA::value_type, char>));
}

// 测试 integral_constant 的布尔特化
TEST(IntegralConstantTest, BooleanSpecialization) {
    using TrueType = AscendC::Std::true_type;
    using FalseType = AscendC::Std::false_type;

    // 测试 value 静态常量
    EXPECT_TRUE(TrueType::value);
    EXPECT_FALSE(FalseType::value);

    // 测试()操作符重载
    EXPECT_TRUE(TrueType()());
    EXPECT_FALSE(FalseType()());

    // 测试类型定义
    EXPECT_TRUE((AscendC::Std::is_same_v<typename TrueType::value_type, bool>));
    EXPECT_TRUE((AscendC::Std::is_same_v<typename TrueType::type, TrueType>));
    EXPECT_TRUE((AscendC::Std::is_same_v<typename FalseType::type, FalseType>));
}

// 使用 integral_constant 实现简单的模板条件
template <typename T>
struct IsIntegralConstant : AscendC::Std::false_type {};

template <typename T, T v>
struct IsIntegralConstant<AscendC::Std::integral_constant<T, v>> : AscendC::Std::true_type {};

// 测试 integral_constant 在模板元编程中的使用
TEST(IntegralConstantTest, TemplateMetaprogramming) {
    EXPECT_TRUE((IsIntegralConstant<AscendC::Std::integral_constant<int, 42>>::value));
    EXPECT_TRUE((IsIntegralConstant<AscendC::Std::true_type>::value));
    EXPECT_TRUE((IsIntegralConstant<AscendC::Std::false_type>::value));
    EXPECT_FALSE((IsIntegralConstant<int>::value));
    EXPECT_FALSE((IsIntegralConstant<std::string>::value));

    // 测试使用 integral_constant 进行静态断言
    static_assert(AscendC::Std::integral_constant<int, 10>::value == 10, "Should be 10");
    static_assert(AscendC::Std::integral_constant<bool, true>::value, "Should be true");
}

// 测试 integral_constant 作为函数参数
TEST(IntegralConstantTest, AsFunctionParameter) {
    // 接受 integral_constant 作为参数的函数
    auto checkValue = [](auto constant) {
        return constant.value;
    };

    EXPECT_EQ(checkValue(AscendC::Std::integral_constant<int, 5>{}), 5);
    EXPECT_EQ(checkValue(AscendC::Std::integral_constant<char, 'Z'>{}), 'Z');
    EXPECT_EQ(checkValue(AscendC::Std::true_type{}), true);
    EXPECT_EQ(checkValue(AscendC::Std::false_type{}), false);

    // 带条件分支的函数
    auto conditionalFunction = [](auto cond) {
        if constexpr (cond.value) {
            return 100;
        } else {
            return 200;
        }
    };

    EXPECT_EQ(conditionalFunction(AscendC::Std::true_type{}), 100);
    EXPECT_EQ(conditionalFunction(AscendC::Std::false_type{}), 200);
}

// 测试基本属性
TEST(IntegralConstantTest, IntBasicProperties) {
    using Zero = AscendC::Std::Int<0>;
    using One = AscendC::Std::Int<1>;
    using Large = AscendC::Std::Int<0xFFFFFFFF>;

    // 验证 value 静态常量
    EXPECT_EQ(Zero::value, 0);
    EXPECT_EQ(One::value, 1);
    EXPECT_EQ(Large::value, 0xFFFFFFFF);

    // 验证类型定义
    EXPECT_TRUE((AscendC::Std::is_same_v<typename Zero::value_type, size_t>));
    EXPECT_TRUE((AscendC::Std::is_same_v<typename Zero::type, Zero>));
    EXPECT_TRUE((AscendC::Std::is_same_v<Zero, AscendC::Std::integral_constant<size_t, 0>>));

    // 验证()操作符重载
    EXPECT_EQ(Zero()(), 0);
    EXPECT_EQ(One()(), 1);
    EXPECT_EQ(Large()(), 0xFFFFFFFF);
}

// 测试编译时计算
TEST(IntegralConstantTest, CompileTimeOperations) {
    // 加法
    static_assert((AscendC::Std::Int<5>::value + AscendC::Std::Int<3>::value) == 8, "Addition failed");
    
    // 乘法
    static_assert((AscendC::Std::Int<4>::value * AscendC::Std::Int<6>::value) == 24, "Multiplication failed");
    
    // 比较
    static_assert(AscendC::Std::Int<10>::value > AscendC::Std::Int<5>::value, "Comparison failed");
    static_assert(AscendC::Std::Int<7>::value != AscendC::Std::Int<77>::value, "Equality check failed");
}

// 模板元函数：计算阶乘
template <typename N>
struct Factorial : AscendC::Std::Int<N::value * Factorial<AscendC::Std::Int<N::value - 1>>::value> {};

template <>
struct Factorial<AscendC::Std::Int<0>> : AscendC::Std::Int<1> {};

// 模板元函数：条件选择
template <typename Condition, typename Then, typename Else>
struct IfThenElse : Then {};

template <typename Then, typename Else>
struct IfThenElse<AscendC::Std::Int<0>, Then, Else> : Else {};

// 测试作为模板参数
TEST(IntegralConstantTest, TemplateParameter) {
    // 验证编译时计算结果
    static_assert(Factorial<AscendC::Std::Int<0>>::value == 1, "Factorial(0) failed");
    static_assert(Factorial<AscendC::Std::Int<1>>::value == 1, "Factorial(1) failed");
    static_assert(Factorial<AscendC::Std::Int<5>>::value == 120, "Factorial(5) failed");

    // 验证条件选择
    using Result1 = IfThenElse<AscendC::Std::Int<1>, AscendC::Std::Int<100>, AscendC::Std::Int<200>>;
    using Result2 = IfThenElse<AscendC::Std::Int<0>, AscendC::Std::Int<100>, AscendC::Std::Int<200>>;
    EXPECT_EQ(Result1::value, 100);
    EXPECT_EQ(Result2::value, 200);
}

// 验证与 AscendC::Std::enable_if 的兼容性
template <typename T>
auto getValue(T t, AscendC::Std::enable_if_t<T::value <= 10, int>* = nullptr) {
    return t.value * 2;
}

template <typename T>
auto getValue(T t, AscendC::Std::enable_if_t<T::value >= 10, int>* = nullptr) {
    return t.value / 2;
}

// 测试与标准库特性结合
TEST(IntegralConstantTest, StdTraitsIntegration) {
    // 验证与 AscendC::Std::is_same 的兼容性
    EXPECT_TRUE((AscendC::Std::is_same_v<AscendC::Std::Int<42>, AscendC::Std::integral_constant<size_t, 42>>));
    EXPECT_FALSE((AscendC::Std::is_same_v<AscendC::Std::Int<1>, AscendC::Std::Int<2>>));

    // 验证与 AscendC::Std::conditional 的兼容性
    using Result = AscendC::Std::conditional_t<
        AscendC::Std::is_same_v<AscendC::Std::Int<1>, AscendC::Std::Int<1>>,
        AscendC::Std::Int<100>,
        AscendC::Std::Int<200>
    >;
    EXPECT_EQ(Result::value, 100);

    EXPECT_EQ(getValue(AscendC::Std::Int<5>{}), 10);
    EXPECT_EQ(getValue(AscendC::Std::Int<20>{}), 10);
}

// 测试运行时使用场景
TEST(IntegralConstantTest, RuntimeUsage) {
    // 函数参数
    auto add = [](auto a, auto b) {
        return a.value + b.value;
    };
    
    EXPECT_EQ(add(AscendC::Std::Int<3>{}, AscendC::Std::Int<7>{}), 10);

    // 数组大小
    std::array<int, AscendC::Std::Int<5>::value> arr;
    EXPECT_EQ(arr.size(), 5);

    // 循环次数
    int sum = 0;
    for (size_t i = 0; i < AscendC::Std::Int<4>::value; ++i) {
        sum += i;
    }
    EXPECT_EQ(sum, 6); // 0+1+2+3=6
}